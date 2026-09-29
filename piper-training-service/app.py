"""Piper Voice Training Service.

Orchestrates VITS neural-network training for custom Piper TTS voices.
Workflow: upload audio -> STT segmentation -> mel generation -> VITS training -> ONNX export.
"""

import os
import json
import asyncio
import aiofiles
import math
import tempfile
import shutil
import numpy as np
import uuid
import logging
import inspect
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
import librosa
import soundfile as sf
import aiohttp

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.datastructures import Headers
from pydantic import BaseModel
import uvicorn

import training_utils as _training_utils
from training_pipeline import OptimizedTrainingPipeline, TrainingCancelled
from data_processor import DataProcessor
from model_exporter import ModelExporter
from audio_sources import AudioSourceError, max_upload_bytes
from job_control import ActiveRuns, ACTIVE_STATUSES, TERMINAL_STATUSES
from phonemization import PhonemizationError, normalize_language, stt_language
from stt_processor import STTError, STTProcessor, confidence_from_result, default_min_confidence
from validation import (
    safe_name as _safe_name,
    coerce_resume_int as _coerce_resume_int,
    coerce_resume_path as _coerce_resume_path,
    validate_epochs as _validate_epochs,
)

# Configure logging so all logger.info() calls actually output to stdout
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _env_number(name: str, default, cast):
    """A positive number from the environment, or `default` when unset or unusable."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = cast(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using %s", name, raw, default)
        return default
    if not 0 < value < float("inf"):        # also rejects nan, which compares false to everything
        logger.warning("%s=%r must be a positive number; using %s", name, raw, default)
        return default
    return value


# What the trainer is, taken from training_utils when it says. Surfaced on
# /health, in every job status and in the startup log so that nobody mistakes an
# exported bundle for a finished Piper voice. getattr with no default would make
# an older training_utils a startup crash; the honesty fields are simply absent.
TRAINER_KIND = getattr(_training_utils, "TRAINER_KIND", None)
TRAINER_CAVEAT = getattr(_training_utils, "TRAINER_CAVEAT", None) if TRAINER_KIND else None
if TRAINER_KIND and not TRAINER_CAVEAT:
    TRAINER_CAVEAT = (
        "experimental: the text encoder is only trained through a duration loss with synthetic "
        "targets; exported voices are NOT production quality"
    )

# Jobs that may train at the same time. One: the trainer takes the whole GPU, and
# two jobs on one card each fail with OOM half the time.
MAX_CONCURRENT_JOBS = int(_env_number("TRAINING_MAX_CONCURRENT", 1, int))
active_runs = ActiveRuns(MAX_CONCURRENT_JOBS)

# The dataset language used when a request does not name one. It used to be
# implied by three different code paths ("en" in one, "de" in another).
_default_language_raw = os.getenv("TRAINING_DEFAULT_LANGUAGE", "de").strip() or "de"
try:
    DEFAULT_LANGUAGE = normalize_language(_default_language_raw)
except ValueError as _exc:
    # A bad default would silently mis-phonemise every job that does not say.
    raise RuntimeError(f"TRAINING_DEFAULT_LANGUAGE: {_exc}") from _exc

# Longest transcript accepted for one segment (/prepare-dataset). Real segments
# are 1-15 s of speech, a few hundred characters at most.
MAX_TEXT_CHARS = int(_env_number("MAX_TEXT_CHARS", 1000, int))

# The mel features and the model are fixed at this rate (training_pipeline
# _get_config); audio cut or resampled at another rate would be paired with a
# filterbank built for this one.
SUPPORTED_SAMPLE_RATE = 22050

# How long shutdown waits for running jobs to notice they were asked to stop.
# Must stay below the container's stop grace period, or the kill lands first.
SHUTDOWN_GRACE_S = _env_number("TRAINING_SHUTDOWN_GRACE_S", 30.0, float)

# Bodies that are not audio uploads (forms, /prepare-dataset JSON) are small.
MAX_SMALL_BODY_BYTES = 16 * 1024 * 1024
_MULTIPART_SLACK_BYTES = 1024 * 1024

# Strong references to the running job tasks; the loop only keeps weak ones.
_runner_tasks: set = set()
# Datasets being rewritten by a maintenance endpoint: model name -> what is doing it.
_dataset_ops: Dict[str, str] = {}


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Restore interrupted jobs on startup; stop running jobs cleanly on shutdown."""
    if TRAINER_KIND:
        logger.warning("Trainer kind '%s'. %s", TRAINER_KIND, TRAINER_CAVEAT)
    logger.info(
        "Training service starting: default language %s, max concurrent jobs %d, max upload %.0f MB",
        DEFAULT_LANGUAGE, MAX_CONCURRENT_JOBS, max_upload_bytes() / (1024 * 1024),
    )
    await restore_interrupted_jobs()
    yield
    await _stop_running_jobs()


app = FastAPI(
    title="Piper Voice Training Service",
    description="VITS neural network training pipeline for custom Piper TTS voice models",
    lifespan=_lifespan,
)

class _BodyTooLarge(Exception):
    """Raised into the app from `receive()` once the body passes its limit."""


def _body_limit_for(path: str) -> int:
    """Largest request body a route accepts: audio uploads get MAX_UPLOAD_MB, the rest 16 MB."""
    if path in ("/train", "/test-upload"):
        return max_upload_bytes() + _MULTIPART_SLACK_BYTES
    return MAX_SMALL_BODY_BYTES


class _BodyLimitMiddleware:
    """Refuse an oversized request body with 413 before it is spooled to disk.

    Starlette writes a multipart upload to a temp file before the handler runs,
    so a cap inside the handler cannot stop a client filling the disk. Two
    checks, because the client controls both: Content-Length rejects an honest
    oversized request without reading a byte, and a running count catches a
    chunked upload or an understated length.
    """

    def __init__(self, app):
        self.app = app

    @staticmethod
    async def _reject(scope, receive, send, limit: int):
        response = JSONResponse(
            status_code=413,
            content={"detail": f"Request body too large. Maximum is {limit / (1024 * 1024):g} MB."},
            # The unread rest of the body is still on the wire.
            headers={"Connection": "close"},
        )
        await response(scope, receive, send)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] in ("GET", "HEAD", "OPTIONS"):
            await self.app(scope, receive, send)
            return

        limit = _body_limit_for(scope["path"])
        try:
            declared = int(Headers(scope=scope).get("content-length", ""))
        except ValueError:
            declared = None
        if declared is not None and declared > limit:
            await self._reject(scope, receive, send, limit)
            return

        received = 0
        exceeded = False
        started = False

        async def limited_receive():
            nonlocal received, exceeded
            if exceeded:
                raise _BodyTooLarge()
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    exceeded = True
                    raise _BodyTooLarge()
            return message

        async def guarded_send(message):
            nonlocal started
            if exceeded and not started:
                return  # the app's own answer to the aborted read; the 413 replaces it
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except _BodyTooLarge:
            pass
        if exceeded and not started:
            await self._reject(scope, receive, send, limit)


# Added before CORS so that CORS wraps it and a 413 still carries the CORS headers.
app.add_middleware(_BodyLimitMiddleware)

allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",")] if allowed_origins_str else ["*"]
allow_credentials = os.getenv("ALLOW_CREDENTIALS", "false").strip().lower() in {"1", "true", "yes", "on"}
if "*" in allowed_origins and allow_credentials:
    allow_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

training_pipeline = OptimizedTrainingPipeline()
data_processor = DataProcessor()
model_exporter = ModelExporter()

PIPER_TTS_SERVICE_URL = os.getenv("PIPER_TTS_SERVICE_URL", "http://piper-tts-service:5000")
SHARED_MODELS_DIR = os.getenv("SHARED_MODELS_DIR", "/app/shared_models")

# Where this service sends audio for transcription.
#
# compose has always set STT_SERVICE_URL here and this module never read it, so
# the documented knob did nothing. The URL came from a form field on /train and
# /retrain-from-segments instead — which is the wrong source twice over. It is
# deployment configuration, not per-request input; and honouring it means any
# caller can point the service at a host of their choosing, have it POST the
# upload there, and read the resulting connection error back out of the job
# status. That is a working probe of whatever network the container sits on.
#
# The env is now the source of truth. A caller may still pass the field, but
# only with the value the deployment is already configured for, so existing
# clients that echo the default keep working.
STT_SERVICE_URL = os.getenv("STT_SERVICE_URL", "http://stt-service:8000")
ALLOW_CLIENT_STT_URL = os.getenv("ALLOW_CLIENT_STT_URL", "false").strip().lower() in {"1", "true", "yes", "on"}


def resolve_stt_service_url(requested: Optional[str]) -> str:
    """Return the STT URL to use, rejecting an unexpected client override."""
    candidate = (requested or "").strip().rstrip("/")
    if not candidate or candidate == STT_SERVICE_URL.rstrip("/"):
        return STT_SERVICE_URL
    if ALLOW_CLIENT_STT_URL:
        logger.warning("Using client-supplied STT URL %r (ALLOW_CLIENT_STT_URL=true)", candidate)
        return candidate
    raise HTTPException(
        status_code=400,
        detail=(
            "stt_service_url must match this deployment's configured STT service "
            f"({STT_SERVICE_URL}). Set it with the STT_SERVICE_URL environment "
            "variable, or set ALLOW_CLIENT_STT_URL=true to accept per-request URLs."
        ),
    )


def _build_deployment_target_registry() -> dict:
    """Describe where trained model bundles can be deployed after export."""
    targets = {
        "none": {
            "display_name": "Manual download only",
            "deployment_contract": "manual-artifact-v1",
            "kind": "manual",
            "capabilities": ["download_only"],
        },
        "piper-volume": {
            "display_name": "Piper shared volume",
            "deployment_contract": "piper-shared-volume-v1",
            "kind": "tts-runtime",
            "capabilities": ["copy_bundle", "refresh_runtime"],
            "shared_models_dir": SHARED_MODELS_DIR,
            "runtime_url": PIPER_TTS_SERVICE_URL,
            "refresh_path": "/refresh_voices",
        },
        "piper-http": {
            "display_name": "Piper upload API",
            "deployment_contract": "piper-upload-api-v1",
            "kind": "tts-runtime",
            "capabilities": ["upload_bundle", "delete_model", "refresh_runtime"],
            "runtime_url": PIPER_TTS_SERVICE_URL,
            "upload_path": "/upload_model",
            "delete_path": "/voice/{model_name}",
            "refresh_path": "/refresh_voices",
        },
    }

    registry = {
        "default_target": os.getenv("DEFAULT_DEPLOYMENT_TARGET", "piper-volume"),
        "targets": targets,
    }

    override = os.getenv("DEPLOYMENT_TARGETS_JSON", "").strip()
    if override:
        parsed = json.loads(override)
        registry["targets"].update(parsed.get("targets", {}))
        if parsed.get("default_target"):
            registry["default_target"] = parsed["default_target"]

    return registry


DEPLOYMENT_TARGET_REGISTRY = _build_deployment_target_registry()


class TrainingRequest(BaseModel):
    """Parameters for starting or resuming a VITS training job."""

    model_name: str
    language: str = "en"
    sample_rate: int = 22050
    quality: str = "medium"  # low, medium, high
    speaker_name: Optional[str] = None
    # Resolved from STT_SERVICE_URL at request time; never taken from a client.
    stt_service_url: Optional[str] = None
    audio_files: Optional[List[str]] = None # To pass file info internally
    epochs: int = 1000
    batch_size: int = 32
    deployment_target: Optional[str] = None
    
class SegmentData(BaseModel):
    """Single STT-derived segment used for dataset preparation."""

    audio_path: str
    text: str
    start_time: float
    end_time: float
    
class DatasetUpload(BaseModel):
    """Payload for creating a dataset from pre-segmented clips."""

    segments: List[SegmentData]
    model_name: str
    # Dataset language ("de"). Omitted: TRAINING_DEFAULT_LANGUAGE.
    language: Optional[str] = None

class TrainingStatus(BaseModel):
    """In-memory status record returned by the training job API."""

    job_id: str
    status: str
    progress: float
    current_epoch: int
    total_epochs: int
    loss: Optional[float]
    message: str
    model_name: Optional[str] = None
    deployment_target: Optional[str] = None
    # What produced the weights (training_utils.TRAINER_KIND) and what that
    # means for the result; null when training_utils does not say.
    trainer_kind: Optional[str] = None
    trainer_caveat: Optional[str] = None

# Store training jobs
training_jobs = {}

# Old chunking functions removed - now using AudioSegmenter for all STT processing

def resolve_deployment_target(target_id: Optional[str] = None) -> tuple[str, dict]:
    """Resolve a deployment target id to a registry entry."""
    resolved_id = (target_id or DEPLOYMENT_TARGET_REGISTRY["default_target"]).strip()
    target = DEPLOYMENT_TARGET_REGISTRY["targets"].get(resolved_id)
    if not target:
        raise HTTPException(status_code=400, detail=f"Unknown deployment target: {resolved_id}")
    return resolved_id, target


async def refresh_deployment_target(target: dict):
    """Refresh a runtime after deployment when the target supports it."""
    runtime_url = target.get("runtime_url")
    refresh_path = target.get("refresh_path")
    if not runtime_url or not refresh_path:
        return

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{runtime_url}{refresh_path}",
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"Refresh failed: {resp.status} - {body}")


async def deploy_model_bundle(job_id: str, model_name: str, onnx_path: Path, target_id: Optional[str] = None) -> dict:
    """Deploy an exported model bundle to the selected target."""
    resolved_target_id, target = resolve_deployment_target(target_id)
    config_path = onnx_path.parent / f"{onnx_path.stem}.json"

    if resolved_target_id == "none":
        return {
            "target": resolved_target_id,
            "status": "skipped",
            "message": "Export completed; model bundle retained for manual download.",
        }

    if target.get("deployment_contract") == "piper-shared-volume-v1":
        shared_models_dir = Path(target["shared_models_dir"])
        custom_dir = shared_models_dir / "custom" / model_name

        # Off the event loop (the ONNX is tens of MB, on a network share more),
        # and through a temp name + rename: the Piper runtime scans this
        # directory, and must never find a half-copied .onnx.
        def _publish() -> None:
            custom_dir.mkdir(parents=True, exist_ok=True)
            for source, name, required in (
                (onnx_path, f"{model_name}.onnx", True),
                (config_path, f"{model_name}.json", False),
            ):
                if not required and not source.exists():
                    continue
                partial = custom_dir / f"{name}.partial"
                shutil.copy2(source, partial)
                os.replace(partial, custom_dir / name)

        await asyncio.to_thread(_publish)

        try:
            await refresh_deployment_target(target)
        except Exception as refresh_error:
            logger.warning(f"Deployment refresh failed for {resolved_target_id}: {refresh_error}")

        return {
            "target": resolved_target_id,
            "status": "deployed",
            "message": f"Model deployed to {target['display_name']}",
            "location": str(custom_dir),
        }

    if target.get("deployment_contract") == "piper-upload-api-v1":
        async with aiofiles.open(onnx_path, 'rb') as f:
            model_data = await f.read()

        config_data = None
        if config_path.exists():
            async with aiofiles.open(config_path, 'rb') as f:
                config_data = await f.read()

        async with aiohttp.ClientSession() as session:
            form = aiohttp.FormData()
            form.add_field('model_file', model_data, filename=f"{model_name}.onnx", content_type='application/octet-stream')
            if config_data is not None:
                form.add_field('config_file', config_data, filename=f"{model_name}.json", content_type='application/json')
            form.add_field('voice_name', model_name)
            form.add_field('model_name', model_name)

            async with session.post(
                f"{target['runtime_url']}{target['upload_path']}",
                data=form,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"Upload failed: {resp.status} - {body}")

        try:
            await refresh_deployment_target(target)
        except Exception as refresh_error:
            logger.warning(f"Deployment refresh failed for {resolved_target_id}: {refresh_error}")

        return {
            "target": resolved_target_id,
            "status": "deployed",
            "message": f"Model deployed to {target['display_name']}",
            "location": target['runtime_url'],
        }

    raise HTTPException(status_code=400, detail=f"Unsupported deployment contract: {target.get('deployment_contract')}")


async def remove_model_from_deployment_target(model_name: str, target_id: Optional[str] = None):
    """Remove a deployed model from the selected target when supported."""
    resolved_target_id, target = resolve_deployment_target(target_id)

    if resolved_target_id == "none":
        return

    if target.get("deployment_contract") == "piper-shared-volume-v1":
        model_dir = Path(target["shared_models_dir"]) / "custom" / model_name
        if model_dir.exists():
            await asyncio.to_thread(shutil.rmtree, model_dir)
        try:
            await refresh_deployment_target(target)
        except Exception as refresh_error:
            logger.warning(f"Deployment refresh failed for {resolved_target_id}: {refresh_error}")
        return

    if target.get("deployment_contract") == "piper-upload-api-v1":
        delete_path = target["delete_path"].format(model_name=model_name)
        async with aiohttp.ClientSession() as session:
            async with session.delete(
                f"{target['runtime_url']}{delete_path}",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status not in {200, 404}:
                    body = await resp.text()
                    logger.warning(f"Deployment delete failed: {resp.status} - {body}")
        return

    logger.info(f"No removal implementation for deployment target {resolved_target_id}")

def _model_name_from_disk(job_id: str) -> Optional[str]:
    """Read a job's model name from its checkpoint state file.

    The in-memory registry only survives until a restart, and completed jobs are
    deliberately not restored — so after any restart `delete_trained_model` could
    not tell which dataset belonged to the job it was deleting, and quietly left
    both the dataset and the deployed voice behind.
    """
    state_path = Path("checkpoints") / job_id / "job_state.json"
    try:
        with open(state_path) as f:
            return json.load(f).get("model_name")
    except (OSError, json.JSONDecodeError):
        return None


def _other_jobs_using_model(model_name: str, excluding_job_id: str) -> list[str]:
    """Job ids other than *excluding_job_id* that trained the same model name.

    Retraining a voice is the normal workflow, so several jobs routinely share
    one `data/<model_name>` directory and one deployed voice. Deleting any of
    them used to take the dataset and the live voice with it.
    """
    others: list[str] = []
    for state_file in Path("checkpoints").glob("*/job_state.json"):
        if state_file.parent.name == excluding_job_id:
            continue
        try:
            with open(state_file) as f:
                state = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if state.get("model_name") == model_name:
            others.append(state.get("job_id") or state_file.parent.name)

    for other_id, job in training_jobs.items():
        if other_id != excluding_job_id and job.model_name == model_name:
            if other_id not in others:
                others.append(other_id)
    return others


async def restore_interrupted_jobs():
    """Scan checkpoints/ for jobs interrupted by a container restart and restore them."""
    checkpoints_root = Path("checkpoints")
    if not checkpoints_root.exists():
        return
    for state_file in checkpoints_root.glob("*/job_state.json"):
        try:
            with open(state_file) as f:
                state = json.load(f)
            job_id = state.get("job_id")
            status = state.get("status", "unknown")
            if not job_id or status == "completed":
                continue  # Skip finished jobs
            epoch = state.get("epoch", 0)
            # A job that ended on its own keeps the outcome it had; only a job the
            # process died under (state still says "training") is "interrupted".
            if status == "failed":
                restored, message = "failed", f"Failed at epoch {epoch}: {state.get('error') or 'see the service log'}"
            elif status == "cancelled":
                restored, message = "cancelled", (
                    f"Cancelled at epoch {epoch} — use POST /resume-training to continue.")
            else:
                restored, message = "interrupted", (
                    f"Interrupted at epoch {epoch} — use POST /resume-training to continue.")
            # Restore into in-memory dict so /status and /jobs endpoints work
            training_jobs[job_id] = TrainingStatus(
                job_id=job_id,
                status=restored,
                progress=round(epoch / max(state.get("total_epochs", 1), 1) * 100, 1),
                current_epoch=epoch,
                total_epochs=state.get("total_epochs", 10000),
                loss=state.get("loss"),
                message=message,
                model_name=state.get("model_name"),
                trainer_kind=state.get("trainer_kind"),
                trainer_caveat=state.get("trainer_caveat"),
            )
            logger.info(f"Restored {restored} job {job_id} (epoch {epoch})")
        except Exception as e:
            logger.warning(f"Could not restore job from {state_file}: {e}")


# --- job bookkeeping -----------------------------------------------------------
#
# Every phase change of a job goes through _advance, and every claim on the
# training slot through _claim. Both exist because the endpoints, the training
# thread and the user's DELETE /job all write the same status string: without a
# rule about who may overwrite what, a job the user cancelled could be flipped
# back to "training" by the very task the cancel was meant to stop.

def _new_job(job_id: str, status: str = "initializing", *, model_name: str, total_epochs: int, message: str,
             deployment_target: Optional[str], progress: float = 0, current_epoch: int = 0,
             loss: Optional[float] = None) -> TrainingStatus:
    """Register a job record, stamped with what the trainer is."""
    job = TrainingStatus(
        job_id=job_id,
        status=status,
        progress=progress,
        current_epoch=current_epoch,
        total_epochs=total_epochs,
        loss=loss,
        message=message,
        model_name=model_name,
        deployment_target=deployment_target,
        trainer_kind=TRAINER_KIND,
        trainer_caveat=TRAINER_CAVEAT,
    )
    training_jobs[job_id] = job
    return job


def _is_cancelled(job_id: str) -> bool:
    """Has the user cancelled this job?"""
    job = training_jobs.get(job_id)
    return job is not None and job.status == "cancelled"


def _advance(job_id: str, status: str, message: Optional[str] = None,
             progress: Optional[float] = None) -> None:
    """Move a job to its next phase, or raise TrainingCancelled if it was cancelled.

    A phase change is where a runner checks for a cancel it would otherwise only
    see at the next epoch (or never, for the phases before training). Overwriting
    "cancelled" instead is what used to turn a cancelled retrain back into a
    running one.
    """
    job = training_jobs.get(job_id)
    if job is None:
        raise TrainingCancelled("The job was removed while it was running")
    if job.status == "cancelled":
        raise TrainingCancelled("Cancelled by user")
    job.status = status
    if message is not None:
        job.message = message
    if progress is not None:
        job.progress = progress


def _settle(job_id: str, status: str, message: str, progress: Optional[float] = None) -> None:
    """Record how a job ended, unless the user already ended it as "cancelled"."""
    job = training_jobs.get(job_id)
    if job is None or (job.status == "cancelled" and status != "cancelled"):
        return
    job.status = status
    job.message = message
    if progress is not None:
        job.progress = progress


def _cancel_message(job_id: str, cancelled: Exception) -> str:
    """What to tell the user after a cancel, depending on what survives on disk."""
    base = str(cancelled).rstrip(". ") or "Cancelled"
    checkpoint_dir = Path("checkpoints") / job_id
    if (checkpoint_dir / "final_model.pt").exists():
        return (f"{base}. Training had finished; the model was not deployed. "
                f"Export it with POST /export/{job_id}.")
    if (checkpoint_dir / "job_state.json").exists():
        return f"{base}. Resume from the last checkpoint to continue."
    return f"{base}. No checkpoint had been written yet, so start a new job to train this voice."


def _claim(job_id: str, model_name: str) -> None:
    """Take the training slot for *job_id* or raise 409 naming what is in the way."""
    busy = _dataset_ops.get(model_name)
    if busy:
        raise HTTPException(
            status_code=409,
            detail=f"The dataset for '{model_name}' is being modified by {busy}; try again when it has finished.",
        )
    active_runs.claim(job_id, model_name)


@contextmanager
def _dataset_operation(model_name: str, what: str):
    """Exclusive access to data/<model_name> for a maintenance endpoint (409 if busy)."""
    running = active_runs.job_for_model(model_name)
    if running:
        raise HTTPException(
            status_code=409,
            detail=f"Job {running} is training '{model_name}'; its dataset cannot be changed while it runs.",
        )
    if model_name in _dataset_ops:
        raise HTTPException(
            status_code=409,
            detail=f"The dataset for '{model_name}' is already being modified by {_dataset_ops[model_name]}.",
        )
    _dataset_ops[model_name] = what
    try:
        yield
    finally:
        _dataset_ops.pop(model_name, None)


def _launch(job_id: str, runner, *args, **kwargs) -> None:
    """Run *runner* as this job's background task and free its slot when it ends.

    An asyncio task rather than a FastAPI BackgroundTask. uvicorn waits for
    background tasks before it runs the lifespan shutdown, so a multi-hour
    training run held shutdown until the container was killed and the shutdown
    hook that asks jobs to stop never ran. This task is not part of any request,
    so shutdown reaches _stop_running_jobs first.
    """
    async def _guarded():
        try:
            await runner(*args, **kwargs)
        except asyncio.CancelledError:
            # The event loop is going away under the job.
            _settle(job_id, "interrupted", "The service stopped while this job ran. Resume with POST /resume-training.")
            raise
        except Exception as exc:  # a runner is meant to catch its own; this is the net
            logger.exception(f"[{job_id}] runner crashed")
            _settle(job_id, "failed", f"Job failed unexpectedly: {exc}")
        finally:
            active_runs.release(job_id)

    task = asyncio.get_running_loop().create_task(_guarded())
    _runner_tasks.add(task)
    task.add_done_callback(_runner_tasks.discard)


def _start_job(job_id: str, model_name: str, runner, args: tuple, **job_fields) -> TrainingStatus:
    """Claim the slot, register the job record and launch its runner: all or nothing.

    Synchronous on purpose (no await), so nothing else can claim the slot in
    between. If registering or launching fails, the slot and the record are
    given back rather than leaked, and a record this replaces is restored.
    """
    _claim(job_id, model_name)
    previous = training_jobs.get(job_id)
    try:
        job = _new_job(job_id, model_name=model_name, **job_fields)
        _launch(job_id, runner, *args)
    except BaseException:
        active_runs.release(job_id)
        if previous is not None:
            training_jobs[job_id] = previous
        else:
            training_jobs.pop(job_id, None)
        raise
    return job


async def _stop_running_jobs() -> None:
    """Shutdown: ask every running job to stop and give them a moment to do so.

    A stop is a cancel: the training loop notices it at the next epoch boundary,
    writes ``cancelled`` into job_state.json and exits, and the job stays
    resumable from its last periodic checkpoint. If the container's stop grace
    period is shorter than an epoch the process is killed first, which is the
    same outcome the crash path always had ("interrupted" after the restart).
    """
    for job_id in active_runs.job_ids():
        job = training_jobs.get(job_id)
        if job is not None and job.status not in TERMINAL_STATUSES:
            job.status = "cancelled"
            job.message = "Stopped because the service is shutting down. Resume with POST /resume-training."
    tasks = [t for t in _runner_tasks if not t.done()]
    if not tasks:
        return
    logger.info(f"Shutdown: waiting up to {SHUTDOWN_GRACE_S:g}s for {len(tasks)} job(s) to stop")
    _, pending = await asyncio.wait(tasks, timeout=SHUTDOWN_GRACE_S)
    if pending:
        logger.warning(
            f"{len(pending)} job(s) did not stop within {SHUTDOWN_GRACE_S:g}s; "
            "a resume will start from their last periodic checkpoint"
        )


def _resolve_language(value: Optional[str]) -> str:
    """The dataset language of a request: what it names, else TRAINING_DEFAULT_LANGUAGE (400 if unsupported)."""
    text = (value or "").strip()
    if not text:
        return DEFAULT_LANGUAGE
    try:
        return normalize_language(text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _check_sample_rate(value: int) -> int:
    """Only the rate the model and the mel features are built for is accepted."""
    if value != SUPPORTED_SAMPLE_RATE:
        raise HTTPException(
            status_code=400,
            detail=(
                f"sample_rate must be {SUPPORTED_SAMPLE_RATE} (got {value}): the mel features "
                "and the model are fixed at that rate, and other rates would be silently mismatched."
            ),
        )
    return value


def _check_batch_size(value: int) -> int:
    """A batch size the loader can build (the trainer caps the upper end itself)."""
    if not 1 <= value <= 1024:
        raise HTTPException(status_code=400, detail=f"batch_size must be between 1 and 1024 (got {value})")
    return value


# --- long-running work, kept off the event loop --------------------------------

def _generate_mels(audio_files: list, mel_dir: Path, sample_rate: int, skip_existing: bool = True):
    """Compute mel spectrograms for *audio_files*. Blocking: run it with ``asyncio.to_thread``.

    Returns ``(generated, failed)`` where *failed* lists the files that could not
    be processed. The endpoints ran this loop inline, so /health and /status
    stopped answering for as long as a dataset took (minutes) and Docker marked
    the container unhealthy.
    """
    mel_dir.mkdir(parents=True, exist_ok=True)
    generated = 0
    failed = []
    for audio_file in audio_files:
        audio_file = Path(audio_file)
        mel_file = mel_dir / f"{audio_file.stem}.npy"
        if skip_existing and mel_file.exists():
            continue
        try:
            audio, _ = librosa.load(str(audio_file), sr=sample_rate)
            mel_spec = data_processor._compute_mel_spectrogram(audio)
            # Written under another name and renamed: a job reading the dataset
            # must never open a half-written .npy.
            partial = mel_file.with_name(f"{audio_file.stem}.partial.npy")
            np.save(partial, mel_spec)
            os.replace(partial, mel_file)
            generated += 1
            if generated % 100 == 0:
                logger.info(f"Generated {generated} mel spectrograms...")
        except Exception as e:
            logger.error(f"Error processing {audio_file}: {e}")
            failed.append(audio_file)
    return generated, failed


async def _export_off_loop(job_id: str) -> Path:
    """Run the ONNX export away from the event loop.

    ``export_to_onnx`` is a coroutine whose work (loading the checkpoint,
    tracing, verifying in onnxruntime) is CPU-bound, so awaiting it directly
    stalls /health and /status for the whole export. It is run on its own event
    loop in a worker thread, which is correct whether the exporter offloads
    internally or not, and if it ever becomes a plain function.
    """
    def _run():
        result = model_exporter.export_to_onnx(job_id)
        if inspect.isawaitable(result):
            async def _await():
                return await result
            return asyncio.run(_await())
        return result

    return await asyncio.to_thread(_run)


class _ExportFailed(Exception):
    """Training finished and its checkpoint is saved, but the ONNX export failed."""


async def _export_and_deploy(job_id: str, model_name: str, deployment_target: Optional[str],
                             progress: float = 90) -> dict:
    """Export the finished checkpoint and deploy the bundle.

    Raises TrainingCancelled if the user cancelled first (nothing is deployed
    for a job the user stopped) and ``_ExportFailed`` if the export itself
    fails. A deployment failure is returned, not raised: the ONNX exists and is
    downloadable, so the job still completed.
    """
    _advance(job_id, "exporting", "Exporting model to ONNX format...", progress)
    logger.info(f"Exporting model to ONNX for job {job_id}")
    try:
        onnx_path = await _export_off_loop(job_id)
    except Exception as export_error:
        logger.error(f"ONNX export failed for job {job_id}: {export_error}")
        raise _ExportFailed(str(export_error)) from export_error
    logger.info(f"Model exported to ONNX: {onnx_path}")

    if _is_cancelled(job_id):
        raise TrainingCancelled("Cancelled during export; the voice was not deployed")
    try:
        result = await deploy_model_bundle(job_id, model_name, onnx_path, deployment_target)
        logger.info(f"Deployment result for {job_id}: {result}")
        return result
    except Exception as deploy_error:
        logger.warning(f"Model export succeeded but deployment failed for {job_id}: {deploy_error}")
        detail = deploy_error.detail if isinstance(deploy_error, HTTPException) else str(deploy_error)
        return {"target": deployment_target, "status": "failed", "message": str(detail)}


def _complete(job_id: str, deployment_result: dict, summary: str = "") -> None:
    """Final status of a job whose model was exported."""
    status = deployment_result.get("status")
    if status == "deployed":
        message = f"Training completed and model deployed to {deployment_result.get('target')}."
    elif status == "skipped":
        message = "Training completed. Exported model retained for manual download."
    else:
        message = (f"Training completed but deployment failed: {deployment_result.get('message')}. "
                   f"Exported model is still available for download.")
    _settle(job_id, "completed", f"{message} {summary}".strip(), 100)


def _fail_export(job_id: str, failure: _ExportFailed) -> None:
    """A finished training run whose export failed is not a completed job.

    The UI reads "completed" as "there is a model to download". The checkpoint is
    intact, so the message says how to retry.
    """
    _settle(
        job_id, "failed",
        f"Training finished and the checkpoint is saved, but the ONNX export failed: {failure}. "
        f"Retry with POST /export/{job_id}.",
        100,
    )


def _cleanup_uploads(dataset_path: Path) -> None:
    """Remove the raw uploads once they have been cut into segments (or the job ended)."""
    temp_audio_dir = dataset_path / "temp_uploads"
    try:
        if temp_audio_dir.exists():
            shutil.rmtree(temp_audio_dir)
            logger.info("Cleaned up temporary files")
    except OSError as cleanup_err:
        logger.warning(f"Cleanup of {temp_audio_dir} failed: {cleanup_err}")


@app.get("/")
async def root():
    """Root endpoint — service identity."""
    return {"service": "PiperTTS Training Service", "status": "ready", "version": "1.0.0"}

@app.get("/health")
async def health():
    """Liveness probe. Cheap and independent of the GPU, storage and STT."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_jobs": len(active_runs),
        "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
        "trainer_kind": TRAINER_KIND,
        "trainer_caveat": TRAINER_CAVEAT,
    }


def _probe_storage() -> dict:
    """Can the working directories be written? Blocking (a network share may hang)."""
    checks = {}
    for name in ("data", "checkpoints", "models"):
        path = Path(name)
        try:
            path.mkdir(parents=True, exist_ok=True)
            # A real write, not os.access(): ACLs and NFS squash make W_OK lie,
            # and unwritable bind mounts are the usual first-run failure.
            with tempfile.NamedTemporaryFile(dir=path, prefix=".ready-"):
                pass
            checks[name] = {"ok": True}
        except OSError as exc:
            checks[name] = {"ok": False, "error": f"{path.resolve()} is not writable: {exc}"}
    try:
        checks["data"]["free_mb"] = int(shutil.disk_usage("data").free / (1024 * 1024))
    except OSError:
        pass
    return checks


def _check_device() -> dict:
    """Is the compute device the container was started for the one PyTorch is using?

    start.sh records the accelerator it found in TRAINING_DEVICE_TYPE. An image
    whose torch build has no kernels for the card (a cu118 wheel on Blackwell)
    or a missing driver mount leaves torch on the CPU: training would start and
    then crawl. That is a broken deployment, and readiness is the place to say so.
    """
    device = getattr(training_pipeline, "device", None)
    actual = getattr(device, "type", None)
    expected = os.getenv("TRAINING_DEVICE_TYPE", "").strip().lower()
    result = {"ok": True, "device": str(device) if device is not None else None,
              "expected": expected or None}
    hidden = os.environ.get("CUDA_VISIBLE_DEVICES") == "" or os.environ.get("HIP_VISIBLE_DEVICES") == ""
    if hidden:
        # An empty CUDA_VISIBLE_DEVICES is the operator asking for the CPU.
        result["note"] = "the GPU is hidden on purpose (empty *_VISIBLE_DEVICES)"
    elif expected in ("cuda", "hip") and actual != "cuda":
        result.update(
            ok=False,
            error=(f"a {expected} GPU was detected at startup but PyTorch is running on "
                   f"'{actual}': the driver is not visible to the container, or the torch build "
                   f"has no kernels for this GPU (Blackwell needs a cu128 build)"),
        )
    return result


@app.get("/ready")
async def ready():
    """Readiness: 200 when this service can accept a training job, 503 when its environment is broken.

    Mirrors /health and adds what a job would need: writable data, checkpoint and
    model directories, and the accelerator the container was started for. A
    running job does not make the service "not ready" (that would take a busy
    trainer out of rotation); ``accepting_jobs`` says whether the slot is free.
    """
    problems: list = []
    warnings: list = []
    try:
        checks = await asyncio.wait_for(asyncio.to_thread(_probe_storage), timeout=5.0)
    except asyncio.TimeoutError:
        checks = {}
        problems.append("storage probe timed out; a mounted volume may be hung")
    for name, check in list(checks.items()):
        if not check.get("ok", True):
            problems.append(check["error"])

    device = _check_device()
    checks["device"] = device
    if not device["ok"]:
        problems.append(device["error"])

    # Only affects where finished voices go, not whether a job can run.
    default_target = DEPLOYMENT_TARGET_REGISTRY["targets"].get(DEPLOYMENT_TARGET_REGISTRY["default_target"], {})
    if (default_target.get("deployment_contract") == "piper-shared-volume-v1"
            and not os.access(SHARED_MODELS_DIR, os.W_OK)):
        warnings.append(
            f"{SHARED_MODELS_DIR} is not writable; deploying to the default target "
            f"'{DEPLOYMENT_TARGET_REGISTRY['default_target']}' will fail (the export stays downloadable)"
        )

    is_ready = not problems
    body = {
        "status": "ready" if is_ready else "not_ready",
        "timestamp": datetime.now().isoformat(),
        "accepting_jobs": is_ready and len(active_runs) < MAX_CONCURRENT_JOBS,
        "active_jobs": len(active_runs),
        "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
        "checks": checks,
        "problems": problems,
        "warnings": warnings,
        "trainer_kind": TRAINER_KIND,
        "trainer_caveat": TRAINER_CAVEAT,
    }
    return JSONResponse(status_code=200 if is_ready else 503, content=body)


@app.get("/deployment-targets")
async def deployment_targets():
    """Return the configured deployment targets for exported model bundles."""
    return DEPLOYMENT_TARGET_REGISTRY

def _replace_tree(source: Path, target: Path) -> None:
    """Replace *target* with a copy of *source* (blocking)."""
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


@app.post("/restore-backup")
async def restore_backup():
    """Restore training data from a local backup directory."""
    try:
        # Check if backup exists in current directory
        backup_dir = Path("./backup_stst_data")
        target_dir = Path("data/stst")

        if backup_dir.exists():
            # Deletes and recreates a dataset: never under a job that reads it.
            with _dataset_operation("stst", "restore-backup"):
                await asyncio.to_thread(_replace_tree, backup_dir, target_dir)
            return {"message": "Backup restored successfully", "status": "success"}
        else:
            return {"message": "No backup found in current directory", "status": "error"}
    except HTTPException:
        raise
    except Exception as e:
        return {"message": f"Error restoring backup: {str(e)}", "status": "error"}

@app.post("/generate-missing-mels")
async def generate_missing_mels(model_name: str = "stst"):
    """Generate mel spectrograms for audio files that are missing them."""
    model_name = _safe_name(model_name, "model_name")
    try:
        dataset_dir = Path(f"data/{model_name}")
        audio_dir = dataset_dir / "audio"
        mel_dir = dataset_dir / "mel"

        if not audio_dir.exists():
            raise HTTPException(status_code=404, detail="Audio directory not found")

        audio_files = list(audio_dir.glob("*.wav"))

        with _dataset_operation(model_name, "generate-missing-mels"):
            processed, failed = await asyncio.to_thread(
                _generate_mels, audio_files, mel_dir, SUPPORTED_SAMPLE_RATE, True)

        return {
            "message": f"Generated {processed} missing mel spectrograms",
            "total_audio_files": len(audio_files),
            "processed": processed,
            "failed": len(failed),
            "status": "success" if not failed else "partial",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating mel spectrograms: {str(e)}")

@app.post("/prepare-dataset")
async def prepare_dataset(dataset: DatasetUpload):
    """Create a training dataset from pre-segmented STT results."""
    model_name = _safe_name(dataset.model_name, "model_name")
    language = _resolve_language(dataset.language)
    if not dataset.segments:
        raise HTTPException(status_code=400, detail="segments must not be empty")
    for index, segment in enumerate(dataset.segments):
        if len(segment.text) > MAX_TEXT_CHARS:
            raise HTTPException(
                status_code=400,
                detail=f"segments[{index}].text has {len(segment.text)} characters; the limit is "
                       f"{MAX_TEXT_CHARS} (MAX_TEXT_CHARS)",
            )
    try:
        stats: dict = {}
        with _dataset_operation(model_name, "prepare-dataset"):
            dataset_path = await data_processor.prepare_dataset(
                segments=dataset.segments,
                model_name=model_name,
                language=language,
                stats=stats,
            )
        return {
            "status": "success",
            "dataset_path": str(dataset_path),
            # Samples actually written; this used to echo the number received
            # even when segments were skipped.
            "num_samples": stats.get("prepared", len(dataset.segments)),
            "num_segments_received": len(dataset.segments),
            "num_skipped": stats.get("skipped", 0),
            "language": language,
        }
    except HTTPException:
        # A 400 from the split validator ("too few segments") is the caller's
        # problem to fix and must not be relabelled as an internal error.
        raise
    except AudioSourceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (PhonemizationError, ValueError) as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test-upload")
async def test_upload(
    model_name: str = Form(...),
    audio_files: List[UploadFile] = File(...)
):
    """Dry-run upload — returns filenames and content types without processing."""
    try:
        return {
            "model_name": model_name,
            "file_count": len(audio_files),
            "filenames": [f.filename for f in audio_files],
            "content_types": [f.content_type for f in audio_files]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test upload failed: {str(e)}")


def _upload_filename(raw: str, position: int, used_stems: set) -> str:
    """A safe, unique on-disk name for an uploaded file.

    Segments are named after the upload's stem, so two uploads that share one
    ("take1.wav" from two folders, or "a.wav" and "a.mp3") overwrote each
    other's file and then each other's segments, silently.
    """
    name = Path(raw.replace("\\", "/")).name
    if name in ("", ".", ".."):
        name = f"upload_{position:03d}"
    path = Path(name)
    stem, suffix = path.stem, path.suffix
    candidate = stem
    if candidate.lower() in used_stems:
        candidate = f"{stem}_{position:03d}"
    used_stems.add(candidate.lower())
    return f"{candidate}{suffix}"


@app.post("/train")
async def train_model(
    model_name: str = Form(...),
    audio_files: List[UploadFile] = File(...),
    language: str = Form(DEFAULT_LANGUAGE),
    sample_rate: int = Form(SUPPORTED_SAMPLE_RATE),
    quality: str = Form("medium"),
    stt_service_url: str = Form(""),
    epochs: int = Form(1000),
    batch_size: int = Form(32),
    deployment_target: str = Form(""),
):
    """
    New STT-based training endpoint that processes audio files through STT service
    for proper segmentation before training.

    Workflow:
    1. Save uploaded audio files
    2. Process through STT service for segmentation
    3. Create training segments using ffmpeg
    4. Generate training dataset
    5. Start training with optimized pipeline

    Answers 409 while another job holds the training slot.
    """
    job_id = str(uuid.uuid4())

    model_name = _safe_name(model_name, "model_name")
    epochs = _validate_epochs(epochs)
    batch_size = _check_batch_size(batch_size)
    resolved_target_id, _ = resolve_deployment_target(deployment_target or None)
    stt_service_url = resolve_stt_service_url(stt_service_url)
    language = _resolve_language(language)
    sample_rate = _check_sample_rate(sample_rate)

    # No await between the claim and the job record, so two requests cannot both
    # pass the check.
    _claim(job_id, model_name)
    dataset_path = Path(f"data/{model_name}")
    temp_audio_dir = dataset_path / "temp_uploads"

    try:
        logger.info(f"Starting STT-based training for model: {model_name}")
        logger.info(f"Received {len(audio_files)} audio files")

        # Initialize training job status
        _new_job(
            job_id, "initializing",
            model_name=model_name,
            total_epochs=epochs,
            message="Initializing STT-based training pipeline...",
            deployment_target=resolved_target_id,
        )

        # Create dataset directory
        dataset_path.mkdir(parents=True, exist_ok=True)

        # Create temporary directory for uploaded files
        temp_audio_dir.mkdir(exist_ok=True)

        # Save uploaded audio files
        uploaded_files = []
        total_bytes = 0
        limit_bytes = max_upload_bytes()
        used_stems: set = set()

        for position, audio_file in enumerate(audio_files):
            if not audio_file.filename:
                continue

            safe_filename = _upload_filename(audio_file.filename, position, used_stems)
            file_path = temp_audio_dir / safe_filename

            logger.info(f"Saving uploaded file: {safe_filename}")

            # Save file
            async with aiofiles.open(file_path, 'wb') as f:
                while content := await audio_file.read(1024 * 1024):  # 1MB chunks
                    total_bytes += len(content)
                    if total_bytes > limit_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Upload exceeds {limit_bytes / (1024 * 1024):g} MB (MAX_UPLOAD_MB).",
                        )
                    await f.write(content)

            uploaded_files.append(file_path)
            logger.info(f"Saved {safe_filename} ({file_path.stat().st_size / (1024 * 1024):.1f}MB)")

        if not uploaded_files:
            raise HTTPException(status_code=400, detail="No audio files were uploaded")

        total_size_mb = total_bytes / (1024 * 1024)
        logger.info(f"Total uploaded: {len(uploaded_files)} files, {total_size_mb:.1f}MB")

        # Update status
        job = training_jobs[job_id]
        job.message = f"Processing {len(uploaded_files)} audio files through STT service..."
        job.progress = 10

        # Start background processing
        _launch(
            job_id,
            run_stt_based_training,
            job_id,
            model_name,
            uploaded_files,
            dataset_path,
            language,
            sample_rate,
            stt_service_url,
            quality,
            epochs,
            batch_size,
            resolved_target_id,
        )

        return JSONResponse(
            status_code=202,
            content={
                "message": f"STT-based training started for model: {model_name}",
                "job_id": job_id,
                "files_uploaded": len(uploaded_files),
                "total_size_mb": round(total_size_mb, 1)
            }
        )

    except HTTPException:
        # Never started (too large, nothing uploaded): the caller fixes the
        # request and retries, so it leaves neither a job record nor a claim.
        active_runs.release(job_id)
        training_jobs.pop(job_id, None)
        await asyncio.to_thread(_cleanup_uploads, dataset_path)
        raise
    except Exception as e:
        logger.error(f"Error starting STT-based training: {e}")
        active_runs.release(job_id)
        if job_id in training_jobs:
            training_jobs[job_id].status = "failed"
            training_jobs[job_id].message = f"Failed to start training: {str(e)}"
        await asyncio.to_thread(_cleanup_uploads, dataset_path)
        raise HTTPException(status_code=500, detail=str(e))

async def run_stt_based_training(job_id: str,
                               model_name: str,
                               audio_files: List[Path],
                               dataset_path: Path,
                               language: str,
                               sample_rate: int,
                               stt_service_url: str,
                               quality: str = "medium",
                               epochs: int = 1000,
                               batch_size: int = 32,
                               deployment_target: Optional[str] = None):
    """Background task for STT-based training workflow"""
    try:
        from audio_segmenter import AudioSegmenter

        logger.info(f"Starting STT-based processing for job {job_id}")

        # Update status
        _advance(job_id, "processing", "Processing audio files through STT service...", 20)

        # Initialize audio segmenter
        segmenter = AudioSegmenter()

        # Quality filters for training segments
        quality_filters = {
            'min_duration': 1.0,      # Minimum 1 second
            'max_duration': 15.0,     # Maximum 15 seconds
            # exp(avg_logprob) from the STT service; STT_MIN_CONFIDENCE, default 0.6
            'min_confidence': default_min_confidence(),
            'min_text_length': 10     # Minimum 10 characters
        }

        # Process audio files through STT and create segments
        training_segments, stats = await segmenter.process_multiple_audio_files(
            audio_files,
            dataset_path,
            model_name,
            stt_service_url,
            sample_rate,
            quality_filters,
            language=stt_language(language),
            should_stop=lambda: _is_cancelled(job_id),
        )

        _advance(job_id, "processing",
                 f"Created {len(training_segments)} training segments. Generating metadata...", 60)

        if not training_segments:
            raise RuntimeError(
                "No valid training segments were created from the audio files: "
                f"{stats['total_segments_found']} STT segment(s) found, "
                f"{stats['segments_after_quality_filter']} passed the quality filter "
                f"(1-15 s, at least 10 characters, confidence >= {quality_filters['min_confidence']:g}; "
                "lower STT_MIN_CONFIDENCE to admit less certain transcripts)"
            )

        # Generate mel spectrograms for all training segments. Regenerated, not
        # skipped when a file exists: segment names repeat across uploads of the
        # same recording, so an existing mel may belong to different audio.
        logger.info(f"Generating mel spectrograms for {len(training_segments)} segments...")
        mel_dir = dataset_path / "mel"
        mel_generated, mel_failed = await asyncio.to_thread(
            _generate_mels, [seg.audio_path for seg in training_segments], mel_dir, sample_rate, False)
        logger.info(f"Generated {mel_generated} mel spectrograms")
        if mel_failed:
            # A segment without its mel would fail the dataset load hours in.
            failed_names = {Path(p).stem for p in mel_failed}
            training_segments = [s for s in training_segments if s.audio_path.stem not in failed_names]
            logger.warning(f"Dropped {len(mel_failed)} segment(s) whose mel spectrogram could not be computed")
            if not training_segments:
                raise RuntimeError("No mel spectrogram could be computed for any segment")

        # Generate training metadata (train.json + val.json) with train/val split
        metadata_path = await asyncio.to_thread(
            segmenter.generate_training_metadata,
            training_segments,
            dataset_path,
            model_name,
            language,
        )

        logger.info(f"Generated training metadata: {metadata_path}")
        logger.info(f"Training dataset ready with {len(training_segments)} segments")

        # Clean up temporary upload directory
        await asyncio.to_thread(_cleanup_uploads, dataset_path)

        # Update status before training
        _advance(job_id, "training", "Starting model training with optimized pipeline...", 70)

        # Create training request
        training_request = TrainingRequest(
            model_name=model_name,
            language=language,
            sample_rate=sample_rate,
            quality=quality,
            epochs=epochs,
            batch_size=batch_size,
            deployment_target=deployment_target,
        )

        # Start actual training with optimized pipeline
        logger.info(f"Starting optimized training for model: {model_name}")

        # Run training in a separate thread to avoid blocking the event loop
        # This allows health checks and status queries to work during training
        await asyncio.to_thread(
            training_pipeline.train_sync,
            job_id=job_id,
            request=training_request,
            callback=lambda update: update_training_status(job_id, update),
        )

        # Export model to ONNX format and deploy to the selected target
        deployment_result = await _export_and_deploy(job_id, model_name, deployment_target)

        _complete(
            job_id, deployment_result,
            f"Created from {len(training_segments)} segments ({stats['training_audio_duration']:.1f}s of audio)",
        )
        logger.info(f"STT-based training completed for job {job_id}")

    except TrainingCancelled as cancelled:
        # Not a failure: the checkpoints are intact and the job is resumable.
        # Falling through to export here is what used to deploy the voice the
        # user had just cancelled.
        logger.info(f"Training cancelled for job {job_id}: {cancelled}")
        _settle(job_id, "cancelled", _cancel_message(job_id, cancelled))
    except _ExportFailed as export_failed:
        logger.error(f"Export failed for job {job_id}: {export_failed}")
        _fail_export(job_id, export_failed)
    except Exception as e:
        logger.error(f"STT-based training failed for job {job_id}: {e}")
        _settle(job_id, "failed", f"Training failed: {str(e)}")
    finally:
        await asyncio.to_thread(_cleanup_uploads, dataset_path)


@app.post("/resume-training")
async def resume_training(
    model_name: str = Form(...),
    job_id: str = Form(""),          # optional — auto-detect latest if blank
    extra_epochs: int = Form(0),     # 0 = continue to original total_epochs
    deployment_target: str = Form(""),
):
    """
    Resume an interrupted training job from its latest checkpoint.
    Finds the newest checkpoint_epoch_N.pt for the job and continues from there.

    A job that is still running (or still stopping after a cancel) cannot be
    resumed: 409. So can a job that already completed: it is exported, not resumed.
    """
    model_name = _safe_name(model_name, "model_name")
    if extra_epochs:
        _validate_epochs(extra_epochs, "extra_epochs")
    if job_id.strip():
        job_id = _safe_name(job_id, "job_id")
    resolved_target_id, _ = resolve_deployment_target(deployment_target or None)

    checkpoints_root = Path("checkpoints")

    # Find the job_id from disk if not provided
    target_job_id = job_id.strip() or None
    target_state = None

    for state_file in checkpoints_root.glob("*/job_state.json"):
        try:
            with open(state_file) as f:
                state = json.load(f)
            if state.get("status") == "completed":
                continue
            name_match = state.get("model_name") == model_name
            id_match   = target_job_id and state.get("job_id") == target_job_id
            if id_match or (not target_job_id and name_match):
                # Pick the most-recently-written one if multiple
                state_epoch = _coerce_resume_int(state.get("epoch", 0), 0)
                current_best_epoch = _coerce_resume_int(target_state.get("epoch", 0), 0) if target_state else 0
                if target_state is None or state_epoch > current_best_epoch:
                    target_state = state
        except Exception:
            continue

    if not target_state:
        raise HTTPException(
            status_code=404,
            detail=f"No interrupted job found for model '{model_name}'. "
                   "Train a new model first."
        )

    resumed_job_id   = str(target_state.get("job_id") or "").strip()
    latest_ckpt_path = _coerce_resume_path(target_state.get("latest_checkpoint"))
    saved_epoch      = _coerce_resume_int(target_state.get("epoch", 0), 0)
    total_epochs     = _coerce_resume_int(target_state.get("total_epochs", 10000), 10000)
    language         = target_state.get("language") or DEFAULT_LANGUAGE
    config           = target_state.get("config") if isinstance(target_state.get("config"), dict) else {}

    if not resumed_job_id:
        raise HTTPException(
            status_code=404,
            detail=f"Interrupted job state for model '{model_name}' is missing a valid job id."
        )

    # A job id names a job of one model. Without this, resuming job A "as" model
    # B trained A's weights and then exported and deployed them under B's name.
    recorded_model = target_state.get("model_name")
    if recorded_model and recorded_model != model_name:
        raise HTTPException(
            status_code=400,
            detail=f"Job {resumed_job_id} belongs to model '{recorded_model}', not '{model_name}'.",
        )

    live = training_jobs.get(resumed_job_id)
    if active_runs.is_active(resumed_job_id) or (live is not None and live.status in ACTIVE_STATUSES):
        raise HTTPException(
            status_code=409,
            detail=(f"Job {resumed_job_id} is still running (status "
                    f"'{live.status if live else 'unknown'}'); a second thread on the same "
                    "checkpoints would corrupt them. Wait for it to stop first."),
        )

    if extra_epochs > 0:
        total_epochs = saved_epoch + extra_epochs

    total_epochs = max(total_epochs, saved_epoch or 1)

    if latest_ckpt_path is None or not latest_ckpt_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Checkpoint file not found: {target_state.get('latest_checkpoint')}"
        )

    training_request = TrainingRequest(
        model_name=model_name,
        language=language,
        sample_rate=config.get("sample_rate", SUPPORTED_SAMPLE_RATE),
        quality=config.get("quality", "medium"),
        epochs=total_epochs,
        batch_size=config.get("batch_size", 16),
        deployment_target=resolved_target_id,
    )

    # After every check that can refuse, so a refused request never holds the slot.
    _start_job(
        resumed_job_id, model_name, run_resume_training,
        (resumed_job_id, model_name, training_request, latest_ckpt_path, resolved_target_id),
        status="training",
        total_epochs=total_epochs,
        message=f"Resuming from epoch {saved_epoch}/{total_epochs}...",
        deployment_target=resolved_target_id,
        progress=round(saved_epoch / total_epochs * 100, 1),
        current_epoch=saved_epoch,
        loss=target_state.get("loss") if isinstance(target_state.get("loss"), (int, float)) else None,
    )

    return JSONResponse(status_code=202, content={
        "message": f"Resuming training for '{model_name}' from epoch {saved_epoch}",
        "job_id": resumed_job_id,
        "resume_from_epoch": saved_epoch,
        "total_epochs": total_epochs,
        "checkpoint": str(latest_ckpt_path),
    })


async def run_resume_training(job_id: str, model_name: str, request: TrainingRequest,
                              checkpoint_path: Path, deployment_target: Optional[str]):
    """Resume training in a background thread, then export the updated model."""
    # Guarded: an exception escaping a background task is swallowed by the event
    # loop and the job would sit at "training" forever with no error anywhere
    # the caller could see.
    try:
        await asyncio.to_thread(
            training_pipeline.train_sync,
            job_id=job_id,
            request=request,
            callback=lambda update: update_training_status(job_id, update),
            resume_from=str(checkpoint_path),
        )

        deployment_result = await _export_and_deploy(job_id, model_name, deployment_target, progress=95)
        _complete(job_id, deployment_result)
    except TrainingCancelled as cancelled:
        logger.info(f"Resumed training cancelled for {job_id}: {cancelled}")
        _settle(job_id, "cancelled", _cancel_message(job_id, cancelled))
    except _ExportFailed as export_failed:
        # This used to leave the job at "exporting" for good.
        logger.error(f"Export after resume failed for {job_id}: {export_failed}")
        _fail_export(job_id, export_failed)
    except Exception as train_error:
        logger.error(f"Resumed training failed for {job_id}: {train_error}")
        _settle(job_id, "failed", f"Resumed training failed: {train_error}")


@app.post("/train-from-dataset")
async def train_from_dataset(
    model_name: str = Form(...),
    language: str = Form(DEFAULT_LANGUAGE),
    epochs: int = Form(10000),
    batch_size: int = Form(32),
    deployment_target: str = Form(""),
):
    """
    Start training directly from an already-prepared dataset.
    Requires data/{model_name}/train.json and val.json to exist.
    Skips upload and STT — goes straight to VITS training.
    """
    model_name = _safe_name(model_name, "model_name")
    epochs = _validate_epochs(epochs)
    batch_size = _check_batch_size(batch_size)
    language = _resolve_language(language)
    train_json = Path(f"data/{model_name}/train.json")
    val_json = Path(f"data/{model_name}/val.json")

    if not train_json.exists() or not val_json.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Dataset not ready: need data/{model_name}/train.json and val.json"
        )

    try:
        n_train = len(json.loads(train_json.read_text()))
        n_val   = len(json.loads(val_json.read_text()))
    except (OSError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"Dataset for '{model_name}' is unreadable: {exc}")

    resolved_target_id, _ = resolve_deployment_target(deployment_target or None)

    job_id = str(uuid.uuid4())
    training_request = TrainingRequest(
        model_name=model_name,
        language=language,
        sample_rate=SUPPORTED_SAMPLE_RATE,
        quality="medium",
        epochs=epochs,
        batch_size=batch_size,
        deployment_target=resolved_target_id,
    )
    _start_job(
        job_id, model_name, run_training, (job_id, training_request),
        status="training",
        total_epochs=epochs,
        message=f"Starting training with {n_train} train / {n_val} val samples...",
        deployment_target=resolved_target_id,
    )

    return JSONResponse(status_code=202, content={
        "message": f"Training started for model '{model_name}'",
        "job_id": job_id,
        "train_samples": n_train,
        "val_samples": n_val,
        "epochs": epochs,
    })


@app.post("/retrain-from-segments")
async def retrain_from_segments(
    model_name: str = Form(...),
    language: str = Form(DEFAULT_LANGUAGE),
    epochs: int = Form(10000),
    batch_size: int = Form(32),
    prefix_filter: str = Form(""),
    stt_service_url: str = Form(""),
    deployment_target: str = Form(""),
):
    """
    Rebuild metadata from existing pre-segmented audio files and retrain.

    Skips upload and re-segmentation — uses whatever WAV files are already
    in data/{model_name}/audio/.  Runs STT on each clip to get transcriptions,
    then calls generate_training_metadata() and train_sync().
    """
    model_name = _safe_name(model_name, "model_name")
    epochs = _validate_epochs(epochs)
    batch_size = _check_batch_size(batch_size)
    language = _resolve_language(language)
    stt_service_url = resolve_stt_service_url(stt_service_url)
    dataset_path = Path(f"data/{model_name}")
    audio_dir = dataset_path / "audio"

    if not audio_dir.exists():
        raise HTTPException(status_code=404, detail=f"No audio directory found at {audio_dir}")

    wav_files = sorted(audio_dir.glob("*.wav"))
    if prefix_filter:
        wav_files = [f for f in wav_files if f.name.startswith(prefix_filter)]

    if not wav_files:
        raise HTTPException(status_code=404, detail=f"No WAV files found (prefix_filter='{prefix_filter}')")

    resolved_target_id, _ = resolve_deployment_target(deployment_target or None)

    job_id = str(uuid.uuid4())
    _start_job(
        job_id, model_name, _run_retrain_from_segments,
        (job_id, model_name, wav_files, dataset_path,
         language, epochs, batch_size, stt_service_url, resolved_target_id),
        status="initializing",
        total_epochs=epochs,
        message=f"Found {len(wav_files)} audio segments — starting STT transcription...",
        deployment_target=resolved_target_id,
    )

    return JSONResponse(status_code=202, content={
        "message": f"Retrain job started for model '{model_name}'",
        "job_id": job_id,
        "audio_segments": len(wav_files),
    })


async def _run_retrain_from_segments(
    job_id: str,
    model_name: str,
    wav_files: list,
    dataset_path: Path,
    language: str,
    epochs: int,
    batch_size: int,
    stt_service_url: str,
    deployment_target: Optional[str],
):
    """Background task: STT-transcribe existing clips, rebuild metadata, train."""
    from audio_segmenter import AudioSegmenter, TrainingSegment

    try:
        total = len(wav_files)
        logger.info(f"[{job_id}] Retraining from {total} existing segments for model '{model_name}'")

        _advance(job_id, "transcribing", f"Running STT on {total} audio segments (0/{total})...")

        min_confidence = default_min_confidence()
        # Semaphore limits concurrent STT calls to avoid overloading the service
        CONCURRENCY = 8
        sem = asyncio.Semaphore(CONCURRENCY)
        completed = 0
        training_segments = []
        errors: list = []
        rejected = {"text": 0, "confidence": 0}
        lock = asyncio.Lock()

        async with STTProcessor(stt_service_url, language=stt_language(language)) as stt:

            async def transcribe_one(audio_path: Path):
                """Transcribe one pre-segmented WAV clip and append valid results."""
                nonlocal completed
                async with sem:
                    try:
                        if _is_cancelled(job_id):
                            return  # stop spending STT time on a job nobody wants
                        result = await stt.transcribe_audio_file(audio_path, return_segments=True)

                        text = ""
                        # Prefer the full text field (most reliable for short clips)
                        if result.get("text", "").strip():
                            text = result["text"].strip()
                        elif result.get("segments"):
                            text = " ".join(s.get("text", "") for s in result["segments"]).strip()

                        if not text or len(text) < 5:
                            rejected["text"] += 1
                            return  # Skip clips with no / too-short transcription

                        # exp(avg_logprob), the same scale as the /train filter. None
                        # when the service reported nothing: kept, as before.
                        confidence = confidence_from_result(result)
                        if confidence is not None and confidence < min_confidence:
                            rejected["confidence"] += 1
                            return

                        duration = await asyncio.to_thread(librosa.get_duration, path=str(audio_path))
                        seg = TrainingSegment(
                            audio_path=audio_path,
                            text=text,
                            duration=duration,
                            speaker_id=0,
                            confidence=1.0 if confidence is None else confidence,
                            original_file=audio_path.stem,
                            start_time=0.0,
                            end_time=duration,
                        )
                        async with lock:
                            training_segments.append(seg)

                    except Exception as e:
                        async with lock:
                            errors.append(f"{audio_path.name}: {e}")
                            if len(errors) <= 5:
                                logger.warning(f"[{job_id}] STT failed for {audio_path.name}: {e}")
                    finally:
                        async with lock:
                            completed += 1
                            if completed % 100 == 0 or completed == total:
                                pct = 10 + int(50 * completed / total)
                                job = training_jobs.get(job_id)
                                if job is not None and job.status == "transcribing":
                                    job.progress = pct
                                    job.message = (
                                        f"STT transcription: {completed}/{total} clips done, "
                                        f"{len(training_segments)} valid so far..."
                                    )
                                logger.info(f"[{job_id}] {completed}/{total} transcribed, {len(training_segments)} valid")

            await asyncio.gather(*[transcribe_one(p) for p in wav_files])

        if _is_cancelled(job_id):
            raise TrainingCancelled("Cancelled during transcription")

        if not training_segments:
            if errors:
                # An outage or a misconfiguration, not a dataset without speech.
                raise STTError(f"STT failed for {len(errors)} of {total} clips; first error: {errors[0]}")
            raise RuntimeError(
                f"STT produced no usable transcriptions from {total} clips "
                f"({rejected['text']} empty or too short, {rejected['confidence']} below "
                f"confidence {min_confidence:g}; lower STT_MIN_CONFIDENCE to admit less certain ones)"
            )

        logger.info(f"[{job_id}] STT done: {len(training_segments)}/{total} segments kept")

        # Mels for clips that have none yet (existing ones belong to the same audio).
        await asyncio.to_thread(
            _generate_mels, [seg.audio_path for seg in training_segments],
            dataset_path / "mel", SUPPORTED_SAMPLE_RATE, True)

        # Rebuild metadata files
        _advance(job_id, "transcribing", f"Building metadata for {len(training_segments)} segments...", 62)

        segmenter = AudioSegmenter()
        await asyncio.to_thread(
            segmenter.generate_training_metadata,
            training_segments, dataset_path, model_name, language,
        )

        # Training
        note = f" ({len(errors)} clip(s) failed STT and were left out)" if errors else ""
        _advance(job_id, "training", f"Starting VITS training...{note}", 65)

        training_request = TrainingRequest(
            model_name=model_name,
            language=language,
            sample_rate=SUPPORTED_SAMPLE_RATE,
            quality="medium",
            epochs=epochs,
            batch_size=batch_size,
            deployment_target=deployment_target,
        )

        await asyncio.to_thread(
            training_pipeline.train_sync,
            job_id=job_id,
            request=training_request,
            callback=lambda update: update_training_status(job_id, update),
        )

        # Export
        deployment_result = await _export_and_deploy(job_id, model_name, deployment_target, progress=92)
        _complete(job_id, deployment_result, f"{len(training_segments)} segments.{note}")
        logger.info(f"[{job_id}] Retrain from segments completed for '{model_name}'")

    except TrainingCancelled as cancelled:
        logger.info(f"[{job_id}] Retrain cancelled: {cancelled}")
        _settle(job_id, "cancelled", _cancel_message(job_id, cancelled))
    except _ExportFailed as export_failed:
        logger.error(f"[{job_id}] Retrain export failed: {export_failed}")
        _fail_export(job_id, export_failed)
    except Exception as e:
        logger.error(f"[{job_id}] Retrain from segments failed: {e}")
        _settle(job_id, "failed", f"Failed: {e}")


_exports_in_flight: set = set()


@app.post("/export/{job_id}")
async def manual_export_model(job_id: str, model_name: str = Form(...), deployment_target: str = Form("")):
    """Manually export a completed training checkpoint and deploy it to a configured target."""
    job_id = _safe_name(job_id, "job_id")
    model_name = _safe_name(model_name, "model_name")
    if active_runs.is_active(job_id) or job_id in _exports_in_flight:
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} is still running or exporting; wait for it to finish first.",
        )
    _exports_in_flight.add(job_id)
    try:
        resolved_target_id, _ = resolve_deployment_target(deployment_target or None)
        # Check if checkpoint exists
        checkpoint_path = Path(f"checkpoints/{job_id}/final_model.pt")
        if not checkpoint_path.exists():
            raise HTTPException(status_code=404, detail=f"Model checkpoint not found: {checkpoint_path}")

        # Export model to ONNX format
        onnx_path = await _export_off_loop(job_id)

        # Deploy model bundle to selected target
        deployment_result = await deploy_model_bundle(job_id, model_name, onnx_path, resolved_target_id)

        return {
            "message": f"Model '{model_name}' exported successfully.",
            "job_id": job_id,
            "model_name": model_name,
            "onnx_path": str(onnx_path),
            "deployment": deployment_result,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")
    finally:
        _exports_in_flight.discard(job_id)

@app.delete("/model/{job_id}")
async def delete_trained_model(job_id: str):
    """Delete a trained model and all associated files (checkpoint, export, dataset).

    Refused (409) while the job is running: the training thread writes into the
    directories this removes, and would recreate a half-deleted job.
    """
    job_id = _safe_name(job_id, "job_id")
    live = training_jobs.get(job_id)
    if job_id in _exports_in_flight:
        raise HTTPException(status_code=409, detail=f"Job {job_id} is being exported; try again when that has finished.")
    if active_runs.is_active(job_id) or (live is not None and live.status in ACTIVE_STATUSES):
        raise HTTPException(
            status_code=409,
            detail=(f"Job {job_id} is still {'stopping' if live and live.status == 'cancelled' else 'running'}. "
                    f"Cancel it with DELETE /job/{job_id} and wait until it has stopped, then delete it."),
        )
    try:
        # EVERYTHING THAT READS DISK HAPPENS FIRST.
        #
        # _model_name_from_disk reads checkpoints/<job_id>/job_state.json, and
        # the checkpoint directory is about to be deleted below. Resolving the
        # name after that rmtree meant the disk fallback always returned None —
        # which is precisely the after-a-restart case it was added for, since
        # completed jobs are not restored into memory. The dataset and the
        # deployed voice were then silently left behind, exactly as before.
        model_name = _model_name_from_disk(job_id)
        deployment_target = None
        if job_id in training_jobs:
            model_name = training_jobs[job_id].model_name or model_name
            deployment_target = training_jobs[job_id].deployment_target
            del training_jobs[job_id]
            logger.info(f"Removed job {job_id} from active jobs")

        # Also resolved before the deletes. This one scans OTHER jobs' state
        # files so it would survive, but keeping the whole read phase together
        # is what stops the next edit from reintroducing the same ordering bug.
        siblings = _other_jobs_using_model(model_name, job_id) if model_name else []

        # Remove checkpoint directory
        checkpoint_dir = Path(f"checkpoints/{job_id}")
        if checkpoint_dir.exists():
            await asyncio.to_thread(shutil.rmtree, checkpoint_dir)
            logger.info(f"Deleted checkpoint directory: {checkpoint_dir}")

        # Remove exported model directory
        model_dir = Path(f"models/{job_id}")
        if model_dir.exists():
            await asyncio.to_thread(shutil.rmtree, model_dir)
            logger.info(f"Deleted model directory: {model_dir}")

        deleted_items = ["checkpoint directory", "model directory", "training job record"]

        # The dataset and the deployed voice are keyed on the MODEL NAME, not on
        # the job id, and retraining a voice produces several jobs that share
        # both. Deleting one job used to take the dataset the others still need
        # and undeploy a voice a newer job had just published.
        if model_name and siblings:
            logger.info(
                f"Keeping dataset and deployed voice for '{model_name}': still used "
                f"by job(s) {', '.join(siblings)}"
            )
        elif model_name:
            dataset_dir = Path(f"data/{model_name}")
            if dataset_dir.exists():
                await asyncio.to_thread(shutil.rmtree, dataset_dir)
                logger.info(f"Deleted dataset directory: {dataset_dir}")
                deleted_items.append("dataset directory")

            try:
                await remove_model_from_deployment_target(model_name, deployment_target)
                deleted_items.append("deployed voice")
            except Exception as tts_error:
                logger.warning(f"Could not remove from deployment target: {tts_error}")

        return {
            "message": f"Model {job_id} deleted successfully",
            "model_name": model_name,
            "deleted_items": deleted_items,
            # Named so the caller can see why the dataset survived, rather than
            # concluding the delete silently half-failed.
            "retained_for_jobs": siblings,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

@app.get("/status/{job_id}")
async def get_training_status(job_id: str):
    """Return the current status of a training job."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return training_jobs[job_id]

@app.get("/jobs")
async def list_jobs():
    """List all training jobs (active and historical)."""
    return list(training_jobs.values())

@app.get("/download/{job_id}")
async def download_model(job_id: str):
    """Download the exported ONNX model for a training job.

    Looks up the exported file directly (job_id is path-sanitised) so models
    remain downloadable after a container restart, when completed jobs are no
    longer present in the in-memory job registry.
    """
    job_id = _safe_name(job_id, "job_id")

    model_path = Path(f"models/{job_id}/{job_id}.onnx")
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model file not found")

    return FileResponse(
        path=model_path,
        filename=f"{job_id}.onnx",
        media_type="application/octet-stream"
    )

@app.delete("/job/{job_id}")
async def cancel_training(job_id: str):
    """Request cancellation of a running training job."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = training_jobs[job_id]
    if job.status in TERMINAL_STATUSES:
        raise HTTPException(status_code=400, detail="Job cannot be cancelled")
    if not active_runs.is_active(job_id):
        # e.g. "interrupted" after a restart: no thread exists to stop, and marking
        # it cancelled would only hide that it can still be resumed.
        raise HTTPException(
            status_code=400,
            detail=f"Job is not running (status '{job.status}'); resume it with POST /resume-training or delete it.",
        )

    job.status = "cancelled"
    job.message = "Training cancelled by user"

    return {"message": "Training job cancelled"}

async def run_training(job_id: str, request: TrainingRequest):
    """Background task: run VITS training then export the model."""
    try:
        _advance(job_id, "training", "Training in progress...")

        # Run training in a separate thread to avoid blocking the event loop
        await asyncio.to_thread(
            training_pipeline.train_sync,
            job_id=job_id,
            request=request,
            callback=lambda update: update_training_status(job_id, update),
        )

        # Export model to ONNX format and deploy it
        deployment_result = await _export_and_deploy(job_id, request.model_name, request.deployment_target)
        _complete(job_id, deployment_result)

    except TrainingCancelled as cancelled:
        logger.info(f"Training cancelled for {job_id}: {cancelled}")
        _settle(job_id, "cancelled", _cancel_message(job_id, cancelled))
    except _ExportFailed as export_failed:
        logger.error(f"Model export failed for {job_id}: {export_failed}")
        _fail_export(job_id, export_failed)
    except Exception as e:
        _settle(job_id, "failed", f"Training failed: {str(e)}")
        logger.error(f"Training error for {job_id}: {e}")

def update_training_status(job_id: str, update: dict):
    """Apply a training-loop status update to the in-memory job record."""
    job = training_jobs.get(job_id)
    if job is None:
        return None
    if 'check_status' in update:
        return job
    for key, value in update.items():
        if key == 'status' and (job.status == 'cancelled' or value == 'completed'):
            # The user's cancel is not the trainer's to overwrite, and "completed"
            # from the trainer only means the weights are saved: the runner still
            # exports and deploys, and decides when the job is done. Letting it
            # through made a polling client see "completed" before there was a model.
            continue
        if hasattr(job, key):
            # Sanitize float values — JSON cannot serialize NaN/Inf
            if isinstance(value, float) and not math.isfinite(value):
                value = None
            setattr(job, key, value)
    return job

if __name__ == "__main__":
    # timeout_keep_alive: uvicorn's default of 5 s closes idle connections the
    # frontend's pooled client may still try to reuse ("server disconnected").
    # This is the effective place for it: the image starts through start.sh,
    # which runs this file, not the uvicorn CLI.
    uvicorn.run(
        app, host="0.0.0.0", port=8080,
        timeout_keep_alive=int(_env_number("UVICORN_TIMEOUT_KEEP_ALIVE", 75, int)),
    )
