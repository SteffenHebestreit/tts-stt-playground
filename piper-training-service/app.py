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
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
import librosa
import soundfile as sf
import aiohttp

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from training_pipeline import OptimizedTrainingPipeline
from data_processor import DataProcessor
from model_exporter import ModelExporter
from validation import (
    safe_name as _safe_name,
    coerce_resume_int as _coerce_resume_int,
    coerce_resume_path as _coerce_resume_path,
)

# Configure logging so all logger.info() calls actually output to stdout
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Restore interrupted training jobs from disk on startup."""
    await restore_interrupted_jobs()
    yield


app = FastAPI(
    title="Piper Voice Training Service",
    description="VITS neural network training pipeline for custom Piper TTS voice models",
    lifespan=_lifespan,
)

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
    stt_service_url: str = "http://stt-service:8000"
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
        custom_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(onnx_path, custom_dir / f"{model_name}.onnx")
        if config_path.exists():
            shutil.copy2(config_path, custom_dir / f"{model_name}.json")

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
            shutil.rmtree(model_dir)
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
            # Restore into in-memory dict so /status and /jobs endpoints work
            training_jobs[job_id] = TrainingStatus(
                job_id=job_id,
                status="interrupted",
                progress=round(state.get("epoch", 0) / max(state.get("total_epochs", 1), 1) * 100, 1),
                current_epoch=state.get("epoch", 0),
                total_epochs=state.get("total_epochs", 10000),
                loss=state.get("loss"),
                message=(
                    f"Interrupted at epoch {state.get('epoch', '?')} — "
                    f"use POST /resume-training to continue."
                ),
                model_name=state.get("model_name"),
            )
            logger.info(f"Restored interrupted job {job_id} (epoch {state.get('epoch')})")
        except Exception as e:
            logger.warning(f"Could not restore job from {state_file}: {e}")


@app.get("/")
async def root():
    """Root endpoint — service identity."""
    return {"service": "PiperTTS Training Service", "status": "ready", "version": "1.0.0"}

@app.get("/health")
async def health():
    """Liveness / readiness probe."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/deployment-targets")
async def deployment_targets():
    """Return the configured deployment targets for exported model bundles."""
    return DEPLOYMENT_TARGET_REGISTRY

@app.post("/restore-backup")
async def restore_backup():
    """Restore training data from a local backup directory."""
    try:
        # Check if backup exists in current directory
        backup_dir = Path("./backup_stst_data")
        target_dir = Path("data/stst")
        
        if backup_dir.exists():
            import shutil
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(backup_dir, target_dir)
            return {"message": "Backup restored successfully", "status": "success"}
        else:
            return {"message": "No backup found in current directory", "status": "error"}
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
            
        mel_dir.mkdir(exist_ok=True)
        
        # Get all audio files
        audio_files = list(audio_dir.glob("*.wav"))
        
        # Process missing mel spectrograms
        processed = 0
        for audio_file in audio_files:
            mel_file = mel_dir / f"{audio_file.stem}.npy"
            
            if not mel_file.exists():
                try:
                    # Load audio
                    audio, sr = librosa.load(audio_file, sr=22050)
                    
                    # Generate mel spectrogram
                    mel_spec = data_processor._compute_mel_spectrogram(audio)
                    
                    # Save mel spectrogram
                    np.save(mel_file, mel_spec)
                    processed += 1
                    
                    if processed % 100 == 0:
                        logger.info(f"Processed {processed} mel spectrograms...")
                        
                except Exception as e:
                    logger.error(f"Error processing {audio_file}: {e}")
                    continue
        
        return {
            "message": f"Generated {processed} missing mel spectrograms", 
            "total_audio_files": len(audio_files),
            "processed": processed,
            "status": "success"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating mel spectrograms: {str(e)}")

@app.post("/prepare-dataset")
async def prepare_dataset(dataset: DatasetUpload):
    """Create a training dataset from pre-segmented STT results."""
    model_name = _safe_name(dataset.model_name, "model_name")
    try:
        dataset_path = await data_processor.prepare_dataset(
            segments=dataset.segments,
            model_name=model_name
        )
        return {
            "status": "success",
            "dataset_path": str(dataset_path),
            "num_samples": len(dataset.segments)
        }
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

@app.post("/train")
async def train_model(
    background_tasks: BackgroundTasks,
    model_name: str = Form(...),
    audio_files: List[UploadFile] = File(...),
    language: str = Form("de"),
    sample_rate: int = Form(22050),
    quality: str = Form("medium"),
    stt_service_url: str = Form("http://stt-service:8000"),
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
    """
    job_id = str(uuid.uuid4())

    model_name = _safe_name(model_name, "model_name")
    resolved_target_id, _ = resolve_deployment_target(deployment_target or None)

    try:
        logger.info(f"Starting STT-based training for model: {model_name}")
        logger.info(f"Received {len(audio_files)} audio files")
        
        # Initialize training job status
        training_jobs[job_id] = TrainingStatus(
            job_id=job_id,
            status="initializing",
            progress=0,
            current_epoch=0,
            total_epochs=epochs,
            loss=None,
            message="Initializing STT-based training pipeline...",
            model_name=model_name,
            deployment_target=resolved_target_id,
        )
        
        # Create dataset directory
        dataset_path = Path(f"data/{model_name}")
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Create temporary directory for uploaded files
        temp_audio_dir = dataset_path / "temp_uploads"
        temp_audio_dir.mkdir(exist_ok=True)
        
        # Save uploaded audio files
        uploaded_files = []
        total_size_mb = 0
        
        for audio_file in audio_files:
            if not audio_file.filename:
                continue
                
            # Sanitize filename
            safe_filename = Path(audio_file.filename).name
            file_path = temp_audio_dir / safe_filename
            
            logger.info(f"Saving uploaded file: {safe_filename}")
            
            # Save file
            async with aiofiles.open(file_path, 'wb') as f:
                while content := await audio_file.read(1024 * 1024):  # 1MB chunks
                    await f.write(content)
            
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            total_size_mb += file_size_mb
            
            uploaded_files.append(file_path)
            logger.info(f"Saved {safe_filename} ({file_size_mb:.1f}MB)")
        
        logger.info(f"Total uploaded: {len(uploaded_files)} files, {total_size_mb:.1f}MB")
        
        # Update status
        training_jobs[job_id].message = f"Processing {len(uploaded_files)} audio files through STT service..."
        training_jobs[job_id].progress = 10
        
        # Start background processing
        background_tasks.add_task(
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
        
    except Exception as e:
        logger.error(f"Error starting STT-based training: {e}")
        if job_id in training_jobs:
            training_jobs[job_id].status = "failed"
            training_jobs[job_id].message = f"Failed to start training: {str(e)}"
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
        training_jobs[job_id].status = "processing"
        training_jobs[job_id].message = "Processing audio files through STT service..."
        training_jobs[job_id].progress = 20
        
        # Initialize audio segmenter
        segmenter = AudioSegmenter()
        
        # Quality filters for training segments
        quality_filters = {
            'min_duration': 1.0,      # Minimum 1 second
            'max_duration': 15.0,     # Maximum 15 seconds  
            'min_confidence': 0.6,    # Minimum 60% confidence
            'min_text_length': 10     # Minimum 10 characters
        }
        
        # Process audio files through STT and create segments
        training_segments, stats = await segmenter.process_multiple_audio_files(
            audio_files,
            dataset_path,
            model_name,
            stt_service_url,
            sample_rate,
            quality_filters
        )
        
        # Update status
        training_jobs[job_id].progress = 60
        training_jobs[job_id].message = f"Created {len(training_segments)} training segments. Generating metadata..."
        
        if not training_segments:
            raise Exception("No valid training segments were created from the audio files")
        
        # Generate mel spectrograms for all training segments
        logger.info(f"Generating mel spectrograms for {len(training_segments)} segments...")
        mel_dir = dataset_path / "mel"
        mel_dir.mkdir(exist_ok=True)

        mel_generated = 0
        for seg in training_segments:
            try:
                mel_path = mel_dir / f"{seg.audio_path.stem}.npy"
                if not mel_path.exists():
                    audio, sr = librosa.load(str(seg.audio_path), sr=sample_rate)
                    mel_spec = data_processor._compute_mel_spectrogram(audio)
                    np.save(mel_path, mel_spec)
                    mel_generated += 1
            except Exception as e:
                logger.warning(f"Mel generation failed for {seg.audio_path.name}: {e}")

        logger.info(f"Generated {mel_generated} mel spectrograms")

        # Generate training metadata (train.json + val.json) with train/val split
        metadata_path = segmenter.generate_training_metadata(
            training_segments,
            dataset_path,
            model_name,
            language=language
        )

        logger.info(f"Generated training metadata: {metadata_path}")
        logger.info(f"Training dataset ready with {len(training_segments)} segments")
        
        # Clean up temporary upload directory
        temp_audio_dir = dataset_path / "temp_uploads"
        if temp_audio_dir.exists():
            import shutil
            shutil.rmtree(temp_audio_dir)
            logger.info(f"Cleaned up temporary files")
        
        # Update status before training
        training_jobs[job_id].progress = 70
        training_jobs[job_id].message = "Starting model training with optimized pipeline..."
        
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
        training_jobs[job_id].status = "exporting"
        training_jobs[job_id].progress = 90
        training_jobs[job_id].message = "Exporting model to ONNX format..."
        logger.info(f"Exporting model to ONNX for job {job_id}")

        try:
            onnx_path = await model_exporter.export_to_onnx(job_id)
            logger.info(f"Model exported to ONNX: {onnx_path}")
        except Exception as export_err:
            logger.error(f"ONNX export failed for job {job_id}: {export_err}")
            # Training succeeded but export failed — still mark as completed with warning
            training_jobs[job_id].status = "completed"
            training_jobs[job_id].progress = 100
            training_jobs[job_id].message = f"Training completed but ONNX export failed: {export_err}. Checkpoint saved."
            logger.info(f"STT-based training completed (export failed) for job {job_id}")
            return

        try:
            deployment_result = await deploy_model_bundle(job_id, model_name, onnx_path, deployment_target)
            logger.info(f"Deployment result for {job_id}: {deployment_result}")
        except Exception as deploy_error:
            logger.warning(f"Model export succeeded but deployment failed for {job_id}: {deploy_error}")
            deployment_result = {"status": "failed", "message": str(deploy_error)}

        # Final status update
        training_jobs[job_id].status = "completed"
        training_jobs[job_id].progress = 100
        if deployment_result.get("status") == "deployed":
            training_jobs[job_id].message = (
                f"Training completed and model deployed to {deployment_result['target']}. "
                f"Created from {len(training_segments)} segments ({stats['training_audio_duration']:.1f}s of audio)"
            )
        elif deployment_result.get("status") == "skipped":
            training_jobs[job_id].message = (
                f"Training completed. Exported model retained for manual download. "
                f"Created from {len(training_segments)} segments ({stats['training_audio_duration']:.1f}s of audio)"
            )
        else:
            training_jobs[job_id].message = (
                f"Training completed but deployment failed: {deployment_result.get('message')}. "
                f"Exported model is still available for download."
            )

        logger.info(f"STT-based training completed for job {job_id}")
        
    except Exception as e:
        logger.error(f"STT-based training failed for job {job_id}: {e}")
        training_jobs[job_id].status = "failed"
        training_jobs[job_id].message = f"Training failed: {str(e)}"
        
        # Clean up on failure
        try:
            temp_audio_dir = dataset_path / "temp_uploads"
            if temp_audio_dir.exists():
                import shutil
                shutil.rmtree(temp_audio_dir)
        except OSError as cleanup_err:
            logger.warning(f"Cleanup failed after training error: {cleanup_err}")


@app.post("/resume-training")
async def resume_training(
    background_tasks: BackgroundTasks,
    model_name: str = Form(...),
    job_id: str = Form(""),          # optional — auto-detect latest if blank
    extra_epochs: int = Form(0),     # 0 = continue to original total_epochs
    deployment_target: str = Form(""),
):
    """
    Resume an interrupted training job from its latest checkpoint.
    Finds the newest checkpoint_epoch_N.pt for the job and continues from there.
    """
    model_name = _safe_name(model_name, "model_name")
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
    language         = target_state.get("language", "de")
    config           = target_state.get("config") if isinstance(target_state.get("config"), dict) else {}

    if not resumed_job_id:
        raise HTTPException(
            status_code=404,
            detail=f"Interrupted job state for model '{model_name}' is missing a valid job id."
        )

    if extra_epochs > 0:
        total_epochs = saved_epoch + extra_epochs

    total_epochs = max(total_epochs, saved_epoch or 1)

    if latest_ckpt_path is None or not latest_ckpt_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Checkpoint file not found: {target_state.get('latest_checkpoint')}"
        )

    # Update/create the in-memory job record
    training_jobs[resumed_job_id] = TrainingStatus(
        job_id=resumed_job_id,
        status="training",
        progress=round(saved_epoch / total_epochs * 100, 1),
        current_epoch=saved_epoch,
        total_epochs=total_epochs,
        loss=target_state.get("loss"),
        message=f"Resuming from epoch {saved_epoch}/{total_epochs}...",
        model_name=model_name,
        deployment_target=resolved_target_id,
    )

    training_request = TrainingRequest(
        model_name=model_name,
        language=language,
        sample_rate=config.get("sample_rate", 22050),
        quality=config.get("quality", "medium"),
        epochs=total_epochs,
        batch_size=config.get("batch_size", 16),
        deployment_target=resolved_target_id,
    )

    async def _resume():
        """Resume training in a background thread, then export the updated model."""
        await asyncio.to_thread(
            training_pipeline.train_sync,
            job_id=resumed_job_id,
            request=training_request,
            callback=lambda update: update_training_status(resumed_job_id, update),
            resume_from=str(latest_ckpt_path),
        )
        # Export after training completes
        try:
            training_jobs[resumed_job_id].status = "exporting"
            onnx_path = await model_exporter.export_to_onnx(resumed_job_id)
            deployment_result = await deploy_model_bundle(resumed_job_id, model_name, onnx_path, resolved_target_id)
            training_jobs[resumed_job_id].status = "completed"
            training_jobs[resumed_job_id].progress = 100
            training_jobs[resumed_job_id].message = (
                f"Resumed training complete. Deployment target: {deployment_result['target']} ({deployment_result['status']})."
            )
        except Exception as e:
            logger.error(f"Export after resume failed: {e}")
            training_jobs[resumed_job_id].message = f"Training complete but export failed: {e}"

    background_tasks.add_task(_resume)

    return JSONResponse(status_code=202, content={
        "message": f"Resuming training for '{model_name}' from epoch {saved_epoch}",
        "job_id": resumed_job_id,
        "resume_from_epoch": saved_epoch,
        "total_epochs": total_epochs,
        "checkpoint": str(latest_ckpt_path),
    })


@app.post("/train-from-dataset")
async def train_from_dataset(
    background_tasks: BackgroundTasks,
    model_name: str = Form(...),
    language: str = Form("de"),
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
    train_json = Path(f"data/{model_name}/train.json")
    val_json = Path(f"data/{model_name}/val.json")

    if not train_json.exists() or not val_json.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Dataset not ready: need data/{model_name}/train.json and val.json"
        )

    import json as _json
    n_train = len(_json.loads(train_json.read_text()))
    n_val   = len(_json.loads(val_json.read_text()))

    resolved_target_id, _ = resolve_deployment_target(deployment_target or None)

    job_id = str(uuid.uuid4())
    training_jobs[job_id] = TrainingStatus(
        job_id=job_id,
        status="training",
        progress=0,
        current_epoch=0,
        total_epochs=epochs,
        loss=None,
        message=f"Starting training with {n_train} train / {n_val} val samples...",
        model_name=model_name,
        deployment_target=resolved_target_id,
    )

    background_tasks.add_task(run_training, job_id, TrainingRequest(
        model_name=model_name,
        language=language,
        sample_rate=22050,
        quality="medium",
        epochs=epochs,
        batch_size=batch_size,
        deployment_target=resolved_target_id,
    ))

    return JSONResponse(status_code=202, content={
        "message": f"Training started for model '{model_name}'",
        "job_id": job_id,
        "train_samples": n_train,
        "val_samples": n_val,
        "epochs": epochs,
    })


@app.post("/retrain-from-segments")
async def retrain_from_segments(
    background_tasks: BackgroundTasks,
    model_name: str = Form(...),
    language: str = Form("de"),
    epochs: int = Form(10000),
    batch_size: int = Form(32),
    prefix_filter: str = Form(""),
    stt_service_url: str = Form("http://stt-service:8000"),
    deployment_target: str = Form(""),
):
    """
    Rebuild metadata from existing pre-segmented audio files and retrain.

    Skips upload and re-segmentation — uses whatever WAV files are already
    in data/{model_name}/audio/.  Runs STT on each clip to get transcriptions,
    then calls generate_training_metadata() and train_sync().
    """
    model_name = _safe_name(model_name, "model_name")
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
    training_jobs[job_id] = TrainingStatus(
        job_id=job_id,
        status="initializing",
        progress=0,
        current_epoch=0,
        total_epochs=epochs,
        loss=None,
        message=f"Found {len(wav_files)} audio segments — starting STT transcription...",
        model_name=model_name,
        deployment_target=resolved_target_id,
    )

    background_tasks.add_task(
        _run_retrain_from_segments,
        job_id, model_name, wav_files, dataset_path,
        language, epochs, batch_size, stt_service_url, resolved_target_id,
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
    from stt_processor import STTProcessor
    import librosa

    try:
        total = len(wav_files)
        logger.info(f"[{job_id}] Retraining from {total} existing segments for model '{model_name}'")

        training_jobs[job_id].status = "transcribing"
        training_jobs[job_id].message = f"Running STT on {total} audio segments (0/{total})..."

        # Semaphore limits concurrent STT calls to avoid overloading the service
        CONCURRENCY = 8
        sem = asyncio.Semaphore(CONCURRENCY)
        completed = 0
        training_segments = []
        lock = asyncio.Lock()

        async def transcribe_one(audio_path: Path):
            """Transcribe one pre-segmented WAV clip and append valid results."""
            nonlocal completed
            async with sem:
                try:
                    async with STTProcessor(stt_service_url) as stt:
                        result = await stt.transcribe_audio_file(audio_path, return_segments=True)

                    text = ""
                    # Prefer the full text field (most reliable for short clips)
                    if result.get("text", "").strip():
                        text = result["text"].strip()
                    elif result.get("segments"):
                        text = " ".join(s.get("text", "") for s in result["segments"]).strip()

                    if not text or len(text) < 5:
                        return  # Skip clips with no / too-short transcription

                    duration = librosa.get_duration(path=str(audio_path))
                    seg = TrainingSegment(
                        audio_path=audio_path,
                        text=text,
                        duration=duration,
                        speaker_id=0,
                        confidence=1.0,
                        original_file=audio_path.stem,
                        start_time=0.0,
                        end_time=duration,
                    )
                    async with lock:
                        training_segments.append(seg)

                except Exception as e:
                    logger.warning(f"[{job_id}] STT failed for {audio_path.name}: {e}")
                finally:
                    async with lock:
                        completed += 1
                        if completed % 100 == 0 or completed == total:
                            pct = 10 + int(50 * completed / total)
                            training_jobs[job_id].progress = pct
                            training_jobs[job_id].message = (
                                f"STT transcription: {completed}/{total} clips done, "
                                f"{len(training_segments)} valid so far..."
                            )
                            logger.info(f"[{job_id}] {completed}/{total} transcribed, {len(training_segments)} valid")

        await asyncio.gather(*[transcribe_one(p) for p in wav_files])

        if not training_segments:
            raise RuntimeError("STT produced no valid transcriptions — check STT service logs.")

        logger.info(f"[{job_id}] STT done: {len(training_segments)}/{total} segments kept")

        # Rebuild metadata files
        training_jobs[job_id].progress = 62
        training_jobs[job_id].message = f"Building metadata for {len(training_segments)} segments..."

        segmenter = AudioSegmenter()
        segmenter.generate_training_metadata(
            training_segments, dataset_path, model_name, language=language
        )

        # Training
        training_jobs[job_id].progress = 65
        training_jobs[job_id].status = "training"
        training_jobs[job_id].message = "Starting VITS training..."

        training_request = TrainingRequest(
            model_name=model_name,
            language=language,
            sample_rate=22050,
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
        training_jobs[job_id].status = "exporting"
        training_jobs[job_id].progress = 92
        training_jobs[job_id].message = "Exporting model to ONNX..."

        onnx_path = await model_exporter.export_to_onnx(job_id)
        deployment_result = await deploy_model_bundle(job_id, model_name, onnx_path, deployment_target)

        training_jobs[job_id].status = "completed"
        training_jobs[job_id].progress = 100
        training_jobs[job_id].message = (
            f"Training complete! {len(training_segments)} segments, "
            f"model '{model_name}' deployment status: {deployment_result['status']} on {deployment_result['target']}."
        )
        logger.info(f"[{job_id}] Retrain from segments completed for '{model_name}'")

    except Exception as e:
        logger.error(f"[{job_id}] Retrain from segments failed: {e}")
        training_jobs[job_id].status = "failed"
        training_jobs[job_id].message = f"Failed: {e}"


@app.post("/export/{job_id}")
async def manual_export_model(job_id: str, model_name: str = Form(...), deployment_target: str = Form("")):
    """Manually export a completed training checkpoint and deploy it to a configured target."""
    job_id = _safe_name(job_id, "job_id")
    model_name = _safe_name(model_name, "model_name")
    try:
        resolved_target_id, _ = resolve_deployment_target(deployment_target or None)
        # Check if checkpoint exists
        checkpoint_path = Path(f"checkpoints/{job_id}/final_model.pt")
        if not checkpoint_path.exists():
            raise HTTPException(status_code=404, detail=f"Model checkpoint not found: {checkpoint_path}")
        
        # Export model to ONNX format
        onnx_path = await model_exporter.export_to_onnx(job_id)
        
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

@app.delete("/model/{job_id}")
async def delete_trained_model(job_id: str):
    """Delete a trained model and all associated files (checkpoint, export, dataset)."""
    job_id = _safe_name(job_id, "job_id")
    try:
        import shutil

        # Remove checkpoint directory
        checkpoint_dir = Path(f"checkpoints/{job_id}")
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
            logger.info(f"Deleted checkpoint directory: {checkpoint_dir}")
        
        # Remove exported model directory
        model_dir = Path(f"models/{job_id}")
        if model_dir.exists():
            shutil.rmtree(model_dir)
            logger.info(f"Deleted model directory: {model_dir}")
        
        # Look up model name from training job record
        model_name = None
        deployment_target = None
        if job_id in training_jobs:
            model_name = training_jobs[job_id].model_name
            deployment_target = training_jobs[job_id].deployment_target
            del training_jobs[job_id]
            logger.info(f"Removed job {job_id} from active jobs")

        # Remove dataset directory
        if model_name:
            dataset_dir = Path(f"data/{model_name}")
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)
                logger.info(f"Deleted dataset directory: {dataset_dir}")

        if model_name:
            try:
                await remove_model_from_deployment_target(model_name, deployment_target)
            except Exception as tts_error:
                logger.warning(f"Could not remove from deployment target: {tts_error}")
        
        return {
            "message": f"Model {job_id} deleted successfully",
            "deleted_items": [
                "checkpoint directory",
                "model directory", 
                "dataset directory",
                "training job record"
            ]
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
    
    if training_jobs[job_id].status in ["completed", "failed", "cancelled"]:
        raise HTTPException(status_code=400, detail="Job cannot be cancelled")
    
    training_jobs[job_id].status = "cancelled"
    training_jobs[job_id].message = "Training cancelled by user"
    
    return {"message": "Training job cancelled"}

async def run_training(job_id: str, request: TrainingRequest):
    """Background task: run VITS training then export the model."""
    try:
        training_jobs[job_id].status = "training"
        training_jobs[job_id].message = "Training in progress..."
        
        # Run training in a separate thread to avoid blocking the event loop
        await asyncio.to_thread(
            training_pipeline.train_sync,
            job_id=job_id,
            request=request,
            callback=lambda update: update_training_status(job_id, update),
        )
        
        # Export model to ONNX format
        training_jobs[job_id].status = "exporting"
        training_jobs[job_id].message = "Exporting model to ONNX format..."
        
        try:
            onnx_path = await model_exporter.export_to_onnx(job_id)
            
            deployment_result = await deploy_model_bundle(
                job_id,
                request.model_name,
                onnx_path,
                request.deployment_target,
            )
            
            training_jobs[job_id].status = "completed"
            training_jobs[job_id].progress = 100.0
            training_jobs[job_id].message = (
                f"Training completed. Deployment target: {deployment_result['target']} ({deployment_result['status']})."
            )
            
        except Exception as export_error:
            logger.error(f"Model export failed for {job_id}: {export_error}")
            training_jobs[job_id].status = "completed"
            training_jobs[job_id].progress = 100.0
            training_jobs[job_id].message = f"Training completed, but model export failed: {str(export_error)}"
        
    except Exception as e:
        training_jobs[job_id].status = "failed"
        training_jobs[job_id].message = f"Training failed: {str(e)}"
        logger.error(f"Training error for {job_id}: {e}")

def update_training_status(job_id: str, update: dict):
    """Apply a training-loop status update to the in-memory job record."""
    if job_id in training_jobs:
        if 'check_status' in update:
            return training_jobs[job_id]
        for key, value in update.items():
            if hasattr(training_jobs[job_id], key):
                # Sanitize float values — JSON cannot serialize NaN/Inf
                if isinstance(value, float) and not math.isfinite(value):
                    value = None
                setattr(training_jobs[job_id], key, value)
        return training_jobs[job_id]
    return None

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
