"""Guards for two training-job lifecycle bugs that both ended in the wrong artefact.

**Cancellation did not cancel.** `train_sync` checked the job status once per
epoch and `break`, then fell straight into the block below the loop — which
saves `final_model.pt`, stamps `job_state.json` as `completed`, and calls back
with `status='completed'`, overwriting the `cancelled` the user had just set.
The caller then exported and *deployed* the half-trained model. So
`DELETE /job/{id}` returned 200, the UI said "Training job cancelled", and the
voice the operator was trying to stop went live a few minutes later.

**Deleting a job deleted another job's data.** The dataset directory and the
deployed voice are keyed on the model *name*; the delete endpoint is keyed on
the *job id*. Retraining a voice is the normal workflow and produces several
jobs sharing one `data/<name>` and one deployed voice, so removing an old job
took the current one's dataset and undeployed its voice.

Static, because `app.py` and `training_pipeline.py` both import torch. The
epoch-bound validator these endpoints now call is unit-tested for real in
`test_piper_training_validation.py`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPO_ROOT / "piper-training-service"
APP = SERVICE_DIR / "app.py"
PIPELINE = SERVICE_DIR / "training_pipeline.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _function(path: Path, name: str):
    return next(
        (n for n in ast.walk(_tree(path))
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name),
        None,
    )


# --- cancellation ------------------------------------------------------------


def test_the_pipeline_defines_a_distinct_cancellation_signal():
    """Not an ordinary failure: the checkpoints survive and the job is resumable,
    so the callers must be able to tell the two apart."""
    source = PIPELINE.read_text(encoding="utf-8")
    assert "class TrainingCancelled(Exception)" in source, (
        "training_pipeline has no TrainingCancelled, so a cancelled run is "
        "indistinguishable from a completed one at the call site"
    )


def test_cancellation_raises_instead_of_falling_through_to_the_export_path():
    """The whole bug in one assertion: a `break` alone is not enough."""
    train_sync = _function(PIPELINE, "train_sync")
    assert train_sync is not None, "train_sync not found"
    body = ast.unparse(train_sync)

    assert "raise TrainingCancelled" in body, (
        "train_sync still leaves the epoch loop without raising, so execution "
        "continues into the final_model.pt save and the completed callback"
    )
    # The raise has to come before the artefacts are written, or it changes nothing.
    raise_at = body.index("raise TrainingCancelled")
    for artefact in ("final_model.pt", "'completed'"):
        assert artefact in body, f"train_sync no longer writes {artefact}?"
        assert raise_at < body.index(artefact), (
            f"train_sync writes {artefact} before raising TrainingCancelled — the "
            f"cancelled job still produces a deployable model"
        )


def test_a_cancelled_job_is_not_stamped_completed_on_disk():
    """`restore_interrupted_jobs` and `/resume-training` both skip `completed`,
    so a cancelled job marked that way could never be resumed."""
    body = ast.unparse(_function(PIPELINE, "train_sync"))
    assert "_mark_state('cancelled')" in body or '_mark_state("cancelled")' in body, (
        "train_sync does not record 'cancelled' in job_state.json"
    )


# Every place that runs train_sync and then exports.
TRAINING_CALLERS = [
    "run_stt_based_training",
    "run_training",
    "_run_retrain_from_segments",
    "_resume",
]


@pytest.mark.parametrize("caller", TRAINING_CALLERS)
def test_every_caller_handles_cancellation_separately_from_failure(caller: str):
    node = _function(APP, caller)
    assert node is not None, f"{caller} not found — did it move?"
    body = ast.unparse(node)

    assert "TrainingCancelled" in body, (
        f"{caller} does not handle TrainingCancelled. Either it reports a "
        f"cancelled job as failed, or the exception escapes a BackgroundTask "
        f"and the job sits at 'training' forever."
    )
    assert '"cancelled"' in body or "'cancelled'" in body, (
        f"{caller} catches TrainingCancelled but does not set the job status to "
        f"'cancelled'"
    )


@pytest.mark.parametrize("caller", TRAINING_CALLERS)
def test_every_caller_guards_the_train_sync_call(caller: str):
    """These all run as BackgroundTasks, where an unhandled exception is
    swallowed by the event loop and leaves the job status stuck."""
    node = _function(APP, caller)
    assert node is not None

    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        rendered = ast.unparse(sub)
        if "train_sync" not in rendered:
            continue
        # Walk up: the call must sit inside a Try somewhere in this function.
        guarded = any(
            isinstance(ancestor, ast.Try)
            and "train_sync" in ast.unparse(ancestor.body)
            for ancestor in ast.walk(node)
        )
        assert guarded, (
            f"{caller} calls train_sync outside any try block; a failure there "
            f"vanishes into the BackgroundTask and the job never leaves 'training'"
        )
        return
    pytest.fail(f"{caller} no longer calls train_sync")


# --- delete ------------------------------------------------------------------


def test_delete_checks_for_other_jobs_before_removing_shared_data():
    node = _function(APP, "delete_trained_model")
    assert node is not None, "delete_trained_model not found"
    body = ast.unparse(node)

    assert "_other_jobs_using_model" in body, (
        "delete_trained_model removes data/<model_name> and undeploys the voice "
        "without checking whether another job trained the same name. Retraining "
        "a voice produces exactly that situation."
    )
    # The guard has to gate both destructive steps, not just one.
    assert "rmtree" in body and "remove_model_from_deployment_target" in body


def test_the_sibling_lookup_excludes_the_job_being_deleted():
    """Otherwise every delete finds itself and nothing is ever cleaned up."""
    node = _function(APP, "_other_jobs_using_model")
    assert node is not None, "_other_jobs_using_model not found"
    params = [a.arg for a in node.args.args]
    assert "excluding_job_id" in params, (
        f"_other_jobs_using_model{tuple(params)} takes no exclusion argument, so "
        f"it will always match the job being deleted"
    )
    assert "excluding_job_id" in ast.unparse(node)


def test_delete_resolves_the_model_name_from_disk_too():
    """Completed jobs are deliberately not restored into memory after a restart,
    so the in-memory record cannot be the only source."""
    body = ast.unparse(_function(APP, "delete_trained_model"))
    assert "_model_name_from_disk" in body, (
        "delete_trained_model only reads model_name from training_jobs, which is "
        "empty for completed jobs after a restart — the dataset and the deployed "
        "voice are then silently left behind"
    )


def test_delete_reports_what_it_kept():
    """A caller must be able to tell 'retained on purpose' from 'half-failed'."""
    body = ast.unparse(_function(APP, "delete_trained_model"))
    assert "retained_for_jobs" in body


# --- epoch bounds ------------------------------------------------------------


@pytest.mark.parametrize("endpoint", [
    "train_model", "train_from_dataset", "retrain_from_segments", "resume_training",
])
def test_epoch_counts_are_bounded_at_every_entry_point(endpoint: str):
    """`epochs=0` produced a job reporting 0/0, which divides by zero in the
    resume path's own progress calculation."""
    node = _function(APP, endpoint)
    assert node is not None, f"{endpoint} not found"
    assert "_validate_epochs" in ast.unparse(node), (
        f"{endpoint} accepts an unbounded epoch count"
    )


# --- ffmpeg invocation -------------------------------------------------------
#
# `-ss` after `-i` is *output* seeking: ffmpeg decodes the file from the start
# and discards everything before the cut point. Cutting one recording into N
# segments therefore cost N full decodes, so segmenting a long upload was
# quadratic in the number of segments. Before `-i` it seeks the input directly,
# and has been accurate as well as fast since ffmpeg 2.1.

FFMPEG_CALLERS = [
    (SERVICE_DIR / "audio_segmenter.py", "extract_audio_segment"),
    (REPO_ROOT / "qwen3-tts-service" / "app.py", "_trim_audio_segment"),
]


def _ffmpeg_argv(path: Path, func_name: str) -> list[str]:
    """The literal argv list built inside *func_name*."""
    node = _function(path, func_name)
    assert node is not None, f"{func_name} not found in {path.name}"
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Assign):
            continue
        target = sub.targets[0]
        if getattr(target, "id", None) != "cmd":
            continue
        assert isinstance(sub.value, ast.List), f"{func_name}: cmd is not a list literal"
        return [ast.unparse(el) for el in sub.value.elts]
    pytest.fail(f"{func_name} no longer builds a `cmd` list")


@pytest.mark.parametrize("path,func", FFMPEG_CALLERS, ids=lambda v: getattr(v, "name", v))
def test_ffmpeg_seeks_the_input_not_the_output(path: Path, func: str):
    argv = _ffmpeg_argv(path, func)
    assert "'-ss'" in argv, f"{func}: no -ss in the ffmpeg argv"
    assert "'-i'" in argv, f"{func}: no -i in the ffmpeg argv"
    assert argv.index("'-ss'") < argv.index("'-i'"), (
        f"{func} places -ss after -i, which makes ffmpeg decode the whole file "
        f"up to the cut point every time. Move it before -i."
    )


@pytest.mark.parametrize("path,func", FFMPEG_CALLERS, ids=lambda v: getattr(v, "name", v))
def test_ffmpeg_calls_are_bounded_by_a_timeout(path: Path, func: str):
    """A wedged ffmpeg must not stall segmentation or a request forever."""
    body = ast.unparse(_function(path, func))
    assert "timeout" in body, f"{func} runs ffmpeg with no timeout"


def test_segment_extraction_does_not_block_the_event_loop():
    """It runs inside a BackgroundTask, so a synchronous subprocess.run here
    froze /health and /status for the whole of segmentation."""
    body = ast.unparse(_function(SERVICE_DIR / "audio_segmenter.py", "extract_audio_segment"))
    assert "to_thread" in body, (
        "extract_audio_segment calls subprocess.run directly on the event loop"
    )


def test_the_train_val_split_goes_through_the_shared_helper():
    """Reproducibility lives in validation.split_train_val, which is unit-tested;
    an inline np.random.permutation here would bypass all of it."""
    body = ast.unparse(
        _function(SERVICE_DIR / "audio_segmenter.py", "generate_training_metadata"))
    assert "split_train_val" in body
    assert "np.random.permutation" not in body, (
        "the unseeded permutation is back — the split would stop being "
        "reproducible across retrains"
    )
