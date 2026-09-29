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

Static, because `training_pipeline.py` and `model_exporter.py` import torch. What is
left here are cheap structural backstops that no small behavioural test replaces:

* the pipeline raises `TrainingCancelled` before it writes `final_model.pt`, and every
  one of the four callers handles it and guards `train_sync`: the behaviour is run in
  `test_piper_training_core_pipeline.py` (needs torch, so it runs in CI's gating
  training job, not the unit job) and through the endpoints in
  `test_piper_training_service_jobs.py` for the main path, but not per caller;
* the export refuses a checkpoint with missing weights and the dataset checks its
  vocabulary against `n_vocab`: both live in torch-only code;
* the train/validation split goes through the shared helper.

The delete-sharing, epoch-bound and ffmpeg checks that used to be greps here are
behavioural tests now (see the note below). The epoch-bound validator itself is
unit-tested in `test_piper_training_validation.py`.
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
    "run_resume_training",
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
    """These all run detached from the request that started them, where an
    unhandled exception is swallowed by the event loop and leaves the job
    status stuck."""
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


# --- delete, epoch bounds, ffmpeg invocation --------------------------------
#
# These were source greps here and are behavioural tests now, which fail when the
# behaviour is wrong rather than when a name changes:
#   * deleting one of several jobs for a voice keeps its dataset and deployed voice
#     (sibling lookup, exclusion of the deleted job, name resolved from disk, the
#     `retained_for_jobs` report):
#       test_piper_training_service_jobs.py::test_deleting_one_of_several_jobs_for_a_voice_...
#   * the epoch bound at every entry point:
#       test_piper_training_service_data.py::test_every_other_way_of_starting_a_job_bounds_...
#       (and test_train_validates_what_it_would_otherwise_train_on for /train)
#   * ffmpeg seeks the input, is bounded by a timeout and stays off the event loop:
#       test_piper_training_service_data.py::test_a_segment_is_cut_by_seeking_the_input_...
#       test_qwen3_tts_voices.py::test_a_reference_segment_is_trimmed_by_seeking_the_input_...


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


# --- export integrity ---------------------------------------------------------
#
# `load_state_dict(..., strict=False)` with the result discarded. Training and
# export build the same VITS class, so a missing key means the checkpoint has no
# weights for that layer and the exported ONNX carries its random
# initialisation — a model that loads, runs, and emits noise, reported as a
# successful export and then deployed as a voice.

EXPORTER = SERVICE_DIR / "model_exporter.py"


def test_export_refuses_a_checkpoint_with_missing_weights():
    body = ast.unparse(_function(EXPORTER, "export_to_onnx"))
    assert "load_state_dict" in body, "export no longer loads a state dict?"
    assert "missing_keys" in body, (
        "export_to_onnx calls load_state_dict(strict=False) and ignores the "
        "result, so a checkpoint/architecture mismatch produces an ONNX full of "
        "untrained random values instead of an error"
    )
    assert "raise" in body.split("missing_keys", 1)[1][:400], (
        "missing_keys is inspected but not fatal"
    )


def test_export_tolerates_extra_tensors_in_the_checkpoint():
    """The benign direction: a checkpoint carrying more than inference needs."""
    body = ast.unparse(_function(EXPORTER, "export_to_onnx"))
    assert "unexpected_keys" in body, (
        "unexpected_keys is not distinguished from missing_keys; making both "
        "fatal would reject checkpoints that are perfectly usable"
    )


def test_the_weight_check_is_not_relabelled_as_an_onnx_failure():
    """The generic handler says 'may need more epochs', which is the wrong
    advice for an architecture mismatch and buries the real message."""
    body = ast.unparse(_function(EXPORTER, "export_to_onnx"))
    load_at = body.index("load_state_dict")
    export_at = body.index("torch.onnx.export")
    try_at = body.index("try:")
    assert load_at < try_at < export_at, (
        "the weight check sits inside the try whose handler rewrites every "
        "exception as an ONNX export failure"
    )


def test_the_dataset_checks_its_vocabulary_against_the_embedding_size():
    """An id at or above n_vocab is an out-of-range embedding lookup, which on
    GPU is a device-side assert hours into a run with an unrelated traceback."""
    body = ast.unparse(_function(SERVICE_DIR / "dataset.py", "__init__"))
    assert "n_vocab" in body, (
        "TTSDataset builds a phoneme vocabulary without comparing it to the "
        "configured embedding size"
    )
    assert "raise" in body.split("n_vocab", 1)[1][:600]
