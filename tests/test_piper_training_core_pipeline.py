"""CPU tests for the piper-training-service training loop: failures must be loud.

The loop used to log every per-batch exception and carry on, and with fewer
examples than the batch size its DataLoader was empty (``drop_last=True``). Either
way the job "completed": ``final_model.pt`` was saved with the initial random
weights, exported to ONNX and deployed as a voice. Checkpoints were never pruned
(10000 epochs at interval 5 is 2000 files), ``empty_cache()`` ran on every step,
and the vocabulary was rebuilt from whatever ``train.json`` held at the moment.

These run a real (tiny) model on CPU with synthetic mels; no audio, no GPU, no
downloads. They skip where torch is not installed (an error with REQUIRE_TORCH_TESTS=1, set by
the CI training job). ``PIPER_TRAINING_SERVICE_DIR``
points them at another checkout of the service, which is how they were run
against the pre-fix code to show they fail there.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from optional_deps import importorskip_unless_required as _need

# Skips where torch is missing, except with REQUIRE_TORCH_TESTS=1 (the CI training
# job), where that is an error: see tests/optional_deps.py.
torch = _need("torch")
np = _need("numpy")

REPO = Path(__file__).resolve().parents[1]
SERVICE_DIR = Path(os.environ.get("PIPER_TRAINING_SERVICE_DIR") or REPO / "piper-training-service")

MODULES = ("vits_model", "validation", "dataset", "training_utils", "training_pipeline")

TINY = dict(
    hidden_channels=16, inter_channels=16, n_layers=1, n_heads=2,
    n_vocab=64, vocoder_channels=16, save_interval=1,
)


@pytest.fixture(scope="module")
def svc():
    """The service's modules under their plain names, as the container imports them."""
    saved_path = list(sys.path)
    saved = {name: sys.modules.pop(name) for name in MODULES if name in sys.modules}
    sys.path.insert(0, str(SERVICE_DIR))
    try:
        yield SimpleNamespace(**{name: importlib.import_module(name) for name in MODULES})
    finally:
        for name in MODULES:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
        sys.path[:] = saved_path


# On the pre-fix code there is no TrainingFailed; the run simply does not raise,
# which is what these tests catch. RuntimeError lets them run there.
def _failed(svc):
    return getattr(svc.training_pipeline, "TrainingFailed", RuntimeError)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A working directory with the layout the service expects (data/, checkpoints/)."""
    monkeypatch.chdir(tmp_path)
    for name in ("TRAIN_MAX_CONSECUTIVE_FAILURES", "KEEP_LAST_CHECKPOINTS", "MAX_TRAIN_BATCH_SIZE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TRAIN_NUM_WORKERS", "0")  # load in-process: no fork per test
    return tmp_path


def make_dataset(root: Path, name: str = "voice", n_train: int = 8, n_val: int = 2,
                 phonemes=("ab", "abc", "cab", "bca")):
    """A prepared dataset: train.json, val.json and one synthetic mel per entry."""
    data = root / "data" / name
    (data / "mel").mkdir(parents=True)
    rng = np.random.default_rng(0)

    def entries(count, offset):
        out = []
        for i in range(count):
            idx = offset + i
            np.save(data / "mel" / f"{idx}.npy", rng.normal(size=(80, 12 + idx)).astype("float32"))
            out.append({"mel_path": f"mel/{idx}.npy", "text": "t", "phonemes": phonemes[idx % len(phonemes)]})
        return out

    train = entries(n_train, 0)
    val = entries(n_val, n_train)
    (data / "train.json").write_text(json.dumps(train))
    (data / "val.json").write_text(json.dumps(val))
    return data


def request(name="voice", epochs=2, batch_size=4):
    return SimpleNamespace(model_name=name, batch_size=batch_size, epochs=epochs, language="de")


def run(svc, job_id="job", req=None, resume_from=None, callback=None, **config):
    torch.manual_seed(0)  # weights and shuffling: keep the loss assertions repeatable
    pipeline = svc.training_pipeline.OptimizedTrainingPipeline()
    return pipeline.train_sync(
        job_id, req or request(), callback=callback,
        override_config=dict(TINY, **config), resume_from=resume_from,
    )


def ckpt_dir(root, job_id="job"):
    return root / "checkpoints" / job_id


# --- refusing to start / failing loudly ------------------------------------------


def test_refuses_to_start_when_the_train_split_is_smaller_than_the_batch(svc, workspace):
    """3 examples, batch 4, drop_last=True: zero batches per epoch. The job used
    to finish in milliseconds and deploy random weights."""
    make_dataset(workspace, n_train=3, n_val=1)

    with pytest.raises(_failed(svc), match="batch"):
        run(svc)

    assert not (ckpt_dir(workspace) / "final_model.pt").exists()


def test_a_dataset_whose_mels_are_all_missing_is_refused(svc, workspace):
    data = make_dataset(workspace, n_train=4, n_val=1)
    for mel in (data / "mel").glob("*.npy"):
        mel.unlink()

    with pytest.raises(ValueError, match="mel"):
        run(svc)

    assert not (ckpt_dir(workspace) / "final_model.pt").exists()


def test_aborts_after_consecutive_batch_failures_instead_of_finishing(svc, workspace, monkeypatch):
    make_dataset(workspace, n_train=8)
    monkeypatch.setenv("TRAIN_MAX_CONSECUTIVE_FAILURES", "3")
    calls = []

    def broken(self, batch):
        calls.append(1)
        raise RuntimeError("synthetic bug in the loss")

    monkeypatch.setattr(svc.vits_model.VITS, "compute_loss", broken)

    with pytest.raises(_failed(svc), match="consecutive"):
        run(svc, req=request(epochs=5, batch_size=2))

    assert len(calls) == 3, "must stop at the limit, not run 5 epochs of 4 failing batches"
    assert not (ckpt_dir(workspace) / "final_model.pt").exists()


def test_aborts_when_an_epoch_performed_zero_optimizer_steps(svc, workspace, monkeypatch):
    """Two failing batches per epoch is below the consecutive limit; the epoch
    still trained nothing and must not be counted as progress."""
    make_dataset(workspace, n_train=4)
    monkeypatch.setattr(
        svc.vits_model.VITS, "compute_loss",
        lambda self, batch: (_ for _ in ()).throw(RuntimeError("nothing works")),
    )

    with pytest.raises(_failed(svc), match="zero optimizer steps"):
        run(svc, req=request(epochs=3, batch_size=2))

    assert not (ckpt_dir(workspace) / "final_model.pt").exists()


def test_non_finite_losses_abort_the_run(svc, workspace, monkeypatch):
    make_dataset(workspace, n_train=8)
    monkeypatch.setenv("TRAIN_MAX_CONSECUTIVE_FAILURES", "2")
    original = svc.vits_model.VITS.compute_loss

    def nan_loss(self, batch):
        losses = original(self, batch)
        losses["reconstruction"] = losses["reconstruction"] * float("nan")
        return losses

    monkeypatch.setattr(svc.vits_model.VITS, "compute_loss", nan_loss)

    with pytest.raises(_failed(svc), match="non-finite"):
        run(svc, req=request(epochs=4, batch_size=2))

    assert not (ckpt_dir(workspace) / "final_model.pt").exists()


def test_a_non_finite_gradient_is_never_applied(svc, workspace, monkeypatch):
    """The loss check runs before backward and cannot see a NaN gradient; that
    used to pass through clip_grad_norm_ unchanged and turn every weight into NaN."""
    make_dataset(workspace, n_train=8)
    monkeypatch.setenv("TRAIN_MAX_CONSECUTIVE_FAILURES", "2")
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda *a, **k: torch.tensor(float("inf")))
    steps = []
    original_step = torch.optim.AdamW.step
    monkeypatch.setattr(
        torch.optim.AdamW, "step",
        lambda self, *a, **k: (steps.append(1), original_step(self, *a, **k))[1],
    )

    with pytest.raises(_failed(svc), match="non-finite gradient"):
        run(svc, req=request(epochs=4, batch_size=2))

    assert steps == [], "a step with a non-finite gradient was applied"


def test_a_failed_job_is_marked_failed_and_stays_resumable_on_disk(svc, workspace, monkeypatch):
    make_dataset(workspace, n_train=8)
    run(svc, req=request(epochs=2, batch_size=2))  # writes job_state.json
    monkeypatch.setenv("TRAIN_MAX_CONSECUTIVE_FAILURES", "2")
    monkeypatch.setattr(
        svc.vits_model.VITS, "compute_loss",
        lambda self, batch: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(_failed(svc)):
        run(svc, req=request(epochs=4, batch_size=2),
            resume_from=str(ckpt_dir(workspace) / "checkpoint_epoch_2.pt"))

    state = json.loads((ckpt_dir(workspace) / "job_state.json").read_text())
    assert state["status"] == "failed"
    assert "boom" in state["error"]


# --- the loop still trains --------------------------------------------------------


def test_one_batch_per_epoch_still_takes_optimizer_steps(svc, workspace, monkeypatch):
    """With gradient accumulation of 2 and one batch per epoch the accumulated
    gradient was discarded by the next epoch's zero_grad: zero steps, ever."""
    make_dataset(workspace, n_train=4)
    steps = []
    original_step = torch.optim.AdamW.step

    def counting_step(self, *args, **kwargs):
        steps.append(1)
        return original_step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", counting_step)

    run(svc, req=request(epochs=3, batch_size=4))

    assert len(steps) == 3


def test_training_reduces_the_loss_of_a_repeated_batch(svc, workspace):
    """A guard against the loop silently not learning (e.g. detached loss)."""
    make_dataset(workspace, n_train=4, n_val=1, phonemes=("abc",))
    seen = []

    def watch(update):
        if isinstance(update, dict) and "loss" in update:
            seen.append(update["loss"])

    run(svc, req=request(epochs=12, batch_size=4), callback=watch, learning_rate=0.003)

    assert len(seen) == 12 and all(np.isfinite(seen))
    assert seen[-1] < seen[0]


def test_reports_the_trainer_kind_at_start_and_records_it_everywhere(svc, workspace, caplog):
    make_dataset(workspace)
    seen = []
    caplog.set_level(logging.WARNING)

    run(svc, req=request(epochs=2, batch_size=4), callback=lambda u: seen.append(u))

    kind = svc.training_utils.TRAINER_KIND
    assert kind == "experimental-mel-flow"
    assert any(kind in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)
    assert any(isinstance(u, dict) and u.get("trainer_kind") == kind for u in seen)

    state = json.loads((ckpt_dir(workspace) / "job_state.json").read_text())
    assert state["trainer_kind"] == kind and state["trainer_caveat"] == svc.training_utils.TRAINER_CAVEAT
    for name in ("final_model.pt", "checkpoint_epoch_2.pt", "best_model.pt"):
        blob = torch.load(ckpt_dir(workspace) / name, map_location="cpu", weights_only=True)
        assert blob["trainer_kind"] == kind, name
    final = torch.load(ckpt_dir(workspace) / "final_model.pt", map_location="cpu", weights_only=True)
    assert final["optimizer_steps"] >= 1


def test_cancelling_still_withholds_the_final_model_and_marks_the_job_cancelled(svc, workspace):
    """Behavioural companion of the static guards in test_training_job_lifecycle.py:
    the loop was restructured, and cancelling must still not produce a deployable model."""
    make_dataset(workspace)
    polls = []

    def cancel_on_third_epoch(update):
        if isinstance(update, dict) and update.get("check_status"):
            polls.append(1)
            if len(polls) == 3:
                return SimpleNamespace(status="cancelled")
        return None

    with pytest.raises(svc.training_pipeline.TrainingCancelled):
        run(svc, req=request(epochs=6, batch_size=4), callback=cancel_on_third_epoch)

    assert not (ckpt_dir(workspace) / "final_model.pt").exists()
    state = json.loads((ckpt_dir(workspace) / "job_state.json").read_text())
    assert state["status"] == "cancelled" and state["epoch"] == 2


def test_an_oom_halves_the_batch_and_the_run_still_finishes(svc, workspace, monkeypatch):
    make_dataset(workspace, n_train=8)
    original = svc.vits_model.VITS.compute_loss
    calls = []

    def oom_once(self, batch):
        calls.append(batch["text"].shape[0])
        if len(calls) == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (synthetic)")
        return original(self, batch)

    monkeypatch.setattr(svc.vits_model.VITS, "compute_loss", oom_once)

    run(svc, req=request(epochs=3, batch_size=4))

    assert calls[0] == 4 and set(calls[1:]) == {2}, "the retry must use the reduced batch size"
    assert (ckpt_dir(workspace) / "final_model.pt").exists()


def test_an_oom_that_cannot_be_helped_by_a_smaller_batch_fails_the_job(svc, workspace, monkeypatch):
    make_dataset(workspace, n_train=8)

    def always_oom(self, batch):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (synthetic)")

    monkeypatch.setattr(svc.vits_model.VITS, "compute_loss", always_oom)

    with pytest.raises(RuntimeError, match="insufficient GPU memory"):
        run(svc, req=request(epochs=3, batch_size=2))

    assert not (ckpt_dir(workspace) / "final_model.pt").exists()


def test_the_best_model_is_scored_on_the_validation_split_when_there_is_one(svc, workspace):
    """val.json was written by every dataset preparation and read by nothing."""
    make_dataset(workspace, n_val=2)
    run(svc, job_id="with_val", req=request(epochs=2, batch_size=4))
    (workspace / "data" / "voice" / "val.json").unlink()
    run(svc, job_id="without_val", req=request(epochs=2, batch_size=4))

    with_val = torch.load(ckpt_dir(workspace, "with_val") / "best_model.pt", map_location="cpu", weights_only=True)
    without_val = torch.load(ckpt_dir(workspace, "without_val") / "best_model.pt", map_location="cpu", weights_only=True)
    assert with_val["metric_name"] == "val_loss"
    assert without_val["metric_name"] == "train_loss"


# --- checkpoints -----------------------------------------------------------------


def test_only_the_last_n_periodic_checkpoints_are_kept_and_best_and_final_survive(svc, workspace, monkeypatch):
    make_dataset(workspace)
    monkeypatch.setenv("KEEP_LAST_CHECKPOINTS", "2")

    run(svc, req=request(epochs=6, batch_size=4))

    names = sorted(p.name for p in ckpt_dir(workspace).iterdir())
    assert names == sorted([
        "checkpoint_epoch_5.pt", "checkpoint_epoch_6.pt",
        "best_model.pt", "final_model.pt", "job_state.json", "vocab.json",
    ])
    state = json.loads((ckpt_dir(workspace) / "job_state.json").read_text())
    assert Path(state["latest_checkpoint"]).exists()
    assert not list(ckpt_dir(workspace).glob("*.tmp")), "atomic writes leave no temp files"


def test_default_keeps_the_last_five(svc, workspace):
    make_dataset(workspace)

    run(svc, req=request(epochs=8, batch_size=4))

    kept = sorted(int(p.stem.rsplit("_", 1)[1]) for p in ckpt_dir(workspace).glob("checkpoint_epoch_*.pt"))
    assert kept == [4, 5, 6, 7, 8]


def test_keep_last_checkpoints_zero_disables_pruning(svc, workspace, monkeypatch):
    make_dataset(workspace)
    monkeypatch.setenv("KEEP_LAST_CHECKPOINTS", "0")

    run(svc, req=request(epochs=4, batch_size=4))

    assert len(list(ckpt_dir(workspace).glob("checkpoint_epoch_*.pt"))) == 4


# --- vocabulary --------------------------------------------------------------------


def test_the_vocabulary_is_persisted_next_to_the_checkpoints(svc, workspace):
    make_dataset(workspace)

    run(svc, req=request(epochs=1, batch_size=4))

    expected = svc.validation.phoneme_id_map_from_entries(
        json.loads((workspace / "data" / "voice" / "train.json").read_text())
    )
    persisted = json.loads((ckpt_dir(workspace) / "vocab.json").read_text())
    assert persisted["phoneme_id_map"] == expected
    final = torch.load(ckpt_dir(workspace) / "final_model.pt", map_location="cpu", weights_only=True)
    assert final["phoneme_to_id"] == expected


def test_resume_keeps_the_original_vocabulary_when_the_dataset_changed(svc, workspace):
    """Renumbering the symbols of a resumed run makes every embedding row belong
    to a different symbol."""
    data = make_dataset(workspace, phonemes=("ab", "ba"))
    run(svc, req=request(epochs=1, batch_size=4))
    original = json.loads((ckpt_dir(workspace) / "vocab.json").read_text())["phoneme_id_map"]

    # New symbols that sort BEFORE the existing ones would shift every id if the
    # vocabulary were rebuilt.
    changed = json.loads((data / "train.json").read_text())
    for entry in changed:
        entry["phonemes"] = "!!" + entry["phonemes"] + "zz"
    (data / "train.json").write_text(json.dumps(changed))

    run(svc, req=request(epochs=2, batch_size=4),
        resume_from=str(ckpt_dir(workspace) / "checkpoint_epoch_1.pt"))

    resumed = json.loads((ckpt_dir(workspace) / "vocab.json").read_text())["phoneme_id_map"]
    assert resumed == original
    final = torch.load(ckpt_dir(workspace) / "final_model.pt", map_location="cpu", weights_only=True)
    assert final["phoneme_to_id"] == original


def test_resuming_from_a_missing_checkpoint_fails_instead_of_training_from_scratch(svc, workspace):
    make_dataset(workspace)

    with pytest.raises(_failed(svc), match="not found"):
        run(svc, resume_from=str(ckpt_dir(workspace) / "checkpoint_epoch_9.pt"))

    assert not (ckpt_dir(workspace) / "final_model.pt").exists()


def test_resume_from_a_checkpoint_written_before_vocabularies_were_persisted(svc, workspace, caplog):
    """The payload the previous train_sync wrote: no vocabulary, no step count, no
    trainer kind. Resuming must keep working (and say what it cannot guarantee)."""
    make_dataset(workspace)
    model = svc.vits_model.VITS(svc.vits_model.VITSConfig(**TINY))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sum(p.sum() for p in model.parameters()).backward()
    optimizer.step()  # Adam's per-parameter state exists only after a step
    legacy = ckpt_dir(workspace) / "checkpoint_epoch_2.pt"
    legacy.parent.mkdir(parents=True)
    torch.save({
        "epoch": 2, "loss": 0.5, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "config": dict(TINY),
    }, legacy)
    caplog.set_level(logging.WARNING)

    run(svc, req=request(epochs=4, batch_size=4), resume_from=str(legacy))

    assert any("predates persisted vocabularies" in r.getMessage() for r in caplog.records)
    final = torch.load(ckpt_dir(workspace) / "final_model.pt", map_location="cpu", weights_only=True)
    assert final["optimizer_steps"] >= 3, "1 inferred from the optimizer state + 2 new epochs"
    assert final["config"]["boundary_tokens"] is False, "it was trained without <start>/<end>"
    assert json.loads((ckpt_dir(workspace) / "vocab.json").read_text())["boundary_tokens"] is False


def test_a_resume_with_nothing_left_to_train_still_exports_a_trained_model(svc, workspace):
    """The resumed model already has its steps; finishing immediately is fine."""
    make_dataset(workspace)
    run(svc, req=request(epochs=2, batch_size=4))

    run(svc, req=request(epochs=2, batch_size=4),
        resume_from=str(ckpt_dir(workspace) / "checkpoint_epoch_2.pt"))

    final = torch.load(ckpt_dir(workspace) / "final_model.pt", map_location="cpu", weights_only=True)
    assert final["optimizer_steps"] >= 1


# --- dataset ------------------------------------------------------------------------


def test_the_validation_split_uses_the_training_vocabulary(svc, workspace):
    data = make_dataset(workspace, phonemes=("ab", "abc"))
    config = dict(TINY)
    train = svc.dataset.TTSDataset(data, config)
    # A val split with a symbol that would sort first if it were numbered on its own.
    val_entries = [{"mel_path": "mel/0.npy", "text": "t", "phonemes": "!ab"}]
    (data / "val.json").write_text(json.dumps(val_entries))

    val = svc.dataset.TTSDataset(data, config, split="val", phoneme_to_id=train.phoneme_to_id)

    assert val.phoneme_to_id == train.phoneme_to_id
    ids = val[0]["text"].tolist()
    unk = train.phoneme_to_id["<unk>"]
    assert ids[1] == unk, "'!' is not in the training vocabulary"
    assert ids[2] == train.phoneme_to_id["a"]


def test_sequences_are_wrapped_in_start_and_end_like_the_runtime_does(svc, workspace):
    """piper-tts-service feeds [<start>, ..., <end>]; those rows were never trained."""
    data = make_dataset(workspace, phonemes=("ab",))
    dataset = svc.dataset.TTSDataset(data, dict(TINY))
    v = dataset.phoneme_to_id

    ids = dataset[0]["text"].tolist()

    assert ids == [v["<start>"], v["a"], v["b"], v["<end>"]]


def test_duration_targets_are_deterministic(svc, workspace):
    """They carried fresh Gaussian noise on every access."""
    data = make_dataset(workspace)
    dataset = svc.dataset.TTSDataset(data, dict(TINY))

    first, second = dataset[0]["duration_target"], dataset[0]["duration_target"]

    assert torch.equal(first, second)
    assert torch.allclose(first.sum(), torch.tensor(float(dataset[0]["mel_spec"].shape[1])))


def test_entries_with_a_missing_mel_are_dropped_and_counted(svc, workspace, caplog):
    data = make_dataset(workspace, n_train=6)
    (data / "mel" / "0.npy").unlink()
    caplog.set_level(logging.WARNING)

    dataset = svc.dataset.TTSDataset(data, dict(TINY))

    assert len(dataset) == 5
    assert any("dropped 1 of 6" in r.getMessage() for r in caplog.records)


# --- the model ------------------------------------------------------------------------


def _batch(svc, workspace):
    data = make_dataset(workspace, n_train=4, phonemes=("a", "abcdefgh"))
    dataset = svc.dataset.TTSDataset(data, dict(TINY))
    return svc.dataset.collate_fn([dataset[i] for i in range(4)])


def test_padding_does_not_enter_the_duration_loss(svc, workspace):
    """A padded slot predicts 0.0 against a target of log(1e-6) = -13.8, which
    added ~190 per padded position to the squared error."""
    batch = _batch(svc, workspace)
    assert batch["text_lengths"].min() < batch["text"].shape[1], "the batch must contain padding"
    torch.manual_seed(0)
    model = svc.vits_model.VITS(svc.vits_model.VITSConfig(**TINY)).eval()

    with torch.no_grad():
        _, log_duration, _ = model(batch["text"], batch["text_lengths"], batch["mel_spec"], batch["mel_lengths"])
        losses = model.compute_loss(batch)

    valid = torch.arange(batch["text"].shape[1])[None, :] < batch["text_lengths"][:, None]
    target = torch.log(batch["duration_target"] + 1e-6)
    expected = (((log_duration - target) ** 2)[valid]).mean()
    # 0.1 is the weight compute_loss applies.
    assert losses["duration"].item() == pytest.approx(0.1 * expected.item(), rel=1e-4)


def test_the_training_step_never_empties_the_cuda_cache(svc, workspace, monkeypatch):
    """empty_cache() ran before every vocoder call and inside its upsampling loop
    (a device-wide sync per step)."""
    batch = _batch(svc, workspace)
    model = svc.vits_model.VITS(svc.vits_model.VITSConfig(**TINY))
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))

    model.compute_loss(batch)

    assert calls == []


def test_a_vocoder_oom_reaches_the_training_loop_instead_of_being_papered_over(svc, workspace, monkeypatch):
    """The model used to catch it and re-run the vocoder on the CPU under
    no_grad: a waveform with no graph, so that step trained only the duration
    head and nothing said so."""
    batch = _batch(svc, workspace)
    model = svc.vits_model.VITS(svc.vits_model.VITSConfig(**TINY))
    real_forward = svc.vits_model.HiFiGANGenerator.forward
    calls = []

    def oom_first(self, x):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB (synthetic)")
        return real_forward(self, x)

    monkeypatch.setattr(svc.vits_model.HiFiGANGenerator, "forward", oom_first)

    with pytest.raises(RuntimeError, match="out of memory"):
        model.compute_loss(batch)
