"""CPU smoke tests for the piper-training-service ONNX export.

The exported graph used to be length independent in the wrong way: the duration
expansion looped over tokens and called ``.item()``, so ``torch.onnx.export``
froze the durations of its random 100-symbol dummy input into the graph. Measured
on the real model: every text produced the same 3197 frames while PyTorch produced
3124-3277, and a 160-symbol text got 3197 where PyTorch gave 4899. On an unpinned
torch 2.14 the export did not run at all (``onnxscript`` missing; then
``GuardOnDataDependentSymNode`` from the ``.item()`` loop).

These tests build a tiny model (no GPU, no downloads), export it through the real
``ModelExporter`` and run the ONNX file in onnxruntime. They skip where torch,
onnx or onnxruntime are not installed (the unit job installs tests/requirements.txt
only), and fail there when REQUIRE_TORCH_TESTS=1, which the CI training job sets.

``PIPER_TRAINING_SERVICE_DIR`` points them at another checkout of the service,
which is how they were run against the pre-fix code to show they fail there.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from optional_deps import importorskip_unless_required as _need

# Skips where a dependency is missing, except with REQUIRE_TORCH_TESTS=1 (the CI
# training job), where a missing one is an error: see tests/optional_deps.py.
torch = _need("torch")
ort = _need("onnxruntime")
_need("onnx")
np = _need("numpy")

REPO = Path(__file__).resolve().parents[1]
SERVICE_DIR = Path(os.environ.get("PIPER_TRAINING_SERVICE_DIR") or REPO / "piper-training-service")

# 16 -> 8 -> 4 -> 2 -> 1 channels through the four upsampling stages: the smallest
# vocoder that still has every layer type.
TINY = dict(
    hidden_channels=16, inter_channels=16, n_layers=1, n_heads=2,
    n_vocab=64, vocoder_channels=16, sample_rate=22050,
    speaker_name="tiny", language="de", quality="medium",
)
VOCAB = {"<pad>": 0, "<unk>": 1, "<start>": 2, "<end>": 3, " ": 4, "a": 5, "b": 6}
MODULES = ("vits_model", "validation", "phonemization", "dataset", "training_utils", "model_exporter")


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


def _tiny_model(svc, seed=0, frames_per_token=3.0):
    """A tiny model whose duration head predicts about ``frames_per_token`` frames."""
    torch.manual_seed(seed)
    model = svc.vits_model.VITS(svc.vits_model.VITSConfig(**TINY)).eval()
    with torch.no_grad():
        # Untrained, the head outputs ~0 (1 frame per token), which would make
        # every length trivially proportional. Bias it so durations are >1.
        model.duration_predictor.proj.bias.fill_(float(np.log(frames_per_token)))
    return model


def _write_final(root: Path, job_id: str, model, **extra):
    ckpt_dir = root / "checkpoints" / job_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "config": dict(TINY),
        "training_complete": True,
        "phoneme_to_id": dict(VOCAB),
        "trainer_kind": "experimental-mel-flow",
        "optimizer_steps": 7,
    }
    payload.update(extra)
    payload = {k: v for k, v in payload.items() if v is not None}
    torch.save(payload, ckpt_dir / "final_model.pt")


def _export(svc, job_id):
    return asyncio.run(svc.model_exporter.ModelExporter().export_to_onnx(job_id))


def _run_onnx(onnx_path, n_symbols, seed):
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ids = torch.randint(1, TINY["n_vocab"], (1, n_symbols), generator=torch.Generator().manual_seed(seed))
    out = session.run(None, {"text": ids.numpy(), "text_lengths": np.array([n_symbols], dtype=np.int64)})[0]
    return ids, out


@pytest.fixture(scope="module")
def exported(svc, tmp_path_factory):
    """One real export shared by the read-only assertions below."""
    root = tmp_path_factory.mktemp("export")
    previous = os.getcwd()
    os.chdir(root)
    try:
        model = _tiny_model(svc)
        _write_final(root, "job", model)
        onnx_path = _export(svc, "job")
        yield SimpleNamespace(root=root, model=model, onnx=root / onnx_path, job="job")
    finally:
        os.chdir(previous)


# --- the expansion itself ------------------------------------------------------


def test_expansion_repeats_each_token_by_its_duration(svc):
    model = _tiny_model(svc)
    encodings = torch.arange(1, 4, dtype=torch.float32).view(1, 3, 1).expand(1, 3, 4)  # rows 1,2,3
    durations = torch.tensor([[2.0, 1.0, 3.0]])

    out = model.expand_encodings(encodings, durations)

    assert out.shape == (1, 6, 4)
    assert out[0, :, 0].tolist() == [1, 1, 2, 3, 3, 3]


def test_expansion_gives_padding_no_frames_and_pads_shorter_items_with_zeros(svc):
    """Batch of two texts of different length. A padded slot used to be forced up
    to one frame (duration 0 -> max(1, 0)), so the short text was followed by an
    extra frame per padded token."""
    model = _tiny_model(svc)
    encodings = torch.ones(2, 4, 3)
    durations = torch.full((2, 4), 2.0)
    mask = torch.tensor([[True, True, True, True], [True, True, False, False]])

    out = model.expand_encodings(encodings, durations, mask)

    assert out.shape == (2, 8, 3)
    assert out[1, :4].abs().sum() > 0, "the two real tokens produce 4 frames"
    assert out[1, 4:].abs().sum() == 0, "nothing after the real tokens, and no frame for padding"


def test_expansion_length_follows_the_durations_in_eager_mode(svc):
    model = _tiny_model(svc)
    enc = torch.randn(1, 5, 4)
    short = model.expand_encodings(enc, torch.full((1, 5), 2.0))
    long = model.expand_encodings(enc, torch.full((1, 5), 7.0))
    assert (short.shape[1], long.shape[1]) == (10, 35)


# --- the exported graph --------------------------------------------------------


def test_onnx_output_length_follows_text_length_and_matches_pytorch(svc, exported):
    """The measured failure: constant length for every text, nothing past 100 symbols."""
    lengths = {}
    for n_symbols in (30, 150):
        ids, produced = _run_onnx(exported.onnx, n_symbols, seed=n_symbols)
        with torch.no_grad():
            expected = exported.model(ids, torch.tensor([n_symbols])).numpy()
        assert produced.shape == expected.shape, (
            f"{n_symbols} symbols: ONNX gave {produced.shape[-1]} samples, PyTorch {expected.shape[-1]}"
        )
        assert np.abs(produced - expected).max() < 1e-3
        lengths[n_symbols] = produced.shape[-1]

    assert lengths[150] > 3 * lengths[30], lengths
    assert lengths[150] > 100 * 256, "text beyond 100 symbols must not be truncated"


def test_onnx_accepts_a_single_symbol(svc, exported):
    ids, produced = _run_onnx(exported.onnx, 1, seed=1)
    assert produced.shape[-1] > 0 and np.isfinite(produced).all()


def test_the_verifier_rejects_a_graph_with_frozen_durations(svc, tmp_path):
    """The pre-fix expansion, exported, must be caught by the post-export check."""

    def frozen_expand(encodings, frames):
        # The original implementation: a Python loop over tokens using .item().
        rows = []
        for t in range(encodings.shape[1]):
            rows.append(encodings[0, t:t + 1].repeat(int(frames[0, t].item()), 1))
        return torch.cat(rows, dim=0).unsqueeze(0)

    model = _tiny_model(svc)
    dummy = torch.randint(1, TINY["n_vocab"], (1, 20))
    original = svc.vits_model.VITS.__dict__["_expand"]  # the staticmethod object itself
    svc.vits_model.VITS._expand = staticmethod(frozen_expand)
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.no_grad():
                torch.onnx.export(
                    model, (dummy, torch.tensor([20])), str(tmp_path / "frozen.onnx"),
                    input_names=["text", "text_lengths"], output_names=["audio"],
                    dynamic_axes={"text": {0: "b", 1: "s"}, "text_lengths": {0: "b"}, "audio": {0: "b", 1: "t"}},
                    opset_version=15, **svc.model_exporter.onnx_export_kwargs(),
                )
    finally:
        svc.vits_model.VITS._expand = original

    with pytest.raises(RuntimeError, match="length dependent"):
        svc.model_exporter.verify_onnx_export(model, tmp_path / "frozen.onnx", TINY["n_vocab"], lengths=(30, 150))


# --- the bundle ----------------------------------------------------------------


def test_bundle_uses_the_checkpoints_vocabulary_and_records_the_trainer(svc, exported):
    bundle = json.loads((exported.root / "models" / exported.job / f"{exported.job}.json").read_text())

    assert bundle["phoneme_id_map"] == VOCAB
    assert json.loads((exported.root / "models" / exported.job / "phonemes.json").read_text()) == VOCAB
    card = bundle["model_card"]
    assert card["trainer_kind"] == svc.training_utils.TRAINER_KIND == "experimental-mel-flow"
    assert card["trainer_caveat"] == svc.training_utils.TRAINER_CAVEAT
    assert bundle["limits"]["max_input_symbols"] >= 1000
    verification = bundle["export"]["verification"]
    assert set(verification["samples"]) == {"16", "48"}


def test_export_does_not_rebuild_the_vocabulary_from_a_changed_dataset(svc, tmp_path, monkeypatch):
    """The exporter re-derived the vocabulary from train.json at export time, so
    a dataset regenerated after training renumbered the symbols."""
    monkeypatch.chdir(tmp_path)
    _write_final(tmp_path, "job", _tiny_model(svc))
    data = tmp_path / "data" / TINY["speaker_name"]
    data.mkdir(parents=True)
    (data / "train.json").write_text(json.dumps([{"phonemes": "xyz qrs"}]))

    _export(svc, "job")

    exported = json.loads((tmp_path / "models" / "job" / "job.json").read_text())
    assert exported["phoneme_id_map"] == VOCAB


def test_export_never_picks_another_voices_dataset(svc, tmp_path, monkeypatch):
    """A checkpoint without a persisted vocabulary whose own dataset is gone used
    to scan data/ and take the first directory with a train.json: another voice's
    alphabet on this voice's weights."""
    monkeypatch.chdir(tmp_path)
    _write_final(tmp_path, "job", _tiny_model(svc), phoneme_to_id=None)
    other = tmp_path / "data" / "another_voice"
    other.mkdir(parents=True)
    (other / "train.json").write_text(json.dumps([{"phonemes": "wrong alphabet"}]))

    with pytest.raises(RuntimeError, match="vocabulary"):
        _export(svc, "job")

    assert not (tmp_path / "models" / "job" / "job.onnx").exists()


def test_legacy_checkpoint_falls_back_to_the_dataset_named_after_the_voice(svc, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_final(
        tmp_path, "job", _tiny_model(svc), phoneme_to_id=None, optimizer_steps=None, trainer_kind=None
    )
    data = tmp_path / "data" / TINY["speaker_name"]
    data.mkdir(parents=True)
    (data / "train.json").write_text(json.dumps([{"phonemes": "ab"}]))

    _export(svc, "job")

    exported = json.loads((tmp_path / "models" / "job" / "job.json").read_text())
    assert exported["phoneme_id_map"] == svc.validation.phoneme_id_map_from_entries([{"phonemes": "ab"}])
    assert "Trained before the trainer kind was recorded" in exported["model_card"]["trainer_caveat"]


def test_export_refuses_weights_that_never_took_an_optimizer_step(svc, tmp_path, monkeypatch):
    """A job that trained zero steps saved its random initialisation as the final
    model, and the exporter turned that into a deployable voice."""
    monkeypatch.chdir(tmp_path)
    _write_final(tmp_path, "job", _tiny_model(svc), optimizer_steps=0)

    with pytest.raises(RuntimeError, match="optimizer step"):
        _export(svc, "job")

    assert not (tmp_path / "models" / "job").exists(), "nothing may be written for a refused export"


def test_vocabulary_larger_than_the_embedding_is_refused(svc, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    too_big = dict(VOCAB, z=TINY["n_vocab"] + 3)
    _write_final(tmp_path, "job", _tiny_model(svc), phoneme_to_id=too_big)

    with pytest.raises(ValueError, match="embedding"):
        _export(svc, "job")


# --- the language the bundle tells the runtime to phonemise in -----------------------------------


@pytest.mark.parametrize("language, expected", [
    ("pt-br", "pt-br"),      # trained with the espeak voice pt-br; used to be exported as en-us
    ("fr-be", "fr-be"),
    ("en-gb", "en-gb"),
    ("en", "en-us"),
    ("de", "de"),
    ("sv", "en-us"),         # a tag this service cannot phonemise: an old checkpoint, trained as en-us
])
def test_the_bundle_names_the_voice_training_phonemised_with(svc, tmp_path, monkeypatch, language, expected):
    """piper-tts phonemises with `phonemizer_language`; a pt-BR voice exported as en-us read
    Portuguese as English."""
    monkeypatch.chdir(tmp_path)
    _write_final(tmp_path, "job", _tiny_model(svc), config=dict(TINY, language=language))

    _export(svc, "job")

    bundle = json.loads((tmp_path / "models" / "job" / "job.json").read_text())
    assert bundle["phonemizer_language"] == expected
    assert bundle["espeak"]["voice"] == expected, "the espeak voice and phonemizer_language must agree"
    assert bundle["model_card"]["language"] == language, "the card still records what was asked for"
    if language in ("pt-br", "fr-be", "en-gb", "en", "de"):
        assert expected == svc.phonemization.espeak_voice(language), "not the voice the dataset was built with"


def test_a_checkpoint_without_a_language_exports_as_english_like_before(svc, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = {k: v for k, v in TINY.items() if k != "language"}
    _write_final(tmp_path, "job", _tiny_model(svc), config=config)

    _export(svc, "job")

    bundle = json.loads((tmp_path / "models" / "job" / "job.json").read_text())
    assert bundle["phonemizer_language"] == "en-us"


def test_a_speaker_name_read_from_the_checkpoint_cannot_point_the_legacy_lookup_elsewhere(svc, tmp_path, monkeypatch):
    """A checkpoint without a vocabulary looks for data/<speaker_name>/train.json. The name comes
    from the checkpoint, so `../other` must not be followed to some other directory's train.json."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    other = tmp_path / "other"
    other.mkdir()
    (other / "train.json").write_text(json.dumps([{"phonemes": "ab"}]))
    _write_final(tmp_path, "job", _tiny_model(svc), phoneme_to_id=None,
                 config=dict(TINY, speaker_name="../other"))

    with pytest.raises(RuntimeError, match="vocabulary"):
        _export(svc, "job")
