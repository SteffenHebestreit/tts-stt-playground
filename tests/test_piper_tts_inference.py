"""Custom-voice inference through the real ONNX Runtime, librosa and soundfile.

Guards the service's pinned audio stack (numpy, onnxruntime, librosa, soundfile)
end to end: a bumped pin that breaks the ONNX call, the time-stretch or the WAV
encoding fails here, not in a container. Skipped where those packages are not
installed (the CI test image only has the lightweight ones). The phonemizer's
espeak backend needs the system library, so it is replaced by a fixed mapping.
"""

from __future__ import annotations

import io
import json
import sys
import types

import pytest

np = pytest.importorskip("numpy")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
librosa = pytest.importorskip("librosa")
sf = pytest.importorskip("soundfile")

from test_piper_tts_harness import start  # noqa: E402,F401  (fixture)


def _tiny_vits_like_model() -> bytes:
    """An ONNX graph with the exported voices' interface: (text, text_lengths) -> audio.

    The audio is the phoneme ids scaled into [-1, 1] and repeated, so its length
    and content depend on the input the way a real model's would.
    """
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [
            helper.make_node("Cast", ["text"], ["as_float"], to=TensorProto.FLOAT),
            helper.make_node("Mul", ["as_float", "scale"], ["scaled"]),
            helper.make_node("Concat", ["scaled", "scaled", "scaled", "scaled"], ["audio"], axis=1),
        ],
        "tiny",
        [
            helper.make_tensor_value_info("text", TensorProto.INT64, [1, "n"]),
            helper.make_tensor_value_info("text_lengths", TensorProto.INT64, [1]),
        ],
        [helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, "m"])],
        initializer=[helper.make_tensor("scale", TensorProto.FLOAT, [1], [0.01])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model.SerializeToString()


@pytest.fixture
def fake_phonemizer(monkeypatch):
    module = types.ModuleType("phonemizer")
    module.phonemize = lambda text, language, backend, strip: "abab" * 2000  # 8000 IPA characters
    monkeypatch.setitem(sys.modules, "phonemizer", module)


def _upload_tiny_voice(svc):
    config = {
        "audio": {"sample_rate": 22050, "quality": "medium"},
        "model_card": {"language": "de", "speaker": "luna"},
        "phoneme_id_map": {"<pad>": 0, "<start>": 1, "<end>": 2, "a": 3, "b": 4},
        "phonemizer_language": "de",
    }
    r = svc.client.post(
        "/upload_model",
        files={"model_file": ("luna.onnx", _tiny_vits_like_model(), "application/octet-stream"),
               "config_file": ("luna.json", json.dumps(config), "application/json")},
        data={"voice_name": "luna"},
    )
    assert r.status_code == 200, r.text


def _decode(wav: bytes):
    audio, rate = sf.read(io.BytesIO(wav))
    return audio, rate


def test_a_custom_voice_synthesises_a_wav_with_the_real_stack(start, monkeypatch, fake_phonemizer):
    svc = start()
    monkeypatch.setattr(svc.module, "librosa", librosa)
    monkeypatch.setattr(svc.module, "sf", sf)
    _upload_tiny_voice(svc)

    r = svc.client.post("/synthesize", json={"text": "Hallo", "voice_name": "luna"})

    assert r.status_code == 200
    audio, rate = _decode(r.content)
    assert rate == 22050
    assert len(audio) == 4 * (8000 + 2)          # start + 8000 phonemes + end, repeated four times
    assert audio.max() > 0


def test_speed_changes_the_length_of_the_real_audio(start, monkeypatch, fake_phonemizer):
    svc = start()
    monkeypatch.setattr(svc.module, "librosa", librosa)
    monkeypatch.setattr(svc.module, "sf", sf)
    _upload_tiny_voice(svc)

    normal, _ = _decode(svc.client.post("/synthesize", json={"text": "Hallo", "voice_name": "luna"}).content)
    fast, _ = _decode(svc.client.post(
        "/synthesize", json={"text": "Hallo", "voice_name": "luna", "speed": 2.0}).content)

    assert len(fast) == pytest.approx(len(normal) / 2, rel=0.05)


def test_tts_routes_a_custom_voice_through_onnx_runtime(start, monkeypatch, fake_phonemizer):
    svc = start()
    monkeypatch.setattr(svc.module, "librosa", librosa)
    monkeypatch.setattr(svc.module, "sf", sf)
    _upload_tiny_voice(svc)

    r = svc.client.post("/tts", json={"text": "Hallo", "voice": "luna"})

    assert r.status_code == 200
    assert r.headers["X-Voice-Used"] == "luna"
    assert r.content[:4] == b"RIFF"
