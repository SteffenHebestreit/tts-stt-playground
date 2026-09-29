"""/upload_model and /tts against real ONNX graphs and the real onnxruntime.

`test_piper_tts_hardening.py` covers the same rules with a stand-in for the
validation process. These tests build actual models, so they check that the child
process's script is right about onnxruntime itself, and that a model which never
finishes can neither be published nor keep a synthesis slot. Skipped where `onnx`
and `onnxruntime` are not installed.
"""

from __future__ import annotations

import json
import sys
import time
import types

import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from onnx import TensorProto, helper  # noqa: E402

from test_piper_tts_harness import start  # noqa: E402,F401  (fixture)

CONFIG = {
    "audio": {"sample_rate": 22050, "quality": "medium"},
    "model_card": {"language": "de", "speaker": "luna"},
    "phoneme_id_map": {"<pad>": 0, "<start>": 1, "<end>": 2, "a": 3, "b": 4},
    "phonemizer_language": "de",
}


def _graph(inputs, trip=None) -> bytes:
    """A model with the given input names; `trip` puts a Loop with that trip count in front."""
    nodes = [helper.make_node("Cast", [inputs[0]], ["as_float"], to=TensorProto.FLOAT)]
    initializers = [helper.make_tensor("scale", TensorProto.FLOAT, [1], [0.01])]
    multiplier = "scale"
    if trip is not None:
        body = helper.make_graph(
            [helper.make_node("Identity", ["cond_in"], ["cond_out"]),
             helper.make_node("Add", ["acc_in", "one"], ["acc_out"])],
            "body",
            [helper.make_tensor_value_info("iter", TensorProto.INT64, []),
             helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
             helper.make_tensor_value_info("acc_in", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("cond_out", TensorProto.BOOL, []),
             helper.make_tensor_value_info("acc_out", TensorProto.FLOAT, [1])],
            initializer=[helper.make_tensor("one", TensorProto.FLOAT, [1], [1.0])],
        )
        initializers += [
            helper.make_tensor("trip", TensorProto.INT64, [], [trip]),
            helper.make_tensor("cond", TensorProto.BOOL, [], [True]),
            helper.make_tensor("zero", TensorProto.FLOAT, [1], [0.0]),
        ]
        nodes.append(helper.make_node("Loop", ["trip", "cond", "zero"], ["spun"], body=body))
        nodes.append(helper.make_node("Mul", ["scale", "spun"], ["scaled_by_loop"]))
        multiplier = "scaled_by_loop"
    nodes.append(helper.make_node("Mul", ["as_float", multiplier], ["audio"]))
    graph = helper.make_graph(
        nodes, "g",
        [helper.make_tensor_value_info(inputs[0], TensorProto.INT64, [1, "n"]),
         helper.make_tensor_value_info(inputs[1], TensorProto.INT64, [1])],
        [helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, "m"])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model.SerializeToString()


def _upload(svc, model, name="luna", config=CONFIG):
    return svc.client.post(
        "/upload_model",
        files={"model_file": ("m.onnx", model, "application/octet-stream"),
               "config_file": ("m.json", json.dumps(config), "application/json")},
        data={"voice_name": name},
    )


def _published(svc) -> list:
    custom = svc.models / "custom"
    return sorted(p.name for p in custom.iterdir()) if custom.exists() else []


def _free_slots(svc) -> int:
    return svc.module._SLOTS[1]._value


def test_a_model_with_the_training_export_interface_passes_validation(start):
    svc = start(validate_models=True)

    r = _upload(svc, _graph(["text", "text_lengths"]))

    assert r.status_code == 200, r.text
    assert _published(svc) == ["luna"]


def test_a_model_with_another_interface_is_refused(start):
    svc = start(validate_models=True)

    r = _upload(svc, _graph(["input", "input_lengths"]))

    assert r.status_code == 400 and "'text' and 'text_lengths'" in r.json()["detail"]
    assert _published(svc) == []


def test_bytes_that_are_not_a_model_are_refused(start):
    svc = start(validate_models=True)

    r = _upload(svc, b"MZ\x90\x00 definitely not onnx" * 100)

    assert r.status_code == 400 and "not a loadable ONNX model" in r.json()["detail"]
    assert _published(svc) == []


def test_a_model_that_spins_on_a_tiny_input_is_refused_within_the_timeout(start):
    svc = start(validate_models=True, PIPER_ONNX_VALIDATE_TIMEOUT_S="2")

    began = time.monotonic()
    r = _upload(svc, _graph(["text", "text_lengths"], trip=10**12))

    assert r.status_code == 400 and "PIPER_ONNX_VALIDATE_TIMEOUT_S" in r.json()["detail"]
    assert time.monotonic() - began < 30
    assert _published(svc) == []


def test_a_runaway_model_cannot_take_the_slot_for_good(start, monkeypatch):
    """The whole chain with onnxruntime: PIPER_TIMEOUT_S ends the run and the slot comes back."""
    svc = start(PIPER_MAX_CONCURRENCY="1", PIPER_TIMEOUT_S="1")
    # Present as if it had passed validation earlier (an older upload, or a file placed by hand).
    voice = svc.models / "custom" / "spinner"
    voice.mkdir(parents=True)
    (voice / "spinner.onnx").write_bytes(_graph(["text", "text_lengths"], trip=10**12))
    (voice / "spinner.json").write_text(json.dumps(CONFIG))
    assert svc.client.post("/refresh_voices").json()["custom_voices"] == 1
    phonemizer = types.ModuleType("phonemizer")
    phonemizer.phonemize = lambda text, language, backend, strip: "ab"
    monkeypatch.setitem(sys.modules, "phonemizer", phonemizer)

    for _ in range(3):  # as many overruns in a row as there are slots, and one more
        overran = svc.client.post("/tts", json={"text": "Hallo", "voice": "spinner"})
        assert overran.status_code == 504, overran.text
        deadline = time.monotonic() + 10
        while _free_slots(svc) < 1 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert _free_slots(svc) == 1, "an overrun kept running inside onnxruntime and kept its slot"

    assert svc.speak().status_code == 200, "synthesis is dead after the overruns"
