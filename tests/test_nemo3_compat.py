"""The parakeet / canary services against the API shapes of NeMo 2.7.x and 3.x.

nemo_toolkit 3.0.0 (2026-08-07) resolves from the old ``nemo_toolkit[asr]>=2.0.0``
line on its own. Reading both wheels and running a randomly initialised
Parakeet-TDT through the real ``transcribe()`` on each (CPU, torch 2.11) showed
the surface these services call is the same: same signature, ``Hypothesis``
fields and timestamp keys. The helpers in ``nemo_common`` therefore detect
what a build accepts or returns rather than asking for a version, and these
tests pin that behaviour with fakes that have each shape:

* ``transcribe`` without the optional ``timestamps`` / ``verbose`` parameters;
* results as a list, a tuple of lists, an n-best list or plain strings;
* a checkpoint given as a path to a ``.nemo`` file (``from_pretrained`` reads any
  string containing "/" as a Hub repo id, on both releases);
* the forced-alignment model some Canary checkpoints keep outside the module tree.
"""

from __future__ import annotations

import importlib.metadata
import logging
from types import SimpleNamespace

import pytest

from test_nemo_services_harness import (
    FakeCanary, FakeParakeet, hypothesis, install_fake_nemo, install_model, load_app,
    load_nemo_common, make_client, run, wav_bytes,
)

SERVICES = ["parakeet", "canary"]
MODELS = {"parakeet": FakeParakeet, "canary": FakeCanary}
LOADERS = {"parakeet": "_load_parakeet", "canary": "_load_canary"}
ENV_MODEL = {"parakeet": "PARAKEET_ASR_MODEL", "canary": "CANARY_ASR_MODEL"}


@pytest.fixture(params=SERVICES)
def service(request):
    return request.param


class LegacyParakeet(FakeParakeet):
    """A NeMo whose ``transcribe`` has neither ``timestamps`` nor ``verbose``."""

    def transcribe(self, audio, batch_size=4, return_hypotheses=False, num_workers=0,
                   channel_selector=None, augmentor=None):
        return self._run(audio, {"batch_size": batch_size, "num_workers": num_workers})


def build(service, monkeypatch, model):
    module = load_app(service)
    monkeypatch.setattr(module, "_FFMPEG", None)  # native 16 kHz mono WAV
    install_model(module, model)
    return module


def transcribe(module, seconds=1.0):
    async def scenario():
        async with make_client(module) as client:
            return await client.post("/transcribe", files={"audio": ("clip.wav", wav_bytes(seconds), "audio/wav")})

    return run(scenario())


# --- result containers ---------------------------------------------------------------------

@pytest.mark.parametrize("shape", ["list", "tuple of lists", "n-best lists", "n-best objects", "strings"])
def test_every_result_container_gives_the_same_answer(service, monkeypatch, shape):
    best = hypothesis("guten tag", [{"start": 0.0, "end": 1.0, "segment": "guten tag"}])
    worse = hypothesis("guten tach")
    shaped = {
        "list": lambda: [best],
        "tuple of lists": lambda: ([best], [worse]),
        "n-best lists": lambda: [[best, worse]],
        "n-best objects": lambda: [SimpleNamespace(n_best_hypotheses=[best, worse])],
        "strings": lambda: ["guten tag"],
    }[shape]
    model = MODELS[service]()
    model.reply = lambda paths, kwargs: shaped()
    module = build(service, monkeypatch, model)

    response = transcribe(module, seconds=2.0)

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "guten tag"
    if shape == "strings":  # no hypothesis object, so no timestamps: one segment covers the file
        assert body["segments"] == [{"start": 0.0, "end": pytest.approx(2.0), "text": "guten tag"}]
    else:
        assert body["segments"] == [{"start": 0.0, "end": 1.0, "text": "guten tag"}]


def test_a_result_container_with_nothing_in_it_is_an_empty_transcript_not_an_error(service, monkeypatch):
    model = MODELS[service]()
    model.reply = lambda paths, kwargs: ([],)
    module = build(service, monkeypatch, model)

    response = transcribe(module)

    assert response.status_code == 200
    assert response.json()["text"] == ""
    assert response.json()["segments"] == []


def test_the_batch_route_reduces_each_file_to_its_best_hypothesis(monkeypatch):
    model = FakeParakeet()
    model.reply = lambda paths, kwargs: [[hypothesis(f"best:{i}"), hypothesis("worse")] for i in range(len(paths))]
    module = build("parakeet", monkeypatch, model)

    async def scenario():
        async with make_client(module) as client:
            files = [("audios", (f"c{i}.wav", wav_bytes(1.0), "audio/wav")) for i in range(2)]
            return await client.post("/transcribe-batch", files=files)

    results = run(scenario()).json()["results"]

    assert [r["text"] for r in results] == ["best:0", "best:1"]


# --- optional transcribe() parameters -----------------------------------------------------------

def test_a_nemo_without_timestamps_still_answers_with_one_segment(monkeypatch):
    model = LegacyParakeet()
    model.reply = lambda paths, kwargs: ["hallo welt"]
    module = build("parakeet", monkeypatch, model)

    response = transcribe(module, seconds=2.0)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["text"] == "hallo welt"
    assert body["segments"] == [{"start": 0.0, "end": pytest.approx(2.0), "text": "hallo welt"}]
    assert set(model.calls[0]) == {"paths", "batch_size", "num_workers"}, \
        "arguments the installed transcribe() does not name must not be passed"


def test_the_current_signature_receives_every_argument(service, monkeypatch):
    model = MODELS[service]()
    module = build(service, monkeypatch, model)

    assert transcribe(module).status_code == 200

    call = model.calls[0]
    assert call["timestamps"] is True and call["batch_size"] == 1
    if service == "parakeet":
        assert call["num_workers"] == 0 and call["verbose"] is False


class _Nemo:
    def __init__(self, transcribe):
        self.transcribe = transcribe


def test_kwargs_are_reduced_to_the_named_parameters_and_the_drop_is_logged_once(caplog):
    common, _ = load_nemo_common("parakeet")
    model = _Nemo(lambda audio, batch_size=4, timestamps=None: None)
    wanted = {"batch_size": 1, "timestamps": True, "num_workers": 0, "verbose": False}

    with caplog.at_level(logging.WARNING, logger=common.logger.name):
        first = common.filter_transcribe_kwargs(model, wanted)
        second = common.filter_transcribe_kwargs(model, wanted)

    assert first == second == {"batch_size": 1, "timestamps": True}
    dropped = [r.getMessage() for r in caplog.records if "no 'num_workers'" in r.getMessage()
               or "no 'verbose'" in r.getMessage()]
    assert len(dropped) == 2, "one warning per dropped name, not one per request"


def test_a_transcribe_that_takes_star_kwargs_gets_everything():
    common, _ = load_nemo_common("canary")
    model = _Nemo(lambda audio, batch_size=4, **prompt: None)
    wanted = {"batch_size": 1, "source_lang": "de", "pnc": "yes", "timestamps": True}

    assert common.filter_transcribe_kwargs(model, wanted) == wanted


def test_a_transcribe_that_cannot_be_inspected_is_called_as_asked():
    common, _ = load_nemo_common("parakeet")

    assert common.filter_transcribe_kwargs(_Nemo(object()), {"timestamps": True}) == {"timestamps": True}
    assert common.filter_transcribe_kwargs(SimpleNamespace(), {"timestamps": True}) == {"timestamps": True}


@pytest.mark.parametrize("service_name", SERVICES)
def test_transcribe_results_shapes(service_name):
    common, _ = load_nemo_common(service_name)
    a, b = hypothesis("a"), hypothesis("b")

    assert common.transcribe_results(None) == []
    assert common.transcribe_results([]) == []
    assert common.transcribe_results(()) == []
    assert common.transcribe_results([a, b]) == [a, b]
    assert common.transcribe_results(([a, b], [b, a])) == [a, b]
    assert common.transcribe_results([[a, b], [b]]) == [a, b]
    assert common.transcribe_results([[], a]) == ["", a]
    assert common.transcribe_results(["x", "y"]) == ["x", "y"]
    assert common.transcribe_results("x") == ["x"]
    assert common.transcribe_results(a) == [a]
    nbest = SimpleNamespace(n_best_hypotheses=[b, a])
    assert common.transcribe_results([nbest]) == [b]


# --- loading a checkpoint from a path -----------------------------------------------------------

def _load(service, monkeypatch, model_name):
    module = load_app(service, **{ENV_MODEL[service]: model_name})
    model = MODELS[service]()
    calls = []
    install_fake_nemo(
        monkeypatch,
        lambda name: calls.append(("from_pretrained", name)) or model,
        restore=lambda path: calls.append(("restore_from", path)) or model,
    )
    assert getattr(module, LOADERS[service])() is model
    return calls


def test_a_path_to_a_nemo_file_is_restored_not_looked_up_on_the_hub(service, monkeypatch, tmp_path):
    checkpoint = tmp_path / "parakeet-primeline.nemo"
    checkpoint.write_bytes(b"not read by the fake")

    assert _load(service, monkeypatch, str(checkpoint)) == [("restore_from", str(checkpoint))]


@pytest.mark.parametrize("name", ["nvidia/parakeet-tdt-0.6b-v3", "primeline/parakeet-primeline", "stt_de_conformer_ctc_large"])
def test_a_repo_id_or_registry_name_still_goes_through_from_pretrained(service, monkeypatch, name):
    assert _load(service, monkeypatch, name) == [("from_pretrained", name)]


def test_a_path_that_does_not_exist_is_left_to_from_pretrained_to_reject(service, monkeypatch, tmp_path):
    missing = str(tmp_path / "missing.nemo")

    assert _load(service, monkeypatch, missing) == [("from_pretrained", missing)]


# --- /status --------------------------------------------------------------------------------

def test_status_reports_the_stack_the_image_was_built_with(service, monkeypatch):
    real_version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata, "version",
        lambda name: "3.0.0" if name == "nemo-toolkit" else real_version(name),
    )
    module = load_app(service)
    module.torch.__version__ = "2.11.0+cu128"
    module.torch.version.cuda = "12.8"
    module.torch._C = SimpleNamespace(_cuda_getArchFlags=lambda: "sm_90 sm_120")
    module.torch.cuda.get_arch_list = lambda: []  # what torch answers while no GPU is visible

    async def scenario():
        async with make_client(module) as client:
            return await client.get("/status")

    runtime = run(scenario()).json()["runtime"]

    assert runtime == {
        "nemo_toolkit": "3.0.0",
        "torch": "2.11.0+cu128",
        "torch_cuda": "12.8",
        "cuda_arch_list": ["sm_90", "sm_120"],
    }


def test_the_arch_list_falls_back_to_the_gpu_api_when_the_compiled_flags_are_not_readable():
    common, _ = load_nemo_common("parakeet")
    torch = SimpleNamespace(cuda=SimpleNamespace(get_arch_list=lambda: ["sm_86"]))

    assert common.runtime_versions(torch)["cuda_arch_list"] == ["sm_86"]


def test_status_still_answers_when_the_versions_cannot_be_read(service, monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    module = load_app(service)  # the fake torch has no __version__ and no arch list

    async def scenario():
        async with make_client(module) as client:
            return await client.get("/status")

    response = run(scenario())

    assert response.status_code == 200
    assert response.json()["runtime"] == {"nemo_toolkit": None, "torch": None, "torch_cuda": None}


# --- unloading a Canary checkpoint that carries its own aligner ----------------------------------

class _Aligner:
    def __init__(self):
        self.moves = []

    def cpu(self):
        self.moves.append("cpu")
        return self


@pytest.mark.parametrize("service_name", SERVICES)
def test_unload_moves_the_timestamp_aligner_off_the_gpu_too(service_name, monkeypatch):
    """NeMo keeps ``timestamps_asr_model`` outside the module tree, so ``model.cpu()`` misses it."""
    model = MODELS[service_name]()
    model.timestamps_asr_model = _Aligner()
    module = build(service_name, monkeypatch, model)
    assert transcribe(module).status_code == 200  # loads the model and takes it resident

    async def scenario():
        async with make_client(module) as client:
            return await client.post("/unload")

    response = run(scenario())

    assert response.json()["unloaded"] is True
    assert model.moves[-1] == "cpu"
    assert model.timestamps_asr_model.moves == ["cpu"], \
        "the alignment model kept its GPU memory after /unload reported the model released"


def test_a_failing_cpu_move_of_the_model_does_not_strand_the_aligner():
    common, _ = load_nemo_common("canary")
    aligner = _Aligner()

    class Broken:
        timestamps_asr_model = aligner

        def cpu(self):
            raise RuntimeError("cuda error")

    with pytest.raises(RuntimeError, match="cuda error"):
        common.move_to_cpu(Broken())

    assert aligner.moves == ["cpu"]


def test_a_model_without_an_aligner_is_moved_as_before():
    common, _ = load_nemo_common("parakeet")
    model = FakeParakeet()

    common.move_to_cpu(model)

    assert model.moves == ["cpu"]
