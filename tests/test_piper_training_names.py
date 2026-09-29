"""Model names and job ids: one alphabet, re-checked wherever they come back from disk.

`safe_name` used to block path separators only, so CR, LF, ESC, spaces and `?#%`
passed (and were then refused by piper, qwen3 and the gateway, which accept
`[A-Za-z0-9_-]`). Worse, the same names were read back from `job_state.json`
after a restart and built into `data/<name>` for `shutil.rmtree`, unchecked: a
tampered or restored state file naming `../checkpoints` deleted the checkpoints
of every job.

These drive the real endpoints (fake trainer and exporter, see
test_piper_training_service_support.py) with hostile state on disk. Run against
the pre-fix service (`PIPER_TRAINING_SERVICE_DIR`) the destructive ones lose
data, which is how they were checked.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

from test_piper_training_service_support import (  # noqa: F401  (training_service is a fixture)
    asgi_client, install_fake_stt, make_dataset, settle, stt_result, training_service, write_job_state,
    write_wav,
)

# What a hostile or damaged state file can hold in place of a name. None of these
# may reach the filesystem: "..", "../x" and "sub/dir" walk out of data/, the rest
# are not in [A-Za-z0-9_-].
HOSTILE_NAMES = [
    "../victim", "../checkpoints", "..", "sub/dir", "a b", "a\nb", "x?y", "x#y", "x%2e%2e", "\x1b[31m",
    "voice.v2", "é", "a" * 65,
]


def _tamper(state_dir: Path, **fields) -> None:
    path = state_dir / "job_state.json"
    state = json.loads(path.read_text())
    state.update(fields)
    path.write_text(json.dumps(state))


def _victim(root: Path) -> Path:
    """A directory next to data/ and checkpoints/ that nothing in the service owns."""
    victim = root / "victim"
    victim.mkdir(exist_ok=True)
    (victim / "precious.txt").write_text("do not delete")
    return victim


# --- the alphabet at the endpoints -------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["a\nb", "x y", "x?y", "x#y", "\x1b[31m", "voice.v2", "é", "a" * 65])
@pytest.mark.parametrize("call", [
    ("post", "/train-from-dataset", "data"),
    ("post", "/retrain-from-segments", "data"),
    ("post", "/resume-training", "data"),
    ("post", "/export/job-a", "data"),
    ("post", "/generate-missing-mels", "params"),
])
def test_a_name_outside_the_alphabet_is_refused_with_400(training_service, call, name):
    svc = training_service()
    method, url, how = call

    async def run():
        async with asgi_client(svc.module) as client:
            kwargs = {"params": {"model_name": name}} if how == "params" else {"data": {"model_name": name}}
            response = await getattr(client, method)(url, **kwargs)
            assert response.status_code == 400, response.text
            detail = response.json()["detail"]
            assert "model_name" in detail
            assert "letters, digits" in detail, "the refusal should say what is allowed"
            # The name is echoed through repr(), so a control character comes back
            # escaped and never as the raw character.
            if "\x1b" in name:
                assert "\x1b" not in detail and "\\x1b" in detail

    asyncio.run(run())
    assert svc.pipeline.calls == []
    assert not (svc.root / "data").exists() or not any((svc.root / "data").iterdir())


def test_a_dataset_name_outside_the_alphabet_is_refused_by_prepare_dataset(training_service):
    svc = training_service()

    async def run():
        async with asgi_client(svc.module) as client:
            response = await client.post("/prepare-dataset", json={"model_name": "bad\nname", "segments": []})
            assert response.status_code == 400, response.text

    asyncio.run(run())


@pytest.mark.parametrize("path", ["/model/{}", "/download/{}"])
@pytest.mark.parametrize("job_id", ["a%20b", "a%0Ab", "a.b", "x" * 65])
def test_a_job_id_outside_the_alphabet_is_refused(training_service, path, job_id):
    svc = training_service()

    async def run():
        async with asgi_client(svc.module) as client:
            method = client.delete if path.startswith("/model") else client.get
            response = await method(path.format(job_id))
            assert response.status_code == 400, response.text

    asyncio.run(run())


@pytest.mark.parametrize("name", ["voice", "my_voice-2", "Voice-1", "a" * 64, "550e8400-e29b-41d4-a716-446655440000"])
def test_every_valid_name_still_starts_a_job(training_service, name):
    """Nothing that was a good name before (and is in the shared alphabet) got refused."""
    svc = training_service()
    make_dataset(svc.root, name)

    async def run():
        async with asgi_client(svc.module) as client:
            response = await client.post("/train-from-dataset", data={"model_name": name, "epochs": 5})
            assert response.status_code == 202, response.text
            await settle(svc)
            assert svc.module.training_jobs[response.json()["job_id"]].model_name == name

    asyncio.run(run())


def test_surrounding_whitespace_is_still_stripped(training_service):
    svc = training_service()
    make_dataset(svc.root, "voice")

    async def run():
        async with asgi_client(svc.module) as client:
            response = await client.post("/train-from-dataset", data={"model_name": "  voice\n", "epochs": 5})
            assert response.status_code == 202, response.text
            await settle(svc)

    asyncio.run(run())


# --- names read back from disk -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("tampered", HOSTILE_NAMES)
def test_deleting_a_job_never_removes_what_a_tampered_model_name_points_at(training_service, tampered):
    """`../checkpoints` in job_state.json used to become rmtree("data/../checkpoints")."""
    svc = training_service()
    victim = _victim(svc.root)
    bystander = write_job_state(svc.root, "job-other", "other")
    write_job_state(svc.root, "job-a", tampered)
    (svc.root / "data" / "other").mkdir(parents=True)

    async def run():
        async with asgi_client(svc.module) as client:
            response = await client.delete("/model/job-a")
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["model_name"] is None, "a name that is not a name must not be used or reported"
            assert "dataset directory" not in body["deleted_items"]

    asyncio.run(run())

    assert (victim / "precious.txt").exists()
    assert (bystander / "job_state.json").exists(), "another job's checkpoints were deleted"
    assert (svc.root / "data" / "other").exists()
    assert not (svc.root / "checkpoints" / "job-a").exists(), "the job's own files are still deleted"


def test_deleting_a_job_still_removes_its_dataset_when_the_name_is_valid(training_service):
    svc = training_service()
    make_dataset(svc.root, "voice")
    write_job_state(svc.root, "job-a", "voice", status="completed")

    async def run():
        async with asgi_client(svc.module) as client:
            response = await client.delete("/model/job-a")
            assert response.status_code == 200, response.text
            assert response.json()["model_name"] == "voice"

    asyncio.run(run())
    assert not (svc.root / "data" / "voice").exists()


def test_a_symlinked_directory_is_refused_before_anything_is_deleted(training_service):
    """Both roots are checked BEFORE the first removal, so a refusal leaves the job intact."""
    svc = training_service()
    elsewhere = svc.root / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "precious.txt").write_text("x")
    write_job_state(svc.root, "job-a", "voice", status="failed")
    (svc.root / "data").mkdir()
    os.symlink(elsewhere, svc.root / "data" / "voice")
    (svc.root / "models" / "job-a").mkdir(parents=True)
    asyncio.run(svc.module.restore_interrupted_jobs())
    assert "job-a" in svc.module.training_jobs

    async def run():
        async with asgi_client(svc.module) as client:
            response = await client.delete("/model/job-a")
            assert response.status_code == 400, response.text
            assert "outside" in response.json()["detail"]

    asyncio.run(run())

    assert (elsewhere / "precious.txt").exists()
    assert (svc.root / "checkpoints" / "job-a" / "job_state.json").exists(), "the request half-deleted the job"
    assert (svc.root / "models" / "job-a").exists()
    assert "job-a" in svc.module.training_jobs


def test_restoring_jobs_drops_names_and_ids_that_are_not_names(training_service):
    """The restored record is what DELETE /model reads the name from afterwards."""
    svc = training_service()
    write_job_state(svc.root, "job-ok", "voice", status="training")
    write_job_state(svc.root, "job-badname", "../victim", status="training")
    bad_id = write_job_state(svc.root, "job-badid", "voice2", status="training")
    _tamper(bad_id, job_id="../../x")

    asyncio.run(svc.module.restore_interrupted_jobs())

    jobs = svc.module.training_jobs
    assert set(jobs) == {"job-ok", "job-badname"}, "a state whose job_id is a path was restored"
    assert jobs["job-ok"].model_name == "voice"
    assert jobs["job-badname"].model_name is None


def test_a_restored_tampered_job_cannot_delete_outside_data_through_memory_either(training_service):
    svc = training_service()
    victim = _victim(svc.root)
    write_job_state(svc.root, "job-a", "../victim", status="failed")
    asyncio.run(svc.module.restore_interrupted_jobs())

    async def run():
        async with asgi_client(svc.module) as client:
            assert (await client.delete("/model/job-a")).status_code == 200

    asyncio.run(run())
    assert (victim / "precious.txt").exists()


def test_a_tampered_record_in_memory_is_rechecked_before_it_reaches_rmtree(training_service):
    """Belt and braces: whatever put the record there, DELETE /model checks the name itself."""
    svc = training_service()
    victim = _victim(svc.root)
    write_job_state(svc.root, "job-a", "voice", status="failed")
    asyncio.run(svc.module.restore_interrupted_jobs())
    svc.module.training_jobs["job-a"].model_name = "../victim"   # pydantic does not validate assignment

    async def run():
        async with asgi_client(svc.module) as client:
            assert (await client.delete("/model/job-a")).status_code == 200

    asyncio.run(run())
    assert (victim / "precious.txt").exists()


def test_the_sibling_lookup_reports_only_ids_it_can_vouch_for(training_service):
    svc = training_service()
    write_job_state(svc.root, "job-a", "voice", status="failed")
    forged = write_job_state(svc.root, "job-b", "voice", status="failed")
    _tamper(forged, job_id="forged\nINFO injected log line")

    assert svc.module._other_jobs_using_model("voice", "job-a") == ["job-b"], (
        "an id read from a file was passed on into the response and the log")


def test_resume_refuses_a_state_whose_job_id_is_not_a_job_id(training_service):
    svc = training_service()
    state_dir = write_job_state(svc.root, "job-a", "voice", status="training")
    _tamper(state_dir, job_id="../job-a")

    async def run():
        async with asgi_client(svc.module) as client:
            response = await client.post("/resume-training", data={"model_name": "voice"})
            assert response.status_code == 404, response.text

    asyncio.run(run())
    assert svc.pipeline.calls == []
    assert "../job-a" not in svc.module.training_jobs


def test_resume_will_not_load_a_checkpoint_from_outside_the_jobs_own_directory(training_service):
    """`latest_checkpoint` is a path in a file; torch.load must not be pointed at any file on disk."""
    svc = training_service()
    outside = svc.root / "elsewhere.pt"
    outside.write_bytes(b"not one of this job's checkpoints")
    state_dir = write_job_state(svc.root, "job-a", "voice", status="training", with_checkpoint=False)
    _tamper(state_dir, latest_checkpoint=str(outside))

    async def run():
        async with asgi_client(svc.module) as client:
            response = await client.post("/resume-training", data={"model_name": "voice"})
            assert response.status_code == 400, response.text
            assert "checkpoints/job-a" in response.json()["detail"]

    asyncio.run(run())
    assert svc.pipeline.calls == []


def test_resume_still_accepts_the_checkpoint_the_trainer_recorded(training_service):
    svc = training_service()
    write_job_state(svc.root, "job-a", "voice", status="training")

    async def run():
        async with asgi_client(svc.module) as client:
            response = await client.post("/resume-training", data={"model_name": "voice"})
            assert response.status_code == 202, response.text
            await settle(svc)

    asyncio.run(run())
    assert [c.job_id for c in svc.pipeline.calls] == ["job-a"]


def test_resume_accepts_the_relative_path_the_trainer_actually_writes(training_service):
    """training_pipeline records `checkpoints/<id>/checkpoint_epoch_N.pt`, relative to the working directory."""
    svc = training_service()
    state_dir = write_job_state(svc.root, "job-a", "voice", status="training")
    _tamper(state_dir, latest_checkpoint="checkpoints/job-a/checkpoint_epoch_5.pt")

    async def run():
        async with asgi_client(svc.module) as client:
            response = await client.post("/resume-training", data={"model_name": "voice"})
            assert response.status_code == 202, response.text
            await settle(svc)

    asyncio.run(run())


# --- the second guard in front of the shared-volume deploy and undeploy -----------------------------------------------


def test_deploying_and_undeploying_refuse_a_name_that_leaves_custom(training_service, tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    svc = training_service({"SHARED_MODELS_DIR": str(shared), "DEFAULT_DEPLOYMENT_TARGET": "piper-volume"})
    victim = shared / "victim"
    victim.mkdir()
    (victim / "precious.txt").write_text("x")
    onnx = svc.root / "job.onnx"
    onnx.write_bytes(b"onnx")

    async def run():
        for bad in ("../victim", "..", "a b"):
            with pytest.raises(HTTPException) as refused:
                await svc.module.deploy_model_bundle("job", bad, onnx, "piper-volume")
            assert refused.value.status_code == 400
            with pytest.raises(HTTPException) as refused:
                await svc.module.remove_model_from_deployment_target(bad, "piper-volume")
            assert refused.value.status_code == 400

        deployed = await svc.module.deploy_model_bundle("job", "voice", onnx, "piper-volume")
        assert deployed["status"] == "deployed"
        assert (shared / "custom" / "voice" / "voice.onnx").exists()
        await svc.module.remove_model_from_deployment_target("voice", "piper-volume")
        assert not (shared / "custom" / "voice").exists()

    asyncio.run(run())
    assert (victim / "precious.txt").exists()
    assert not (shared / "custom" / "victim").exists()


# --- log lines ---------------------------------------------------------------------------------------------------------


def test_an_upload_name_with_control_characters_never_reaches_the_log_or_the_disk_raw(
        training_service, monkeypatch, tmp_path, caplog):
    """The upload's own file name is client-controlled and is logged by the segmenter, the
    STT client and the dataset builder alike; it is cleaned once, where it is accepted."""
    svc = training_service()
    import audio_segmenter

    async def extract(self, input_path, output_path, start_time, end_time, sample_rate=22050, channels=1):
        write_wav(output_path, seconds=end_time - start_time)
        return True

    monkeypatch.setattr(audio_segmenter.AudioSegmenter, "_find_ffmpeg", lambda self: "ffmpeg")
    monkeypatch.setattr(audio_segmenter.AudioSegmenter, "extract_audio_segment", extract)

    async def respond(processor, path):
        return stt_result(count=5)

    install_fake_stt(monkeypatch, respond)
    wav = write_wav(tmp_path / "rec.wav", seconds=12.0).read_bytes()
    evil = "take\x1b[2J\x1b[31mFORGED\x9b.wav"

    async def run():
        async with asgi_client(svc.module) as client:
            response = await client.post(
                "/train", data={"model_name": "voice", "epochs": 5},
                files=[("audio_files", (evil, wav, "audio/wav"))])
            assert response.status_code == 202, response.text
            await settle(svc)
            assert svc.module.training_jobs[response.json()["job_id"]].status == "completed"

    with caplog.at_level(logging.INFO):
        asyncio.run(run())

    logged = [r.getMessage() for r in caplog.records if "FORGED" in r.getMessage()]
    assert logged, "the upload was not logged at all"
    assert not any(c in line for line in logged for c in "\x1b\x9b"), logged
    on_disk = [p.name for p in (svc.root / "data" / "voice").rglob("*") if "FORGED" in p.name]
    assert on_disk and not any(c in name for name in on_disk for c in "\x1b\x9b"), on_disk


@pytest.mark.parametrize("raw, expected", [
    ("take1.wav", "take1.wav"),
    ("C:\\Users\\me\\take1.wav", "take1.wav"),
    ("../../etc/take1.wav", "take1.wav"),
    ("a\nb\rc\x1b[0m.wav", "a_b_c_[0m.wav"),
    ("..", "upload_003"),
    ("", "upload_003"),
    ("\x00.wav", "_.wav"),
])
def test_upload_names_lose_directories_and_control_characters(training_service, raw, expected):
    svc = training_service()
    assert svc.module._upload_filename(raw, 3, set()) == expected


# --- the module's own contract, without an app --------------------------------------------------------------------------


def _validation():
    sys.modules.pop("validation", None)
    from importlib.util import module_from_spec, spec_from_file_location
    path = Path(__file__).resolve().parents[1] / "piper-training-service" / "validation.py"
    spec = spec_from_file_location("pt_validation_names", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", HOSTILE_NAMES + ["", "   ", "\x00", "a\x00b"])
def test_stored_name_and_confined_path_reject_everything_outside_the_alphabet(tmp_path, name):
    validation = _validation()
    assert validation.stored_name(name) is None
    assert not validation.is_safe_name(name)
    with pytest.raises(ValueError):
        validation.confined_path(tmp_path, name)


@pytest.mark.parametrize("value", [None, 5, ["voice"], b"voice"])
def test_stored_name_never_coerces(value):
    assert _validation().stored_name(value) is None


def test_confined_path_refuses_a_symlink_that_leaves_the_root(tmp_path):
    validation = _validation()
    root, elsewhere = tmp_path / "data", tmp_path / "elsewhere"
    root.mkdir()
    elsewhere.mkdir()
    os.symlink(elsewhere, root / "link")
    (root / "real").mkdir()

    assert validation.confined_path(root, "real") == root / "real"
    assert validation.confined_path(root, "not-yet-created") == root / "not-yet-created"
    with pytest.raises(ValueError, match="outside"):
        validation.confined_path(root, "link")


def test_is_within_resolves_dots_and_symlinks(tmp_path):
    validation = _validation()
    job = tmp_path / "checkpoints" / "job"
    job.mkdir(parents=True)
    (job / "c.pt").write_bytes(b"x")
    (tmp_path / "other.pt").write_bytes(b"x")
    os.symlink(tmp_path / "other.pt", job / "link.pt")

    assert validation.is_within(job / "c.pt", job)
    assert validation.is_within(job / "missing.pt", job), "a path that does not exist yet is judged by where it would be"
    assert not validation.is_within(job / ".." / ".." / "other.pt", job)
    assert not validation.is_within(job / "link.pt", job)
    assert not validation.is_within(tmp_path / "other.pt", job)
