"""Job lifecycle of the piper-training-service, through its real endpoints.

Everything here drives `app.py` over ASGI with a fake trainer and exporter (see
test_piper_training_service_support.py), so the registry, the background
runners, cancellation and the slot accounting are the real code.

The defects these pin down all came from one place: the job status string is
written by the endpoints, the training thread and the user's cancel, and nothing
said who may overwrite what or whether a thread was still alive. Two starts
could share the GPU, a resume could add a second thread to a running job,
deleting a running job removed the files under it, a cancelled retrain was
flipped back to "training", and a failed export left the job at "exporting" or
"completed" with no model.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from pathlib import Path

import pytest

from test_piper_training_service_support import (  # noqa: F401  (training_service is a fixture)
    LoopLag, asgi_client, install_fake_stt, make_dataset, settle, status_of, stt_result,
    training_service, wait_until, write_job_state, write_wav,
)


@contextlib.asynccontextmanager
async def client_for(svc):
    """A client; on the way out the fake trainer is released and every runner has finished."""
    async with asgi_client(svc.module) as client:
        try:
            yield client
        finally:
            await settle(svc)


def job_id_for(svc, model_name: str) -> str:
    return next(j.job_id for j in svc.module.training_jobs.values() if j.model_name == model_name)


async def start_blocked_job(svc, client, model_name: str = "alpha"):
    """Start a dataset job whose trainer blocks; returns (pending POST, job id).

    The POST is left as a task: with FastAPI background tasks it only completes
    when the job does, with the new runner it completes at once. Either way the
    job is running by the time this returns.
    """
    make_dataset(svc.root, model_name)
    svc.pipeline.block = True
    svc.pipeline.started.clear()
    post = asyncio.create_task(
        client.post("/train-from-dataset", data={"model_name": model_name, "epochs": 5}))
    await wait_until(svc.pipeline.started.is_set)
    return post, job_id_for(svc, model_name)


# --- one active job at a time -----------------------------------------------------


def test_a_second_job_is_refused_with_409_naming_the_running_one(training_service):
    svc = training_service()

    async def run():
        async with client_for(svc) as client:
            _, running = await start_blocked_job(svc, client, "alpha")
            make_dataset(svc.root, "beta")
            try:
                second = await asyncio.wait_for(
                    client.post("/train-from-dataset", data={"model_name": "beta", "epochs": 5}), 3)
            except asyncio.TimeoutError:
                pytest.fail("the second job was accepted and is running next to the first")
            assert second.status_code == 409
            assert running in second.json()["detail"]
            assert len(svc.pipeline.calls) == 1, "a second training thread was started"
            assert not any(j.model_name == "beta" for j in svc.module.training_jobs.values()), (
                "a refused request left a job record behind")

    asyncio.run(run())


def test_every_way_of_starting_a_job_respects_the_limit(training_service, monkeypatch):
    svc = training_service()
    install_fake_stt(monkeypatch, lambda p, path: None)
    write_wav(svc.root / "data" / "gamma" / "audio" / "gamma_a_0000.wav")
    write_job_state(svc.root, "old-job", "delta", status="training")

    async def run():
        async with client_for(svc) as client:
            _, running = await start_blocked_job(svc, client, "alpha")
            attempts = {
                "/train": {"data": {"model_name": "epsilon"},
                           "files": {"audio_files": ("a.wav", b"RIFFxxxx", "audio/wav")}},
                "/retrain-from-segments": {"data": {"model_name": "gamma"}},
                "/resume-training": {"data": {"model_name": "delta"}},
            }
            for path, kwargs in attempts.items():
                try:
                    response = await asyncio.wait_for(client.post(path, **kwargs), 3)
                except asyncio.TimeoutError:
                    pytest.fail(f"{path} was accepted while a job held the training slot")
                assert response.status_code == 409, f"{path}: {response.status_code} {response.text}"
                assert running in response.json()["detail"]
            assert len(svc.pipeline.calls) == 1

    asyncio.run(run())


def test_capacity_can_be_raised_but_one_model_never_runs_twice(training_service):
    svc = training_service({"TRAINING_MAX_CONCURRENT": "2"})

    async def run():
        async with client_for(svc) as client:
            _, first = await start_blocked_job(svc, client, "alpha")
            make_dataset(svc.root, "beta")
            svc.pipeline.started.clear()
            beta = asyncio.create_task(
                client.post("/train-from-dataset", data={"model_name": "beta", "epochs": 5}))
            await wait_until(svc.pipeline.started.is_set)
            assert len(svc.pipeline.calls) == 2, "the second slot was not usable"

            # Two jobs on one model would write the same dataset and voice.
            again = await client.post("/train-from-dataset", data={"model_name": "alpha", "epochs": 5})
            assert again.status_code == 409
            assert first in again.json()["detail"]

            make_dataset(svc.root, "gamma")
            third = await client.post("/train-from-dataset", data={"model_name": "gamma", "epochs": 5})
            assert third.status_code == 409, "capacity of 2 was exceeded"
            await settle(svc)
            await beta

    asyncio.run(run())


@pytest.mark.parametrize("outcome", ["completes", "fails", "is cancelled"])
def test_the_slot_is_freed_however_the_job_ends(training_service, outcome):
    svc = training_service()
    make_dataset(svc.root, "alpha")
    make_dataset(svc.root, "beta")

    async def run():
        async with client_for(svc) as client:
            if outcome == "fails":
                svc.pipeline.error = RuntimeError("diverged")
            if outcome == "is cancelled":
                post, job_id = await start_blocked_job(svc, client, "alpha")
                assert (await client.delete(f"/job/{job_id}")).status_code == 200
                await settle(svc)
                await post
            else:
                first = await client.post("/train-from-dataset", data={"model_name": "alpha", "epochs": 5})
                assert first.status_code == 202
                await settle(svc)
            svc.pipeline.error = None
            svc.pipeline.block = False
            second = await client.post("/train-from-dataset", data={"model_name": "beta", "epochs": 5})
            assert second.status_code == 202, f"the slot stayed taken after a job that {outcome}: {second.text}"

    asyncio.run(run())


# --- resume -------------------------------------------------------------------------


def test_resuming_a_job_that_is_running_is_refused(training_service):
    svc = training_service()

    async def run():
        async with client_for(svc) as client:
            _, running = await start_blocked_job(svc, client, "alpha")
            # What a checkpoint written a few epochs ago looks like on disk.
            write_job_state(svc.root, running, "alpha", status="training")
            try:
                response = await asyncio.wait_for(
                    client.post("/resume-training", data={"model_name": "alpha"}), 3)
            except asyncio.TimeoutError:
                pytest.fail("a second thread was started on a job that is already training")
            assert response.status_code == 409
            assert running in response.json()["detail"]
            assert len(svc.pipeline.calls) == 1

    asyncio.run(run())


def test_a_job_the_registry_says_is_running_cannot_be_resumed_even_without_a_slot(training_service):
    """The status string alone is enough to refuse: it is what every client sees."""
    svc = training_service()
    write_job_state(svc.root, "job-x", "alpha", status="training")
    svc.module.training_jobs["job-x"] = svc.module.TrainingStatus(
        job_id="job-x", status="training", progress=10, current_epoch=5, total_epochs=10,
        loss=0.5, message="running", model_name="alpha")

    async def run():
        async with client_for(svc) as client:
            response = await asyncio.wait_for(
                client.post("/resume-training", data={"model_name": "alpha"}), 3)
            assert response.status_code == 409
            assert svc.pipeline.calls == []

    asyncio.run(run())


@pytest.mark.parametrize("disk_status", ["training", "cancelled", "failed"])
def test_an_interrupted_job_resumes_from_its_checkpoint_and_finishes(training_service, disk_status):
    svc = training_service()
    write_job_state(svc.root, "job-r", "alpha", status=disk_status, epoch=5, total_epochs=10)

    async def run():
        async with client_for(svc) as client:
            response = await client.post("/resume-training", data={"model_name": "alpha"})
            assert response.status_code == 202
            assert response.json()["job_id"] == "job-r"
            assert response.json()["resume_from_epoch"] == 5
            await settle(svc)
            assert status_of(svc, "job-r") == "completed"
            call = svc.pipeline.calls[0]
            assert call.resume_from.endswith("checkpoint_epoch_5.pt")
            assert call.request.epochs == 10
            assert svc.exporter.calls == ["job-r"]

    asyncio.run(run())


def test_a_completed_job_is_exported_not_resumed(training_service):
    svc = training_service()
    write_job_state(svc.root, "job-c", "alpha", status="completed")

    async def run():
        async with client_for(svc) as client:
            response = await client.post("/resume-training", data={"model_name": "alpha"})
            assert response.status_code == 404
            assert svc.pipeline.calls == []

    asyncio.run(run())


def test_resume_by_job_id_cannot_relabel_the_job_as_another_model(training_service):
    """Resuming job A "as" model B trained A's weights and deployed them as B."""
    svc = training_service()
    write_job_state(svc.root, "job-a", "alpha", status="training")

    async def run():
        async with client_for(svc) as client:
            response = await client.post(
                "/resume-training", data={"model_name": "beta", "job_id": "job-a"})
            assert response.status_code == 400
            assert "alpha" in response.json()["detail"]
            assert svc.pipeline.calls == []

    asyncio.run(run())


def test_a_refused_resume_does_not_keep_the_training_slot(training_service):
    svc = training_service()
    write_job_state(svc.root, "job-m", "alpha", status="training", with_checkpoint=False)
    make_dataset(svc.root, "beta")

    async def run():
        async with client_for(svc) as client:
            missing = await client.post("/resume-training", data={"model_name": "alpha"})
            assert missing.status_code == 404
            ok = await client.post("/train-from-dataset", data={"model_name": "beta", "epochs": 5})
            assert ok.status_code == 202

    asyncio.run(run())


# --- failures leave a terminal status --------------------------------------------------


def test_a_failed_export_after_resume_leaves_the_job_failed_not_exporting(training_service):
    svc = training_service()
    write_job_state(svc.root, "job-r", "alpha", status="training")
    svc.exporter.error = RuntimeError("onnx exploded")

    async def run():
        async with client_for(svc) as client:
            await client.post("/resume-training", data={"model_name": "alpha"})
            await settle(svc)
            job = svc.module.training_jobs["job-r"]
            assert job.status == "failed", f"stuck at {job.status!r}: {job.message}"
            assert "onnx exploded" in job.message
            assert "/export/job-r" in job.message, "the message should say how to retry"

    asyncio.run(run())


def test_a_failed_export_is_a_failed_job_with_a_way_to_retry(training_service):
    """'completed' means a model exists to download; here none does."""
    svc = training_service()
    make_dataset(svc.root, "alpha")
    svc.exporter.error = RuntimeError("no kernel image")

    async def run():
        async with client_for(svc) as client:
            response = await client.post("/train-from-dataset", data={"model_name": "alpha", "epochs": 5})
            job_id = response.json()["job_id"]
            await settle(svc)
            job = svc.module.training_jobs[job_id]
            assert job.status == "failed"
            assert "no kernel image" in job.message and f"/export/{job_id}" in job.message
            assert not (svc.root / "models" / job_id / f"{job_id}.onnx").exists()

    asyncio.run(run())


def test_a_failed_deployment_still_completes_because_the_export_exists(training_service):
    svc = training_service({
        "DEFAULT_DEPLOYMENT_TARGET": "piper-http",
        "PIPER_TTS_SERVICE_URL": "http://piper.invalid:5000",
    })
    make_dataset(svc.root, "alpha")

    async def run():
        async with client_for(svc) as client:
            response = await client.post("/train-from-dataset", data={"model_name": "alpha", "epochs": 5})
            job_id = response.json()["job_id"]
            await settle(svc)
            job = svc.module.training_jobs[job_id]
            assert job.status == "completed"
            assert "deployment failed" in job.message
            assert (svc.root / "models" / job_id / f"{job_id}.onnx").exists()

    asyncio.run(run())


def test_a_crashing_trainer_marks_the_job_failed_with_the_cause(training_service):
    svc = training_service()
    make_dataset(svc.root, "alpha")
    svc.pipeline.error = RuntimeError("CUDA out of memory")

    async def run():
        async with client_for(svc) as client:
            response = await client.post("/train-from-dataset", data={"model_name": "alpha", "epochs": 5})
            job_id = response.json()["job_id"]
            await settle(svc)
            job = svc.module.training_jobs[job_id]
            assert job.status == "failed" and "CUDA out of memory" in job.message
            assert svc.exporter.calls == []

    asyncio.run(run())


# --- cancel ------------------------------------------------------------------------------


def test_a_cancelled_job_stays_cancelled_and_is_never_exported(training_service):
    svc = training_service()

    async def run():
        async with client_for(svc) as client:
            post, job_id = await start_blocked_job(svc, client)
            assert (await client.delete(f"/job/{job_id}")).status_code == 200
            await settle(svc)
            await post
            assert status_of(svc, job_id) == "cancelled"
            assert svc.exporter.calls == [], "a cancelled job was exported"
            assert not (svc.root / "models" / job_id).exists()

    asyncio.run(run())


def test_cancelling_during_transcription_is_not_overwritten_by_the_retrain(training_service, monkeypatch):
    """The retrain runner set status "training" unconditionally after STT, undoing the cancel and
    then training a voice nobody wanted."""
    svc = training_service()
    for index in range(3):
        write_wav(svc.root / "data" / "voice" / "audio" / f"voice_clip_{index}.wav")

    async def run():
        gate = asyncio.Event()

        async def respond(processor, path):
            return stt_result()

        fake = install_fake_stt(monkeypatch, respond, gate)
        async with client_for(svc) as client:
            post = asyncio.create_task(client.post("/retrain-from-segments", data={"model_name": "voice"}))
            await wait_until(lambda: fake.waiting >= 1)
            job_id = job_id_for(svc, "voice")
            assert status_of(svc, job_id) == "transcribing"
            assert (await client.delete(f"/job/{job_id}")).status_code == 200

            gate.set()
            await settle(svc)
            await post
            assert status_of(svc, job_id) == "cancelled", (
                f"the retrain ignored the cancel and went on to {status_of(svc, job_id)!r}")
            assert svc.pipeline.calls == [], "a cancelled retrain still trained"
            assert svc.exporter.calls == []

    asyncio.run(run())


def test_a_job_that_is_not_running_cannot_be_cancelled(training_service):
    svc = training_service()
    write_job_state(svc.root, "job-old", "alpha", status="training")

    async def run():
        async with client_for(svc) as client:
            await svc.module.restore_interrupted_jobs()
            assert status_of(svc, "job-old") == "interrupted"
            response = await client.delete("/job/job-old")
            assert response.status_code == 400
            assert status_of(svc, "job-old") == "interrupted", "a stale job was relabelled cancelled"
            assert (await client.delete("/job/unknown")).status_code == 404

    asyncio.run(run())


def test_the_trainer_cannot_overwrite_a_cancel_or_announce_completion_early(training_service):
    svc = training_service()
    module = svc.module
    module.training_jobs["j"] = module.TrainingStatus(
        job_id="j", status="training", progress=1, current_epoch=1, total_epochs=5,
        loss=None, message="m", model_name="alpha")

    module.update_training_status("j", {"status": "completed", "message": "Training completed successfully"})
    assert module.training_jobs["j"].status == "training", (
        "the trainer's 'completed' only means the weights are saved; the export has not run")
    assert module.training_jobs["j"].message == "Training completed successfully"

    module.training_jobs["j"].status = "cancelled"
    module.update_training_status("j", {"status": "training", "current_epoch": 3})
    assert module.training_jobs["j"].status == "cancelled"
    assert module.training_jobs["j"].current_epoch == 3

    module.update_training_status("j", {"loss": float("nan")})
    assert module.training_jobs["j"].loss is None


def test_a_polling_client_never_sees_completed_before_the_model_exists(training_service):
    svc = training_service()
    make_dataset(svc.root, "alpha")
    svc.exporter.blocking_seconds = 0.4

    async def run():
        async with client_for(svc) as client:
            response = await client.post("/train-from-dataset", data={"model_name": "alpha", "epochs": 5})
            job_id = response.json()["job_id"]
            seen = set()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                status = (await client.get(f"/status/{job_id}")).json()["status"]
                seen.add(status)
                if status == "completed":
                    break
                await asyncio.sleep(0.005)
            assert "completed" in seen
            assert svc.exporter.calls == [job_id]
            # 'completed' is only ever visible once the ONNX is on disk.
            assert (svc.root / "models" / job_id / f"{job_id}.onnx").exists()

    asyncio.run(run())


# --- delete / export while running -----------------------------------------------------------


def test_deleting_a_running_job_is_refused_and_removes_nothing(training_service):
    svc = training_service()

    async def run():
        async with client_for(svc) as client:
            _, job_id = await start_blocked_job(svc, client)
            write_job_state(svc.root, job_id, "alpha", status="training")
            response = await client.delete(f"/model/{job_id}")
            assert response.status_code == 409
            assert (svc.root / "checkpoints" / job_id / "job_state.json").exists()
            assert (svc.root / "data" / "alpha" / "train.json").exists()
            assert job_id in svc.module.training_jobs

    asyncio.run(run())


def test_a_job_can_be_deleted_once_it_has_stopped(training_service):
    svc = training_service()

    async def run():
        async with client_for(svc) as client:
            post, job_id = await start_blocked_job(svc, client)
            await client.delete(f"/job/{job_id}")
            await settle(svc)
            await post
            response = await client.delete(f"/model/{job_id}")
            assert response.status_code == 200, response.text
            assert job_id not in svc.module.training_jobs
            assert not (svc.root / "data" / "alpha").exists()

    asyncio.run(run())


def test_a_running_job_cannot_be_exported_by_hand(training_service):
    svc = training_service()

    async def run():
        async with client_for(svc) as client:
            _, job_id = await start_blocked_job(svc, client)
            response = await client.post(f"/export/{job_id}", data={"model_name": "alpha"})
            assert response.status_code == 409
            assert svc.exporter.calls == []

    asyncio.run(run())


def test_a_finished_job_can_be_exported_by_hand(training_service):
    svc = training_service()
    make_dataset(svc.root, "alpha")

    async def run():
        async with client_for(svc) as client:
            response = await client.post("/train-from-dataset", data={"model_name": "alpha", "epochs": 5})
            job_id = response.json()["job_id"]
            await settle(svc)
            exported = await client.post(f"/export/{job_id}", data={"model_name": "alpha"})
            assert exported.status_code == 200, exported.text
            assert exported.json()["deployment"]["status"] == "skipped"

    asyncio.run(run())


def test_a_job_being_exported_by_hand_is_neither_deleted_nor_exported_twice(training_service):
    svc = training_service()
    make_dataset(svc.root, "alpha")

    async def run():
        async with client_for(svc) as client:
            response = await client.post("/train-from-dataset", data={"model_name": "alpha", "epochs": 5})
            job_id = response.json()["job_id"]
            await settle(svc)
            svc.exporter.calls.clear()
            svc.exporter.blocking_seconds = 0.4
            export = asyncio.create_task(client.post(f"/export/{job_id}", data={"model_name": "alpha"}))
            await wait_until(lambda: svc.exporter.calls == [job_id])
            assert (await client.delete(f"/model/{job_id}")).status_code == 409
            assert (await client.post(f"/export/{job_id}", data={"model_name": "alpha"})).status_code == 409
            assert (await export).status_code == 200
            assert svc.exporter.calls == [job_id], "the export ran twice"

    asyncio.run(run())


# --- shutdown --------------------------------------------------------------------------------


def test_shutdown_stops_running_jobs_so_they_stay_resumable(training_service):
    """The lifespan exit is the only place a running job can be asked to stop before the
    process goes; uvicorn used to sit waiting for the job instead."""
    svc = training_service({"TRAINING_SHUTDOWN_GRACE_S": "5"})

    async def run():
        async with client_for(svc) as client:
            async with svc.module.app.router.lifespan_context(svc.module.app):
                post, job_id = await start_blocked_job(svc, client)
                # The periodic checkpoint the job has written by now.
                write_job_state(svc.root, job_id, "alpha", status="training")
            # Back from the shutdown hook: the job has been asked to stop and has.
            assert status_of(svc, job_id) == "cancelled"
            assert "resume" in svc.module.training_jobs[job_id].message.lower()
            assert "cancelled" in svc.pipeline.observed_status
            assert len(svc.module._runner_tasks) == 0 or all(t.done() for t in svc.module._runner_tasks)
            assert svc.exporter.calls == []
            await post

    asyncio.run(run())


def test_jobs_are_restored_with_the_outcome_they_had(training_service):
    svc = training_service()
    write_job_state(svc.root, "j-killed", "a", status="training")
    write_job_state(svc.root, "j-failed", "b", status="failed", error="weights are NaN")
    write_job_state(svc.root, "j-cancelled", "c", status="cancelled")
    write_job_state(svc.root, "j-done", "d", status="completed")

    async def run():
        await svc.module.restore_interrupted_jobs()

    asyncio.run(run())
    jobs = svc.module.training_jobs
    assert jobs["j-killed"].status == "interrupted"
    assert jobs["j-failed"].status == "failed" and "weights are NaN" in jobs["j-failed"].message
    assert jobs["j-cancelled"].status == "cancelled"
    assert "j-done" not in jobs


# --- dataset maintenance is exclusive with training ---------------------------------------------


def test_the_dataset_of_a_training_model_cannot_be_rewritten(training_service):
    svc = training_service()
    write_wav(svc.root / "data" / "alpha" / "audio" / "clip.wav")

    async def run():
        async with client_for(svc) as client:
            await start_blocked_job(svc, client, "alpha")
            mels = await client.post("/generate-missing-mels", params={"model_name": "alpha"})
            assert mels.status_code == 409
            prepared = await client.post("/prepare-dataset", json={
                "model_name": "alpha",
                "segments": [{"audio_path": "data/alpha/audio/clip.wav", "text": "hallo welt",
                              "start_time": 0.0, "end_time": 1.0}]})
            assert prepared.status_code == 409
            assert not (svc.root / "data" / "alpha" / "mel" / "clip.npy").exists()

    asyncio.run(run())


# --- honesty about the trainer ----------------------------------------------------------------------


def test_health_and_job_status_say_what_the_trainer_is(training_service):
    svc = training_service()
    make_dataset(svc.root, "alpha")

    async def run():
        async with client_for(svc) as client:
            health = (await client.get("/health")).json()
            assert health["trainer_kind"] == "experimental-mel-flow"
            assert "not intelligible" in health["trainer_caveat"]

            response = await client.post("/train-from-dataset", data={"model_name": "alpha", "epochs": 5})
            job_id = response.json()["job_id"]
            await settle(svc)
            status = (await client.get(f"/status/{job_id}")).json()
            assert status["trainer_kind"] == "experimental-mel-flow"
            assert status["trainer_caveat"] == health["trainer_caveat"]
            assert (await client.get("/jobs")).json()[0]["trainer_kind"] == "experimental-mel-flow"

    asyncio.run(run())


def test_without_a_trainer_kind_the_fields_are_null_not_invented(training_service):
    svc = training_service(trainer_kind=None)

    async def run():
        async with client_for(svc) as client:
            health = (await client.get("/health")).json()
            assert health["trainer_kind"] is None and health["trainer_caveat"] is None

    asyncio.run(run())


def test_startup_logs_the_trainer_caveat(training_service, caplog):
    svc = training_service()

    async def run():
        async with svc.module.app.router.lifespan_context(svc.module.app):
            pass

    with caplog.at_level(logging.WARNING):
        asyncio.run(run())
    startup = [r for r in caplog.records if "experimental-mel-flow" in r.getMessage()]
    assert startup and startup[0].levelno == logging.WARNING
    assert "not intelligible" in startup[0].getMessage()


# --- /ready -------------------------------------------------------------------------------------------


def test_ready_is_200_when_the_service_can_take_a_job(training_service):
    svc = training_service()

    async def run():
        async with client_for(svc) as client:
            response = await client.get("/ready")
            body = response.json()
            assert response.status_code == 200
            assert body["status"] == "ready" and body["accepting_jobs"] is True
            assert body["checks"]["data"]["ok"] and body["checks"]["checkpoints"]["ok"]
            assert body["trainer_kind"] == "experimental-mel-flow"

    asyncio.run(run())


def test_ready_stays_200_while_busy_but_says_it_is_not_accepting(training_service):
    """A busy trainer is healthy; taking it out of rotation would be wrong."""
    svc = training_service()

    async def run():
        async with client_for(svc) as client:
            await start_blocked_job(svc, client)
            response = await client.get("/ready")
            assert response.status_code == 200
            body = response.json()
            assert body["accepting_jobs"] is False and body["active_jobs"] == 1

    asyncio.run(run())


def test_ready_fails_when_a_working_directory_is_unusable(training_service):
    svc = training_service()
    # A file where the directory should be: what a broken bind mount looks like.
    (svc.root / "checkpoints").write_text("not a directory")

    async def run():
        async with client_for(svc) as client:
            response = await client.get("/ready")
            assert response.status_code == 503
            body = response.json()
            assert body["status"] == "not_ready" and body["accepting_jobs"] is False
            assert any("checkpoints" in p for p in body["problems"])
            assert (await client.get("/health")).status_code == 200, "liveness must not depend on storage"

    asyncio.run(run())


def test_ready_fails_when_a_gpu_was_expected_but_torch_is_on_the_cpu(training_service):
    """A cu118 wheel on Blackwell, or a driver that is not mounted: training would start and crawl."""
    svc = training_service({"TRAINING_DEVICE_TYPE": "cuda"})

    async def run():
        async with client_for(svc) as client:
            response = await client.get("/ready")
            assert response.status_code == 503
            assert any("cpu" in p for p in response.json()["problems"])

    asyncio.run(run())


def test_ready_accepts_a_gpu_that_was_hidden_on_purpose(training_service, monkeypatch):
    """CUDA_VISIBLE_DEVICES="" is how an operator asks for the CPU; that is not a broken deployment."""
    svc = training_service({"TRAINING_DEVICE_TYPE": "cuda"})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    async def run():
        async with client_for(svc) as client:
            response = await client.get("/ready")
            assert response.status_code == 200, response.text
            assert "on purpose" in response.json()["checks"]["device"]["note"]

    asyncio.run(run())


# --- the event loop stays free ------------------------------------------------------------------------------


def _slow_loads(svc, monkeypatch, seconds: float):
    """librosa.load that blocks its thread, like decoding a real file does."""
    original = svc.librosa.load

    def slow(*args, **kwargs):
        time.sleep(seconds)
        return original(*args, **kwargs)

    monkeypatch.setattr(svc.librosa, "load", slow)


def test_generating_mels_does_not_freeze_the_service(training_service, monkeypatch):
    svc = training_service()
    for index in range(4):
        write_wav(svc.root / "data" / "voice" / "audio" / f"clip_{index}.wav")
    _slow_loads(svc, monkeypatch, 0.25)

    async def run():
        async with client_for(svc) as client:
            async with LoopLag() as lag:
                response = await client.post("/generate-missing-mels", params={"model_name": "voice"})
            assert response.status_code == 200 and response.json()["processed"] == 4
            assert lag.max_gap < 0.4, (
                f"the event loop was blocked for {lag.max_gap:.2f}s: /health could not have answered")
            assert len(list((svc.root / "data" / "voice" / "mel").glob("clip_*.npy"))) == 4

    asyncio.run(run())


def test_preparing_a_dataset_does_not_freeze_the_service(training_service, monkeypatch):
    svc = training_service()
    for index in range(3):
        write_wav(svc.root / "data" / "voice" / "raw" / f"clip_{index}.wav", seconds=2.0)
    _slow_loads(svc, monkeypatch, 0.25)
    segments = [
        {"audio_path": f"data/voice/raw/clip_{i}.wav", "text": f"das ist satz {i}",
         "start_time": 0.0, "end_time": 1.5}
        for i in range(3)
    ]

    async def run():
        async with client_for(svc) as client:
            async with LoopLag() as lag:
                response = await client.post("/prepare-dataset", json={"model_name": "voice", "segments": segments})
            assert response.status_code == 200, response.text
            assert lag.max_gap < 0.4, f"the event loop was blocked for {lag.max_gap:.2f}s"

    asyncio.run(run())


def test_the_onnx_export_does_not_freeze_the_service(training_service):
    svc = training_service()
    make_dataset(svc.root, "alpha")
    svc.exporter.blocking_seconds = 0.6  # a CPU-bound body inside an `async def`, like torch's

    async def run():
        async with client_for(svc) as client:
            first = await client.post("/train-from-dataset", data={"model_name": "alpha", "epochs": 5})
            job_id = first.json()["job_id"]
            await settle(svc)
            svc.exporter.calls.clear()
            async with LoopLag() as lag:
                response = await client.post(f"/export/{job_id}", data={"model_name": "alpha"})
            assert response.status_code == 200, response.text
            assert svc.exporter.calls == [job_id]
            assert lag.max_gap < 0.4, f"the event loop was blocked for {lag.max_gap:.2f}s during export"

    asyncio.run(run())
