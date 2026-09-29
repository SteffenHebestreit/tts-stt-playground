"""Tests for the Piper Training service."""


def test_health(training_client):
    """Health endpoint reports service readiness."""
    r = training_client.get("/health")
    assert r.status_code == 200


def test_list_jobs(training_client):
    """List training jobs — returns a flat list (may be empty)."""
    r = training_client.get("/jobs")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)


def test_deployment_targets(training_client):
    """Deployment target registry should be available."""
    r = training_client.get("/deployment-targets")
    assert r.status_code == 200
    data = r.json()
    assert "targets" in data
    assert data["default_target"] in data["targets"]


def test_train_requires_audio(training_client):
    """Starting training without audio files should fail."""
    r = training_client.post(
        "/train",
        data={"model_name": "test_voice", "language": "en"},
    )
    assert r.status_code in (400, 422)


def test_status_unknown_job(training_client):
    """Querying a non-existent job should return 404."""
    r = training_client.get("/status/nonexistent-job-id")
    assert r.status_code == 404


def test_train_from_dataset_rejects_path_traversal(training_client):
    """A model_name containing path separators must be rejected, not used to build a path."""
    r = training_client.post(
        "/train-from-dataset",
        data={"model_name": "../../etc", "language": "en"},
    )
    assert r.status_code == 400


def test_generate_missing_mels_rejects_path_traversal(training_client):
    """A model_name with traversal segments must be rejected before any filesystem access."""
    r = training_client.post(
        "/generate-missing-mels",
        params={"model_name": "../secrets"},
    )
    assert r.status_code == 400


def test_health_names_the_trainer(training_client):
    """The trainer is experimental; /health must say so rather than imply a finished VITS."""
    body = training_client.get("/health").json()
    assert "trainer_kind" in body and "trainer_caveat" in body
    if body["trainer_kind"]:
        assert body["trainer_caveat"]


def test_ready_reports_storage_and_device(training_client):
    """200 when jobs can be accepted, 503 with the reasons when the environment is broken."""
    r = training_client.get("/ready")
    assert r.status_code in (200, 503)
    body = r.json()
    assert {"data", "checkpoints", "models", "device"} <= set(body["checks"])
    assert (r.status_code == 200) == (body["status"] == "ready")

