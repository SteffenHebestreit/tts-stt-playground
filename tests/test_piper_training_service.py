"""Tests for the Piper Training service."""

import pytest


def test_health(training_client):
    r = training_client.get("/health")
    assert r.status_code == 200


def test_list_jobs(training_client):
    """List training jobs — returns a flat list (may be empty)."""
    r = training_client.get("/jobs")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)


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
