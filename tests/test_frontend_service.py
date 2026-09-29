"""Tests for the frontend service."""


def test_health(frontend_client):
    """Health endpoint returns platform status and service URLs."""
    r = frontend_client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert "services" in data


def test_index_returns_html(frontend_client):
    """Root endpoint serves the HTML frontend."""
    r = frontend_client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "TTS" in r.text


def test_api_docs_returns_html(frontend_client):
    """API docs endpoint serves an HTML page."""
    r = frontend_client.get("/api-docs")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_security_headers_are_sent(frontend_client):
    """Every response carries the framing / sniffing / referrer headers."""
    r = frontend_client.get("/health")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["referrer-policy"] == "same-origin"
    assert r.headers["x-frame-options"] == "SAMEORIGIN"


def test_cross_origin_state_change_is_refused(frontend_client):
    """A page on another site must not be able to delete a training job.

    Skipped when the deployment opted in to every origin (ALLOWED_ORIGINS=*),
    which is a legitimate but explicit choice; the offline suite covers the rest.
    """
    probe = frontend_client.options(
        "/api/training/job/nonexistent",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "DELETE"},
    )
    if probe.headers.get("access-control-allow-origin"):
        import pytest
        pytest.skip("this deployment allows every origin (ALLOWED_ORIGINS=*)")

    r = frontend_client.delete(
        "/api/training/job/nonexistent", headers={"Origin": "https://evil.example"})
    assert r.status_code == 403


def test_v1_errors_use_the_openai_envelope(frontend_client):
    """An unknown /v1 route answers in the envelope, not FastAPI's {"detail": ...}."""
    r = frontend_client.get("/v1/does-not-exist")
    assert r.status_code == 404
    assert set(r.json()["error"]) == {"message", "type", "param", "code"}
