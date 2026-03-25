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
