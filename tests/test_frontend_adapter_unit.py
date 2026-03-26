"""Unit tests for frontend provider registry and adapter endpoints."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import os

import httpx
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def frontend_module():
    """Load the frontend app module directly from disk for unit testing."""
    repo_root = Path(__file__).resolve().parents[1]
    app_path = repo_root / "frontend-service" / "app.py"

    spec = spec_from_file_location("frontend_service_app", app_path)
    module = module_from_spec(spec)

    previous_cwd = os.getcwd()
    try:
        os.chdir(app_path.parent)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        os.chdir(previous_cwd)

    return module


@pytest.fixture()
def frontend_test_client(frontend_module):
    """Create a TestClient for the frontend FastAPI app."""
    with TestClient(frontend_module.app) as client:
        yield client


def test_provider_registry_endpoint(frontend_test_client):
    """Provider registry should expose provider metadata and defaults."""
    response = frontend_test_client.get("/providers")
    assert response.status_code == 200
    data = response.json()
    assert "providers" in data
    assert data["ui"]["default_tts_provider"] in data["providers"]
    assert data["providers"]["piper"]["settings"]["defaults"]["quality"] == "medium"
    assert data["providers"]["qwen3"]["settings"]["defaults"]["speaker"] == "Vivian"
    assert data["providers"]["qwen3"]["contracts"]["model_selection"] == "model-selection-v1"
    assert data["providers"]["qwen3"]["contracts"]["runtime_status"] == "runtime-status-v1"
    assert data["providers"]["qwen3"]["contracts"]["voice_design"] == "voice-design-tts-v1"
    assert data["providers"]["piper-training"]["contracts"]["training"] == "voice-training-job-v1"
    assert data["providers"]["piper-training"]["settings"]["defaults"]["batch_size"] == "32"
    assert data["ui"]["copy"]["stt_title"] == "Speech-to-Text"
    assert data["providers"]["piper"]["ui"]["sections"]["tts"]["title"] == "Text-to-Speech (PiperTTS)"
    assert data["providers"]["qwen3"]["ui"]["forms"]["voice_clone"]["fields"]["voice_file"]["label"] == "Voice Sample Audio:"
    assert data["providers"]["piper-training"]["ui"]["forms"]["continue_training"]["actions"]["resume"] == "Resume from Checkpoint"
    assert data["providers"]["qwen3"]["ui"]["messages"]["model_switching"]["success"] == "Model switched to {model}"
    assert data["providers"]["qwen3"]["ui"]["messages"]["model_catalog"]["unavailable_option"] == "Service unavailable"
    assert data["providers"]["qwen3"]["ui"]["messages"]["saved_voice_library"]["empty_option"] == "No saved voices - upload a sample first"
    assert data["providers"]["piper"]["ui"]["messages"]["custom_voice_library"]["loading"] == "Loading voices..."
    assert data["providers"]["piper"]["ui"]["messages"]["tts_generation"]["start"] == "Generating speech with {provider}..."
    assert data["providers"]["whisper"]["ui"]["messages"]["transcription"]["segmented_heading"] == "Transcription with Segmentation"
    assert data["providers"]["piper-training"]["ui"]["messages"]["model_management"]["deploy_success"] == "Model \"{model_name}\" deployment status: {status} on {deployment_target}."
    assert data["providers"]["piper-training"]["ui"]["messages"]["job_details"]["configuration_heading"] == "Configuration"


def test_tts_adapter_rejects_unknown_provider(frontend_test_client):
    """Unknown providers should be rejected before any backend call is attempted."""
    response = frontend_test_client.post(
        "/api/tts",
        json={"provider": "does-not-exist", "text": "Hello world"},
    )
    assert response.status_code == 404


def test_stt_adapter_rejects_unknown_provider(frontend_test_client, test_audio_bytes):
    """Unknown STT providers should be rejected before any backend call is attempted."""
    response = frontend_test_client.post(
        "/api/stt",
        files={"audio": ("test.wav", test_audio_bytes, "audio/wav")},
        data={"provider": "does-not-exist"},
    )
    assert response.status_code == 404


class _MockAsyncClient:
    """Minimal async httpx client stub used by the adapter tests."""

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url):
        if url.endswith('/deployment-targets'):
            return httpx.Response(
                200,
                json={
                    'default_target': 'none',
                    'targets': {
                        'none': {
                            'display_name': 'Manual download only',
                            'deployment_contract': 'manual-artifact-v1',
                            'capabilities': ['download_only'],
                        }
                    },
                },
                request=httpx.Request('GET', url),
            )

        if url.endswith('/jobs'):
            return httpx.Response(
                200,
                json=[{
                    'job_id': 'job-1',
                    'status': 'completed',
                    'model_name': 'demo',
                    'deployment_target': 'none',
                    'progress': 100.0,
                    'created_at': '2025-01-01T00:00:00Z',
                }],
                request=httpx.Request('GET', url),
            )

        if '/status/' in url:
            return httpx.Response(
                200,
                json={
                    'job_id': 'job-1',
                    'status': 'completed',
                    'deployment_target': 'none',
                    'progress': 100.0,
                    'current_epoch': 100,
                    'total_epochs': 100,
                    'loss': 0.12345,
                    'model_name': 'demo',
                    'created_at': '2025-01-01T00:00:00Z',
                    'config': {
                        'speaker_name': 'demo',
                        'epochs': 100,
                        'batch_size': 16,
                        'learning_rate': 0.0001,
                    },
                    'logs': [
                        {'timestamp': '2025-01-01T00:05:00Z', 'message': 'Training completed successfully'},
                    ],
                },
                request=httpx.Request('GET', url),
            )

        if '/download/' in url:
            return httpx.Response(
                200,
                content=b'onnx-binary',
                headers={
                    'content-type': 'application/octet-stream',
                    'content-disposition': 'attachment; filename="job-1.onnx"',
                },
                request=httpx.Request('GET', url),
            )

        if url.endswith('/models'):
            return httpx.Response(
                200,
                json={
                    'current_model': 'Qwen3-0.6B',
                    'models': {
                        'Qwen3-0.6B': {
                            'name': 'Qwen3 0.6B',
                            'description': 'Fast model',
                            'capabilities': ['voice_clone'],
                        },
                        'Qwen3-1.7B': {
                            'name': 'Qwen3 1.7B',
                            'description': 'High quality model',
                            'capabilities': ['voice_clone', 'voice_design'],
                        },
                    },
                },
                request=httpx.Request('GET', url),
            )

        if url.endswith('/status'):
            return httpx.Response(
                200,
                json={
                    'device': 'cuda',
                    'cuda_available': True,
                    'model_loaded': True,
                    'current_model': 'Qwen3-0.6B',
                    'current_model_info': {'name': 'Qwen3 0.6B'},
                    'builtin_speakers': ['Vivian', 'Ryan'],
                    'gpu_memory_allocated': 2147483648,
                },
                request=httpx.Request('GET', url),
            )

        if url.endswith("/voices") and 'qwen3-tts-service' not in url:
            return httpx.Response(
                200,
                json={
                    "voices": {
                        "en_US-lessac-medium": {
                            "name": "lessac",
                            "language": "en_US",
                            "quality": "medium",
                            "model_type": "default",
                        },
                        "demo_custom_voice": {
                            "name": "demo_custom_voice",
                            "language": "en_US",
                            "quality": "medium",
                            "model_type": "custom",
                        }
                    }
                },
                request=httpx.Request("GET", url),
            )

        if url.endswith('/voices') and 'qwen3-tts-service' in url:
            return httpx.Response(
                200,
                json={
                    'voices': [
                        {
                            'id': 'voice-1',
                            'name': 'Demo Voice',
                            'lang': 'English',
                            'ref_text': 'sample reference text',
                            'created_at': '2025-01-01T00:00:00Z',
                        }
                    ]
                },
                request=httpx.Request('GET', url),
            )

        if url.endswith("/speakers"):
            return httpx.Response(
                200,
                json={"speakers": ["Vivian", "Ryan"], "languages": ["English", "German"]},
                request=httpx.Request("GET", url),
            )

        raise AssertionError(f"Unexpected GET url: {url}")

    async def post(self, url, json=None, data=None, files=None):
        if url.endswith('/train'):
            return httpx.Response(
                202,
                json={'job_id': 'job-1', 'message': 'started'},
                request=httpx.Request('POST', url),
            )

        if url.endswith('/train-from-dataset'):
            return httpx.Response(
                202,
                json={'job_id': 'job-2', 'message': 'started'},
                request=httpx.Request('POST', url),
            )

        if url.endswith('/resume-training'):
            return httpx.Response(
                202,
                json={'job_id': 'job-3', 'message': 'resumed'},
                request=httpx.Request('POST', url),
            )

        if '/export/' in url:
            return httpx.Response(
                200,
                json={'deployment': {'target': 'none', 'status': 'skipped'}},
                request=httpx.Request('POST', url),
            )

        if url.endswith('/load_model'):
            model = json['model'] if json else 'unknown'
            return httpx.Response(
                200,
                json={'message': 'switched', 'model_info': {'name': model}},
                request=httpx.Request('POST', url),
            )

        if url.endswith('/voices/save'):
            return httpx.Response(
                200,
                json={'voice_id': 'voice-1', 'status': 'saved'},
                request=httpx.Request('POST', url),
            )

        if '/voices/' in url and url.endswith('/tts'):
            return httpx.Response(
                200,
                content=b'RIFFsaved-voice-audio',
                headers={'content-type': 'audio/wav', 'x-generation-time': '1.2'},
                request=httpx.Request('POST', url),
            )

        if url.endswith('/transcribe'):
            file_keys = [entry[0] for entry in files or []]
            data_map = dict(data or [])
            assert file_keys == ['audio']
            assert data_map.get('language') == 'de'
            return httpx.Response(
                200,
                json={'text': 'hallo welt', 'segments': [{'start': 0.0, 'end': 1.0, 'text': 'hallo welt'}], 'language': 'de'},
                request=httpx.Request('POST', url),
            )

        if url.endswith('/v1/audio/transcriptions'):
            file_keys = [entry[0] for entry in files or []]
            data_map = dict(data or [])
            assert file_keys == ['file']
            assert data_map.get('response_format') == 'json'
            assert data_map.get('language') == 'en'
            return httpx.Response(
                200,
                json={
                    'transcript': 'hello world',
                    'language': 'en',
                    'duration': '1.75',
                    'segments': [{'start': '0.0', 'end': '1.75', 'text': 'hello world'}],
                },
                request=httpx.Request('POST', url),
            )

        if url.endswith('/clone') or url.endswith('/clone-with-ref-text'):
            return httpx.Response(
                200,
                content=b'RIFFclone-audio',
                headers={'content-type': 'audio/wav'},
                request=httpx.Request('POST', url),
            )

        if url.endswith('/voice_design'):
            return httpx.Response(
                200,
                content=b'RIFFdesign-audio',
                headers={'content-type': 'audio/wav'},
                request=httpx.Request('POST', url),
            )

        if url.endswith("/tts"):
            return httpx.Response(
                200,
                content=b"RIFFmock-audio",
                headers={"content-type": "audio/wav", "x-test-header": "ok"},
                request=httpx.Request("POST", url),
            )

        raise AssertionError(f"Unexpected POST url: {url}")

    async def delete(self, url):
        if '/voices/' in url or '/voice/' in url:
            return httpx.Response(
                200,
                json={'status': 'deleted'},
                request=httpx.Request('DELETE', url),
            )

        if '/model/' in url or '/job/' in url:
            return httpx.Response(
                200,
                json={'status': 'success'},
                request=httpx.Request('DELETE', url),
            )

        raise AssertionError(f'Unexpected DELETE url: {url}')


def test_piper_voice_catalog_adapter(frontend_module, frontend_test_client, monkeypatch):
    """Piper voice catalogs should be normalized by the frontend adapter."""
    monkeypatch.setattr(frontend_module.httpx, "AsyncClient", _MockAsyncClient)

    response = frontend_test_client.get("/api/providers/piper/voices")
    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "piper"
    assert data["voices"][0]["id"] == "en_US-lessac-medium"
    assert data["voices"][0]["name"] == "lessac"


def test_qwen3_speaker_catalog_adapter(frontend_module, frontend_test_client, monkeypatch):
    """Qwen3 speaker catalogs should be normalized into generic voice entries."""
    monkeypatch.setattr(frontend_module.httpx, "AsyncClient", _MockAsyncClient)

    response = frontend_test_client.get("/api/providers/qwen3/voices")
    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "qwen3"
    assert data["voices"][0]["id"] == "Vivian"
    assert data["voices"][0]["kind"] == "builtin"


def test_qwen3_model_catalog_adapter(frontend_module, frontend_test_client, monkeypatch):
    """Advanced Qwen3 model catalogs should be normalized by the frontend adapter."""
    monkeypatch.setattr(frontend_module.httpx, 'AsyncClient', _MockAsyncClient)

    response = frontend_test_client.get('/api/providers/qwen3/models')
    assert response.status_code == 200
    data = response.json()
    assert data['provider'] == 'qwen3'
    assert data['models'][0]['id'] == 'Qwen3-0.6B'
    assert data['models'][0]['is_current'] is True
    assert data['models'][0]['capabilities_text'] == 'voice_clone'
    assert data['current_model']['id'] == 'Qwen3-0.6B'


def test_qwen3_runtime_status_adapter(frontend_module, frontend_test_client, monkeypatch):
    """Qwen3 runtime status should be normalized into a stable frontend status shape."""
    monkeypatch.setattr(frontend_module.httpx, 'AsyncClient', _MockAsyncClient)

    response = frontend_test_client.get('/api/providers/qwen3/status')
    assert response.status_code == 200
    assert response.json() == {
        'provider': 'qwen3',
        'device_name': 'CUDA',
        'device_type': 'gpu',
        'model_loaded': True,
        'model_name': 'Qwen3 0.6B',
        'gpu_memory_gb': 2.0,
        'speakers': ['Vivian', 'Ryan'],
    }


def test_qwen3_model_select_adapter(frontend_module, frontend_test_client, monkeypatch):
    """Model selection responses should be normalized for frontend notifications."""
    monkeypatch.setattr(frontend_module.httpx, 'AsyncClient', _MockAsyncClient)

    response = frontend_test_client.post('/api/providers/qwen3/models/select', json={'model': 'Qwen3-1.7B'})
    assert response.status_code == 200
    assert response.json() == {
        'provider': 'qwen3',
        'message': 'switched',
        'model': {'id': 'Qwen3-1.7B', 'name': 'Qwen3-1.7B'},
    }


def test_qwen3_saved_voices_adapter(frontend_module, frontend_test_client, monkeypatch):
    """Saved voice libraries should be exposed through the provider adapter."""
    monkeypatch.setattr(frontend_module.httpx, 'AsyncClient', _MockAsyncClient)

    response = frontend_test_client.get('/api/providers/qwen3/saved-voices')
    assert response.status_code == 200
    data = response.json()
    assert data['provider'] == 'qwen3'
    assert data['voices'][0]['id'] == 'voice-1'
    assert data['voices'][0]['language'] == 'English'
    assert data['voices'][0]['reference_preview'] == 'sample reference text'
    assert data['voices'][0]['created_at_display'] == '2025-01-01 00:00 UTC'


def test_piper_custom_voices_adapter(frontend_module, frontend_test_client, monkeypatch):
    """Managed custom Piper voices should be exposed through the frontend adapter."""
    monkeypatch.setattr(frontend_module.httpx, 'AsyncClient', _MockAsyncClient)

    response = frontend_test_client.get('/api/providers/piper/custom-voices')
    assert response.status_code == 200
    data = response.json()
    assert data['provider'] == 'piper'
    assert data['voices'][0]['id'] == 'demo_custom_voice'
    assert data['voices'][0]['kind'] == 'custom'


def test_piper_custom_voice_delete_adapter(frontend_module, frontend_test_client, monkeypatch):
    """Managed custom Piper voices should be deletable through the frontend adapter."""
    monkeypatch.setattr(frontend_module.httpx, 'AsyncClient', _MockAsyncClient)

    response = frontend_test_client.delete('/api/providers/piper/custom-voices/demo_custom_voice')
    assert response.status_code == 200
    assert response.json()['status'] == 'deleted'


def test_qwen3_saved_voice_tts_adapter(frontend_module, frontend_test_client, monkeypatch):
    """Saved-voice synthesis should stream audio back through the frontend adapter."""
    monkeypatch.setattr(frontend_module.httpx, 'AsyncClient', _MockAsyncClient)

    response = frontend_test_client.post(
        '/api/providers/qwen3/saved-voices/voice-1/tts',
        data={'text': 'Hello world', 'lang': 'English'},
    )
    assert response.status_code == 200
    assert response.content.startswith(b'RIFF')
    assert response.headers['x-provider'] == 'qwen3'


def test_qwen3_voice_clone_adapter(frontend_module, frontend_test_client, monkeypatch):
    """Voice clone requests should proxy multipart uploads through the frontend adapter."""
    monkeypatch.setattr(frontend_module.httpx, 'AsyncClient', _MockAsyncClient)

    response = frontend_test_client.post(
        '/api/providers/qwen3/voice-clone',
        data={'text': 'Hello world', 'lang': 'English', 'ref_text': 'reference text'},
        files={'file': ('sample.wav', b'wave-data', 'audio/wav')},
    )
    assert response.status_code == 200
    assert response.content.startswith(b'RIFF')
    assert response.headers['x-provider'] == 'qwen3'


def test_qwen3_voice_design_adapter(frontend_module, frontend_test_client, monkeypatch):
    """Voice design requests should proxy JSON payloads through the frontend adapter."""
    monkeypatch.setattr(frontend_module.httpx, 'AsyncClient', _MockAsyncClient)

    response = frontend_test_client.post(
        '/api/providers/qwen3/voice-design',
        json={'text': 'Hello world', 'lang': 'English', 'voice_description': 'Warm narrator'},
    )
    assert response.status_code == 200
    assert response.content.startswith(b'RIFF')
    assert response.headers['x-provider'] == 'qwen3'


def test_frontend_tts_adapter_passthrough(frontend_module, frontend_test_client, monkeypatch):
    """Frontend TTS adapter should return backend audio and passthrough headers."""
    monkeypatch.setattr(frontend_module.httpx, "AsyncClient", _MockAsyncClient)

    response = frontend_test_client.post(
        "/api/tts",
        json={
            "provider": "piper",
            "text": "Hello world",
            "voice": "en_US-lessac-medium",
            "language": "en",
        },
    )
    assert response.status_code == 200
    assert response.content.startswith(b"RIFF")
    assert response.headers["x-provider"] == "piper"


def test_stt_form_adapter(frontend_module, frontend_test_client, monkeypatch, test_audio_bytes):
    """The normalized STT adapter should translate form uploads for stt-form providers."""
    monkeypatch.setattr(frontend_module.httpx, 'AsyncClient', _MockAsyncClient)

    response = frontend_test_client.post(
        '/api/stt',
        data={'provider': 'whisper', 'language': 'de'},
        files={'audio': ('test.wav', test_audio_bytes, 'audio/wav')},
    )

    assert response.status_code == 200
    assert response.headers['x-provider'] == 'whisper'
    assert response.json() == {
        'text': 'hallo welt',
        'segments': [{'start': 0.0, 'end': 1.0, 'text': 'hallo welt'}],
        'language': 'de',
        'duration': None,
    }


def test_openai_stt_adapter(frontend_module, frontend_test_client, monkeypatch, test_audio_bytes):
    """The normalized STT adapter should translate form uploads for OpenAI-compatible STT providers."""
    monkeypatch.setattr(frontend_module.httpx, 'AsyncClient', _MockAsyncClient)

    response = frontend_test_client.post(
        '/api/stt',
        data={'provider': 'whisper-cpp', 'language': 'en'},
        files={'audio': ('test.wav', test_audio_bytes, 'audio/wav')},
    )

    assert response.status_code == 200
    assert response.headers['x-provider'] == 'whisper-cpp'
    assert response.json() == {
        'text': 'hello world',
        'segments': [{'start': 0.0, 'end': 1.75, 'text': 'hello world'}],
        'language': 'en',
        'duration': 1.75,
    }


def test_training_targets_adapter(frontend_module, frontend_test_client, monkeypatch):
    """Frontend should proxy training deployment target metadata."""
    monkeypatch.setattr(frontend_module.httpx, 'AsyncClient', _MockAsyncClient)

    response = frontend_test_client.get('/api/training/deployment-targets')
    assert response.status_code == 200
    data = response.json()
    assert data['default_target'] == 'none'
    assert 'none' in data['targets']


def test_training_jobs_adapter(frontend_module, frontend_test_client, monkeypatch):
    """Frontend should proxy training job listings."""
    monkeypatch.setattr(frontend_module.httpx, 'AsyncClient', _MockAsyncClient)

    response = frontend_test_client.get('/api/training/jobs')
    assert response.status_code == 200
    data = response.json()
    assert data[0]['job_id'] == 'job-1'
    assert data[0]['voice_name'] == 'demo'
    assert data[0]['deployment_target_label'] == 'Manual download only'
    assert data[0]['created_at_display'] == '2025-01-01 00:00 UTC'


def test_training_status_adapter(frontend_module, frontend_test_client, monkeypatch):
    """Frontend should normalize training job details for the browser."""
    monkeypatch.setattr(frontend_module.httpx, 'AsyncClient', _MockAsyncClient)

    response = frontend_test_client.get('/api/training/status/job-1')
    assert response.status_code == 200
    data = response.json()
    assert data['job_id'] == 'job-1'
    assert data['voice_name'] == 'demo'
    assert data['deployment_target_label'] == 'Manual download only'
    assert data['config_summary'] == {
        'epochs': 100,
        'batch_size': 16,
        'learning_rate': 0.0001,
    }
    assert data['best_loss'] == pytest.approx(0.12345)
    assert data['best_loss_display'] == '0.1235'
    assert data['recent_logs'][0]['display'] == '2025-01-01 00:05 UTC: Training completed successfully'


def test_training_download_adapter(frontend_module, frontend_test_client, monkeypatch):
    """Frontend should proxy model downloads with upstream headers."""
    monkeypatch.setattr(frontend_module.httpx, 'AsyncClient', _MockAsyncClient)

    response = frontend_test_client.get('/api/training/download/job-1')
    assert response.status_code == 200
    assert response.content == b'onnx-binary'
    assert 'attachment;' in response.headers['content-disposition']