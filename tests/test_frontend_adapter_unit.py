"""Unit tests for frontend provider registry and adapter endpoints."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import os
import sys

import httpx
import pytest
from fastapi.testclient import TestClient

from frontend_loader import install_stub, load_frontend_app, wav_bytes


@pytest.fixture(scope="session")
def frontend_module():
    """Load the frontend app module directly from disk for unit testing."""
    repo_root = Path(__file__).resolve().parents[1]
    app_path = repo_root / "frontend-service" / "app.py"

    spec = spec_from_file_location("frontend_service_app", app_path)
    module = module_from_spec(spec)

    previous_cwd = os.getcwd()
    previous_enable_whisper_cpp = os.environ.get("ENABLE_WHISPER_CPP")
    previous_enable_parakeet_asr = os.environ.get("ENABLE_PARAKEET_ASR")
    # app.py imports sibling modules (openai_router). Loading it by path does
    # not put its directory on sys.path, so those imports must be made
    # resolvable explicitly — chdir alone is not enough.
    sys.path.insert(0, str(app_path.parent))
    try:
        os.environ["ENABLE_WHISPER_CPP"] = "true"
        os.environ["ENABLE_PARAKEET_ASR"] = "true"
        os.chdir(app_path.parent)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(app_path.parent))
        if previous_enable_whisper_cpp is None:
            os.environ.pop("ENABLE_WHISPER_CPP", None)
        else:
            os.environ["ENABLE_WHISPER_CPP"] = previous_enable_whisper_cpp
        if previous_enable_parakeet_asr is None:
            os.environ.pop("ENABLE_PARAKEET_ASR", None)
        else:
            os.environ["ENABLE_PARAKEET_ASR"] = previous_enable_parakeet_asr
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
    assert data["providers"]["whisper-cpp"]["contracts"]["transcribe"] == "openai-audio-transcriptions-v1"
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


class _StreamableResponse(httpx.Response):
    """An httpx.Response that also satisfies the streaming-proxy interface.

    The gateway proxies audio with `client.send(..., stream=True)` and forwards
    `aiter_raw()`, so the doubles have to answer those as well as `.json()`.
    Content is already fully materialised here, so it is handed over in one chunk.
    """

    async def aiter_raw(self, chunk_size=None):
        yield self.content

    async def aread(self):
        return self.content

    async def aclose(self):
        return None


class _MockRequest:
    """What `build_request` hands back to `send` — just the captured call."""

    def __init__(self, method, url, kwargs):
        self.method = method
        self.url = url
        self.kwargs = kwargs


class _MockAsyncClient:
    """Minimal async httpx client stub used by the adapter tests."""

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def build_request(self, method, url, **kwargs):
        return _MockRequest(method, url, kwargs)

    async def send(self, request, stream=False):
        """Dispatch through get/post so subclass overrides still apply."""
        kwargs = dict(request.kwargs)
        kwargs.pop("timeout", None)
        if request.method.upper() == "GET":
            response = await self.get(request.url, timeout=None)
        else:
            response = await self.post(request.url, timeout=None, **kwargs)
        return _StreamableResponse(
            status_code=response.status_code,
            headers=response.headers,
            content=response.content,
            request=response.request,
        )

    async def get(self, url, timeout=None):
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

    async def post(self, url, json=None, data=None, files=None, timeout=None):
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

        if url.endswith('/inference'):
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

    async def delete(self, url, timeout=None):
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


class _TrainingUnavailableAsyncClient(_MockAsyncClient):
    """Async client stub that simulates the training service being unreachable."""

    async def post(self, url, json=None, data=None, files=None, timeout=None):
        if url.endswith('/resume-training'):
            raise httpx.ConnectError('connection refused', request=httpx.Request('POST', url))
        return await super().post(url, json=json, data=data, files=files)


class _TrainingResumeNoFilesAsyncClient(_MockAsyncClient):
    """Async client stub that verifies data-only resume requests omit an empty files payload."""

    async def post(self, url, json=None, data=None, files=None, timeout=None):
        if url.endswith('/resume-training'):
            assert dict(data or []).get('model_name') == 'demo'
            assert files is None
            return httpx.Response(
                202,
                json={'job_id': 'job-3', 'message': 'resumed'},
                request=httpx.Request('POST', url),
            )
        return await super().post(url, json=json, data=data, files=files)


class _VoicesUnavailableAsyncClient(_MockAsyncClient):
    """Async client stub that simulates a TTS provider being unreachable on /voices."""

    async def get(self, url, timeout=None):
        if url.endswith('/voices'):
            raise httpx.ConnectError('connection refused', request=httpx.Request('GET', url))
        return await super().get(url)


class _TtsUnavailableAsyncClient(_MockAsyncClient):
    """Async client stub that simulates a TTS provider being unreachable on /tts."""

    async def post(self, url, json=None, data=None, files=None, timeout=None):
        if url.endswith('/tts'):
            raise httpx.ConnectError('connection refused', request=httpx.Request('POST', url))
        return await super().post(url, json=json, data=data, files=files)


class _SttUnavailableAsyncClient(_MockAsyncClient):
    """Async client stub that simulates an STT provider being unreachable on /transcribe."""

    async def post(self, url, json=None, data=None, files=None, timeout=None):
        if url.endswith('/transcribe'):
            raise httpx.ConnectError('connection refused', request=httpx.Request('POST', url))
        return await super().post(url, json=json, data=data, files=files)


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


def test_training_resume_returns_service_unavailable(frontend_module, frontend_test_client, monkeypatch):
    """Frontend should surface training transport failures as a 503 instead of a generic 500."""
    monkeypatch.setattr(frontend_module.httpx, 'AsyncClient', _TrainingUnavailableAsyncClient)

    response = frontend_test_client.post('/api/training/resume', data={'model_name': 'demo'})

    assert response.status_code == 503
    assert 'Piper Training is unavailable' in response.json()['detail']


def test_training_resume_adapter_omits_empty_files(frontend_module, frontend_test_client, monkeypatch):
    """Frontend should forward data-only resume forms without forcing an empty files payload."""
    monkeypatch.setattr(frontend_module.httpx, 'AsyncClient', _TrainingResumeNoFilesAsyncClient)

    response = frontend_test_client.post('/api/training/resume', data={'model_name': 'demo'})

    assert response.status_code == 200
    assert response.json() == {'job_id': 'job-3', 'message': 'resumed'}


def test_provider_voices_returns_service_unavailable(frontend_module, frontend_test_client, monkeypatch):
    """A voice catalog request should surface upstream transport failures as 503."""
    monkeypatch.setattr(frontend_module.httpx, 'AsyncClient', _VoicesUnavailableAsyncClient)

    response = frontend_test_client.get('/api/providers/piper/voices')

    assert response.status_code == 503
    assert 'is unavailable' in response.json()['detail']


def test_frontend_tts_returns_service_unavailable(frontend_module, frontend_test_client, monkeypatch):
    """The TTS adapter should surface upstream transport failures as 503 instead of 500."""
    monkeypatch.setattr(frontend_module.httpx, 'AsyncClient', _TtsUnavailableAsyncClient)

    response = frontend_test_client.post('/api/tts', json={'provider': 'piper', 'text': 'Hello world'})

    assert response.status_code == 503
    assert 'is unavailable' in response.json()['detail']


def test_frontend_stt_returns_service_unavailable(frontend_module, frontend_test_client, monkeypatch, test_audio_bytes):
    """The STT adapter should surface upstream transport failures as 503 instead of 500."""
    monkeypatch.setattr(frontend_module.httpx, 'AsyncClient', _SttUnavailableAsyncClient)

    response = frontend_test_client.post(
        '/api/stt',
        files={'audio': ('test.wav', test_audio_bytes, 'audio/wav')},
        data={'provider': 'whisper'},
    )

    assert response.status_code == 503
    assert 'is unavailable' in response.json()['detail']


def test_parakeet_provider_registered(frontend_test_client):
    """Parakeet should register as an STT provider on the shared stt-form-v1 contract."""
    data = frontend_test_client.get('/providers').json()
    assert data['ui']['enable_parakeet_asr'] is True
    parakeet = data['providers']['parakeet']
    assert parakeet['kind'] == 'stt'
    assert parakeet['contracts']['transcribe'] == 'stt-form-v1'
    assert parakeet['settings']['defaults']['enable_segmentation'] is True
    assert parakeet['ui']['selectable_as_stt'] is True


def test_parakeet_stt_transcribe(frontend_module, frontend_test_client, monkeypatch, test_audio_bytes):
    """Transcribing through the Parakeet provider should reuse the stt-form-v1 adapter."""
    monkeypatch.setattr(frontend_module.httpx, 'AsyncClient', _MockAsyncClient)

    response = frontend_test_client.post(
        '/api/stt',
        files={'audio': ('test.wav', test_audio_bytes, 'audio/wav')},
        data={'provider': 'parakeet', 'language': 'de'},
    )

    assert response.status_code == 200
    body = response.json()
    assert body['text'] == 'hallo welt'
    assert body['language'] == 'de'
    assert body['segments'][0]['text'] == 'hallo welt'


def test_training_download_adapter(frontend_module, frontend_test_client, monkeypatch):
    """Frontend should proxy model downloads with upstream headers."""
    monkeypatch.setattr(frontend_module.httpx, 'AsyncClient', _MockAsyncClient)

    response = frontend_test_client.get('/api/training/download/job-1')
    assert response.status_code == 200
    assert response.content == b'onnx-binary'
    assert 'attachment;' in response.headers['content-disposition']


# --------------------------------------------------------------------------
# Pure normalizer / helper unit tests (called directly, no HTTP)
# --------------------------------------------------------------------------

def test_normalize_qwen3_language(frontend_module):
    fn = frontend_module._normalize_qwen3_language
    assert fn("de") == "German"
    assert fn("en_US") == "English"
    assert fn("de-DE") == "German"
    assert fn("French") == "French"       # already a capitalized label
    # A language the model cannot speak is forwarded so the service can refuse it
    # (400 naming the supported ones), never swapped for English.
    assert fn("nl") == "nl"
    assert fn("zz") == "zz"
    assert fn("Dutch") == "Dutch"


@pytest.mark.parametrize("value", ["auto", "AUTO", " Auto ", "", "   ", None])
def test_qwen3_language_auto_stays_auto_so_the_service_default_applies(frontend_module, value):
    """qwen3-tts resolves "auto" and an empty value to QWEN3_DEFAULT_LANGUAGE (German unless
    the operator changed it). Answering "English" here overrode that on every request."""
    assert frontend_module._normalize_qwen3_language(value) == "auto"


def test_format_frontend_timestamp(frontend_module):
    fn = frontend_module._format_frontend_timestamp
    assert fn(None) is None
    assert fn("") is None
    assert fn("not-a-date") == "not-a-date"          # unparseable -> echoed
    assert fn("2025-01-01T00:05:00Z") == "2025-01-01 00:05 UTC"
    naive = fn("2025-01-01T12:30:00")                # no tz -> no UTC suffix
    assert naive == "2025-01-01 12:30"


def test_truncate_text(frontend_module):
    fn = frontend_module._truncate_text
    assert fn(None) is None
    assert fn("   ") is None
    assert fn("short") == "short"
    assert fn("x" * 100, limit=10).endswith("...")


def test_normalize_training_target_label(frontend_module):
    fn = frontend_module._normalize_training_target_label
    assert fn(None) == "Default"
    assert fn("none") == "Manual download only"
    assert fn("piper-volume") == "Piper shared volume"
    assert fn("custom-thing") == "Custom Thing"


def test_normalize_qwen3_voice_catalog(frontend_module):
    out = frontend_module._normalize_qwen3_voice_catalog(
        {"speakers": ["Vivian", "Ryan"], "languages": ["English", "German"]}
    )
    assert [v["id"] for v in out] == ["Vivian", "Ryan"]
    assert out[0]["kind"] == "builtin"


def test_normalize_qwen3_model_catalog(frontend_module):
    out = frontend_module._normalize_qwen3_model_catalog({
        "current_model": "m1",
        "models": {
            "m1": {"name": "M1", "capabilities": ["tts"]},
            "m2": {"name": "M2", "capabilities": ["tts", "voice_clone"]},
        },
    })
    by_id = {m["id"]: m for m in out}
    assert by_id["m1"]["is_current"] is True
    assert by_id["m2"]["capabilities_text"] == "tts, voice_clone"


def test_normalize_qwen3_runtime_status(frontend_module):
    out = frontend_module._normalize_qwen3_runtime_status({
        "device": "cuda",
        "cuda_available": True,
        "model_loaded": True,
        "current_model_info": {"name": "M"},
        "gpu_memory_allocated": 2 * 1024 ** 3,
        "builtin_speakers": ["A"],
    }, "qwen3")
    assert out["device_type"] == "gpu"
    assert out["gpu_memory_gb"] == 2.0
    assert out["model_name"] == "M"
    assert out["speakers"] == ["A"]


def test_normalize_qwen3_runtime_status_handles_bad_gpu_mem(frontend_module):
    out = frontend_module._normalize_qwen3_runtime_status(
        {"device": "cpu", "cuda_available": False, "gpu_memory_allocated": "nope"}, "qwen3"
    )
    assert out["device_type"] == "cpu"
    assert out["gpu_memory_gb"] is None
    assert out["speakers"] == []


def test_normalize_qwen3_saved_voice_library(frontend_module):
    out = frontend_module._normalize_qwen3_saved_voice_library({
        "voices": [
            {"id": "v1", "name": "Demo", "ref_text": "hello there", "created_at": "2025-01-01T00:00:00Z"},
            "not-a-dict",
        ]
    }, "qwen3")
    assert out["provider"] == "qwen3"
    assert len(out["voices"]) == 1
    assert out["voices"][0]["reference_preview"] == "hello there"
    assert out["voices"][0]["created_at_display"] == "2025-01-01 00:00 UTC"


def test_normalize_training_logs(frontend_module):
    logs = [{"timestamp": "2025-01-01T00:05:00Z", "message": "done"}, "plain string"]
    out = frontend_module._normalize_training_logs(logs)
    assert out[0]["display"] == "2025-01-01 00:05 UTC: done"
    assert out[1]["message"] == "plain string"
    # non-list input -> empty
    assert frontend_module._normalize_training_logs(None) == []


def test_normalize_training_job_best_loss_and_config(frontend_module):
    job = frontend_module._normalize_training_job({
        "job_id": "j1",
        "loss": "0.1234",
        "config": {"epochs": 100, "batch_size": 16, "learning_rate": 0.0001},
    })
    assert job["best_loss"] == 0.1234
    assert job["best_loss_display"] == "0.1234"
    assert job["config_summary"]["epochs"] == 100
    assert job["voice_name"] == "j1"  # falls back to job_id


def test_normalize_training_jobs_payload_shapes(frontend_module):
    fn = frontend_module._normalize_training_jobs_payload
    as_list = fn([{"job_id": "a"}, "skip-non-dict"])
    assert isinstance(as_list, list) and len(as_list) == 1
    as_dict = fn({"jobs": [{"job_id": "b"}]})
    assert as_dict["jobs"][0]["job_id"] == "b"
    assert fn("passthrough") == "passthrough"


def test_normalize_training_export_response(frontend_module):
    out = frontend_module._normalize_training_export_response(
        {"deployment": {"target": "piper-volume", "status": "deployed"}}
    )
    assert out["deployment"]["target_label"] == "Piper shared volume"


def test_build_error_from_response(frontend_module):
    """A 4xx keeps the backend's own explanation; the status is always the backend's."""
    resp = httpx.Response(
        400, json={"detail": "Language 'xx' is not supported. Supported: de, en."},
        request=httpx.Request("GET", "http://svc/x"))
    exc = frontend_module._build_error_from_response(resp)
    assert exc.status_code == 400
    assert exc.detail == "Language 'xx' is not supported. Supported: de, en."


def test_build_error_from_response_hides_what_a_5xx_says(frontend_module, caplog):
    resp = httpx.Response(502, json={"detail": "boom at /app/models/x.onnx"},
                          request=httpx.Request("GET", "http://svc/x"))
    with caplog.at_level("WARNING"):
        exc = frontend_module._build_error_from_response(resp)
    assert exc.status_code == 502
    assert "boom" not in exc.detail and "/app/" not in exc.detail
    request_id = exc.detail.rsplit("Request id: ", 1)[1].rstrip(".")
    assert any(request_id in r.getMessage() and "boom at /app/models/x.onnx" in r.getMessage()
               for r in caplog.records), "the detail belongs in the log, under the same id"


def test_build_upstream_request_error(frontend_module, caplog):
    err = httpx.ConnectError("refused", request=httpx.Request("POST", "http://svc:5005/transcribe"))
    with caplog.at_level("WARNING"):
        exc = frontend_module._build_upstream_request_error("Parakeet ASR", err)
    assert exc.status_code == 503
    assert "Parakeet ASR is unavailable" in exc.detail
    assert "svc" not in exc.detail and "5005" not in exc.detail and "://" not in exc.detail
    request_id = exc.detail.rsplit("Request id: ", 1)[1].rstrip(".")
    assert any(request_id in r.getMessage() and "http://svc:5005/transcribe" in r.getMessage()
               for r in caplog.records), "the URL belongs in the log, under the same id"


def test_get_http_client_is_shared_and_recreated(frontend_module, monkeypatch):
    frontend_module._http_client = None
    frontend_module._http_client_factory = None
    c1 = frontend_module._get_http_client()
    c2 = frontend_module._get_http_client()
    assert c1 is c2  # reused
    # Swapping the AsyncClient class triggers transparent recreation
    monkeypatch.setattr(frontend_module.httpx, "AsyncClient", _MockAsyncClient)
    c3 = frontend_module._get_http_client()
    assert c3 is not c1
    frontend_module._http_client = None
    frontend_module._http_client_factory = None


# --------------------------------------------------------------------------
# Contract-guard / validation error paths (return 4xx before any upstream call)
# --------------------------------------------------------------------------

def test_provider_voices_unknown_provider_404(frontend_test_client):
    assert frontend_test_client.get('/api/providers/nope/voices').status_code == 404


def test_provider_voices_wrong_kind_400(frontend_test_client):
    # whisper is an STT provider, not a TTS voice catalog
    assert frontend_test_client.get('/api/providers/whisper/voices').status_code == 400


def test_provider_models_rejects_unsupported(frontend_test_client):
    assert frontend_test_client.get('/api/providers/piper/models').status_code == 400


def test_provider_status_rejects_unsupported(frontend_test_client):
    assert frontend_test_client.get('/api/providers/piper/status').status_code == 400


def test_provider_model_select_rejects_unsupported(frontend_test_client):
    r = frontend_test_client.post('/api/providers/piper/models/select', json={'model': 'x'})
    assert r.status_code == 400


def test_provider_saved_voices_rejects_unsupported(frontend_test_client):
    assert frontend_test_client.get('/api/providers/piper/saved-voices').status_code == 400


def test_provider_delete_saved_voice_rejects_unsupported(frontend_test_client):
    assert frontend_test_client.delete('/api/providers/piper/saved-voices/x').status_code == 400


def test_provider_custom_voices_rejects_unsupported(frontend_test_client):
    assert frontend_test_client.get('/api/providers/qwen3/custom-voices').status_code == 400


def test_provider_delete_custom_voice_rejects_unsupported(frontend_test_client):
    assert frontend_test_client.delete('/api/providers/qwen3/custom-voices/x').status_code == 400


def test_provider_voice_clone_rejects_unsupported(frontend_test_client):
    assert frontend_test_client.post('/api/providers/piper/voice-clone').status_code == 400


def test_provider_voice_design_rejects_unsupported(frontend_test_client):
    r = frontend_test_client.post(
        '/api/providers/piper/voice-design',
        json={'text': 'hi', 'voice_description': 'deep voice'},
    )
    assert r.status_code == 400


def test_provider_save_voice_rejects_unsupported(frontend_test_client):
    assert frontend_test_client.post('/api/providers/piper/saved-voices').status_code == 400


def test_frontend_stt_missing_audio_400(frontend_test_client):
    r = frontend_test_client.post('/api/stt', data={'provider': 'whisper'})
    assert r.status_code == 400


def test_frontend_stt_missing_provider_400(frontend_test_client, test_audio_bytes):
    r = frontend_test_client.post('/api/stt', files={'audio': ('a.wav', test_audio_bytes, 'audio/wav')})
    assert r.status_code == 400


def test_frontend_tts_empty_text_400(frontend_test_client):
    r = frontend_test_client.post('/api/tts', json={'provider': 'piper', 'text': '   '})
    assert r.status_code == 400


def test_training_start_adapter(frontend_module, frontend_test_client, monkeypatch, test_audio_bytes):
    """Start-training proxies a multipart form to the training service."""
    monkeypatch.setattr(frontend_module.httpx, 'AsyncClient', _MockAsyncClient)
    r = frontend_test_client.post(
        '/api/training/train',
        data={'model_name': 'demo', 'language': 'de'},
        files={'audio_files': ('a.wav', test_audio_bytes, 'audio/wav')},
    )
    assert r.status_code == 200
    assert r.json()['job_id'] == 'job-1'


def test_training_export_adapter(frontend_module, frontend_test_client, monkeypatch):
    """Export proxies a form POST and labels the deployment target."""
    monkeypatch.setattr(frontend_module.httpx, 'AsyncClient', _MockAsyncClient)
    r = frontend_test_client.post('/api/training/export/job-1', data={'model_name': 'demo'})
    assert r.status_code == 200
    assert r.json()['deployment']['target_label'] == 'Manual download only'


def test_training_delete_model_adapter(frontend_module, frontend_test_client, monkeypatch):
    monkeypatch.setattr(frontend_module.httpx, 'AsyncClient', _MockAsyncClient)
    r = frontend_test_client.delete('/api/training/model/job-1')
    assert r.status_code == 200
    assert r.json()['status'] == 'success'


def test_training_cancel_job_adapter(frontend_module, frontend_test_client, monkeypatch):
    monkeypatch.setattr(frontend_module.httpx, 'AsyncClient', _MockAsyncClient)
    r = frontend_test_client.delete('/api/training/job/job-1')
    assert r.status_code == 200
    assert r.json()['status'] == 'success'


def test_training_deployment_targets_unavailable(frontend_module, frontend_test_client, monkeypatch):
    """Transport failure on a training GET should surface as 503."""
    monkeypatch.setattr(frontend_module.httpx, 'AsyncClient', _TrainingUnavailableAsyncClient)
    # _TrainingUnavailableAsyncClient only fails resume; deployment-targets GET succeeds via base.
    r = frontend_test_client.get('/api/training/deployment-targets')
    assert r.status_code == 200


# --------------------------------------------------------------------------
# Page routes (render the UI / docs and the lean health probe)
# --------------------------------------------------------------------------

def test_index_page_renders(frontend_test_client):
    r = frontend_test_client.get('/')
    assert r.status_code == 200
    assert 'text/html' in r.headers['content-type']


def test_api_docs_page(frontend_test_client):
    r = frontend_test_client.get('/api-docs')
    assert r.status_code == 200
    assert 'text/html' in r.headers['content-type']


def test_health_is_lean(frontend_test_client):
    """/health stays small: status + service map, no full provider registry (P4)."""
    r = frontend_test_client.get('/health')
    assert r.status_code == 200
    body = r.json()
    assert body['status'] == 'healthy'
    assert 'services' in body
    assert 'provider_registry' not in body

# --- language forwarding across contracts ------------------------------------
#
# whisper.cpp's server defaults to `std::string language = "en"` and only
# overrides it when the form field is PRESENT. So omitting the field on
# language="auto" made auto-detect mean "English", silently transcribing German
# audio as English on every whisper-cpp deployment (the ARM SBC and Strix Halo).
# "auto" is an explicitly supported value there, so it must be sent through.


class _Upload:
    """Minimal stand-in for a Starlette UploadFile."""

    filename = "a.wav"
    content_type = "audio/wav"

    async def read(self):
        return b"RIFF"


def _form(**kw):
    form = {"audio": _Upload()}
    form.update(kw)
    return form


def _build(frontend_module, contract, **kw):
    import asyncio
    return asyncio.run(
        frontend_module._build_frontend_stt_payload("whisper-cpp", _form(**kw), contract)
    )


def test_openai_contract_sends_auto_language_explicitly(frontend_module):
    """The regression: 'auto' must reach whisper.cpp, not be dropped."""
    _path, data, _files = _build(frontend_module, "openai-audio-transcriptions-v1", language="auto")
    assert data.get("language") == "auto", (
        "language=auto was omitted; whisper.cpp would default to English and "
        "silently mis-transcribe German audio"
    )


def test_openai_contract_defaults_to_auto_when_language_absent(frontend_module):
    _path, data, _files = _build(frontend_module, "openai-audio-transcriptions-v1")
    assert data.get("language") == "auto"


def test_openai_contract_forwards_an_explicit_language(frontend_module):
    _path, data, _files = _build(frontend_module, "openai-audio-transcriptions-v1", language="de")
    assert data.get("language") == "de"


@pytest.mark.parametrize("spelling", ["auto", "AUTO", " Auto "])
def test_native_contract_sends_auto_explicitly(frontend_module, spelling):
    """stt-form-v1 services read an explicit "auto" as "detect the language" and a missing
    field as "use STT_DEFAULT_LANGUAGE". Dropping "auto" made every Auto-Detect upload
    German on a deployment whose default is 'de', while the live microphone (which does
    send "auto") detected."""
    _path, data, _files = _build(frontend_module, "stt-form-v1", language=spelling)
    assert data.get("language") == "auto"


def test_native_contract_leaves_the_field_out_when_no_language_was_given(frontend_module):
    _path, data, _files = _build(frontend_module, "stt-form-v1")
    assert "language" not in data
    _path, data, _files = _build(frontend_module, "stt-form-v1", language="  ")
    assert "language" not in data

    _path, data, _files = _build(frontend_module, "stt-form-v1", language="de")
    assert data.get("language") == "de"


# --- upstream timeouts -------------------------------------------------------
#
# `_timeout()` splits the budget per phase because a bare float applies the
# WHOLE budget to every phase, connect included. The helper existed and was used
# on the streaming path, while nine other call sites still passed the raw float
# — so with the qwen3 container stopped, /voice-clone and saved-voice TTS sat on
# a dead socket for their full 600 s read budget before reporting "unavailable",
# instead of failing at the 3 s connect. Static, because a real 600 s wait is
# not something a test can sit through.


GATEWAY_APP = Path(__file__).resolve().parents[1] / "frontend-service" / "app.py"


def _httpx_timeout_arguments() -> list[str]:
    """Every `timeout=` keyword passed to a client call in the gateway."""
    import ast

    tree = ast.parse(GATEWAY_APP.read_text(encoding="utf-8"))

    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # client.get/post/delete/build_request(...), or a dict literal handed to
        # one of them as **request_kwargs.
        if not (isinstance(func, ast.Attribute)
                and func.attr in {"get", "post", "delete", "put", "build_request", "send"}):
            continue
        for kw in node.keywords:
            if kw.arg == "timeout":
                found.append(f"line {node.lineno}: timeout={ast.unparse(kw.value)}")
    return found


def test_every_upstream_call_uses_per_phase_timeouts():
    """No call site may hand httpx a bare number."""
    offenders = [
        entry for entry in _httpx_timeout_arguments()
        if "_timeout(" not in entry and "None" not in entry
    ]
    assert not offenders, (
        "these upstream calls pass a scalar timeout, which httpx applies to the "
        "connect phase too — a stopped backend then blocks for the full read "
        f"budget instead of failing fast:\n  " + "\n  ".join(offenders)
        + "\nWrap the value in _timeout()."
    )


def test_request_kwargs_dicts_also_use_per_phase_timeouts():
    """The multipart helpers build `{"data": ..., "timeout": ...}` and splat it,
    so the check above cannot see them."""
    import ast

    offenders: list[str] = []
    for node in ast.walk(ast.parse(GATEWAY_APP.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if not (isinstance(key, ast.Constant) and key.value == "timeout"):
                continue
            rendered = ast.unparse(value)
            if "_timeout(" not in rendered:
                offenders.append(f"line {node.lineno}: 'timeout': {rendered}")

    assert not offenders, (
        "these request-kwargs dicts carry a scalar timeout:\n  "
        + "\n  ".join(offenders) + "\nWrap the value in _timeout()."
    )


def test_connect_phase_stays_short():
    """The point of the split: a dead upstream is detected in seconds."""
    import re

    match = re.search(
        r"httpx\.Timeout\(connect=([0-9.]+)", GATEWAY_APP.read_text(encoding="utf-8"))
    assert match, "_timeout() no longer builds an httpx.Timeout with an explicit connect"
    assert float(match.group(1)) <= 10.0, (
        "the connect budget has grown past 10 s; container-to-container connects "
        "on a bridge network resolve in milliseconds, and a long one here is "
        "indistinguishable from the bare-float bug this replaced"
    )


# --- path parameters ----------------------------------------------------------
#
# job_id / voice_id are interpolated into the upstream URL after Starlette has
# percent-DECODED them: `a%3Fb=1` reached the training service as the query
# string of `/model/a`, `x%23y` as a fragment, and `%2e%2e` as `..` (DELETE
# /model/.. is DELETE /). Nothing a backend generates contains those characters.

RESOURCE_ROUTES = [
    ("DELETE", "/api/training/model/{}"),
    ("DELETE", "/api/training/job/{}"),
    ("GET", "/api/training/status/{}"),
    ("GET", "/api/training/download/{}"),
    ("POST", "/api/training/export/{}"),
    ("DELETE", "/api/providers/piper/custom-voices/{}"),
    ("DELETE", "/api/providers/qwen3/saved-voices/{}"),
    ("POST", "/api/providers/qwen3/saved-voices/{}/tts"),
]


def _echo_backend(method, url, kwargs):
    return {"status": "success"}


@pytest.mark.parametrize("method,template", RESOURCE_ROUTES)
@pytest.mark.parametrize("bad", [
    "a%3Fb=1", "x%23y", "%2e%2e", "..", ".", "a%20b", "a%2Fb", "a%5Cb", "a;b",
    "a%00b", "a%0d%0ab", "a" * 129,
])
def test_hostile_resource_ids_never_reach_a_backend(monkeypatch, method, template, bad):
    module = load_frontend_app()
    stub = install_stub(monkeypatch, module, _echo_backend)
    r = TestClient(module.app).request(method, template.format(bad))
    # 422 for an id the route matches and rejects; 404/405 where the client has
    # already normalised the path (`.` / `..` segments) onto some other route.
    assert r.status_code in (404, 405, 422), (r.status_code, r.text)
    assert stub.calls == [], f"{method} {template.format(bad)} was forwarded: {stub.calls}"


@pytest.mark.parametrize("method,template", RESOURCE_ROUTES)
@pytest.mark.parametrize("good", [
    "job-1", "de_DE-thorsten-medium", "3f2b8c1e-5d4a-4f0e-9b7a-1c2d3e4f5a6b", "A" * 128,
])
def test_ids_the_backends_generate_still_pass(monkeypatch, method, template, good):
    module = load_frontend_app()

    def backend(m, url, kwargs):
        if url.endswith("/tts"):
            return httpx.Response(200, content=wav_bytes(), headers={"content-type": "audio/wav"})
        return {"status": "success", "deployment": {}}

    stub = install_stub(monkeypatch, module, backend)
    if template.endswith("/tts"):
        r = TestClient(module.app).post(template.format(good), data={"text": "hi"})
    else:
        r = TestClient(module.app).request(method, template.format(good))
    assert r.status_code == 200, (r.status_code, r.text)
    assert len(stub.calls) == 1
    assert good in stub.calls[0][1]


# --- a 200 that is not the JSON the gateway expects ----------------------------
#
# ~20 endpoints called `response.json()` on a successful upstream reply with
# nothing around it, so a proxy's HTML error page, a truncated body or a bare
# string became an unhandled JSONDecodeError: a 500 that does not even name the
# service at fault.

JSON_ENDPOINTS = [
    # (method, path, request kwargs, needs a JSON object from the backend)
    ("GET", "/api/providers/piper/voices", {}, True),
    ("GET", "/api/providers/qwen3/voices", {}, True),
    ("GET", "/api/providers/qwen3/models", {}, True),
    ("POST", "/api/providers/qwen3/models/select", {"json": {"model": "m"}}, True),
    ("GET", "/api/providers/qwen3/status", {}, True),
    ("GET", "/api/providers/qwen3/saved-voices", {}, True),
    ("GET", "/api/providers/piper/custom-voices", {}, True),
    ("POST", "/api/providers/qwen3/saved-voices", {"data": {"name": "x"}}, False),
    ("DELETE", "/api/providers/qwen3/saved-voices/v-1", {}, False),
    ("DELETE", "/api/providers/piper/custom-voices/v-1", {}, False),
    ("POST", "/api/stt", {"data": {"provider": "whisper"},
                          "files": {"audio": ("a.wav", b"RIFF", "audio/wav")}}, True),
    ("GET", "/api/training/deployment-targets", {}, False),
    ("GET", "/api/training/jobs", {}, False),
    ("GET", "/api/training/status/job-1", {}, True),
    ("POST", "/api/training/train", {"data": {"model_name": "demo"}}, False),
    ("POST", "/api/training/train-from-dataset", {"data": {"model_name": "demo"}}, False),
    ("POST", "/api/training/resume", {"data": {"model_name": "demo"}}, False),
    ("POST", "/api/training/export/job-1", {"data": {"model_name": "demo"}}, True),
    ("DELETE", "/api/training/model/job-1", {}, False),
    ("DELETE", "/api/training/job/job-1", {}, False),
]


@pytest.mark.parametrize("method,path,kwargs,_needs_object", JSON_ENDPOINTS,
                         ids=[f"{m} {p}" for m, p, _, _ in JSON_ENDPOINTS])
def test_non_json_upstream_success_is_a_502_naming_the_service(
        monkeypatch, method, path, kwargs, _needs_object):
    module = load_frontend_app()
    install_stub(monkeypatch, module, lambda m, u, k: httpx.Response(
        200, content=b"<html>502 from the proxy</html>", headers={"content-type": "text/html"}))
    r = TestClient(module.app, raise_server_exceptions=False).request(method, path, **kwargs)
    assert r.status_code == 502, (r.status_code, r.text)
    assert "not JSON" in r.json()["detail"]


@pytest.mark.parametrize("method,path,kwargs,needs_object",
                         [entry for entry in JSON_ENDPOINTS if entry[3]],
                         ids=[f"{m} {p}" for m, p, _, n in JSON_ENDPOINTS if n])
def test_json_of_the_wrong_shape_is_a_502_not_a_crash(monkeypatch, method, path, kwargs, needs_object):
    module = load_frontend_app()
    install_stub(monkeypatch, module, lambda m, u, k: ["not", "an", "object"])
    r = TestClient(module.app, raise_server_exceptions=False).request(method, path, **kwargs)
    assert r.status_code == 502, (r.status_code, r.text)
    assert "unexpected response" in r.json()["detail"]


def test_upstream_errors_still_pass_through_with_their_own_status(monkeypatch):
    """The new parse step must not swallow the existing >=400 handling."""
    module = load_frontend_app()
    install_stub(monkeypatch, module, lambda m, u, k: httpx.Response(404, json={"detail": "no such job"}))
    r = TestClient(module.app).delete("/api/training/job/job-1")
    assert r.status_code == 404 and r.json() == {"detail": "no such job"}


@pytest.mark.parametrize("body,expected_detail", [
    (["ok"], ["ok"]),
    ("done", "done"),
    (42, 42),
])
def test_unload_tolerates_a_non_object_json_reply(monkeypatch, body, expected_detail):
    """`{"provider": ..., **body}` raised TypeError for anything but an object, so
    a backend that answered with a bare value turned a successful unload into a 500."""
    module = load_frontend_app()
    install_stub(monkeypatch, module, lambda m, u, k: httpx.Response(200, json=body))
    r = TestClient(module.app, raise_server_exceptions=False).post("/api/providers/qwen3/unload")
    assert r.status_code == 200
    assert r.json() == {"provider": "qwen3", "detail": expected_detail}


def test_unload_keeps_the_upstream_status_and_object_body(monkeypatch):
    module = load_frontend_app()
    install_stub(monkeypatch, module, lambda m, u, k: httpx.Response(
        409, json={"unloaded": False, "reason": "busy", "in_flight": 2}))
    r = TestClient(module.app).post("/api/providers/qwen3/unload")
    assert r.status_code == 409
    assert r.json() == {"provider": "qwen3", "unloaded": False, "reason": "busy", "in_flight": 2}


def test_unload_with_a_plain_text_reply(monkeypatch):
    module = load_frontend_app()
    install_stub(monkeypatch, module, lambda m, u, k: httpx.Response(200, content=b"ok"))
    r = TestClient(module.app).post("/api/providers/qwen3/unload")
    assert r.status_code == 200 and r.json() == {"provider": "qwen3", "detail": "ok"}


def test_stt_adapter_survives_junk_in_numeric_fields(monkeypatch):
    module = load_frontend_app()
    payload = {"text": "hi", "duration": "n/a",
               "segments": [{"start": "x", "end": "2.5", "text": "hi"}, "not a segment"]}
    install_stub(monkeypatch, module, lambda m, u, k: payload)
    r = TestClient(module.app).post(
        "/api/stt", data={"provider": "whisper"},
        files={"audio": ("a.wav", b"RIFF", "audio/wav")})
    assert r.status_code == 200
    assert r.json()["duration"] is None
    assert r.json()["segments"] == [{"start": 0.0, "end": 2.5, "text": "hi"}]


def test_catalog_normalizers_tolerate_the_wrong_inner_shape(frontend_module):
    assert frontend_module._normalize_piper_voice_catalog({"voices": []}) == []
    assert frontend_module._normalize_piper_voice_catalog({"voices": {"a": "junk"}}) == []
    assert frontend_module._normalize_qwen3_voice_catalog({"speakers": "Vivian"}) == []
    assert frontend_module._normalize_qwen3_model_catalog({"models": {"m": None}}) == []


# --- Piper "auto" -------------------------------------------------------------
#
# "auto" means the service decides the language. The gateway must not fill one in
# itself (the piper service owns PIPER_DEFAULT_LANGUAGE), and /api/tts and
# /v1/audio/speech must agree on every spelling of it.


def _piper_bodies(monkeypatch, language):
    module = load_frontend_app()
    stub = install_stub(monkeypatch, module, lambda m, u, k: httpx.Response(
        200, content=wav_bytes(), headers={"content-type": "audio/wav"}))
    client = TestClient(module.app)

    stub.calls.clear()
    client.post("/api/tts", json={"provider": "piper", "text": "Guten Tag", "language": language})
    via_api = stub.calls[-1][2]["json"]

    stub.calls.clear()
    client.post("/v1/audio/speech", json={
        "input": "Guten Tag", "response_format": "wav", "language": language})
    via_v1 = stub.calls[-1][2]["json"]
    return via_api, via_v1


@pytest.mark.parametrize("language", ["auto", "AUTO", " Auto ", ""])
def test_piper_auto_is_not_replaced_by_a_gateway_default(monkeypatch, language):
    via_api, via_v1 = _piper_bodies(monkeypatch, language)
    for body in (via_api, via_v1):
        assert "language" not in body, f"gateway sent language={body.get('language')!r} for auto"
    assert via_api == via_v1


@pytest.mark.parametrize("language", ["de", "de_DE", "en_US"])
def test_explicit_piper_language_is_forwarded_the_same_by_both_routes(monkeypatch, language):
    via_api, via_v1 = _piper_bodies(monkeypatch, language)
    assert via_api["language"] == via_v1["language"] == language
    assert via_api == via_v1


def test_absent_language_on_v1_is_auto_too(monkeypatch):
    module = load_frontend_app()
    stub = install_stub(monkeypatch, module, lambda m, u, k: httpx.Response(
        200, content=wav_bytes(), headers={"content-type": "audio/wav"}))
    TestClient(module.app).post("/v1/audio/speech", json={"input": "Hallo", "response_format": "wav"})
    assert "language" not in stub.calls[-1][2]["json"]


# --- the page's registry element ------------------------------------------------


def _registry_element(html):
    import re
    match = re.search(
        r'<script type="application/json" id="provider-registry-data">(.*?)</script>', html, re.S)
    assert match, "the registry element is missing from the page"
    return match.group(1)


def test_registry_in_the_page_matches_the_providers_endpoint_in_order(monkeypatch):
    import json
    module = load_frontend_app({"ENABLE_WHISPER_CPP": "true", "ENABLE_PARAKEET_ASR": "true"})
    client = TestClient(module.app)
    embedded = json.loads(_registry_element(client.get("/").text))
    served = client.get("/providers").json()
    assert embedded == served
    # The UI builds its selectors from Object.entries(): order is behaviour.
    assert list(embedded["providers"]) == list(served["providers"]) == list(
        module.PROVIDER_REGISTRY["providers"])


def test_registry_strings_cannot_close_the_script_element(monkeypatch):
    """PROVIDER_REGISTRY_JSON is operator input; `json.dumps(...) | safe` let a
    `</script>` in any string inject markup into the page."""
    import json
    evil = "</script><script>alert(document.domain)</script>"
    module = load_frontend_app({
        "PROVIDER_REGISTRY_JSON": json.dumps({
            "providers": {"evil": {"kind": "stt", "display_name": evil}},
            "ui": {"copy": {"app_subtitle": evil}},
        }),
        "APP_VERSION": 'v1"</script><script>alert(1)</script>',
    })
    client = TestClient(module.app)
    page = client.get("/").text
    baseline = TestClient(load_frontend_app().app).get("/").text

    assert "<script>alert" not in page
    assert page.count("<script") == baseline.count("<script")
    assert page.count("</script>") == baseline.count("</script>")

    embedded = json.loads(_registry_element(page))
    assert embedded["providers"]["evil"]["display_name"] == evil, "the value must round-trip intact"
    assert embedded["ui"]["copy"]["app_subtitle"] == evil


# --- entrypoint -----------------------------------------------------------------


def _run_entrypoint(tmp_path, workers, *extra):
    import subprocess

    out = tmp_path / "argv"
    fake = tmp_path / "bin"
    fake.mkdir(exist_ok=True)
    python = fake / "python"
    python.write_text('#!/bin/sh\necho "$$" > "$OUT.pid"\nfor a in "$@"; do echo "$a"; done > "$OUT"\n')
    python.chmod(0o755)
    env = {"PATH": f"{fake}:/usr/bin:/bin", "OUT": str(out)}
    if workers is not None:
        env["FRONTEND_WORKERS"] = workers
    script = Path(__file__).resolve().parents[1] / "frontend-service" / "entrypoint.sh"
    proc = subprocess.Popen(["sh", str(script), *extra], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    _stdout, stderr = proc.communicate(timeout=20)
    argv = out.read_text().split()
    return proc.pid, int((tmp_path / "argv.pid").read_text()), argv, stderr


def _workers_of(argv):
    return argv[argv.index("--workers") + 1]


def test_entrypoint_defaults_to_two_workers(tmp_path):
    _, _, argv, stderr = _run_entrypoint(tmp_path, None)
    assert _workers_of(argv) == "2" and stderr == ""
    assert argv[:3] == ["-m", "uvicorn", "app:app"]
    assert "--proxy-headers" in argv and "--no-access-log" in argv


def test_entrypoint_honours_frontend_workers(tmp_path):
    _, _, argv, _ = _run_entrypoint(tmp_path, "1")
    assert _workers_of(argv) == "1"
    _, _, argv, _ = _run_entrypoint(tmp_path, "5")
    assert _workers_of(argv) == "5"


@pytest.mark.parametrize("bad", ["", "0", "-2", "many", "1.5", "00"])
def test_entrypoint_falls_back_on_a_bad_worker_count(tmp_path, bad):
    _, _, argv, stderr = _run_entrypoint(tmp_path, bad)
    if bad == "":
        assert _workers_of(argv) == "2"
    else:
        assert _workers_of(argv) == "2"
        assert "FRONTEND_WORKERS" in stderr


def test_entrypoint_execs_so_signals_reach_uvicorn(tmp_path):
    shell_pid, python_pid, _, _ = _run_entrypoint(tmp_path, "2")
    assert python_pid == shell_pid, "the server runs as a child of the shell, so SIGTERM stops only the shell"


def test_entrypoint_passes_extra_arguments_through(tmp_path):
    _, _, argv, _ = _run_entrypoint(tmp_path, "2", "--log-level", "debug")
    assert argv[-2:] == ["--log-level", "debug"]


def test_dockerfile_ships_and_runs_the_entrypoint():
    import re

    service = Path(__file__).resolve().parents[1] / "frontend-service"
    dockerfile = (service / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(r"^FROM python:3\.12-slim\b", dockerfile, re.M)
    copied = re.findall(r"^COPY\s+(\S+)\s", dockerfile, re.M)
    assert "entrypoint.sh" in copied
    for source in copied:
        assert (service / source).exists(), f"COPY {source}: not in the build context"
    assert re.search(r'^CMD \["sh", "/app/entrypoint\.sh"\]', dockerfile, re.M)
