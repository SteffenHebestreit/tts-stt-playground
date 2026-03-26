"""Contract tests for provider registry metadata and shared provider interfaces."""

import os

import httpx
import pytest


def _assert_stt_form_contract(data):
    """Validate the normalized STT response contract used by the UI."""
    assert isinstance(data.get("text"), str)

    segments = data.get("segments")
    if segments is None:
        return

    assert isinstance(segments, list)
    for segment in segments:
        assert "text" in segment
        assert "start" in segment
        assert "end" in segment


def test_frontend_provider_registry(frontend_client):
    """Frontend should publish provider metadata and contract names."""
    response = frontend_client.get("/providers")
    assert response.status_code == 200

    registry = response.json()
    providers = registry.get("providers", {})
    ui = registry.get("ui", {})

    assert "piper" in providers
    assert "whisper" in providers
    assert "qwen3-asr" in providers
    assert "whisper-cpp" in providers
    assert ui.get("default_tts_provider") in providers
    assert ui.get("default_stt_provider") in providers

    assert providers["whisper"]["contracts"]["transcribe"] == "stt-form-v1"
    assert providers["qwen3-asr"]["contracts"]["transcribe"] == "stt-form-v1"
    assert providers["whisper-cpp"]["contracts"]["transcribe"] == "openai-audio-transcriptions-v1"
    assert providers["piper"]["contracts"]["tts"] == "simple-json-tts-v1"
    assert providers["qwen3"]["contracts"]["tts"] == "simple-json-tts-v1"
    assert providers["qwen3"]["contracts"]["model_catalog"] == "model-catalog-v1"
    assert providers["qwen3"]["contracts"]["voice_clone"] == "voice-clone-tts-v1"
    assert providers["piper-training"]["contracts"]["training"] == "voice-training-job-v1"
    assert providers["piper"]["settings"]["defaults"]["quality"] == "medium"
    assert providers["qwen3"]["settings"]["defaults"]["speaker"] == "Vivian"
    assert providers["piper-training"]["settings"]["defaults"]["batch_size"] == "32"
    assert ui.get("copy", {}).get("stt_tab_label") == "Speech-to-Text"
    assert providers["qwen3"]["ui"]["sections"]["cloning"]["modes"]["design"]["button_label"] == "Design & Generate"
    assert providers["qwen3"]["ui"]["forms"]["builtin_tts"]["fields"]["instruction"]["label"] == "Voice Instruction (Optional):"
    assert providers["qwen3"]["ui"]["messages"]["voice_clone"]["progress_auto_transcribe"] == "Auto-transcribing + cloning... {elapsed}s"
    assert providers["qwen3"]["ui"]["messages"]["saved_voice_library"]["delete_confirm"] == "Delete saved voice \"{voice_name}\"?"
    assert providers["piper"]["ui"]["messages"]["custom_voice_library"]["action_delete"] == "Delete"
    assert providers["piper"]["ui"]["messages"]["tts_generation"]["voice_auto_option"] == "Auto-Select Best Voice"
    assert providers["whisper"]["ui"]["messages"]["transcription"]["copy_success"] == "Transcription copied to clipboard"
    assert providers["whisper-cpp"]["ui"]["messages"]["transcription"]["result_heading"] == "Transcription Result"
    assert providers["piper-training"]["ui"]["forms"]["start_training"]["fields"]["files"]["hint"] == "Recommended: 10+ minutes of high-quality speech data"
    assert providers["piper-training"]["ui"]["messages"]["start_training"]["progress"] == "Progress: {progress}% (Epoch {current_epoch}/{total_epochs})"
    assert providers["piper-training"]["ui"]["messages"]["model_list"]["empty"] == "No trained models found. Start training to create your first model!"


def test_frontend_normalizes_piper_voice_catalog(frontend_client):
    """Frontend should expose a normalized Piper voice catalog."""
    response = frontend_client.get("/api/providers/piper/voices")
    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "piper"
    assert isinstance(data.get("voices"), list)
    if data["voices"]:
        voice = data["voices"][0]
        assert "id" in voice
        assert "name" in voice


def test_frontend_normalizes_qwen3_speaker_catalog(frontend_client):
    """Frontend should expose a normalized Qwen3 speaker catalog."""
    response = frontend_client.get("/api/providers/qwen3/voices")
    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "qwen3"
    assert isinstance(data.get("voices"), list)


def test_frontend_normalizes_qwen3_runtime_status(frontend_client):
    """Frontend should expose normalized runtime status fields for advanced TTS providers."""
    response = frontend_client.get("/api/providers/qwen3/status")
    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "qwen3"
    assert data["device_type"] in {"gpu", "cpu"}
    assert "model_loaded" in data
    assert "speakers" in data


def test_frontend_normalizes_saved_voice_library(frontend_client):
    """Frontend should expose normalized saved-voice fields when provider data is available."""
    response = frontend_client.get("/api/providers/qwen3/saved-voices")
    assert response.status_code == 200

    data = response.json()
    assert data["provider"] == "qwen3"
    assert isinstance(data.get("voices"), list)
    if data["voices"]:
        voice = data["voices"][0]
        assert "language" in voice
        assert "reference_preview" in voice
        assert "created_at_display" in voice


def test_frontend_normalizes_training_job_list(frontend_client):
    """Frontend should expose stable training job summary fields when jobs are present."""
    response = frontend_client.get("/api/training/jobs")
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
    if data:
        job = data[0]
        assert "voice_name" in job
        assert "deployment_target_label" in job
        assert "created_at_display" in job


def test_frontend_tts_rejects_unknown_provider(frontend_client):
    """Frontend TTS adapter should reject unknown providers cleanly."""
    response = frontend_client.post(
        "/api/tts",
        json={"provider": "unknown", "text": "Hello world"},
    )
    assert response.status_code == 404


def test_whisper_provider_matches_stt_form_contract(stt_client, test_audio_bytes):
    """faster-whisper should satisfy the shared STT form contract."""
    response = stt_client.post(
        "/transcribe",
        files={"audio": ("test.wav", test_audio_bytes, "audio/wav")},
    )
    assert response.status_code == 200
    _assert_stt_form_contract(response.json())


def test_qwen3_asr_matches_stt_form_contract(qwen3_asr_client, test_audio_bytes):
    """Qwen3-ASR should satisfy the same STT form contract used by the UI."""
    response = qwen3_asr_client.post(
        "/transcribe",
        files={"audio": ("test.wav", test_audio_bytes, "audio/wav")},
    )
    assert response.status_code == 200
    _assert_stt_form_contract(response.json())


def test_whisper_cpp_matches_openai_transcriptions_contract(test_audio_bytes):
    """Optional contract test for an OpenAI-compatible whisper.cpp deployment."""
    whisper_cpp_url = os.getenv("WHISPER_CPP_URL")
    if not whisper_cpp_url:
        pytest.skip("WHISPER_CPP_URL not configured")

    try:
        with httpx.Client(base_url=whisper_cpp_url, timeout=60.0) as client:
            response = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", test_audio_bytes, "audio/wav")},
                data={"response_format": "json"},
            )
    except httpx.HTTPError as exc:
        pytest.skip(f"whisper.cpp unavailable: {exc}")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data.get("text"), str)