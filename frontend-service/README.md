# Frontend Service

Browser-facing UI and API hub for the TTS-STT platform. Built with FastAPI, Jinja2, and static assets served from the container.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Main web application |
| GET | `/api-docs` | Static API documentation page |
| GET | `/health` | Service health plus configured backend URLs |
| GET | `/providers` | Provider registry with capabilities and contract metadata |
| GET | `/api/providers/{provider_id}/voices` | Normalized voice or speaker catalog for a TTS provider |
| GET | `/api/providers/{provider_id}/custom-voices` | Managed custom voice catalog for providers that support custom model deletion |
| GET | `/api/providers/{provider_id}/models` | Normalized model catalog for providers that support model switching |
| POST | `/api/providers/{provider_id}/models/select` | Switch the active provider model variant |
| GET | `/api/providers/{provider_id}/status` | Provider runtime status for advanced TTS backends |
| GET | `/api/providers/{provider_id}/saved-voices` | List saved provider voice profiles |
| POST | `/api/providers/{provider_id}/saved-voices` | Save a reusable provider voice profile |
| DELETE | `/api/providers/{provider_id}/custom-voices/{voice_id}` | Delete a managed custom provider voice |
| DELETE | `/api/providers/{provider_id}/saved-voices/{voice_id}` | Delete a saved provider voice profile |
| POST | `/api/providers/{provider_id}/saved-voices/{voice_id}/tts` | Generate speech with a saved provider voice |
| POST | `/api/providers/{provider_id}/voice-clone` | Run provider-scoped voice cloning via the frontend adapter |
| POST | `/api/providers/{provider_id}/voice-design` | Run provider-scoped voice design via the frontend adapter |
| POST | `/api/tts` | Normalized frontend TTS adapter for basic synthesis |
| POST | `/api/stt` | Normalized frontend STT adapter for basic transcription |
| GET | `/api/training/deployment-targets` | Training deployment target metadata through the frontend adapter |
| POST | `/api/training/train` | Start a training job through the frontend adapter |
| POST | `/api/training/train-from-dataset` | Start dataset-backed training through the frontend adapter |
| POST | `/api/training/resume` | Resume training through the frontend adapter |
| GET | `/api/training/jobs` | List training jobs through the frontend adapter |
| GET | `/api/training/status/{job_id}` | Get training status through the frontend adapter |
| POST | `/api/training/export/{job_id}` | Export and deploy through the frontend adapter |
| GET | `/api/training/download/{job_id}` | Download exported model through the frontend adapter |
| DELETE | `/api/training/model/{job_id}` | Delete a trained model through the frontend adapter |
| DELETE | `/api/training/job/{job_id}` | Cancel a job through the frontend adapter |

## Features

- Frontend entrypoint for STT, TTS, Qwen3 voice cloning, and training flows
- Injects a provider registry plus browser-facing service URLs into the HTML template
- Serves static assets with restart-based cache busting
- Exposes health data for all backend services used by the UI
- Exposes optional providers such as `whisper-cpp` only when explicitly enabled in the frontend environment
- Supports multiple STT request contracts, including OpenAI-compatible `/v1/audio/transcriptions`
- Exposes a normalized basic TTS adapter so multiple providers can share one browser-facing synthesis contract
- Exposes a normalized basic STT adapter so multiple providers can share one browser-facing transcription contract
- Exposes provider-scoped advanced TTS adapters for model switching, saved voices, voice cloning, and voice design
- Exposes provider-scoped custom voice management adapters so the browser does not call Piper directly for trained-voice operations
- Normalizes saved-voice library responses into stable preview and display-timestamp fields for the browser
- Uses provider `settings` metadata to drive language, quality, speaker, and training defaults in the UI
- Uses registry `ui.copy` and provider `ui.sections` metadata to drive tab labels, section copy, and provider-specific placeholders in the UI
- Uses provider `ui.forms` schemas for advanced feature controls such as training and voice cloning labels, hints, and action text
- Uses provider `ui.messages` schemas for runtime workflow messaging such as validation, progress, status, notifications, list empty states, management actions, and the basic TTS/STT submit flows
- Loads training deployment targets from the training service so export/deploy actions are target-aware rather than Piper-only
- Proxies training actions through `/api/training/*` so the browser no longer needs to call the training service directly
- Normalizes training job, training detail, and export deployment payloads so the browser consumes stable labels and summaries instead of raw training-service fields

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_SERVICE_URL` | `http://piper-tts-service:5000` | Internal PiperTTS URL |
| `STT_SERVICE_URL` | `http://stt-service:8000` | Internal STT URL |
| `VOICE_TRAINING_URL` | `http://piper-training-service:8080` | Internal training URL |
| `QWEN3_TTS_SERVICE_URL` | `http://qwen3-tts-service:5004` | Internal Qwen3-TTS URL |
| `QWEN3_ASR_SERVICE_URL` | `http://qwen3-asr-service:5002` | Internal Qwen3-ASR URL |
| `WHISPER_CPP_SERVICE_URL` | `http://whisper-cpp:8080` | Internal whisper.cpp URL |
| `ENABLE_WHISPER_CPP` | `false` | Include whisper.cpp in the provider registry, STT selector, and health polling |
| `BROWSER_TTS_URL` | `http://localhost:5000` | Browser-visible PiperTTS URL |
| `BROWSER_STT_URL` | `http://localhost:5001` | Browser-visible STT URL |
| `BROWSER_TRAINING_URL` | `http://localhost:8080` | Browser-visible training URL |
| `BROWSER_QWEN3_TTS_URL` | `http://localhost:5004` | Browser-visible Qwen3-TTS URL |
| `BROWSER_QWEN3_ASR_URL` | `http://localhost:5002` | Browser-visible Qwen3-ASR URL |
| `BROWSER_WHISPER_CPP_URL` | `http://localhost:5003` | Browser-visible whisper.cpp URL |
| `DEFAULT_TTS_PROVIDER` | `piper` | Default TTS provider ID selected by the UI |
| `DEFAULT_STT_PROVIDER` | `whisper` | Default STT provider ID selected by the UI |
| `TRAINING_PROVIDER` | `piper-training` | Provider ID used for training workflows |
| `PROVIDER_REGISTRY_JSON` | unset | Optional JSON override for provider metadata, per-service settings, and defaults |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `ALLOW_CREDENTIALS` | `false` | Enables CORS credentials when origins are explicit |

`whisper-cpp` remains optional even when its backend container is running. To surface it in the browser UI, set `ENABLE_WHISPER_CPP=true` for `frontend-service` and start the `whisper-cpp` compose profile.

Access the UI at `http://localhost:3000`.
