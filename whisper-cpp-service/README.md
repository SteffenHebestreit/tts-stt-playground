# whisper-cpp Service

Lightweight speech-to-text service based on `whisper.cpp`. This backend is useful when you want a smaller C++ stack, GGML/GGUF model files, or Vulkan acceleration instead of the Python `faster-whisper` service.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/audio/transcriptions` | OpenAI-compatible transcription endpoint |
| POST | `/inference` | Native whisper.cpp inference endpoint |
| GET | `/` | Basic HTTP response used by the health check |

## Model Handling

- The container downloads `ggml-${WHISPER_MODEL}.bin` into `/models` on first start
- Models are persisted in the `whisper-cpp-models` volume
- Supported model names include `tiny`, `base`, `small`, `medium`, `large-v1`, `large-v2`, `large-v3`, and `large-v3-turbo`

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | `large-v3` | GGML model name to download and serve |
| `EXTRA_ARGS` | empty | Extra `whisper-server` CLI flags |
| `GGML_VULKAN_DEVICE` | unset | Vulkan device index when using the Vulkan overlay |

## Deployment Modes

### CPU

```bash
docker compose --profile whisper-cpp up -d
```

### Vulkan

```bash
docker compose -f docker-compose.yml -f docker-compose.vulkan.yml --profile whisper-cpp up -d
```

Use the Vulkan overlay only on native Linux or WSL2 with Docker Engine running inside WSL. Docker Desktop does not expose `/dev/dri`, so the Vulkan build will not work there.