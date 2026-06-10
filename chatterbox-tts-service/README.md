# Chatterbox-TTS Service

Multilingual text-to-speech and zero-shot voice cloning built on [Resemble AI Chatterbox Multilingual](https://github.com/resemble-ai/chatterbox) (MIT license) — 23 languages **including German**, with built-in PerTh audio watermarking.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/tts` | JSON `{text, language, exaggeration?, cfg_weight?}` → WAV (default voice) |
| POST | `/tts-stream` | Same body → chunked WAV stream; sentences are generated and streamed one by one, so first audio arrives after ~1.5 s instead of after the whole text |
| POST | `/clone` | Form `text`, `lang`, `file` (reference clip) → WAV in the cloned voice |
| POST | `/clone-with-ref-text` | Contract alias of `/clone`; `ref_text` is ignored (not needed by Chatterbox) |
| GET | `/languages` | Supported language ids + default |
| GET | `/status` | Device, GPU memory, model state |
| GET | `/health` | Health check |

`language` accepts ISO codes (`de`, `en`, …) or English names (`German`); `auto` falls back to `CHATTERBOX_DEFAULT_LANGUAGE`.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CHATTERBOX_DEFAULT_LANGUAGE` | `de` | Language used when the request says `auto` |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `ALLOW_CREDENTIALS` | `false` | Enables CORS credentials when origins are explicit |

## Running

Opt-in profile:

```bash
ENABLE_CHATTERBOX_TTS=true docker compose --profile chatterbox-tts --profile frontend up -d
```

Set `ENABLE_CHATTERBOX_TTS=true` on the frontend service so the provider appears as a selectable TTS engine in the browser UI. Model weights (~3 GB) download from Hugging Face on first start into the `chatterbox-tts-cache` volume.
