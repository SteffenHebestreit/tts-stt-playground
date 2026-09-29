# PiperTTS Service

Text-to-speech synthesis using [Piper TTS](https://github.com/rhasspy/piper). The service exposes the Piper voices installed in its models directory plus custom ONNX models exported by the training service.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/tts` | Generate speech with automatic or explicit voice selection |
| POST | `/synthesize` | Generate speech with a specific custom voice |
| GET | `/voices` | List installed and custom voices |
| GET | `/voice/{voice_name}` | Get metadata for a specific voice |
| POST | `/upload_model` | Upload a custom ONNX model and optional config |
| DELETE | `/voice/{voice_name}` | Delete a custom voice |
| POST | `/refresh_voices` | Re-scan the default and custom model directories |
| POST | `/analyze_audio` | Inspect an uploaded audio file via ffprobe and librosa |
| GET | `/health` | Liveness: the process is up |
| GET | `/ready` | Readiness: `503` with the reasons while no voice model is installed, the `piper` binary is missing or the output directory is not writable; misconfigured defaults are listed as `warnings` |

## Language and voice selection

A `/tts` request that names a `voice` gets that voice. Otherwise the service picks one:

1. **Language.** An explicit `language` (`de`, `de_DE`, `de-DE`, `en_GB`, ...) is used as given; a region (`en_GB`) prefers a voice of that region. If `language` is omitted, empty or `auto`, the service decides: it guesses German or English from the text (umlauts, eszett and a short stop-word list; only when the evidence is clear and a voice for that language is installed), otherwise it uses `PIPER_DEFAULT_LANGUAGE` (default `de`). The UI and the gateway omit `language` for "Auto-Detect", so this is the path they take.
2. **Voice.** Among the installed voices of that language: `PIPER_DEFAULT_VOICE` if it serves the language and the request did not ask for a different `quality` or `gender`; otherwise the first voice with the requested `quality` (default `medium`), preferring the requested `gender` (`any` means no preference).

If no installed voice serves the language, the service substitutes an English voice (or the default language's) and says so: `X-Language-Fallback: true`, with `X-Language` naming the voice's language and `X-Language-Requested` what was asked (`auto` when nothing was). The fallback is judged against the language the service was trying to serve, so a missing default language is reported too. Set `PIPER_STRICT_LANGUAGE=true` to answer `400` with the available languages instead.

## Installed voices

The installed voices are the `<id>.onnx` + `<id>.onnx.json` pairs in `models/default/`, read when the service starts, on `POST /refresh_voices`, and whenever the directory changes (a pair copied in while the service runs is picked up by the next request; editing a file in place needs `/refresh_voices`). Language, speaker, quality and sample rate come from the voice's JSON, then from its file name (`de_DE-thorsten-medium`).

To add a voice, copy both files from [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) into the models volume. The image ships the voices listed in the `PIPER_VOICES` build argument of the Dockerfile (a bind-mounted models volume hides them, so an empty volume needs the files copied in). While the directory holds no voice at all, `/voices` lists the built-in catalog (`"catalog_only": true`) and `/ready` answers `503`.

## Usage

### Automatic voice selection

```bash
curl -X POST "http://localhost:5000/tts" \
  -H "Content-Type: application/json" \
  -d '{"text":"Guten Tag, das ist ein Test."}' \
  --output speech.wav

curl -X POST "http://localhost:5000/tts" \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world","language":"en_US","quality":"medium"}' \
  --output speech.wav
```

### Specific custom voice

```bash
curl -X POST "http://localhost:5000/synthesize" \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world","voice_name":"my_voice"}' \
  --output speech.wav
```

### Upload a trained model

```bash
curl -X POST "http://localhost:5000/upload_model" \
  -F "model_file=@my_voice.onnx" \
  -F "config_file=@my_voice.json" \
  -F "model_name=my_voice"
```

Custom voice names are restricted to letters, numbers, `_`, and `-` (at most 100 characters) to avoid path traversal and inconsistent model paths. An upload is written to a private staging directory and moved into place only when the whole upload checked out, so a failed re-upload leaves the existing voice untouched.

An uploaded model is code that runs in this process (ONNX Runtime), so before it is published it is checked:

- **Name.** The id of a built-in voice (`de_DE-thorsten-medium`, in any letter case) is refused with `409`: an upload can no longer replace a built-in voice. A built-in voice also wins over a custom voice that was uploaded under its id before this check existed; that one is no longer offered for `/tts` (it still answers `/synthesize` and can be deleted).
- **Model.** In a separate, killable child process: it must load under ONNX Runtime, take exactly the inputs `text` and `text_lengths` (what the training service exports) and answer one test synthesis within `PIPER_ONNX_VALIDATE_TIMEOUT_S`. Otherwise `400` (`503` if the runtime is missing); the child is killed on timeout.
- **Config.** `phoneme_id_map` is required; `audio.sample_rate` must be an integer from 8000 to 96000; `model_card`, `limits` and `audio` must be objects with short strings; `phonemizer_language` is a plain language code. Otherwise `400`.
- **Quotas.** At most `PIPER_MAX_CUSTOM_VOICES` custom voices (`409`) using at most `PIPER_MAX_CUSTOM_MB` in total (`413`); a re-upload counts against what is left without the old files. One upload is worked on at a time, a second gets `503` with `Retry-After`.

A synthesis with a custom voice that overruns `PIPER_TIMEOUT_S` is told to stop (ONNX Runtime's `terminate` flag), so its slot comes back.

Errors that are not the caller's fault (`500`) answer with a generic message and a request id (also in the `X-Request-ID` header); the detail (the `piper` process's stderr, exception text, paths) is in the service log under the same id.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PIPER_DATA_DIR` | `/app/models` | Directory containing default and custom models |
| `PIPER_OUTPUT_DIR` | `/app/output` | Directory for generated files |
| `PIPER_DEFAULT_LANGUAGE` | `de` | Language for requests that name none (omitted, empty or `auto`) and the text gives no clear hint |
| `PIPER_DEFAULT_VOICE` | unset | Voice id preferred for its language when the request names no voice, quality or gender. Ignored (and listed in `/ready` warnings) if not installed |
| `PIPER_AUTO_DETECT` | `true` | Guess German/English from the text for an omitted or `auto` language; `false` always uses `PIPER_DEFAULT_LANGUAGE` |
| `PIPER_STRICT_LANGUAGE` | `false` | `400` instead of substituting a voice of another language |
| `PIPER_TIMEOUT_S` | `60` | Longest a synthesis may run (`504`, and the `piper` process is killed). Also the longest a request waits for a free slot (`503` with `Retry-After`) |
| `PIPER_MAX_CONCURRENCY` | half the CPU cores, at least 1 | Syntheses (piper processes, ONNX inferences, audio analyses) running at once |
| `MAX_TEXT_CHARS` | `20000` | Longest `text` (`422`, the message names this variable); request bodies far beyond it get `413` |
| `MAX_UPLOAD_MB` | `100` | Largest model file accepted by `/upload_model` (`413`); configs are capped at 5 MB |
| `PIPER_MAX_CUSTOM_VOICES` | `20` | Custom voices that may exist (`409` beyond it; `0` refuses every upload) |
| `PIPER_MAX_CUSTOM_MB` | `2048` | Disk all custom voices may use together (`413` beyond it) |
| `PIPER_ONNX_VALIDATE_TIMEOUT_S` | `30` | Longest an uploaded model gets to load and answer a test synthesis before it is refused |
| `MAX_ANALYZE_UPLOAD_MB` | `25` | Largest file accepted by `/analyze_audio` (`413`) |
| `PIPER_ANALYZE_MAX_SECONDS` | `600` | Longest audio `/analyze_audio` analyses (`413`). The length is read from the header first, and the decode itself stops at the limit, so a small, highly compressible file cannot expand into hours of samples |
| `ONNX_NUM_THREADS` | `2` | Threads per custom-voice ONNX session |
| `ONNX_SESSION_CACHE_SIZE` | `4` | Custom-voice ONNX sessions kept loaded |
| `OUTPUT_RETENTION_HOURS` | `24` | Age after which leftover generated files are pruned |
| `OUTPUT_PRUNE_INTERVAL_S` | `3600` | How often the prune runs |
| `ALLOWED_ORIGINS` | *(empty)* | Comma-separated origins a browser page may call this service from. **Unset or empty: no CORS headers at all.** `*` opens it to every page and must be written down (it logs a warning). Independently of this, a state-changing request (anything but `GET`/`HEAD`/`OPTIONS`) that carries an `Origin` header for another host than the one it was sent to is refused with `403` unless the origin is listed; requests without an `Origin` (the gateway, curl) are not affected |
| `ALLOW_CREDENTIALS` | `false` | Enables CORS credentials when origins are explicit |

Built-in voice models live under `models/default/`; custom voices are stored under `models/custom/{voice_name}/`.

Size limits are enforced from `Content-Length` before the body is read, and per file while it is copied to disk; a client that sends no `Content-Length` is still bounded per file. The service has no authentication: keep `/upload_model`, `/analyze_audio` and `DELETE /voice/{name}` reachable only from the gateway network. The checks above make an upload harder to abuse, not safe to expose: a model is still executed by ONNX Runtime.

`/ready` names the setting to look at (`PIPER_DATA_DIR`, `PIPER_OUTPUT_DIR`), never a resolved path.

## Notes

- The Piper engine is the `rhasspy/piper` v1.2.0 binary. That repository is archived; its successor is [OHF-Voice/piper1-gpl](https://github.com/OHF-Voice/piper1-gpl) (a Python package with a different command line). Migrating is a follow-up, not part of this service yet.
- The image is built on Python 3.12; `requirements.txt` is resolved and the tests run against exactly those pins.
- Inference for custom voices runs in a worker thread, which cannot be killed: on `PIPER_TIMEOUT_S` the request is answered `504` at once and the run is told to stop through its `RunOptions.terminate` flag (checked between operators, loop iterations included); the concurrency slot stays taken until the thread really ends, which is soon after. A single operator that runs for minutes on its own would still hold its slot that long.
- Tests: `tests/test_piper_tts_*.py` (`test_piper_tts_inference.py` and `test_piper_tts_upload_real.py` need numpy, onnx, onnxruntime, librosa and soundfile and skip without them).
