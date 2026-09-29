# Qwen3-TTS Service

Text-to-speech, voice cloning, and saved-voice playback using [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base). This service replaces the older XTTS path and supports both direct cloning and a persistent voice library.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/tts` | Generate speech with a built-in speaker (CustomVoice models only) |
| POST | `/clone` | Clone voice from a reference audio file |
| POST | `/clone-with-ref-text` | Clone with explicit transcript for higher quality |
| POST | `/voice_design` | Generate speech from a text-described voice |
| GET | `/voices` | List saved voice profiles |
| POST | `/voices/save` | Save a reusable voice profile |
| POST | `/voices/{voice_id}/tts` | Speak using a saved voice profile |
| DELETE | `/voices/{voice_id}` | Delete a saved voice profile |
| GET | `/models` | List available Qwen3-TTS model variants |
| POST | `/load_model` | Switch to a different model variant |
| POST | `/unload` | Release the model's VRAM now (`409` while a request is using it) |
| GET | `/speakers` | List the loaded model's built-in speakers |
| GET | `/status` | Detailed device and model information |
| GET | `/health` | Liveness. `200` even when the model is unloaded on purpose |
| GET | `/ready` | Can it serve a request now? `200` resident or reloadable, `503` while loading or after a failed load |

## Model Variants

| Model | Use Case | Approx. VRAM |
|-------|----------|--------------|
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (default) | Lower-VRAM TrueNAS / shared-GPU deployments, cloning | ~2.5 to 3 GB |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Best general TTS + cloning quality | ~4.5 to 5 GB |
| `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | Built-in speakers on a small card (no instruction control) | ~2.5 to 3 GB |
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Built-in speakers with instruction control | ~4.5 to 5 GB |
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | Text-guided voice design | ~4.5 to 5 GB |

## Usage

### Built-in speaker TTS

Needs a CustomVoice model (`POST /load_model {"model": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"}`).
`GET /speakers` lists what the loaded model offers.

```bash
curl -X POST "http://localhost:5004/tts" \
	-H "Content-Type: application/json" \
	-d '{"text":"Hallo Welt","lang":"German","speaker":"Ryan"}' \
	--output speech.wav
```

### Voice cloning

```bash
curl -X POST "http://localhost:5004/clone" \
	-F "text=Hallo Welt" \
	-F "lang=German" \
	-F "file=@reference.wav" \
	--output clone.wav
```

### Saving a voice

```bash
curl -X POST "http://localhost:5004/voices/save" \
	-F "name=My Voice" -F "file=@reference.wav"
```

`mode` picks how the voice is stored:

- `xvector` (default): speaker embedding only. Fast, works with any recording.
- `icl`: also stores the reference codes and transcript, so generation is
  conditioned on the actual clip. Usually clones more faithfully, but each
  request carries the longer prompt. Needs the transcript: pass `ref_text`, or
  the Qwen3-ASR service is asked for one (`502` if it cannot answer).

Without `ref_text` the Qwen3-ASR service also finds the best 3 to 10 second
segment to trim the upload to. In `xvector` mode that service is optional: if it
is unreachable the voice is saved from the untrimmed upload. If it answers with
no text at all, the request is rejected (`400`) as a recording without speech.

## Languages

`lang` takes the names Qwen3-TTS uses (`German`, `English`, `French`, `Spanish`,
`Italian`, `Portuguese`, `Russian`, `Japanese`, `Korean`, `Chinese`, any case) or
their ISO codes (`de`, `de-DE`, ...).

- `auto`, empty or omitted means `QWEN3_DEFAULT_LANGUAGE` (default **German**).
  Set it to `Auto` to let the model infer the language from the text instead.
- Anything else (`nl`, `pl`, ...) is a `400` listing the supported languages. It is
  not silently mapped to English.
- `X-Language` on the response is the language actually used.

## Limits

| Limit | Default | Answer |
|-------|---------|--------|
| Text (`text`, `ref_text`, `voice_description`, `instruct`) | 5000 characters (`MAX_TEXT_CHARS`) | `413` |
| Reference upload | 20 MB (`MAX_UPLOAD_MB`) | `413`, also from `Content-Length` before the body is read |
| Empty reference upload | | `400` |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `QWEN3_TTS_MODEL` | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | Initial model variant |
| `QWEN3_DEFAULT_LANGUAGE` | `German` | Language for `auto` / omitted `lang`. `Auto` = model inference. |
| `QWEN3_TTS_ATTN_IMPLEMENTATION` | `auto` | `auto` (FlashAttention-2 only if `flash_attn` imports on CUDA, else SDPA), `sdpa`, `eager` or `flash_attention_2`. The image ships no `flash-attn`, so `auto` means SDPA. |
| `MAX_TEXT_CHARS` | `5000` | Longest accepted text field |
| `MAX_UPLOAD_MB` | `20` | Largest accepted reference upload |
| `QWEN3_ASR_SERVICE_URL` | `http://qwen3-asr-service:5002` | ASR service used for transcribing and trimming reference audio |
| `VOICES_DIR` | `/app/voices` | Persistent directory for saved voice profiles |
| `TTS_MODEL_TTL` / `MODEL_TTL` | `300` | Seconds idle before the weights are released. `-1` pins them resident, `0` unloads as soon as idle (and skips the startup preload). A reload restores whichever model was last selected via `/load_model`, not the env default. |
| `TTS_MAX_CONCURRENCY` | `1` | Concurrent generations against the shared model |
| `TTS_MAX_BATCH` | `8` | Sentences generated in one forward pass when a long text is chunked. Peak VRAM scales with it, so it bounds the cost of a long request rather than letting the caller's text length decide. |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `ALLOW_CREDENTIALS` | `false` | Enables CORS credentials when origins are explicit |

## Model variants and capabilities

The variants share one class, so every generation method exists on all of
them and only raises when called on a variant that cannot do the work. The
service therefore routes on the declared capability table rather than probing
with `hasattr`, and answers with the model to switch to when the loaded variant
cannot serve a request:

| Endpoint | Requires | Served by | Wrong model |
|---|---|---|---|
| `/tts` | `custom_voice` | 1.7B / 0.6B CustomVoice | `409` |
| `/clone`, `/clone-with-ref-text`, `/voices/save`, `/voices/{id}/tts` | `voice_clone` | 1.7B Base, 0.6B Base | `400` |
| `/voice_design` | `voice_design` | VoiceDesign | `400` |

The default deployment model is a Base model, which has no preset speakers, so
`/tts` answers `409` there until a CustomVoice model is loaded; `/speakers` is
empty in the meantime. Use `/clone` or a saved voice on a Base model.

Switch with `POST /load_model`; it waits for requests already using the current
model to finish (`409` after three minutes). The choice survives an idle unload.
An unexpected failure is a `500`, never a `400`.

## Startup and readiness

The model loads in the background, so the port is open and `/health` answers
during a first-time download. `/ready` is `503 loading` until the model is in,
`200` after, and `503 load_failed` (with the error) if the last attempt failed;
the next request retries. It stays `200` for a model that was unloaded on
purpose. A request that arrives during the load waits for it rather than loading
a second copy.

## Requirements

- CUDA or ROCm GPU strongly recommended
- CPU mode works, but is not suitable for real-time use
- The service downloads model weights automatically on first start
- Saved voices are persisted in the mounted `/app/voices` volume
- The CUDA image pins torch/torchaudio (`--build-arg TORCH_VERSION=...`) and the
  build fails unless the installed torch has `sm_120` (Blackwell) kernels
