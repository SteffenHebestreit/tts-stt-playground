# Chatterbox-TTS Service

Multilingual text-to-speech and zero-shot voice cloning built on [Resemble AI Chatterbox Multilingual](https://github.com/resemble-ai/chatterbox) (MIT license) — 23 languages **including German**, with built-in PerTh audio watermarking.

Runs the **v3** multilingual checkpoint by default (`CHATTERBOX_T3_MODEL`); see [Checkpoint and package version](#checkpoint-and-package-version).

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/tts` | JSON `{text, language, exaggeration?, cfg_weight?}` → WAV (default voice). Long text is generated sentence by sentence and joined |
| POST | `/tts-stream` | Same body → chunked WAV stream; sentences are generated and streamed one by one, so first audio arrives after ~1.5 s instead of after the whole text |
| POST | `/clone` | Form `text`, `lang`, `file` (reference clip), `exaggeration?`, `cfg_weight?` → WAV in the cloned voice |
| POST | `/clone-with-ref-text` | Contract alias of `/clone`; `ref_text` is ignored (not needed by Chatterbox) |
| GET | `/languages` | Supported language ids + default |
| GET | `/status` | Device, GPU memory, model state, active checkpoint, package/ref, effective repetition penalty |
| GET | `/health` | Liveness. Always 200 while the process is up, also while the model downloads or is unloaded (`model_resident`, `t3_model`) |
| GET | `/ready` | Readiness. 200 once the weights have loaded at least once (a later idle unload does not change that); 503 `loading` during the first load, 503 `load_failed` (with the error) if it failed and never succeeded |
| POST | `/unload` | Free the model and its VRAM now; 409 while a request (or a generation whose client already left) is still running |

`language` accepts ISO codes (`de`, `en`, …) or English names (`German`); `auto` falls back to `CHATTERBOX_DEFAULT_LANGUAGE`.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CHATTERBOX_DEFAULT_LANGUAGE` | `de` | Language used when the request says `auto` |
| `CHATTERBOX_T3_MODEL` | `v3` | T3 checkpoint: `v3`, `v2`, `default` (the library default = v2) or a `.safetensors` filename. If the installed package cannot select a checkpoint, `v3` logs a WARNING and falls back to the default; `/health` `t3_model` says what really loaded |
| `CHATTERBOX_REPETITION_PENALTY` | unset | Overrides `generate()`'s repetition penalty (must be >= 1). Unset = the library's own default: 2.0 in chatterbox-tts 0.1.7, 1.2 on GitHub master |
| `CHATTERBOX_HF_REVISION` | `main` | Hugging Face revision (branch, tag or commit) of the `ResembleAI/chatterbox` weights. Applied by wrapping the library's download call; startup fails loudly if a future library version no longer allows that |
| `CHATTERBOX_CHUNK_GAP_MS` | `150` | Silence inserted between the separately generated chunks of one request (`/tts`, `/clone`, `/tts-stream`); `0` disables |
| `CHATTERBOX_REF_MAX_SECONDS` | `30` | Only this much of a reference clip is decoded for `/clone` (the voice is conditioned on the first 6–10 s) |
| `MAX_TEXT_CHARS` | `5000` | Longest accepted `text`; longer requests get HTTP 413 |
| `MAX_UPLOAD_MB` | `20` | Largest accepted reference clip; larger uploads get HTTP 413 (declared sizes are refused before the body is read) |
| `TTS_MODEL_TTL` / `MODEL_TTL` | `300` | Seconds idle before the model is unloaded; `0` = right after each request, `-1` = never |
| `TTS_MAX_CONCURRENCY` | `1` | Generation slots. Calls into the model are serialised regardless: the library keeps the voice and decoder state on the shared model object |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `ALLOW_CREDENTIALS` | `false` | Enables CORS credentials when origins are explicit |

Build arguments (see below): `TORCH_VERSION`, `TORCHAUDIO_VERSION`, `CHATTERBOX_REPO`, `CHATTERBOX_REF`.

## Behaviour worth knowing

- **Long text is complete.** One `generate()` call in the library silently stops after 1000 speech tokens (~40 s). The text is split into sentence-sized chunks (German-aware: `3. Mai`, `z. B.`, `Dr.`, `Nr.`, `ca.`, `3,5` are not sentence ends), each chunk is generated, and the audio is joined with `CHATTERBOX_CHUNK_GAP_MS` of silence.
- **Voices do not bleed.** The library stores the active voice on the model, and `audio_prompt_path` overwrites it. Every default-voice generation starts from a private copy of the built-in voice taken at load time; a clone lives for its own request only. A checkpoint that ships no built-in voice answers `/tts` with an error that points to `/clone`.
- **Disconnects cannot leak the model.** A client that disconnects (before or during a stream) never leaves the model pinned, and a generation that is still running keeps its generation slot and its model reference until the worker thread has really finished.
- **PerTh watermarking is always on.** Every generated file carries Resemble's imperceptible PerTh watermark, self-hosted included. It is embedded by the library and cannot be switched off here; v3 does not change that.

## Checkpoint and package version

v3 (`t3_mtl23ls_v3.safetensors` in the same Hugging Face repo; opt-in in the library since 2026-05-01, its recommended multilingual model since 2026-06-10) is selected with `ChatterboxMultilingualTTS.from_pretrained(device, t3_model="v3")`. **That argument does not exist in the PyPI release** (`chatterbox-tts` 0.1.7, 2026-03-26): only GitHub `master` has it. German is part of the general v3 model; the six *Single Language Pack* finetunes (zh, es-mx, es-es, pt-br, pt-pt, hi) are separate repositories and not used here.

The Dockerfile therefore installs chatterbox from GitHub, pinned to a commit (`CHATTERBOX_REF`, currently `5de7a54aa4e5e2baadb0182dde554908b48b85c2`, master of 2026-07-21). To move to a newer master:

```bash
git ls-remote https://github.com/resemble-ai/chatterbox.git HEAD
docker compose build --build-arg CHATTERBOX_REF=<sha> chatterbox-tts-service
```

Master also lowers `repetition_penalty` from 2.0 to 1.2 and drops the last speech token's audio (a ~40 ms noise burst); `/status` shows the value in effect.

### Why the dependencies are installed by hand

`chatterbox-tts` hard-pins `torch==2.6.0`, `torchaudio==2.6.0`, `gradio==6.8.0` and `transformers==5.2.0`. Installing it normally downgrades the CUDA 12.8 torch of the image to PyPI's 2.6.0 build, which has **no `sm_120` kernels** (RTX 50 series / Blackwell) and fails on the card with "no kernel image is available". The image therefore

1. installs `torch`/`torchaudio` from the `cu128` index (`TORCH_VERSION`, `TORCHAUDIO_VERSION`, default 2.8.0),
2. installs `requirements.txt` (chatterbox's runtime dependencies without torch, torchaudio and gradio) under a constraint that keeps that torch,
3. installs chatterbox itself with `--no-deps`,
4. fails the build if `torch.version.cuda` is not 12.8, `sm_120` is missing from `torch.cuda.get_arch_list()`, the watermarker cannot import, or the chosen ref has no `t3_model` support.

Two more pins come from the dependency set: `setuptools<82` (resemble-perth imports `pkg_resources`, removed in setuptools 82; a missing one shows up as `'NoneType' object is not callable` at the first model load) and `numpy<2`.

**Not verified in CI:** the image is not built by any test here. torch 2.8.0 is used with a library that upstream tests with 2.6.0; watch the first start on a new card.

## Running

Opt-in profile:

```bash
ENABLE_CHATTERBOX_TTS=true docker compose --profile chatterbox-tts --profile frontend up -d
```

Set `ENABLE_CHATTERBOX_TTS=true` on the frontend service so the provider appears as a selectable TTS engine in the browser UI. Model weights (~3 GB) download from Hugging Face on first start into the `chatterbox-tts-cache` volume. The model loads in the background: the container answers `/health` at once, and `/ready` reports `loading` until the weights are in.
