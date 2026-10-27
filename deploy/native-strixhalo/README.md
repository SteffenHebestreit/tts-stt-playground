# Native speech stack on Strix Halo (no Docker)

On a Strix Halo host that already serves LLMs natively through [llama-swap](https://github.com/mostlygeek/llama-swap), the speech services run the same way: as two more llama-swap models. They are reached with the same API keys and on the same base URL as the LLMs.

This is the API-only deployment. The web UI and voice training need the Docker stack (`deploy/profiles/strixhalo.env`).

Running on **evox2** (Ryzen AI Max+ 395, Radeon 8060S, Ubuntu 26.04) since 2026-10-17.

| Endpoint | Model id (aliases) | Engine |
|---|---|---|
| `POST /v1/audio/transcriptions` | `whisper-large-v3-turbo` (`whisper-1`, `whisper`, `stt`) | whisper.cpp `whisper-server`, Vulkan on the iGPU, Whisper large-v3-turbo **q8_0**, Silero VAD |
| `POST /v1/audio/speech` | `piper` (`tts-1`, `tts-1-hd`, `tts`) | Piper on the CPU behind [`piper_openai_shim.py`](piper_openai_shim.py) |

## Why native and why these engines

- **No Docker on the host.** The LLMs run natively under llama-swap, which supervises every process, so the speech services do too.
  - Same keys and the same TLS front door; nothing new is exposed.
  - Only `127.0.0.1` is bound; llama-swap is the only way in.
- **STT: whisper.cpp via Vulkan,** as in the Docker preset, for the same reason. CTranslate2's PyPI wheels are CUDA-only, and Vulkan needs no per-target kernels (gfx1151).
  - **q8_0 rather than the preset's q5_0:** on a German two-speaker dialogue, q8_0 had 1.7 % WER against 5.0 % for q5_0, at the same speed, about 0.8 s for 25 s of audio.
  - The memory difference, 0.9 GB against 0.6 GB, is irrelevant on 96 GB of GPU memory.
- **TTS: Piper on the CPU.** It costs the iGPU nothing, so the LLMs keep their memory bandwidth. A sentence takes about 0.3 s.

## The OpenAI surface

- **`/v1/audio/transcriptions`:** whisper-server's own OpenAI-compatible route.
  - Multipart fields: `file` (wav, mp3 or flac, any rate; no ffmpeg needed), `language`, `prompt`, `temperature`, `response_format` (`json`, `text`, `srt`, `vtt`, `verbose_json`).
  - `verbose_json` returns segments with `avg_logprob`, `no_speech_prob` and **per-segment `words`** with timestamps.
  - `language` comes back as a full name (`german`).
- **`/v1/audio/speech`:** the shim. JSON fields `input` (≤ 4096 characters), `voice`, `response_format`, `speed` (0.25–4.0).
  - **`voice`:** an installed voice id (`de_DE-thorsten-high`), a language or locale (`de`, `en_US`), or an OpenAI voice name (`alloy`, …), which selects the default voice.
  - **`response_format`:** `mp3` (the default, via lameenc), `wav`, or `pcm` (raw 16-bit at 24 kHz, via soxr).
  - **Refused with `code: unsupported_value`:** `opus`, `aac` and `flac`, because there is no encoder on the host. Also refused: `instructions`, and `stream_format=sse`.

## Install

```bash
bash deploy/native-strixhalo/install.sh        # no sudo; idempotent
```

Then add [`llama-swap.speech.yaml`](llama-swap.speech.yaml) to `~/.config/llama-swap/config.yaml`: under `models:`, plus both names in a group and in `hooks.on_startup.preload`.

- **Back up the config first.** With `-watch-config`, llama-swap reloads within ~2 s and **restarts every model** in about 30 s, LLMs included.
- **Preload probe:** llama-swap's preload probes `GET /`. The shim answers it, and so does whisper-server, with its HTML page.

## Check

```bash
KEY=...   # an llama-swap API key
curl -s https://<host>/v1/audio/speech -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
     -d '{"model":"tts-1","input":"Guten Tag.","response_format":"wav"}' -o hallo.wav
curl -s https://<host>/v1/audio/transcriptions -H "Authorization: Bearer $KEY" \
     -F file=@hallo.wav -F model=whisper-1 -F language=de -F response_format=json
```

## Who uses it

LUNA-Cloud's transcription worker sends speech recognition (`whisper-1`) and its summaries (`qwen3.6-35b-a3b`) to evox2, with its own key (`HALO_KEY_LUNA`). See the luna-cloud repository, `deploy/k8s/README.md`, "Transcription".
