# Magpie-TTS Service

Text-to-speech with NVIDIA's **Magpie-TTS Multilingual** (`nvidia/magpie_tts_multilingual_357m`,
364M parameters) through NeMo. Port **5008**. Opt-in: nothing starts it by default, and the web UI only
shows it when `ENABLE_MAGPIE_TTS=true` is set on the frontend.

|  |  |
|---|---|
| Voices | Five built-in speakers: `Aria`, `Jason`, `John`, `Leo`, `Sofia` (index 0 to 4). **No voice cloning**: NVIDIA removed it from the released checkpoint. |
| Languages | `de`, `en`, `es`, `fr`, `ja`, `zh` with NeMo 3.0.x (see below). |
| Output | WAV, mono, 22.05 kHz. No streaming: audio arrives when the whole text is done. |
| License | [NVIDIA Open Model License](https://huggingface.co/nvidia/magpie_tts_multilingual_357m), a custom license, not MIT or Apache. Read it before shipping. |
| Cost | About **4.2 GB of GPU memory** while loaded (measured at the card on an RTX 4080: 1.75 GB of tensors plus the CUDA context and allocator cache); about 0.6 s of compute per second of audio. First start downloads **~2.5 GB**: the model, its audio codec and two helper models (`google/byt5-small`, `microsoft/wavlm-base-plus`). |

## Endpoints

| | |
|---|---|
| `POST /tts` | `{"text": "...", "language": "auto", "speaker": "Sofia"}` returns `audio/wav`. `language` is a code (`de`, `de-DE`) or a name (`German`); `auto` or blank is `MAGPIE_DEFAULT_LANGUAGE`. `speaker` is a name (any case) or an index; blank is `MAGPIE_DEFAULT_SPEAKER`. Response headers: `X-Sample-Rate`, `X-Language`, `X-Speaker`, `X-Chunk-Count`, `X-Generation-Time`. |
| `GET /speakers` | The speakers, in the `speaker-catalog-v1` shape the gateway turns into the voice list. |
| `GET /languages` | The languages this deployment can speak and the default. |
| `GET /health`, `GET /ready`, `GET /status` | Liveness, readiness (`loading` / `load_failed` as 503), details with GPU memory. |
| `POST /unload` | Free the VRAM now (409 while a request is running). The next request reloads. |

An unsupported language or an unknown speaker is `400` and names what is available. A full queue, or
a wait longer than `TTS_QUEUE_TIMEOUT_S`, is `503` with `Retry-After`.

## Things measured on the real model (nemo_toolkit 3.0.0, RTX 4080)

* **Numbers are only spoken with text normalization.** With it off, the model's tokenizer drops every
  digit: "Am 3. Mai 2026 kostet das Ticket 12,50 Euro um 14:30 Uhr" came back as "Am Mai kostet das
  Ticketeuro um Uhr". `MAGPIE_APPLY_TN` therefore defaults to **true**. The normalizer for a language is
  built on its first use, so the model load runs one short warm-up generation in the default language. Another
  language pays that cost on its first request: measured 12 s for German, 18 s English, 37 s French, 48 s
  Spanish, and every later request in it takes 1 to 4 s. `MAGPIE_WARM_LANGUAGES=en,fr` moves that wait to the load.
* **An unknown language is not an error in NeMo, it is English.** `do_tts` falls back to the English
  tokenizer for any language it has no route for, so Dutch would be read with English rules and answered 200.
  The service lists only the languages whose tokenizer NeMo really routes to and refuses the rest. The
  checkpoint also holds Italian, Vietnamese and Hindi tokenizers (and its newest release Arabic, Korean and
  Portuguese), but NeMo 3.0.0's language map cannot reach them; they need a newer NeMo.
* **Long text is generated sentence group by sentence group.** NeMo's own chunking of a 585-character
  German text left a hesitation ("Uh, jedes Jahr"), a repeated word ("dauern, dauern geht") and a stray
  trailing token at the points where it ended a chunk by force. Groups of whole sentences (at most
  `MAGPIE_MAX_GROUP_CHARS`, 200 by default) transcribed back word for word.
  Chinese and Japanese go through whole: they have no spaces to split on and NeMo has sentence rules for them.

## Settings

| Variable | Default | |
|---|---|---|
| `MAGPIE_MODEL` | `nvidia/magpie_tts_multilingual_357m` | A Hub id, or the path of a `.nemo` file on a mounted volume. |
| `MAGPIE_DEFAULT_LANGUAGE` | `de` | What `auto` means. Must be one of the supported languages or requests without a language are refused. |
| `MAGPIE_DEFAULT_SPEAKER` | `Sofia` | |
| `MAGPIE_SPEAKERS` | `Aria,Jason,John,Leo,Sofia` | Names in baked-embedding order, if a different checkpoint orders them differently. |
| `MAGPIE_WARM_LANGUAGES` | empty | More languages to warm at load, comma-separated (the default language is always warmed). `/ready` answers 503 `loading` meanwhile. |
| `MAGPIE_APPLY_TN` | `true` | Text normalization; see above. |
| `MAGPIE_USE_CFG` | `true` | Classifier-free guidance: better speech, roughly twice the compute. |
| `MAGPIE_MAX_GROUP_CHARS` | `200` | Longest group of sentences per generation call. |
| `MAGPIE_GROUP_GAP_MS` | `150` | Silence between groups. |
| `MAX_TEXT_CHARS` | `5000` | Longest request text (413 above it). |
| `TTS_MODEL_TTL` / `MODEL_TTL` | `300` | Seconds idle before the model is unloaded; `-1` keeps it resident, `0` unloads at once. |
| `TTS_MAX_CONCURRENCY`, `TTS_MAX_QUEUE`, `TTS_QUEUE_TIMEOUT_S` | `1`, `4`, `60` | Admission: one generation at a time, how many may wait, for how long. |
| `ALLOWED_ORIGINS`, `ALLOW_CREDENTIALS` | empty, `false` | CORS; the browser talks to the gateway, so leave empty. |

## Try it locally

The gateway and Magpie are enough to see it in the web UI; nothing else has to run (the other providers
then show as unavailable, which is expected). From the repository root, on a machine with an NVIDIA GPU
and the NVIDIA container runtime:

```bash
ENABLE_MAGPIE_TTS=true docker compose --env-file deploy/profiles/workstation-4080.env \
  -f docker-compose.yml --profile frontend --profile magpie-tts up -d --build
```

The first build takes 10 to 15 minutes (torch and NeMo) and the first start downloads about 2.5 GB. Open
http://localhost:3000, pick **Magpie TTS** under *TTS Engine*, open *Text-to-Speech*, choose a voice and press
*Generate Speech*. The first request in a language other than German adds a one-time wait (see above). To call
the service directly: `curl -s -H 'Content-Type: application/json' -d '{"text":"Hallo Welt.","speaker":"Leo"}' http://127.0.0.1:5008/tts -o out.wav`.

To make `/v1/audio/speech` use it as well, add `DEFAULT_TTS_PROVIDER=magpie`. Stop it with `docker compose down`
(add `-v` to delete the model cache).

## Image

`nemo_toolkit[tts]` on CUDA 12.8 with torch 2.11 (cu128 wheels, so Blackwell cards work). Python 3.11 comes
from the deadsnakes PPA because Ubuntu 22.04's own `python3.11` is a release candidate that NeMo 3 cannot
load a model on (the build checks for it). NVIDIA GPUs only: there is no ROCm or Vulkan variant.
