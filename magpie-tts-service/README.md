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
| `POST /tts` | `{"text": "...", "language": "auto", "speaker": "Sofia"}` returns `audio/wav`. `language` is a code (`de`, `de-DE`) or a name (`German`); `auto` or blank is `MAGPIE_DEFAULT_LANGUAGE`. `speaker` is a name (any case) or an index; blank is `MAGPIE_DEFAULT_SPEAKER`. Response headers: `X-Sample-Rate`, `X-Language`, `X-Speaker`, `X-Chunk-Count` (the generations joined; a group generated again in two halves counts twice), `X-Generation-Time`. |
| `GET /speakers` | The speakers, in the `speaker-catalog-v1` shape the gateway turns into the voice list. |
| `GET /languages` | The languages this deployment can speak and the default. |
| `GET /health`, `GET /ready`, `GET /status` | Liveness, readiness (`loading` / `load_failed` as 503), details with GPU memory. |
| `POST /unload` | Free the VRAM now (409 while a request is running). The next request reloads. |

An unsupported language or an unknown speaker is `400` and names what is available. A full queue, or
a wait longer than `TTS_QUEUE_TIMEOUT_S`, is `503` with `Retry-After`. The gateway passes both on with
their sentence (on `/v1` as `400` `invalid_request_error` and `503` `server_busy`).

`/tts` answers only once the whole text is done, so the gateway waits for it up to
max(600 s, 180 s + 0.1 s per character), or 0.25 s per character for a request in Chinese or Japanese
(`auto` counts as neither, even when `MAGPIE_DEFAULT_LANGUAGE` is `zh` or `ja`). When that runs out, or
its caller hangs up, the gateway closes its connection to Magpie, and Magpie stops after the group it is
generating; a request still waiting for its turn leaves the queue at once.

## Things measured on the real model (nemo_toolkit 3.0.0, RTX 4080)

* **Numbers are only spoken with text normalization.** With it off, the model's tokenizer drops every
  digit: "Am 3. Mai 2026 kostet das Ticket 12,50 Euro um 14:30 Uhr" came back as "Am Mai kostet das
  Ticketeuro um Uhr". `MAGPIE_APPLY_TN` therefore defaults to **true**. The normalizer for a language is
  built on its first use: measured 12 s for German, 18 s English, 37 s French, 48 s Spanish, and every later
  request in it takes 1 to 4 s. So the first model load runs one short warm-up generation in the default
  language (`/ready` answers 503 `loading` during that first load only), `MAGPIE_WARM_LANGUAGES=en,fr` adds
  more languages to it, and any other language pays the build on its first request. A normalizer, once built
  by the warm-up or by a request, stays in host RAM across idle unloads until the container stops (a few
  hundred MB per language, no VRAM), so a reload after `TTS_MODEL_TTL` (300 s by default, 120 s in
  `deploy/profiles/truenas-5060ti.env`) repeats neither the warm-up nor a normalizer build.
* **An unknown language is not an error in NeMo, it is English.** `do_tts` falls back to the English
  tokenizer for any language it has no route for, so Dutch would be read with English rules and answered 200.
  The service lists only the languages whose tokenizer NeMo really routes to and refuses the rest. The
  checkpoint also holds Italian, Vietnamese and Hindi tokenizers (and its newest release Arabic, Korean and
  Portuguese), but NeMo 3.0.0's language map cannot reach them; they need a newer NeMo.
* **Long text is generated sentence group by sentence group.** NeMo's own chunking of a 585-character
  German text left a hesitation ("Uh, jedes Jahr"), a repeated word ("dauern, dauern geht") and a stray
  trailing token at the points where it ended a chunk by force. Groups of whole sentences (at most
  `MAGPIE_MAX_GROUP_CHARS`, 200 by default) transcribed back word for word.
  Chinese and Japanese are grouped too, in groups of at most a third of that (66 characters by default,
  14 to 16 s of speech at the measured 4.2 to 4.6 characters per second). A sentence ends after 。！？…
  or `!` `?` (closing marks such as 」 stay with it), at a `.` only before a space or the end of the
  text (so `3.5` stays whole), and at a line break. A longer sentence is cut after its last clause mark
  (，、；： or `,;:`, but not inside `14:30` or `1,000`) that leaves at least a third of the limit before
  the cut, else at a space, and only else hard at the limit. NeMo 3.0.0 splits these languages itself
  only above 100 characters (Chinese) or 80 words (Japanese), and only at 。？！…, so a long sentence
  joined only by commas used to be one chunk, cut off by the decoder at about 23 s while the request
  still answered 200.
* **A group that NeMo cuts off is generated again.** NeMo stops decoding a call at 500 frames, 23.2 s
  of audio. A group that reaches that limit (number-dense text, which normalization makes much longer,
  can) is split in two and the halves are generated instead, with a warning in the log; if it cannot
  be split, or a half reaches the limit too, that audio is kept as it is, again with a warning, rather
  than failing the request.

## Settings

| Variable | Default | |
|---|---|---|
| `MAGPIE_MODEL` | `nvidia/magpie_tts_multilingual_357m` | A Hub id, or the path of a `.nemo` file on a mounted volume. |
| `MAGPIE_DEFAULT_LANGUAGE` | `de` | What `auto` means. Must be one of the supported languages or requests without a language are refused. |
| `MAGPIE_DEFAULT_SPEAKER` | `Sofia` | |
| `MAGPIE_SPEAKERS` | empty (= `Aria,Jason,John,Leo,Sofia`) | Names in baked-embedding order, for a `MAGPIE_MODEL` that orders or names them differently: comma- or line-separated (a YAML block scalar works). A control character is removed from a name, which keeps its position. |
| `MAGPIE_WARM_LANGUAGES` | empty | More languages to warm at the first load, comma-separated (the default language is always warmed). `/ready` answers 503 `loading` until that first load is done; a reload after an idle unload needs no warm-up (see above). |
| `MAGPIE_APPLY_TN` | `true` | Text normalization; see above. |
| `MAGPIE_USE_CFG` | `true` | Classifier-free guidance: better speech, roughly twice the compute. |
| `MAGPIE_MAX_GROUP_CHARS` | `200` | Longest group of sentences per generation call, 40 to 250 (a value outside that range falls back to 200 with a warning: above about 250 an English group crosses NeMo's 45-word threshold and NeMo splits it again itself). Chinese and Japanese groups are a third of it. |
| `MAGPIE_GROUP_GAP_MS` | `150` | Silence between groups. |
| `MAX_TEXT_CHARS` | `5000` | Longest request text (413 above it). |
| `TTS_MODEL_TTL` / `MODEL_TTL` | `300` | Seconds idle before the model is unloaded; `-1` keeps it resident, `0` unloads at once. |
| `TTS_MAX_CONCURRENCY`, `TTS_MAX_QUEUE`, `TTS_QUEUE_TIMEOUT_S` | `1`, `4`, `60` | Admission: one generation at a time, how many may wait, for how long. |
| `ALLOWED_ORIGINS`, `ALLOW_CREDENTIALS` | empty, `false` | CORS; the browser talks to the gateway, so leave empty. |

## Try it locally

The gateway and Magpie are enough to see it in the web UI; nothing else has to run (the other providers
then show as unavailable, which is expected; on Docker Desktop for Windows a provider that is not running
takes 2.5 to 4 s to fail, and `/api/health` reports it as `ConnectError` or `ConnectTimeout`). From the
repository root, on a machine with an NVIDIA GPU and the NVIDIA container runtime:

```bash
ENABLE_MAGPIE_TTS=true ENABLE_CHATTERBOX_TTS=false \
  docker compose --env-file deploy/profiles/workstation-4080.env \
  -f docker-compose.yml --profile frontend --profile magpie-tts up -d --build
```

`ENABLE_CHATTERBOX_TTS=false` because the 4080 profile enables Chatterbox, which this command does not start.
The first build takes 10 to 15 minutes (torch and NeMo) and the first start downloads about 2.5 GB. Open
http://localhost:3000, pick **Magpie TTS** under *TTS Engine*, open *Text-to-Speech*, choose a voice and press
*Generate Speech*. The first request in a language other than German waits while its text normalizer is built
(see above); a reload after an idle unload does not repeat that. To call the service directly:
`curl -s -H 'Content-Type: application/json' -d '{"text":"Hallo Welt.","speaker":"Leo"}' http://127.0.0.1:5008/tts -o out.wav`.

Prefix `DEFAULT_TTS_PROVIDER=magpie` to the command to preselect Magpie in the UI and make `/v1/audio/speech`
use it as well. Its `voice` is then `Aria`, `Jason`, `John`, `Leo`, `Sofia` (any case), `0` to `4`, or one of
OpenAI's placeholder names (`alloy`, `nova`, ...), which mean `MAGPIE_DEFAULT_SPEAKER`; anything else, a Piper
voice name included, is a `400` that lists the speakers.

Stop it with the same file and profiles: `docker compose -f docker-compose.yml --profile frontend --profile magpie-tts down`.
Every service in `docker-compose.yml` belongs to a profile, so a bare `docker compose down` selects none and
stops nothing. Adding `-v` also deletes the `magpie-tts-cache` volume, that is the 2.5 GB download.

## Image

`nemo_toolkit[tts]` on CUDA 12.8 with torch 2.11 (cu128 wheels, so Blackwell cards work). Python 3.11 comes
from the deadsnakes PPA because Ubuntu 22.04's own `python3.11` is a release candidate that NeMo 3 cannot
load a model on (the build checks for it). NVIDIA GPUs only: there is no ROCm or Vulkan variant.

NeMo is pinned to 3.0.x. A local build can take another 3.x line with `MAGPIE_NEMO_TOOLKIT_SPEC` (for example
`MAGPIE_NEMO_TOOLKIT_SPEC='>=3.0.0,<4' docker compose build magpie-tts-service`). It is Magpie's own variable:
the NeMo 2.x rollback of the ASR images (`NEMO_TOOLKIT_SPEC='>=2.7.3,<3'`) does not reach it, because NeMo 2.7.3
has `MagpieTTSModel` but not the language map (`LANGUAGE_TOKENIZER_MAP`) the service imports. The build
imports both, and NeMo's text normalizer: an incompatible NeMo, or a missing or broken
`nemo_text_processing` (which NeMo would swallow, and then drop every digit), fails the build instead of
the first model load; NeMo 2.x stops it with `cannot import name 'LANGUAGE_TOKENIZER_MAP'`. The Japanese
OpenJTalk dictionary is built in (about 107 MB, in pyopenjtalk's package directory), so the first Japanese
request downloads nothing and works on a host without internet access. Building the image therefore
needs access to github.com (the dictionary is a release asset of `r9y9/open_jtalk`) as well as to PyPI
and download.pytorch.org.
