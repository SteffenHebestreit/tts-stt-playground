# German ASR evaluation

Three defaults in this project trade memory for accuracy, and none of them has
been measured on German:

| Default | Why it exists | What is unknown |
|---|---|---|
| `WHISPER_COMPUTE_TYPE=int8_float16` (auto on GPU) | ~35% less VRAM than float16 — the difference between fitting and not fitting on a 12 GB card | its accuracy cost on German |
| whisper.cpp `q5_0` / `q5_1` | fits the RK3588S and the Vulkan path | same |
| Qwen3-ASR quantisation | fits alongside TTS | same |

The same tool is the **decision gate** for choosing between ASR backends
(Whisper, Parakeet, Qwen3-ASR, Cohere Transcribe, IBM Granite Speech, a German
fine-tune): see
[Comparing providers](#comparing-whisper-parakeet-qwen3-asr-and-a-german-fine-tune).

German is an acceptance criterion for this project, so "probably fine" is not
good enough. This directory turns each of those into a measurement.

---

## What it measures, and what it does not

**It compares one configuration against another on the same audio.** That is the
question you actually have: *does int8 cost me German accuracy?*

**It is not built to reproduce published WER figures**, and you should not
compare its output to them. Normalisation choices, number formatting and
punctuation conventions move absolute WER by more than the effect being
measured. For example, a reference of `3 Euro` against a hypothesis of
`drei Euro` scores one error here — no number expansion is attempted.

That is a deliberate choice, not a gap. Number formatting inflates *both*
configurations identically, so it cancels in the comparison. A half-correct
number expander would introduce errors that do **not** cancel.

**It measures the file path.** One request per clip, the whole clip at once.
The live WebSocket path (`/ws/transcribe`, windowed decoding) is a different
code path with different failure modes and is not exercised here.

---

## Getting audio

Any set of (audio, transcript) pairs works. [Common Voice
German](https://commonvoice.mozilla.org/de/datasets) is the usual choice, and
its `validated.tsv` is read natively.

A few hundred utterances is enough to separate two configurations. Fewer than
about a hundred and the confidence interval will swamp any realistic
quantisation effect — the tool will tell you so rather than let you decide on
noise.

Supported manifests, detected automatically:

```
# Common Voice TSV — path/sentence, clips resolved from a sibling clips/
client_id	path	sentence	up_votes
…	common_voice_de_123.mp3	Guten Tag, wie geht es Ihnen?	2

# JSONL
{"audio": "clips/a.wav", "text": "Guten Tag"}

# Plain CSV/TSV with any of path|audio|file|filename and sentence|text|reference|transcript
```

Rows with no reference text are skipped in every format. Tab-separated files are
read without CSV quoting (Common Voice sentences contain bare `"` characters),
and a leading byte-order mark from a spreadsheet export is tolerated.

---

## Running one configuration

The stack is used mostly as an API, so this measures the API. Whatever the
endpoint returns to a client is what gets scored.

```bash
pip install httpx

python benchmarks/run_german_eval.py \
  --manifest /data/cv-de/validated.tsv \
  --audio-root /data/cv-de \
  --limit 500 --seed 1234 \
  --label float16 \
  --out results-float16.json
```

Output:

```
float16: WER 8.42%  CER 2.11%  (500 utterances, 0 failed)
  substitutions=284 deletions=61 insertions=39
  latency: warm median 0.41 s, p95 0.90 s; cold start 9.80 s
```

Both rates are reported because **CER is the more informative one for German**.
A single wrong morpheme in a compound (`Rechtschreibprüfung` →
`Rechtschreibprufung`) is a whole word error but only one character error, so
WER swings hard on exactly the words where quantisation damage shows up first.

### Which clips: `--limit` and `--seed`

`--limit N` takes a **random sample** of N clips, not the first N. A corpus is
not shuffled — Common Voice's `validated.tsv` is grouped by speaker, so the first
500 rows are a few dozen voices, and an effect that only shows on some voices
can vanish or be exaggerated depending on whose clips come first.

The sample is reproducible: the same `--seed` over the same manifest always
selects the same clips (default `1234`), kept in manifest order. **Use the same
`--limit` and `--seed` for every run you intend to compare.** The comparison
pairs on the audio path, so runs with different seeds share only the few clips
their samples happen to have in common.

### Which endpoint: `--base-url`, `--provider`, `--url`

| You pass | Request goes to | Protocol | Use it to measure |
|---|---|---|---|
| `--base-url http://host:3000` (default) | `/v1/audio/transcriptions` | `openai` (field `file`) | whatever the gateway's **default** provider is, as an OpenAI client sees it |
| `--base-url http://host:3000 --provider parakeet` | `/api/stt` | `gateway` (fields `audio`, `provider`) | one **named** provider through the gateway |
| `--url http://host:5005/transcribe` | that URL | `stt-form` (field `audio`) | one **backend directly**, no gateway in between |
| `--url http://host:5003/v1/audio/transcriptions` | that URL | `openai` | whisper.cpp's OpenAI-compatible route |

Provider ids are `whisper`, `qwen3-asr`, `parakeet`, `canary`, `cohere-asr`,
`granite-asr`, `vllm-asr` and `whisper-cpp`; the non-default ones only exist in the
gateway when their `ENABLE_*` flag is set.

`/v1/audio/transcriptions` **cannot select a provider**: it always uses the
configured default and ignores form fields it does not know, so a `provider` field
sent there would be dropped and the run mislabelled. The tool refuses
`--provider` on that route rather than let that happen. `--api
openai|stt-form|gateway` overrides the protocol it would infer from the URL.

A gateway started with `API_KEY` answers `/v1` and `/api` calls with 401 unless
they carry the key: pass `--api-key` or set `EVAL_API_KEY`. The key is sent as a
bearer token and is never written to the report. `EVAL_BASE_URL` sets the default
`--base-url`.

**A fallback is not a measurement.** When the default provider is down the
gateway silently answers from another one and only says so in an
`X-Provider-Fallback` header. Those responses are **not scored** — they would
attribute another model's errors to the one the run is labelled with. They are
counted as failed, listed in the report's `errors`, and summarised as
`fallback_responses`; if every response was a fallback the run stops with an
error. The same applies when `/api/stt` reports a provider other than the one
requested.

### Warmup, cold start and latency

Models unload after an idle TTL (`STT_MODEL_TTL`, `ASR_MODEL_TTL`), so the first
request after a pause pays a full model load: seconds, against tens of
milliseconds warm. Folding that into the totals made a run's latency depend on
how long the service had been idle.

So a run **warms up first**: `--warmup N` (default 1) sends N requests for the
first clip, untimed and unscored. The first one is reported separately as the
**cold start**, which is the number that matters for a service that unloads. The
timed requests then give the **warm** latency: mean, median, p95, max.
`--warmup 0` disables it, and then the first request's latency includes any load.

`wall_seconds` in the report is the sum of the timed request latencies, warmup
excluded. **RTF** (request time divided by audio duration) is reported only
when the backend states the clip length, which the `stt-form` backends do (their
response has `duration`) and the gateway's `/v1` route does not. The file size is
never used to guess a duration: it is wrong for every compressed format.

### Interrupting

Ctrl-C during a run stops it and reports the utterances that finished
(`"interrupted": true` in the report) instead of discarding hours of results.
`--strict` aborts on the first failed clip instead.

### The report

`--out` writes JSON. Beyond the counts, `results[]` holds every reference,
hypothesis and per-utterance count, and the top level records how the run was made
so it can be reproduced: `label`, `url`, `api`, `provider`, `model`, `language`,
`manifest`, `manifest_rows`, `limit`, `seed`, `started_at`, plus `latency`,
`warmup`, `cold_start_seconds`, `audio_seconds`, `rtf`, `errors`, `failed`,
`not_run` and `fallback_responses`.

---

## Deciding whether a change is safe

Run the baseline, change the setting, run again, then compare:

```bash
WHISPER_COMPUTE_TYPE=float16      docker compose up -d --force-recreate stt-service
python benchmarks/run_german_eval.py --manifest … --limit 500 --label float16 --out base.json

WHISPER_COMPUTE_TYPE=int8_float16 docker compose up -d --force-recreate stt-service
python benchmarks/run_german_eval.py --manifest … --limit 500 --label int8 --out cand.json

python benchmarks/run_german_eval.py --compare base.json cand.json
```

```
baseline  float16              WER 8.42%
candidate int8                 WER 8.61%

NO SIGNIFICANT DIFFERENCE — candidate is 0.19 points worse, but the 95% CI
[-0.31, +0.68] includes zero on 500 utterances. Do not decide on this.
  candidate was worse in 71% of 2000 resamples
  warm median latency: 0.41 s -> 0.33 s
  cold start:          9.80 s -> 8.10 s
```

**Decide with the interval, never with the two percentages.** On a few hundred
utterances the point estimates routinely differ by more than the effect you are
looking for. The comparison is a *paired* bootstrap: every resample draws the
same utterances from both systems, so per-utterance difficulty cancels instead
of becoming the thing you measure.

`--show-regressions N` prints the utterances that got worst, with reference and
both hypotheses side by side — useful for spotting whether a regression is real
degradation or a formatting difference. Only utterances that actually got worse
are listed.

Add `--character-level` to compare CER instead.

`--compare` refuses two runs scored with different `--fold-umlauts` settings
(their counts are not comparable) and notes clips present in only one run,
clips whose reference text differs between the runs, fallback responses and
interrupted runs.

---

## Comparing Whisper, Parakeet, Qwen3-ASR and a German fine-tune

This is the decision gate of `docs/model-and-optimisation-research-2026-08.md`.
It names the candidates, but for several of them no German WER is published at
all, and the published figures use different test sets and normalisation, so
they cannot be compared with each other. Run every candidate over **one
manifest, one sample** of *your* audio, then compare each against the incumbent.

**1. Fix the audio once.** Pick a manifest and never change `--limit` or
`--seed` while the comparison is open:

```bash
export EVAL="--manifest /data/cv-de/validated.tsv --audio-root /data/cv-de --limit 1000 --seed 1234"
```

A thousand clips is a sensible floor when the candidates are expected to be
within a point of each other; a few hundred separates only large gaps.

**2. Bring up one candidate at a time.** They do not all fit on a 16 GB card
together (and the TTS services also want the card), and running one at a time
also keeps their latencies from interfering. Measure each directly with `--url`,
which needs no `ENABLE_*` flag and has no gateway in the way; ports are the
compose defaults, use your own on TrueNAS.

```bash
# Whisper, the incumbent, exactly as you run it today (mind WHISPER_COMPUTE_TYPE)
python benchmarks/run_german_eval.py $EVAL --url http://localhost:5001/transcribe \
  --label whisper-large-v3-turbo --out r-whisper.json

# A German Whisper fine-tune: point WHISPER_MODEL_SIZE at a CTranslate2 build of it
# (a Hugging Face repo id or a local path), recreate stt-service, run again.
WHISPER_MODEL_SIZE=<ct2-repo-or-path> docker compose up -d --force-recreate stt-service
python benchmarks/run_german_eval.py $EVAL --url http://localhost:5001/transcribe \
  --label whisper-german-ft --out r-whisper-ft.json

# Parakeet-TDT (profile parakeet-asr)
docker compose --profile parakeet-asr up -d parakeet-asr-service
python benchmarks/run_german_eval.py $EVAL --url http://localhost:5005/transcribe \
  --label parakeet --out r-parakeet.json

# Qwen3-ASR (profile qwen3-asr)
docker compose --profile qwen3-asr up -d qwen3-asr-service
python benchmarks/run_german_eval.py $EVAL --url http://localhost:5002/transcribe \
  --label qwen3-asr --out r-qwen3.json

# Cohere Transcribe (profile cohere-asr)
docker compose --profile cohere-asr up -d cohere-asr-service
python benchmarks/run_german_eval.py $EVAL --url http://localhost:5009/transcribe \
  --label cohere-asr --out r-cohere.json

# IBM Granite Speech 5.0 2b (profile granite-asr); the 470m TurboCTC is the same
# service with GRANITE_ASR_MODEL=ibm-granite/granite-speech-5.0-470m-turboctc
docker compose --profile granite-asr up -d granite-asr-service
python benchmarks/run_german_eval.py $EVAL --url http://localhost:5010/transcribe \
  --label granite-2b --out r-granite.json
```

`--language de` (the default) goes to every backend. Parakeet auto-detects and only
echoes the hint, so a German clip it mistakes for another language counts as
errors, which is the right thing to measure.

To measure through the gateway instead (the path a real client takes, including
its response normalisation), use `--base-url http://localhost:3000 --provider
parakeet` and so on. That needs the provider enabled in the gateway
(`ENABLE_PARAKEET_ASR=true` etc.), for example:

```bash
ENABLE_COHERE_ASR=true docker compose --profile frontend --profile cohere-asr up -d
python benchmarks/run_german_eval.py $EVAL --base-url http://localhost:3000 \
  --provider cohere-asr --label cohere-asr --out r-cohere.json
```

Cohere Transcribe and Granite take the language from the request (neither
identifies it), so `--language de` decides what they transcribe in. Both put a
voice activity detector in front of the model; on read speech such as MLS it
changes nothing measurable (Cohere: 5.20 % WER with it off, 5.12 % on), and the
detector itself costs about 0.1 s per clip.

**Measured on the workstation** (RTX 4080, bf16, 100 MLS German clips, seed
1234, through the gateway's `/api/stt`; one run each, so read the differences
with the bootstrap, not as a ranking):

| Provider | WER | CER | Warm median / p95 | RTF | VRAM |
|---|---|---|---|---|---|
| `cohere-asr` (Cohere Transcribe 03-2026) | 5.12 % | 2.65 % | 0.42 s / 0.62 s | 0.028 | 4.2 GB |
| `granite-asr` (granite-speech-5.0-2b) | 6.55 % | 2.71 % | 1.68 s / 2.49 s | 0.108 | 4.7 GB |
| `granite-asr` (granite-speech-5.0-470m-turboctc) | 5.69 % | 2.96 % | 0.16 s / 0.19 s | 0.010 | 1.2 GB |

On one 4-minute German recording (11 chunks): Cohere 3.66 % WER in 4.7 s, the
Granite 2b 4.54 % in 15.8 s, the 470m TurboCTC 3.66 % in 2.1 s. Cohere's output
on read speech is mostly lowercase without punctuation, Granite's punctuated and
cased; the scoring normalises both away.

**The same candidates under vLLM** (`vllm-asr`, one model per service, `VLLM_ASR_MODEL`),
measured the same way, plus throughput (next section):

```bash
ENABLE_VLLM_ASR=true VLLM_ASR_MODEL=Qwen/Qwen3-ASR-1.7B \
  docker compose --profile frontend --profile vllm-asr up -d --build
python benchmarks/run_german_eval.py $EVAL --base-url http://localhost:3000 \
  --provider vllm-asr --label vllm-qwen3 --out r-vllm-qwen3.json
```

Measured the same way (RTX 4080, vLLM 0.31, 0.45 of the card, 100 MLS clips, seed 1234):
Granite 2b 6.55 % WER / 2.68 % CER at 0.60 s median, Cohere 5.04 % / 2.66 % at 0.31 s,
Qwen3-ASR 1.7B 6.85 % / 2.49 % at 0.67 s, Whisper turbo German (primeline) 7.04 % / 2.52 %
at 0.29 s: Granite and Cohere transcribe as their transformers services do, one request
2.8x and 1.4x faster. Details: `vllm-asr-service/README.md`.

**3. Compare each candidate against the incumbent, not against each other:**

```bash
for r in whisper-ft parakeet qwen3; do
  python benchmarks/run_german_eval.py --compare r-whisper.json r-$r.json
done
```

**4. Read the verdict, then the cost.**

| Verdict | Meaning |
|---|---|
| `SIGNIFICANT` and better | The candidate is genuinely more accurate on this audio. Adopt it if its latency, VRAM and the licence are acceptable. |
| `NO SIGNIFICANT DIFFERENCE` | You cannot tell them apart on this many clips. Add clips (raise `--limit`, same `--seed`) or choose on latency, VRAM and licence. Do not pick the smaller WER. |
| `SIGNIFICANT` and worse | Reject it for German, however good its published English figure is. |

The comparison also prints warm median latency and cold start for both runs. For
the `stt-form` backends each report also carries an RTF (the gateway's `/v1`
route does not state clip length, so it has none). Read cold start next to
`ASR_MODEL_TTL`: a model that is accurate but takes eight seconds to load is a
different proposition on a service that unloads after five idle minutes.

Two things this cannot tell you: it does not measure the live WebSocket path, so a
streaming decision (for example whether Qwen3-ASR's incremental API can replace
the windowed Whisper loop) also needs a live test; and Common Voice is read
speech, so if your real audio is conversational or noisy, build a manifest from
that audio too.

**Re-run after a runtime bump.** The same procedure is the regression check when
the model runtime changes rather than the model: a new NeMo release for
Parakeet/Canary, bf16 instead of fp32, a new faster-whisper or CTranslate2.
Keep the last report, re-run with the same `--limit` and `--seed`, and
`--compare` old against new.

---

## Confirm the service really changed

`WHISPER_COMPUTE_TYPE` falls back silently if the GPU does not support the
requested type. Check what actually loaded before trusting a comparison:

```bash
curl -s localhost:3000/api/health | grep -o '"compute_type":"[^"]*"'
```

If both runs report the same `compute_type`, you measured the same thing twice.
The same caution applies to a provider comparison: check the report's `url`,
`api` and `provider` rather than trusting the label.

---

## Umlauts

`--fold-umlauts` maps ä→ae, ö→oe, ü→ue, ß→ss before scoring. **Off by default,
and it should usually stay off**: those distinctions are phonemic, and a model
writing `Grusse` for `Grüße` genuinely got it wrong. Turn it on only when
comparing a backend that cannot emit umlauts at all.

Note that `str.casefold()` — the usual way to lowercase — maps `ß` to `ss` on its
own. The normaliser uses `str.lower()` specifically so that half an umlaut fold
is not applied behind your back. There is a test pinning this.

---

## Throughput under concurrency

`run_german_eval.py` sends one clip at a time. What a batching engine is for shows only
with many requests in flight, so `run_throughput.py` sends the same clips (the same
`--limit`/`--seed` selection) with N requests at once, for each N, and reports audio
seconds per wall second (`rtfx`), requests per second, latency p50/p95 and the WER/CER
of what came back:

```bash
python benchmarks/run_throughput.py --base-url http://localhost:3000 --provider vllm-asr \
  --manifest data/de.tsv --limit 200 --concurrency 1 8 32 --out tp-vllm.json
```

Through the gateway, `MAX_CONCURRENT_UPLOADS` (4 per gateway worker by default) refuses
anything beyond it with 503: set it to at least the highest level for the run, or measure
the backend directly (`--url http://localhost:5011/transcribe`). A failed request is
counted, printed and not scored, and makes the exit status 1.

Measured on the RTX 4080 (first 200 MLS clips, 3,039 s of audio, through `/api/stt` with
`MAX_CONCURRENT_UPLOADS=40`), audio seconds per wall second at 1 / 8 / 32 requests at once:

| `vllm-asr` model | 1 | 8 | 32 |
|---|---|---|---|
| granite-speech-5.0-2b | 25x | 146x | 293x |
| cohere-transcribe-03-2026 | 50x | 224x | 392x |
| Qwen3-ASR-1.7B | 24x | 151x | 343x |
| whisper-large-v3-turbo-german | 53x | 225x | 326x |

For the NAS's RTX 5060 Ti scale by about 0.63.

---

## Long recordings

The edit distance is exact and fast for ordinary utterances and for long-form
audio whose hypothesis is close to the reference: it trims the shared start and
end and runs the dynamic programme in a band around the diagonal, so the work
grows with the number of *errors*, not with the square of the length. A 30,000
character comparison with about 5% errors takes seconds, not the hours the
full matrix would.

If `rapidfuzz` is installed it is used automatically for very large comparisons
(over about four million cells). The error **total** is identical either way; when
several alignments tie, the split into substitutions, deletions and insertions
can differ, and everything below `4,000,000` cells always uses the pure-Python
path so saved reports do not depend on what is installed. Set
`ASR_METRICS_NO_RAPIDFUZZ=1` to force the pure-Python path everywhere.

---

## Files

| File | |
|---|---|
| `asr_metrics.py` | Normalisation, WER/CER, paired bootstrap. Stdlib only (rapidfuzz optional), no I/O. |
| `run_german_eval.py` | CLI: manifests, sampling, warmup, API calls to any of the three protocols, reports, comparison. Needs `httpx` to run. |
| `run_throughput.py` | CLI: the same clips with N requests in flight, for several N; throughput, latency and WER per level. Needs `httpx`. |
| `../tests/test_asr_metrics.py` | Offline tests of the normaliser, the metrics and the manifest reader. |
| `../tests/test_benchmarks_runner.py` | `run()` and `compare()` end to end against a stub server that speaks all three protocols: wire fields, provider selection, sampling, warmup, failures, fallbacks, comparison. |
| `../tests/test_benchmarks_metrics.py` | The fast edit distance and bootstrap against the implementations they replaced, operation for operation. |
| `../tests/test_benchmarks_throughput.py` | `run_throughput.py` against a stub that records how many requests it held at once. |

All of them are offline: no audio, no model, no network.
