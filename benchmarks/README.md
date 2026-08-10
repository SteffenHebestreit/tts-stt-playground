# German ASR evaluation

Three defaults in this project trade memory for accuracy, and none of them has
been measured on German:

| Default | Why it exists | What is unknown |
|---|---|---|
| `WHISPER_COMPUTE_TYPE=int8_float16` (auto on GPU) | ~35% less VRAM than float16 — the difference between fitting and not fitting on a 12 GB card | its accuracy cost on German |
| whisper.cpp `q5_0` / `q5_1` | fits the RK3588S and the Vulkan path | same |
| Qwen3-ASR quantisation | fits alongside TTS | same |

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

---

## Running one configuration

The stack is used mostly as an API, so this measures the API. Whatever the
gateway returns to a client is what gets scored.

```bash
pip install httpx

python benchmarks/run_german_eval.py \
  --manifest /data/cv-de/validated.tsv \
  --audio-root /data/cv-de \
  --limit 500 \
  --label float16 \
  --out results-float16.json
```

Output:

```
float16: WER 8.42%  CER 2.11%  (500 utterances, 0 failed)
  substitutions=284 deletions=61 insertions=39
```

Both rates are reported because **CER is the more informative one for German**.
A single wrong morpheme in a compound (`Rechtschreibprüfung` →
`Rechtschreibprufung`) is a whole word error but only one character error, so
WER swings hard on exactly the words where quantisation damage shows up first.

---

## Deciding whether a change is safe

Run the baseline, change the setting, run again, then compare:

```bash
WHISPER_COMPUTE_TYPE=float16      docker compose up -d --force-recreate stt-service
python benchmarks/run_german_eval.py --manifest … --label float16 --out base.json

WHISPER_COMPUTE_TYPE=int8_float16 docker compose up -d --force-recreate stt-service
python benchmarks/run_german_eval.py --manifest … --label int8 --out cand.json

python benchmarks/run_german_eval.py --compare base.json cand.json
```

```
baseline  float16              WER 8.42%
candidate int8                 WER 8.61%

NO SIGNIFICANT DIFFERENCE — candidate is 0.19 points worse, but the 95% CI
[-0.31, +0.68] includes zero on 500 utterances. Do not decide on this.
  candidate was worse in 71% of 2000 resamples
```

**Decide with the interval, never with the two percentages.** On a few hundred
utterances the point estimates routinely differ by more than the effect you are
looking for. The comparison is a *paired* bootstrap: every resample draws the
same utterances from both systems, so per-utterance difficulty cancels instead
of becoming the thing you measure.

`--show-regressions N` prints the utterances that got worst, with reference and
both hypotheses side by side — useful for spotting whether a regression is real
degradation or a formatting difference.

Add `--character-level` to compare CER instead.

---

## Confirm the service really changed

`WHISPER_COMPUTE_TYPE` falls back silently if the GPU does not support the
requested type. Check what actually loaded before trusting a comparison:

```bash
curl -s localhost:3000/api/health | grep -o '"compute_type":"[^"]*"'
```

If both runs report the same `compute_type`, you measured the same thing twice.

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

## Files

| File | |
|---|---|
| `asr_metrics.py` | Normalisation, WER/CER, paired bootstrap. Pure stdlib, no I/O. |
| `run_german_eval.py` | CLI: manifests, API calls, reports, comparison. Needs `httpx`. |
| `../tests/test_asr_metrics.py` | 33 offline tests — no audio, no model, no network. |
