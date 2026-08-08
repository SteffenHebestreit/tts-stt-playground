# Device deployment matrix

Which services and models to run on each target device, with German as a hard requirement on
every one. Written 2026-08-08.

Every performance figure below either carries its hardware or is marked **(unverified)**. Numbers
without hardware attached are not usable for capacity planning, so they are not presented as if
they were.

---

## The matrix

| | **STT** | **TTS** | **Peak memory** | **Turn latency** |
|---|---|---|---|---|
| **RK3588S 8 GB**<br>ARM64, no GPU | `whisper-cpp` `small-q5_1` | `piper` `de_DE-thorsten-medium` | ~5.3 GB of 8 GB with the LLM stage co-resident | 30 s clip → ~14 s (RTF 0.47, measured) |
| **RK3588S 16 GB**<br>ARM64, no GPU | same, or `medium-q5_1` if you can wait | same, + a second voice | ~6.5 GB with `medium` | 30 s clip → ~45 s at `medium` (RTF 1.51, measured) |
| **RTX 5060 Ti 12 GB**<br>Blackwell sm_120 | `stt-service` `large-v3-turbo`, **`float16`** | `piper` (always-on) + `chatterbox` | ~8 GB of 12 GB | **(unverified)** — no measurement on this GPU |
| **RTX 4080 16 GB**<br>Ada sm_89 | `stt-service` `large-v3-turbo`, `int8_float16` | `chatterbox` primary, `piper` low-latency tier | ~9 GB of 16 GB | **(unverified)** |
| **Strix Halo 128 GB**<br>gfx1151, unified | `whisper-cpp` via Vulkan, `large-v3-turbo-q5_0` | `piper` (CPU) | bandwidth-bound, not capacity-bound | `base.en` 2044 s audio in 35.2 s (≈58× RT, ROCm 7.0.1 HIP, Linux) |

Ready-made presets: [`deploy/profiles/`](../deploy/profiles/) — `rk3588.env`,
`truenas-5060ti.env`, `workstation-4080.env`, `strixhalo.env`.

---

## German evidence, per component

German is an acceptance criterion, not an optimisation target: a German-only fine-tune that
degrades other languages is the **wrong** choice here. Everything below is multilingual *and*
names German.

| Component | German evidence |
|---|---|
| Whisper `small` / `large-v3-turbo` | Multilingual checkpoints (no `.en` suffix). German WER at `small` is **(unverified)** |
| `piper` `de_DE-thorsten-medium` | A German voice by construction; 18 German Piper voices exist under [csukuangfj](https://huggingface.co/api/models?author=csukuangfj&search=vits-piper-de_DE) |
| Chatterbox Multilingual | German named in its language list ([model card](https://huggingface.co/ResembleAI/chatterbox)) |
| `parakeet-tdt-0.6b-v3` | German WER **5.04 % FLEURS / 4.84 % CoVoST**, CC-BY-4.0 ([card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)) |
| `canary-180m-flash` | German WER **4.81 % MLS**, CC-BY-4.0 ([card](https://huggingface.co/nvidia/canary-180m-flash)) |
| `Qwen3-TTS-12Hz-0.6B-Base` | Declares `de` in its HF language metadata |
| Gemma 4 E2B | German among the 35+ out-of-the-box languages ([model card](https://ai.google.dev/gemma/docs/core/model_card_4)) |

> **Disqualified:** `distil-large-v3` is **English-only** despite carrying no `.en` suffix. It
> must never enter the STT fallback ladder — it would silently drop German on exactly the
> constrained devices the ladder exists to serve. `stt-service` excludes it deliberately.
>
> **Also disqualified:** `Qwen3-TTS-12Hz-1.7B-Base` declares **no** language metadata at all.
> The 0.6B is the default partly for this reason, not only for its size.

---

## Can one stack serve all four devices?

**TTS: yes.** Piper is ONNX Runtime on CPU, has aarch64 wheels, and runs at **RTF 0.10–0.12 on
the slowest device in the fleet** (measured, RK3588, medium voices). Anything fast enough on a
2.4 GHz Cortex-A76 is trivially fast enough on a 4080 — and it uses **zero VRAM**, which is
exactly what you want when the GPU is busy with ASR. Piper is the baseline everywhere, with
Chatterbox/Qwen3 layered on top only where expressiveness matters and VRAM exists.

**STT: no.** What differs across the four devices is not the model, it is the *compiled backend*,
and only one spans the set:

| Runtime | ARM64 CPU | CUDA sm_120 | CUDA sm_89 | ROCm gfx1151 |
|---|---|---|---|---|
| CTranslate2 (faster-whisper) | ✅ aarch64 wheels | ✅ **fp16 only** — [INT8 disabled for sm120](https://github.com/OpenNMT/CTranslate2/releases) | ✅ | ❌ no prebuilt ROCm wheel |
| ggml (whisper.cpp) | ✅ NEON | ✅ CUDA | ✅ CUDA | ✅ Vulkan + HIP |
| NeMo (parakeet/canary) | ❌ | ✅ | ✅ | ❌ no AMD backend |

**ggml is the only runtime covering all four.** If you want exactly one STT engine,
`whisper-cpp` is the answer and it already exists in this repo. The cost is that it caps the SBC
at `small`.

So: two universal services (Piper, whisper-cpp) plus one best-of-breed upgrade per device class
— a thin layer, not four bespoke stacks.

---

## The RK3588S sequential pipeline

The SBC runs **STT → LLM → TTS** one stage at a time, with the TTS output potentially in a
different language than the input.

### Co-residency fits — no swap machinery required

The natural assumption is that sequential execution demands load/unload orchestration. On these
numbers it does not:

| Item | Memory |
|---|---|
| OS + dockerd | ~1.00 GB **(unverified est.)** |
| `frontend-service` gateway | ~0.15 GB **(unverified est.)** |
| `piper-tts` + `de_DE-thorsten-medium` | ~0.35 GB **(unverified est.)** |
| `whisper-cpp` `small-q5_1` | **0.89 GB (measured peak RSS)** |
| Gemma 4 E2B `Q4_K_M` via llama.cpp | ~2.87 GB (derived, see below) |
| **Total** | **~5.26 GB of 8 GB — ~2.74 GB free** |

Two things make this comfortable rather than marginal:

- ggml and llama.cpp **mmap** their weights, so a large part of that resident set is reclaimable
  page cache rather than anonymous memory. Real pressure is lower than the table suggests.
- Only one stage is *computing* at a time regardless, because there are only four A76 cores.

**Therefore: run the stages sequentially for compute reasons, but leave the models loaded.**
Building a cross-container eviction coordinator would add real complexity for no gain at this
budget. Revisit only if you move to `Q8_0` (see below) or a larger Whisper.

### Sequential execution does not make any stage faster

This is the trap worth stating explicitly. On the RK3588 the binding constraint is **compute**,
not memory:

| Whisper model | RTF (measured) | 30 s clip |
|---|---|---|
| `small` | 0.47 | ~14 s |
| `medium` | 1.51 | ~45 s |

`medium` already fitted in 8 GB before any of this. It was rejected for being slower than real
time, not for being too big. Freeing memory does not unlock it — only tolerance for turn latency
does. Since the pipeline is turn-based rather than live, RTF > 1 is now *arguably* acceptable;
that is a latency-vs-accuracy decision, and `deploy/profiles/rk3588.env` states both numbers so
it can be made deliberately.

### The LLM stage

**It is Gemma 4 E2B, and the name understates it.** Released 2026-04-02, Apache 2.0, 128 K
context. "E2B" is **2.3 B *effective*** parameters — the real weight count is **5.1 B** including
embeddings, via Per-Layer Embeddings. Budget from 5.1 B:

| Quantisation | Approx. size |
|---|---|
| `Q4_K_M` | ~2.9 GB ← fits the table above |
| `Q5_K_M` | ~3.5 GB |
| `Q8_0` | ~5.4 GB — **does not fit** alongside the rest on 8 GB |

Sizes are derived from the parameter count and bits-per-weight, not measured **(unverified)**.

**Gemma 4 E2B also accepts audio input**, with documented ASR and *speech-to-translated-text*,
capped at 30 seconds. That is architecturally interesting: for short clips it could collapse the
STT and LLM stages into one model. Worth an experiment, with two caveats — the 30 s cap is a hard
limit for an ASR API, and its German ASR quality is unpublished **(unverified)**. Keep
whisper-cpp as the STT path until measured.

**Recommendation on scope:** run the LLM *outside* this repo. This is a TTS/STT platform; adding
an LLM service grows its remit and duplicates what llama.cpp and Ollama already do well. Point
the pipeline at an external endpoint instead. (If a dedicated translation model is wanted rather
than a general LLM, note that `translategemma` exists but the published variant is 27 B — far too
large for this board.)

### The NPU: viable for German, but only with the right export

**This section was revised.** An earlier version of this document concluded the NPU was a
distraction. That was too strong, and the reason is a specific detail worth knowing.

There is a **field-tested working German-on-NPU path**, implemented in the sibling
`screenrecorder` project (`core/src/asr/whisper_rknn.py`). The critical trick:

> The Whisper **decoder** is sensitive to INT8 and produces empty or garbled text when fully
> quantised. Export the **decoder in fp16 and the encoder in INT8**.

That resolves the quantisation failure that
[rknn_model_zoo issue #314](https://github.com/airockchip/rknn_model_zoo/issues/314) reports
(`small` → empty transcriptions, `base` → `(((((((`). The earlier conclusion here — "FP32
converts but discards the reason to use an NPU" — missed that only the *decoder* needs higher
precision. The encoder is the expensive part and stays INT8, so most of the NPU benefit survives.

What remains true, and still argues for CPU as the **default**:

| Consideration | Status |
|---|---|
| SenseVoice on RKNN | Fast, but **zh/en/ja/ko/yue only — no German**. Disqualified here by the acceptance criterion. |
| Whisper on RKNN | **Multilingual incl. German.** ~6× slower than SenseVoice, but it is the one that works for German. |
| Export effort | You must do your own RKNN export; tensor shapes depend on it, so paths must be config-driven. |
| Word timestamps | Lost — `<\|notimestamps\|>` is forced for NPU reliability. Diarisation falls back to whole-segment overlap. |
| sherpa-onnx RKNN backend | Still no European-language models and no TTS ([RKNN docs](https://k2-fsa.github.io/sherpa/onnx/rknn/index.html)). |
| whisper.cpp | Still no NPU backend ([#1557](https://github.com/ggml-org/whisper.cpp/issues/1557)). |

**Recommendation:** keep `whisper-cpp` on the CPU as the default — it needs no custom export, keeps
word timestamps, and four A76 cores already meet the turn-latency budget.

RKNN Whisper is recorded here as a **known-viable option for this hardware**, not as planned work
for this repo. Nothing in this project targets the NPU today, and adding it is not scheduled. The
value of the note is that if it is ever wanted, the quantisation split above is the part that is
easy to get wrong, and the prior art is a private reference implementation rather than something
to be rediscovered from the upstream issue tracker.

Design details worth remembering if that day comes: import the NPU backend lazily so `rknnlite`
never enters the default install; make model paths and feature parameters config-driven, because
tensor shapes belong to *your* export rather than to the model; and fail loudly on a config
mismatch instead of guessing shapes.

### What cannot run on ARM64 at all

`stt-service`, `qwen3-asr`, `qwen3-tts`, `parakeet-asr`, `canary-asr`, `chatterbox-tts`,
`piper-training`. All are built on `nvidia/cuda:*` bases, and NeMo/Qwen/Chatterbox require CUDA.
**Do not add them to the arm64 CI matrix** — they would either fail to build or produce an image
that cannot start, which is worse because it looks supported.

`docker-compose.arm64.yml` covers exactly the three that work, and no more.

---

## Device-specific gotchas

**RTX 5060 Ti (D2) — Blackwell sm_120**
- Requires CUDA 12.8 + cu128 wheels. cu118/cu121 fail with `cudaErrorNoKernelImageForDevice`.
- **INT8 is disabled for sm_120 in CTranslate2**, so the repo's `int8_float16` default cannot
  apply. The preset pins `WHISPER_COMPUTE_TYPE=float16` explicitly rather than relying on the
  probe silently falling back.
- This card is **12 GB**, not 16 — three docs previously said 16, which made every VRAM budget
  read optimistically.

**Strix Halo (D4) — gfx1151**
- Strix Halo is **gfx1151 / RDNA 3.5 / Radeon 8060S**. Not `gfx1201` (RDNA 4 *discrete*), not
  `Radeon 890M` (Strix Point, gfx1150).
- `HSA_OVERRIDE_GFX_VERSION=11.0.0` is needed **only** because the `Dockerfile.rocm` images pin
  the rocm6.2 wheel index, which has no gfx1151 kernels. rocm7.0+ wheels carry them natively —
  clear the override in the same commit that bumps those bases, because forcing it on native
  kernels breaks torchaudio resampling, which every ASR/TTS path uses.
- **Do not use `stt-service` here.** CTranslate2's PyPI wheels are CUDA-only, so
  `FORCE_ACCELERATION=rocm` silently resolves to CPU int8. Use `whisper-cpp` via Vulkan.
- Reported "VRAM" is misleading: `mem_get_info()` returns the GTT pool, i.e. system RAM. Set
  `GPU_MEMORY_BUDGET_GB` explicitly.
- Capacity is huge but **memory bandwidth is shared with the CPU** and is the real limit.

---

## Quantised whisper.cpp model names

Quantised builds are **not** published for every size. A name that does not exist now fails
loudly at startup instead of writing an HTML error page into the model volume.

| Size | Available |
|---|---|
| `large-v3-turbo` | `-q5_0`, `-q8_0` |
| `small`, `base` | `-q5_1`, `-q8_0` — **no `-q5_0`** |

Source: [ggerganov/whisper.cpp on HuggingFace](https://huggingface.co/ggerganov/whisper.cpp).

---

## Deploying

```bash
# RK3588S — note the explicit profiles: `all` includes GPU services and omits whisper-cpp
docker compose --env-file deploy/profiles/rk3588.env \
  -f docker-compose.yml -f docker-compose.arm64.yml \
  --profile frontend --profile piper-tts --profile whisper-cpp up -d

# TrueNAS + 5060 Ti
docker compose --env-file deploy/profiles/truenas-5060ti.env \
  -f docker-compose.yml -f docker-compose.truenas.yml --profile all up -d

# Workstation + 4080
docker compose --env-file deploy/profiles/workstation-4080.env \
  -f docker-compose.yml --profile all --profile chatterbox-tts up -d

# Strix Halo
docker compose --env-file deploy/profiles/strixhalo.env \
  -f docker-compose.yml -f docker-compose.vulkan.yml \
  --profile frontend --profile piper-tts --profile whisper-cpp up -d
```

Verify a non-NVIDIA deployment carries no GPU reservation before starting it:

```bash
docker compose -f docker-compose.yml -f docker-compose.arm64.yml \
  --profile frontend --profile piper-tts --profile whisper-cpp config \
  | grep -c 'driver: nvidia'      # must print 0
```

---

## What still needs measuring on real hardware

Honest list of what this document cannot settle from documentation:

0. **RKNN Whisper German quality and speed** — the `screenrecorder` project has this working with
   an INT8 encoder + fp16 decoder, but no German WER or RTF is recorded for it. Since that path
   would free the CPU for the LLM stage, measuring it is the highest-value experiment on this
   board.
1. **RK3588 RSS for every stage** — only whisper.cpp `small` (889 MB) is measured. Run
   `docker stats` on the board.
2. **Gemma 4 E2B tokens/sec on four A76 cores**, and its real `Q4_K_M` resident size. This
   decides whether the LLM stage costs 3 seconds or 60.
3. **German WER for Whisper `small` vs `large-v3-turbo`** — nobody publishes the delta, and it is
   the whole argument for moving the SBC to sherpa-onnx/Canary later.
4. **`int8_float16` German WER on GPU** — faster-whisper publishes no WER delta for int8. It is
   the default on D3; measure before trusting it.
5. **D2/D3 latency** — no measurement exists for either GPU.

A single German evaluation clip set would settle items 3 and 4 and is the one piece of test work
genuinely required.
