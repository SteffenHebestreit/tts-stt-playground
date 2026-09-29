# Developer Roadmap

This repository started as a set of concrete service integrations: Piper for local training and ONNX playback, faster-whisper for STT, Qwen3 for cloning, and whisper.cpp as an optional OpenAI-compatible backend. That works operationally, but it does not yet make provider replacement cheap.

## Goal

Move from named-service wiring to capability-based integration.

The target architecture is:

- a provider registry that describes available backends, capabilities, browser URLs, and request contracts
- a small set of stable contracts for the frontend and orchestration layers
- provider adapters when a backend does not natively match those contracts
- contract tests that verify providers can be swapped without rewriting UI logic

## Current State

Status as of 2026-09-29. What exists, and what is still hardcoded:

- **Registry and contracts (phase 1): done.** The gateway builds a provider registry (`GET /providers`),
  the contracts are written down in [provider-contracts.md](provider-contracts.md), and nine providers
  are registered: Piper, Qwen3-TTS and Chatterbox (TTS), faster-whisper, Qwen3-ASR, Parakeet, Canary
  and whisper.cpp (STT), and Piper training. `PROVIDER_REGISTRY_JSON`, `TRAINING_PROVIDER` and
  `DEFAULT_*_PROVIDER` shape it at deploy time.
- **STT is generalised.** Five providers speak `stt-form-v1` (four of them) or
  `openai-audio-transcriptions-v1` (whisper.cpp). The OpenAI-compatible `/v1` surface returns one response shape whichever backend
  answers, and falls back to the single healthy alternative when the default cannot be reached.
- **Model services share one lifecycle contract.** `/health` (liveness, never loads a model),
  `/ready` (503 while the first load runs or after it failed), `POST /unload` (409 while busy) and an
  idle TTL, implemented once in `model_lifecycle.ModelSlot` and shared by four of the six model
  services (Qwen3-ASR, Parakeet, Canary, Chatterbox; `stt-service` has its own residency module and
  Qwen3-TTS a bespoke reaper).
- **Contract tests exist for the gateway, not yet for the backends.** `tests/test_openai_v1_api.py`
  asserts the identical response shape across providers and the gateway adapters have unit suites; a
  test that runs the same request against every real STT backend needs a GPU stack and does not exist.
- **The OpenAPI specs are generated** from the apps (`scripts/sync_openapi.py`) and CI fails when a
  committed spec is stale, so the training spec can no longer describe routes that do not exist.
- **TTS and training are still Piper- and Qwen3-specific in the UI**, although adding Chatterbox
  needed only registry data, one adapter mapping and no change to the general TTS request path.
- **Provider metadata is mostly static, and the exceptions are narrow.** Canary's language list and
  display name are refreshed from the service's `/status`, so they follow `CANARY_ASR_MODEL`
  (fallback en/de/es/fr while it is unreachable); `/api/providers/piper/voices` passes on the
  service's `default_language`; and the Qwen3-TTS adapter forwards `auto` and languages the model
  lacks instead of mapping them to English, and its registry defaults to `auto`, so
  `QWEN3_DEFAULT_LANGUAGE` and the service's 400 apply. These were hardcoded in the adapter when
  this release was wired. Canary is still the only registry entry that is updated from its
  service; the others are static, which is what phase 4 is for.
- **Live transcription is not a shared contract.** Only `stt-service` implements
  `/ws/transcribe`; the gateway refuses the other providers with a reason (close code 1008).

## Roadmap

### Phase 1: Registry and Contracts

- Introduce a provider registry in the frontend service.
- Classify providers by `kind`, `capabilities`, and `contracts` rather than only by service name.
- Publish the registry through a machine-readable endpoint.
- Document the supported contract families.

Exit criteria:

- frontend can discover providers and defaults from registry data
- provider docs live in-repo and are versioned with the code

Status: **done.**

### Phase 2: UI Generalization

- Route STT requests by contract type rather than fixed endpoint assumptions.
- Treat OpenAI-compatible STT backends as first-class providers.
- Keep specialized TTS panels, but bind them to provider families through registry metadata.

Exit criteria:

- adding a new STT backend that matches `stt-form-v1` or `openai-audio-transcriptions-v1` is configuration plus provider deployment

Status: **mostly done.** Parakeet and Canary were added this way. The per-provider language lists
and the live-transcription path are the remaining exceptions (see above).

### Phase 3: Contract Tests

- Add tests that assert the frontend registry is coherent.
- Add contract tests for each supported STT contract family.
- Keep provider-specific tests, but supplement them with shared interoperability tests.

Exit criteria:

- provider swaps fail fast in CI when a contract is broken

Status: **partial.** The registry, the `/v1` shape and the adapters are covered; the config-drift
suites additionally fail CI when compose, `.env.example`, the device presets, the OpenAPI specs or
the Dockerfiles disagree with each other. A shared conformance suite that runs against each real
backend is open.

### Phase 4: TTS Generalization

- Define a minimal shared TTS contract for plain synthesis.
- Separate advanced capabilities like cloning, saved voices, and model switching into optional capability contracts.
- Move the frontend from hardcoded provider names toward capability-driven rendering where practical.

Exit criteria:

- basic TTS providers can be added without editing the general TTS request path
- advanced provider-specific features remain isolated behind explicit capability checks

Status: **partial.** `simple-json-tts-v1` is shared by Piper, Qwen3-TTS and Chatterbox, and
`chunked-wav-stream-v1` and `saved-voice-library-v1` isolate the advanced features. Qwen3-TTS still
needs its payload translated by the adapter.

### Phase 5: Training Abstraction

- Split training orchestration from Piper-specific export assumptions.
- Define an export-target contract so training outputs can target Piper or another compatible runtime.
- Replace direct service-name coupling with target-provider metadata.

Exit criteria:

- training no longer assumes Piper as the only deployment target

Status: **open** beyond the deployment-target contracts that already exist. Since the last update the
training service serialises jobs (`TRAINING_MAX_CONCURRENT`), reports `/ready`, and stops cleanly on
`SIGTERM` so that an image update does not lose a running job.

## Near-Term Priorities

1. Move the remaining per-provider metadata (Canary languages, Piper's default language, the Qwen3
   `auto` mapping) out of the gateway adapters and into registry data or the services' own `/status`.
2. Keep growing the provider registry instead of adding new hardcoded frontend URLs.
3. Prefer contract adapters over provider-specific branching when integrating new STT backends.
4. Decide the live-transcription contract (`/ws/transcribe`) before a second provider implements it.
   The benchmark in [`benchmarks/README.md`](../benchmarks/README.md#comparing-whisper-parakeet-qwen3-asr-and-a-german-fine-tune)
   is the decision gate for which engine that is.
5. Move training export and voice management behind explicit capability contracts.

The model and optimisation plan, with the status of each item, is
[model-and-optimisation-research-2026-08.md](model-and-optimisation-research-2026-08.md).

## Non-Goals

- hide meaningful provider differences behind vague abstractions
- force all advanced features into one lowest-common-denominator API
- break existing Piper and Qwen3 workflows in pursuit of genericity