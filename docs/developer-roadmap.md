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

- STT is partly generalized: faster-whisper and Qwen3-ASR both conform to the same `/transcribe` response shape.
- whisper.cpp already exposes an OpenAI-compatible STT API, but most UI and orchestration code is still centered on the local `/transcribe` contract.
- TTS and training are still strongly Piper- and Qwen3-specific.
- No single roadmap or provider-contract document existed before this one.

## Roadmap

### Phase 1: Registry and Contracts

- Introduce a provider registry in the frontend service.
- Classify providers by `kind`, `capabilities`, and `contracts` rather than only by service name.
- Publish the registry through a machine-readable endpoint.
- Document the supported contract families.

Exit criteria:

- frontend can discover providers and defaults from registry data
- provider docs live in-repo and are versioned with the code

### Phase 2: UI Generalization

- Route STT requests by contract type rather than fixed endpoint assumptions.
- Treat OpenAI-compatible STT backends as first-class providers.
- Keep specialized TTS panels, but bind them to provider families through registry metadata.

Exit criteria:

- adding a new STT backend that matches `stt-form-v1` or `openai-audio-transcriptions-v1` is configuration plus provider deployment

### Phase 3: Contract Tests

- Add tests that assert the frontend registry is coherent.
- Add contract tests for each supported STT contract family.
- Keep provider-specific tests, but supplement them with shared interoperability tests.

Exit criteria:

- provider swaps fail fast in CI when a contract is broken

### Phase 4: TTS Generalization

- Define a minimal shared TTS contract for plain synthesis.
- Separate advanced capabilities like cloning, saved voices, and model switching into optional capability contracts.
- Move the frontend from hardcoded provider names toward capability-driven rendering where practical.

Exit criteria:

- basic TTS providers can be added without editing the general TTS request path
- advanced provider-specific features remain isolated behind explicit capability checks

### Phase 5: Training Abstraction

- Split training orchestration from Piper-specific export assumptions.
- Define an export-target contract so training outputs can target Piper or another compatible runtime.
- Replace direct service-name coupling with target-provider metadata.

Exit criteria:

- training no longer assumes Piper as the only deployment target

## Near-Term Priorities

1. Keep growing the provider registry instead of adding new hardcoded frontend URLs.
2. Prefer contract adapters over provider-specific branching when integrating new STT backends.
3. Define a neutral simple-TTS contract before adding more TTS providers.
4. Move training export and voice management behind explicit capability contracts.

## Non-Goals

- hide meaningful provider differences behind vague abstractions
- force all advanced features into one lowest-common-denominator API
- break existing Piper and Qwen3 workflows in pursuit of genericity