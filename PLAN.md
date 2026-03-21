# Implementation Plan

## Phase 1: Contracts (Week 1)

### Done in scaffold
- [x] InferenceBSM with model config, operator registration, GPU validation
- [x] Payment restricted to tsUSD (ShieldedCredits pool token)
- [x] Job schema: input = (text, voice, responseFormat), output = (audioData, characterCount, format)
- [x] Tests for registration, model config, job lifecycle, payment validation

### Remaining
- [ ] Import tnt-core as soldeer dependency instead of inline interface
- [ ] Deploy script (Foundry script, not bash — use FullDeploy pattern from tnt-core)
- [ ] Integration test with actual ShieldedCredits contract
- [ ] Gas optimization pass
- [ ] Slashing logic: slash operators for provably wrong outputs or downtime
- [ ] Upgrade to UUPS proxy pattern for upgradability

## Phase 2: Operator Core (Week 2)

### Done in scaffold
- [x] Config system (file + env vars)
- [x] vLLM-Omni subprocess management (spawn, health check, shutdown)
- [x] OpenAI-compatible HTTP API (/v1/audio/speech, /v1/models, /health)
- [x] SpendAuth off-chain verification (EIP-712 ecrecover)
- [x] Billing client (authorizeSpend + claimPayment via alloy)
- [x] GPU detection via nvidia-smi
- [x] Tangle job handler structure

### Remaining
- [ ] Integrate tangle-blueprint-sdk for on-chain job event subscription
- [ ] vLLM stderr log draining and structured forwarding
- [ ] Graceful shutdown (drain in-flight requests before killing vLLM)
- [ ] Request queuing with backpressure (respect max_concurrent_requests)
- [ ] Prometheus metrics endpoint (/metrics)
- [ ] Request/response logging for auditability (without leaking input text)
- [ ] Rate limiting per credit account commitment
- [ ] Pre-authorization mode: authorize spend before serving (safer for operator)

## Phase 3: SDK (Week 2-3)

### Done in scaffold
- [x] EIP-712 SpendAuth signing with ethers.js
- [x] createVoiceClient() with synthesize(), listModels(), healthCheck()
- [x] Cost estimation (per-character billing)
- [x] Nonce management

### Remaining
- [ ] On-chain credit account query (getAccount via ethers contract)
- [ ] Operator discovery from BSM contract (getOperators + endpoint lookup)
- [ ] Automatic operator selection (closest, cheapest, most available)
- [ ] Retry logic with failover to different operators
- [ ] Audio playback helpers (Web Audio API integration)
- [ ] React hooks package (@tangle-network/voice-inference-react)
- [ ] CLI tool for testing (npx @tangle-network/voice-inference-sdk synthesize "hello")

## Phase 4: DevOps & Testing (Week 3-4)

### Done in scaffold
- [x] Dockerfile with CUDA + vLLM + Rust binary
- [x] docker-compose for local dev
- [x] Example config

### Remaining
- [ ] CI pipeline (GitHub Actions: build contracts, build operator, build SDK)
- [ ] Multi-arch Docker builds (amd64 + arm64)
- [ ] Kubernetes Helm chart for production deployment
- [ ] GPU resource scheduling (NVIDIA device plugin)
- [ ] End-to-end test: deploy contracts on Anvil, start operator, run SDK client
- [ ] Load testing with locust or k6
- [ ] Monitoring stack (Grafana dashboards, alert rules)
- [ ] Secret management (Vault integration for operator key)

## Phase 5: Production Hardening (Week 4+)

- [ ] Audit SpendAuth verification (ensure no signature malleability)
- [ ] Audit billing flow (ensure operator can't overcharge)
- [ ] Multi-model support per operator (serve multiple TTS models, route by request)
- [ ] Voice cloning support (custom voice embeddings)
- [ ] SSML input support for advanced prosody control
- [ ] Cross-chain billing (accept ShieldedCredits from Arbitrum, Base via bridges)

## Architecture Decisions

### Why vLLM subprocess (not in-process)?
- vLLM is Python; the operator is Rust. Subprocess isolation is clean.
- vLLM manages its own CUDA memory, GPU scheduling.
- Crash isolation: vLLM can crash without taking down the operator.
- Upgradable independently: swap vLLM version without rebuilding operator.

### Why off-chain SpendAuth verification before serving?
- On-chain verification (calling authorizeSpend) costs gas and takes ~2-12 seconds.
- Off-chain ecrecover is instant and free.
- The operator verifies the signature is valid, then serves immediately.
- After serving, the operator calls authorizeSpend + claimPayment asynchronously.
- Risk: if the credit account is drained between verification and authorization,
  the operator eats the cost. Mitigated by pre-authorization mode in Phase 2.

### Why tsUSD only?
- ShieldedCredits uses a single pool token per account.
- Stablecoin denomination simplifies pricing (no ETH/USD volatility).
- tsUSD is the canonical wrapped token from the VAnchor shielded pool.
- Operators can price in predictable USD terms.

### Why per-character billing for TTS?
- TTS cost scales with input text length, not output tokens.
- Characters are the natural unit for text-to-speech workloads.
- Simple and predictable for users: they know the cost before submitting.
