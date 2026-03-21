# CLAUDE.md

## Project Overview

Voice Inference Blueprint for Tangle Network. Operators serve TTS inference via vLLM-Omni (Qwen3-TTS), users pay anonymously through ShieldedCredits.

## Architecture

Uses the Tangle Blueprint SDK (`blueprint-sdk` crate) with the canonical lib+bin pattern.

- **contracts/**: Solidity BSM (InferenceBSM) -- validates operator registration (GPU caps), restricts payment to tsUSD, stores model metadata
- **operator/src/lib.rs**: Library crate -- `router()`, `run_tts` job handler (TangleArg/TangleResult), `VoiceInferenceServer` BackgroundService, sol! ABI types
- **operator/src/main.rs**: Binary crate -- BlueprintRunner wiring only (BlueprintEnvironment, TangleProducer, TangleConsumer)
- **operator/src/server.rs**: Axum HTTP server with OpenAI-compatible TTS endpoint (runs as BackgroundService)
- **operator/src/billing.rs**: ShieldedCredits on-chain billing (authorizeSpend/claimPayment) and off-chain EIP-712 SpendAuth signature verification
- **operator/src/voice_engine.rs**: vLLM-Omni subprocess management (spawn, health check, speech synthesis proxy)
- **operator/src/config.rs**: Operator config structs (VoiceModelConfig, billing, GPU, server)
- **operator/src/health.rs**: GPU detection via nvidia-smi
- **sdk/**: TypeScript client -- signs SpendAuth, discovers operators, sends TTS requests

## Build Commands

### Contracts
```bash
cd contracts && forge build && forge test
```

### Operator
```bash
cargo build -p voice-inference
```

### SDK
```bash
cd sdk && npm install && npm run build
```

## SDK Patterns

This project follows the Tangle Blueprint SDK patterns:

- **Router**: `Router::new().route(JOB_ID, handler.layer(TangleLayer))`
- **Job handlers**: `async fn handler(TangleArg(req): TangleArg<SolType>) -> TangleResult<SolType>`
- **ABI types**: Defined via `alloy_sol_types::sol!` macro
- **Background services**: `impl BackgroundService for VoiceInferenceServer`
- **Runner**: `BlueprintRunner::builder(TangleConfig::default(), env).router(...).producer(...).consumer(...).run().await`

## Key Dependencies

```toml
blueprint-sdk = { version = "0.1.0-alpha.22", features = ["std", "tangle", "macros"] }
```

## Billing Flow

1. User funds ShieldedCredits account (one ZK proof)
2. User signs EIP-712 SpendAuth per request (off-chain, cheap)
3. Operator verifies SpendAuth signature off-chain (ecrecover)
4. Operator serves TTS synthesis
5. Operator calls `authorizeSpend` on-chain (reserves payment)
6. Operator calls `claimPayment` (receives tokens)

Billing is per-character: `cost = (characters * price_per_1k_characters) / 1000`

## Testing

- Contracts: `forge test` in contracts/
- Operator: `cargo test` at workspace root
- SDK: `npm test` in sdk/
