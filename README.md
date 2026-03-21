![Tangle Network Banner](https://raw.githubusercontent.com/tangle-network/tangle/refs/heads/main/assets/Tangle%20%20Banner.png)

<h1 align="center">Voice Inference Blueprint</h1>

<p align="center"><em>Anonymous TTS inference on <a href="https://tangle.tools">Tangle</a> — operators serve Qwen3-TTS via vLLM-Omni, users pay anonymously via shielded credits.</em></p>

<p align="center">
  <a href="https://discord.com/invite/cv8EfJu3Tn"><img src="https://img.shields.io/discord/833784453251596298?label=Discord" alt="Discord"></a>
  <a href="https://t.me/tanglenet"><img src="https://img.shields.io/endpoint?color=neon&url=https%3A%2F%2Ftg.sumanjay.workers.dev%2Ftanglenet" alt="Telegram"></a>
</p>

## Overview

A Tangle Blueprint enabling operators to serve TTS (Text-to-Speech) inference via [vLLM-Omni](https://github.com/vllm-project/vllm) with anonymous payments through the [Shielded Payment Gateway](https://github.com/tangle-network/shielded-payment-gateway).

**Dual payment paths:**
- **On-chain jobs** via TangleProducer — verifiable results on Tangle
- **x402 HTTP** — fast private TTS at `/v1/audio/speech`

Single model per instance. Built with [Blueprint SDK](https://github.com/tangle-network/blueprint) `0.1.0-alpha.22`.

## Components

| Component | Language | Description |
|-----------|----------|-------------|
| `operator/` | Rust | Operator binary — vLLM subprocess, HTTP server, SpendAuth billing |
| `contracts/` | Solidity | InferenceBSM — GPU validation, model pricing (24 tests) |
| `sdk/` | TypeScript | Client SDK — OpenAI-compatible TTS interface with spend auth signing |

## Related Repos

- [shielded-payment-gateway](https://github.com/tangle-network/shielded-payment-gateway) — the payment layer
- [blueprint](https://github.com/tangle-network/blueprint) — Blueprint SDK
- [tnt-core](https://github.com/tangle-network/tnt-core) — Tangle core protocol

## License

MIT
