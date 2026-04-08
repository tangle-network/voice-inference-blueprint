//! Voice-specific operator configuration.
//!
//! Shared infrastructure config (`TangleConfig`, `ServerConfig`, `BillingConfig`,
//! `GpuConfig`) lives in `tangle-inference-core` and is re-exported here for
//! convenience.

use blueprint_sdk::std::path::PathBuf;
use serde::{Deserialize, Serialize};

pub use tangle_inference_core::{BillingConfig, GpuConfig, ServerConfig, TangleConfig};

/// Top-level operator configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorConfig {
    /// Tangle network configuration (shared).
    pub tangle: TangleConfig,

    /// vLLM-Omni subprocess + per-character pricing configuration (voice-specific).
    pub vllm: VoiceConfig,

    /// HTTP server configuration (shared).
    pub server: ServerConfig,

    /// Billing / ShieldedCredits configuration (shared).
    pub billing: BillingConfig,

    /// GPU configuration (shared).
    pub gpu: GpuConfig,

    /// STT configuration (preferred — supports all backends).
    #[serde(default)]
    pub stt: Option<SttConfig>,

    /// Legacy Whisper STT configuration (backward compat — maps to SttConfig internally).
    #[serde(default)]
    pub whisper: Option<WhisperConfig>,

    /// RLN Mode configuration (optional — enables RLN payment path).
    #[serde(default)]
    pub rln: Option<RLNConfig>,
}

/// STT configuration — supports multiple open-source backends.
///
/// Backends: "whisper", "faster-whisper", "parakeet", "sensevoice", "moonshine".
/// - "whisper" supports subprocess mode (insanely-fast-whisper) and server mode.
/// - All others require server mode with an HTTP endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SttConfig {
    /// Which backend: "faster-whisper", "parakeet", "sensevoice", "moonshine", "whisper".
    pub backend: String,

    /// Model ID (e.g. "Systran/faster-whisper-large-v3", "nvidia/parakeet-tdt-0.6b",
    /// "FunAudioLLM/SenseVoiceSmall", "moonshine/base").
    pub model: String,

    /// "subprocess" (whisper only) or "server" (HTTP endpoint).
    pub mode: String,

    /// HTTP endpoint URL (required for server mode).
    pub endpoint: Option<String>,

    /// Price per audio second in base token units.
    pub price_per_audio_second: u64,

    /// Optional language hint (default: "en").
    #[serde(default = "default_language")]
    pub language: String,
}

fn default_language() -> String {
    "en".to_string()
}

/// Legacy Whisper STT configuration. Omit entirely to disable STT.
/// Prefer the `[stt]` section for new deployments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhisperConfig {
    /// HuggingFace model ID (e.g. "openai/whisper-large-v3" or "distil-whisper/distil-large-v3").
    pub model: String,

    /// "subprocess" (spawn insanely-fast-whisper per request) or "server" (HTTP endpoint).
    pub mode: String,

    /// Whisper HTTP endpoint URL (required for server mode).
    pub endpoint: Option<String>,

    /// Price per audio second in base token units.
    pub price_per_audio_second: u64,
}

/// vLLM-Omni subprocess + pricing config. This is the only truly voice-specific
/// config section — everything else comes from `tangle-inference-core`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceConfig {
    /// HuggingFace model ID (e.g. "Qwen/Qwen3-TTS-1.7B").
    pub model: String,

    /// Maximum context length the model will serve.
    pub max_model_len: u32,

    /// Host/port vLLM will listen on internally.
    pub host: String,
    pub port: u16,

    /// Number of GPUs for tensor parallelism.
    pub tensor_parallel_size: u32,

    /// Price per 1,000 input characters in base token units.
    pub price_per_1k_chars: u64,

    /// Additional vLLM CLI args.
    #[serde(default)]
    pub extra_args: Vec<String>,

    /// Path to the vLLM Python executable.
    #[serde(default = "default_vllm_command")]
    pub command: String,

    /// HuggingFace token for gated models.
    pub hf_token: Option<String>,

    /// Custom model download directory.
    pub download_dir: Option<PathBuf>,

    /// Startup timeout in seconds.
    #[serde(default = "default_startup_timeout")]
    pub startup_timeout_secs: u64,

    /// Default voice ID for synthesis.
    #[serde(default)]
    pub default_voice: Option<String>,

    /// Supported output formats.
    #[serde(default = "default_supported_formats")]
    pub supported_formats: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RLNConfig {
    /// RLNSettlement contract address.
    pub settlement_address: String,

    /// Path to the snarkjs verification key JSON (optional — MVP skips real verification).
    pub verification_key_path: Option<String>,

    /// How often to batch-settle pending RLN claims (seconds).
    #[serde(default = "default_batch_settle_interval")]
    pub batch_settle_interval_secs: u64,

    /// Maximum claims per batch transaction.
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: usize,
}

fn default_batch_settle_interval() -> u64 {
    60
}

fn default_max_batch_size() -> usize {
    64
}

fn default_vllm_command() -> String {
    "python3 -m vllm.entrypoints.openai.api_server".to_string()
}

fn default_startup_timeout() -> u64 {
    300
}

fn default_supported_formats() -> Vec<String> {
    vec!["mp3".to_string(), "wav".to_string(), "ogg".to_string()]
}

impl OperatorConfig {
    /// Resolve STT config: prefer `stt` section, fall back to legacy `whisper`.
    pub fn resolve_stt(&self) -> Option<SttConfig> {
        self.stt.clone().or_else(|| {
            self.whisper.as_ref().map(|w| SttConfig {
                backend: "whisper".to_string(),
                model: w.model.clone(),
                mode: w.mode.clone(),
                endpoint: w.endpoint.clone(),
                price_per_audio_second: w.price_per_audio_second,
                language: "en".to_string(),
            })
        })
    }

    /// Load config from file, env vars, and CLI overrides.
    pub fn load(path: Option<&str>) -> anyhow::Result<Self> {
        let mut builder = config::Config::builder();

        if let Some(path) = path {
            builder = builder.add_source(config::File::with_name(path));
        }

        // Env vars override file config. Prefix: VOICE_OP_ (e.g. VOICE_OP_TANGLE__RPC_URL).
        builder = builder.add_source(
            config::Environment::with_prefix("VOICE_OP")
                .separator("__")
                .try_parsing(true),
        );

        let cfg = builder.build()?.try_deserialize::<Self>()?;
        Ok(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example_config_json() -> &'static str {
        r#"{
            "tangle": {
                "rpc_url": "http://localhost:8545",
                "chain_id": 31337,
                "operator_key": "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
                "shielded_credits": "0x0000000000000000000000000000000000000002",
                "blueprint_id": 1,
                "service_id": null
            },
            "vllm": {
                "model": "Qwen/Qwen3-TTS-1.7B",
                "max_model_len": 8192,
                "host": "127.0.0.1",
                "port": 8000,
                "tensor_parallel_size": 1,
                "price_per_1k_chars": 10
            },
            "server": {
                "host": "0.0.0.0",
                "port": 8080
            },
            "billing": {
                "max_spend_per_request": 1000000,
                "min_credit_balance": 1000
            },
            "gpu": {
                "expected_gpu_count": 1,
                "min_vram_mib": 16000
            }
        }"#
    }

    #[test]
    fn test_deserialize_full_config() {
        let cfg: OperatorConfig = serde_json::from_str(example_config_json()).unwrap();
        assert_eq!(cfg.tangle.chain_id, 31337);
        assert_eq!(cfg.vllm.model, "Qwen/Qwen3-TTS-1.7B");
        assert_eq!(cfg.vllm.port, 8000);
        assert_eq!(cfg.server.port, 8080);
        assert_eq!(cfg.vllm.price_per_1k_chars, 10);
        assert_eq!(cfg.gpu.expected_gpu_count, 1);
        assert!(cfg.tangle.service_id.is_none());
    }

    #[test]
    fn test_rln_config_optional() {
        let cfg: OperatorConfig = serde_json::from_str(example_config_json()).unwrap();
        assert!(cfg.rln.is_none(), "RLN config should be None by default");
    }

    #[test]
    fn test_defaults_applied() {
        let cfg: OperatorConfig = serde_json::from_str(example_config_json()).unwrap();
        assert_eq!(cfg.server.max_concurrent_requests, 64);
        assert_eq!(
            cfg.vllm.command,
            "python3 -m vllm.entrypoints.openai.api_server"
        );
        assert_eq!(cfg.vllm.startup_timeout_secs, 300);
        assert!(cfg.vllm.extra_args.is_empty());
        assert_eq!(cfg.vllm.supported_formats, vec!["mp3", "wav", "ogg"]);
        assert_eq!(cfg.gpu.monitor_interval_secs, 30);
    }

    #[test]
    fn test_load_from_file() {
        let cfg = OperatorConfig::load(Some("../deploy/config.example")).unwrap();
        assert_eq!(cfg.tangle.chain_id, 31337);
        assert_eq!(cfg.vllm.model, "Qwen/Qwen3-TTS-1.7B");
        assert_eq!(cfg.vllm.price_per_1k_chars, 10);
    }

    #[test]
    fn test_missing_required_field_fails() {
        let bad = r#"{"tangle": {"rpc_url": "http://localhost:8545"}}"#;
        let result = serde_json::from_str::<OperatorConfig>(bad);
        assert!(result.is_err());
    }
}
