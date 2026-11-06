//! Voice-specific operator configuration.
//!
//! Shared infrastructure config (`TangleConfig`, `ServerConfig`, `BillingConfig`,
//! `GpuConfig`) lives in `tangle-inference-core` and is re-exported here for
//! convenience.

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
    #[serde(flatten)]
    pub engine: tangle_inference_core::engine::EngineConfig,
    pub price_per_1k_chars: u64,
    #[serde(default)]
    pub default_voice: Option<String>,
    #[serde(default = "default_supported_formats")]
    pub supported_formats: Vec<String>,
}
impl std::ops::Deref for VoiceConfig {
    type Target = tangle_inference_core::engine::EngineConfig;
    fn deref(&self) -> &Self::Target {
        &self.engine
    }
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
                .prefix_separator("_")
                .separator("__")
                .try_parsing(true),
        );

        let cfg = builder.build()?.try_deserialize::<Self>()?;
        Ok(cfg)
    }
}
