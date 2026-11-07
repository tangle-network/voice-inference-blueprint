pub mod config;
pub mod server;
pub mod speech;
pub mod stt_engine;
pub mod whisper;

// Re-export shared infrastructure so downstream crates can `use voice_inference::*`.
pub use tangle_inference_core::server::{
    error_response, extract_x402_spend_auth, payment_required, validate_spend_auth,
};
pub use tangle_inference_core::{
    detect_gpus, parse_nvidia_smi_output, AppState, AppStateBuilder, BillingClient, CostModel,
    CostParams, GpuInfo, NonceStore, PerCharCostModel, RequestGuard, SpendAuthPayload,
};
// Re-export metrics module for tests/downstream use.
pub use tangle_inference_core::metrics;
// Alias billing module path for downstream callers.
pub use tangle_inference_core::billing;

#[cfg(feature = "chain")]
mod chain;
#[cfg(feature = "chain")]
pub use chain::*;
