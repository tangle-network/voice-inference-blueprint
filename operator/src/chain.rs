use blueprint_sdk::std::sync::{Arc, OnceLock};
use blueprint_sdk::std::time::Duration;

use alloy_sol_types::sol;
use blueprint_sdk::macros::debug_job;
use blueprint_sdk::router::Router;
use blueprint_sdk::runner::error::RunnerError;
use blueprint_sdk::runner::BackgroundService;
use blueprint_sdk::tangle::extract::{TangleArg, TangleResult};
use blueprint_sdk::tangle::layers::TangleLayer;
use blueprint_sdk::Job;
use tokio::sync::oneshot;

use crate::{
    config::OperatorConfig, server, speech::SpeechClient, AppStateBuilder, BillingClient,
    NonceStore,
};

// --- ABI types for on-chain job encoding ---

sol! {
    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    /// Input payload ABI-encoded in the Tangle job call.
    struct TTSRequest {
        string input;
        string voice;
        string responseFormat;
    }

    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    /// Output payload ABI-encoded in the Tangle job result.
    struct TTSResult {
        bytes audioData;
        uint32 characterCount;
        string format;
    }
}

// --- Job IDs ---

pub const TTS_JOB: u8 = 0;

// --- Shared state for the on-chain job handler ---

/// Voice engine connection config set by VoiceInferenceServer::start(), read by run_tts.
static VOICE_ENDPOINT: OnceLock<VoiceEndpoint> = OnceLock::new();

struct VoiceEndpoint {
    url: String,
    model: String,
    client: reqwest::Client,
}

/// Called by VoiceInferenceServer to register the voice engine endpoint for on-chain job handlers.
#[allow(clippy::result_large_err)]
fn register_voice_endpoint(
    config: &OperatorConfig,
    speech: &SpeechClient,
) -> Result<(), RunnerError> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(300))
        .build()
        .map_err(|e| RunnerError::Other(format!("failed to build HTTP client: {e}").into()))?;
    let endpoint = VoiceEndpoint {
        url: speech.speech_url(),
        model: config.vllm.model.clone(),
        client,
    };
    let _ = VOICE_ENDPOINT.set(endpoint);
    Ok(())
}

/// Initialize the voice engine endpoint for testing. Call this before submitting jobs
/// when running without the full VoiceInferenceServer background service (e.g. in
/// BlueprintHarness tests with a mock endpoint).
pub fn init_for_testing(base_url: &str, model: &str) {
    let endpoint = VoiceEndpoint {
        url: format!("{base_url}/v1/audio/speech"),
        model: model.to_string(),
        client: reqwest::Client::new(),
    };
    let _ = VOICE_ENDPOINT.set(endpoint);
}

// --- Router ---

pub fn router() -> Router {
    Router::new().route(TTS_JOB, run_tts.layer(TangleLayer))
}

// --- Job handler ---

/// Handle a TTS job submitted on-chain.
#[debug_job]
pub async fn run_tts(
    TangleArg(request): TangleArg<TTSRequest>,
) -> anyhow::Result<TangleResult<TTSResult>> {
    let endpoint = VOICE_ENDPOINT.get().ok_or_else(|| {
        anyhow::anyhow!("voice engine endpoint not registered — VoiceInferenceServer not started",)
    })?;

    let voice = if request.voice.is_empty() {
        "default".to_string()
    } else {
        request.voice
    };

    let response_format = if request.responseFormat.is_empty() {
        "mp3".to_string()
    } else {
        request.responseFormat
    };

    let body = serde_json::json!({
        "model": endpoint.model,
        "input": request.input,
        "voice": voice,
        "response_format": response_format,
        "speed": 1.0,
    });

    let resp = endpoint
        .client
        .post(&endpoint.url)
        .json(&body)
        .send()
        .await
        .map_err(|e| {
            tracing::error!(error = %e, "TTS request failed");
            anyhow::anyhow!("TTS request failed: {e}")
        })?;

    let status = resp.status();
    if !status.is_success() {
        let error_text = resp.text().await.unwrap_or_default();
        tracing::error!(status = %status, error = %error_text, "TTS request returned error");
        return Err(anyhow::anyhow!(format!(
            "TTS request failed with status {status}: {error_text}"
        ),));
    }

    let audio_bytes = resp.bytes().await.map_err(|e| {
        tracing::error!(error = %e, "TTS response read failed");
        anyhow::anyhow!("TTS response read failed: {e}")
    })?;

    let character_count = request.input.len() as u32;

    Ok(TangleResult(TTSResult {
        audioData: audio_bytes.to_vec(),
        characterCount: character_count,
        format: response_format,
    }))
}

// --- Background service: HTTP server + vLLM-Omni subprocess ---

/// Runs the vLLM-Omni subprocess and the OpenAI-compatible HTTP proxy as a
/// [`BackgroundService`]. Includes a watchdog loop that monitors the vLLM
/// process and respawns it if it exits unexpectedly.
#[derive(Clone)]
pub struct VoiceInferenceServer {
    pub config: Arc<OperatorConfig>,
}

impl BackgroundService for VoiceInferenceServer {
    async fn start(&self) -> Result<oneshot::Receiver<Result<(), RunnerError>>, RunnerError> {
        let (tx, rx) = oneshot::channel();
        let config = self.config.clone();

        tokio::spawn(async move {
            // 1. Start the vLLM-Omni subprocess
            let engine_handle = match SpeechClient::spawn(config.clone()).await {
                Ok(h) => Arc::new(h),
                Err(e) => {
                    tracing::error!(error = %e, "failed to spawn vLLM");
                    let _ = tx.send(Err(RunnerError::Other(e.to_string().into())));
                    return;
                }
            };

            tracing::info!("vLLM process started, waiting for readiness");
            if let Err(e) = engine_handle.wait_ready().await {
                tracing::error!(error = %e, "vLLM failed to become ready");
                let _ = tx.send(Err(RunnerError::Other(e.to_string().into())));
                return;
            }
            tracing::info!("vLLM is ready");

            // Register the voice engine endpoint for on-chain job handlers
            if let Err(e) = register_voice_endpoint(&config, &engine_handle) {
                tracing::error!(error = %e, "failed to register voice engine endpoint");
                let _ = tx.send(Err(e));
                return;
            }

            // 2. Build billing client
            let billing_client = match BillingClient::new(&config.tangle, &config.billing) {
                Ok(b) => Arc::new(b),
                Err(e) => {
                    tracing::error!(error = %e, "failed to create billing client");
                    let _ = tx.send(Err(RunnerError::Other(e.to_string().into())));
                    return;
                }
            };

            let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

            let operator_address = billing_client.operator_address();
            let nonce_store = Arc::new(NonceStore::load(config.billing.nonce_store_path.clone()));
            let backend = server::VoiceBackend::new(config.clone(), engine_handle.clone());

            let state = match AppStateBuilder::new()
                .billing(billing_client)
                .nonce_store(nonce_store)
                .server_config(Arc::new(config.server.clone()))
                .billing_config(Arc::new(config.billing.clone()))
                .tangle_config(Arc::new(config.tangle.clone()))
                .operator_address(operator_address)
                .backend(backend)
                .build()
            {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!(error = %e, "failed to build AppState");
                    let _ = tx.send(Err(RunnerError::Other(e.to_string().into())));
                    return;
                }
            };

            match server::start(state, shutdown_rx).await {
                Ok(_join_handle) => {
                    tracing::info!("HTTP server started — background service ready");
                    let _ = tx.send(Ok(()));
                }
                Err(e) => {
                    tracing::error!(error = %e, "failed to start HTTP server");
                    let _ = tx.send(Err(RunnerError::Other(e.to_string().into())));
                    return;
                }
            }

            tokio::select! {
                _ = tokio::signal::ctrl_c() => { let _ = shutdown_tx.send(true); }
                result = engine_handle.supervise(shutdown_tx.subscribe()) => {
                    if let Err(error) = result { tracing::error!(%error, "engine supervision ended"); }
                    let _ = shutdown_tx.send(true);
                }
            }
            engine_handle.shutdown().await;
        });
        Ok(rx)
    }
}
