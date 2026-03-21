pub mod billing;
pub mod config;
pub mod health;
pub mod metrics;
pub mod voice_engine;

pub mod server;

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};
use std::time::Duration;

use alloy_sol_types::sol;
use blueprint_sdk::macros::debug_job;
use blueprint_sdk::router::Router;
use blueprint_sdk::runner::error::RunnerError;
use blueprint_sdk::runner::BackgroundService;
use blueprint_sdk::tangle::extract::{TangleArg, TangleResult};
use blueprint_sdk::tangle::layers::TangleLayer;
use blueprint_sdk::Job;
use tokio::sync::{oneshot, Semaphore};

use crate::config::OperatorConfig;
use crate::voice_engine::VoiceEngine;

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
fn register_voice_endpoint(config: &OperatorConfig) -> Result<(), RunnerError> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(300))
        .build()
        .map_err(|e| RunnerError::Other(format!("failed to build HTTP client: {e}").into()))?;
    let endpoint = VoiceEndpoint {
        url: format!(
            "http://{}:{}/v1/audio/speech",
            config.vllm.host, config.vllm.port
        ),
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
///
/// Uses the voice engine endpoint registered by [`VoiceInferenceServer`] rather than
/// hardcoded values. The shared reqwest::Client is reused across calls.
#[debug_job]
pub async fn run_tts(
    TangleArg(request): TangleArg<TTSRequest>,
) -> Result<TangleResult<TTSResult>, RunnerError> {
    let endpoint = VOICE_ENDPOINT.get().ok_or_else(|| {
        RunnerError::Other(
            "voice engine endpoint not registered — VoiceInferenceServer not started".into(),
        )
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
            RunnerError::Other(format!("TTS request failed: {e}").into())
        })?;

    let status = resp.status();
    if !status.is_success() {
        let error_text = resp.text().await.unwrap_or_default();
        tracing::error!(status = %status, error = %error_text, "TTS request returned error");
        return Err(RunnerError::Other(
            format!("TTS request failed with status {status}: {error_text}").into(),
        ));
    }

    let audio_bytes = resp.bytes().await.map_err(|e| {
        tracing::error!(error = %e, "TTS response read failed");
        RunnerError::Other(format!("TTS response read failed: {e}").into())
    })?;

    let character_count = request.input.len() as u32;

    Ok(TangleResult(TTSResult {
        audioData: audio_bytes.to_vec().into(),
        characterCount: character_count,
        format: response_format,
    }))
}

// --- Background service: HTTP server + vLLM subprocess ---

/// Runs the vLLM-Omni subprocess and the OpenAI-compatible HTTP proxy as a
/// [`BackgroundService`]. This starts before the BlueprintRunner begins
/// polling for on-chain jobs.
///
/// Includes a watchdog loop that monitors the vLLM process and respawns
/// it if it exits unexpectedly.
#[derive(Clone)]
pub struct VoiceInferenceServer {
    pub config: Arc<OperatorConfig>,
}

impl BackgroundService for VoiceInferenceServer {
    async fn start(&self) -> Result<oneshot::Receiver<Result<(), RunnerError>>, RunnerError> {
        let (tx, rx) = oneshot::channel();
        let config = self.config.clone();

        tokio::spawn(async move {
            // 1. Start the vLLM subprocess
            let engine_handle = match VoiceEngine::spawn(config.clone()).await {
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
            if let Err(e) = register_voice_endpoint(&config) {
                tracing::error!(error = %e, "failed to register voice engine endpoint");
                let _ = tx.send(Err(e));
                return;
            }

            // 2. Build billing client
            let billing_client = match billing::BillingClient::new(config.clone()).await {
                Ok(b) => Arc::new(b),
                Err(e) => {
                    tracing::error!(error = %e, "failed to create billing client");
                    let _ = tx.send(Err(RunnerError::Other(e.to_string().into())));
                    return;
                }
            };

            // 3. Build semaphore from config (0 = unlimited)
            let max_concurrent = config.server.max_concurrent_requests;
            let semaphore = Arc::new(if max_concurrent == 0 {
                Semaphore::new(Semaphore::MAX_PERMITS)
            } else {
                Semaphore::new(max_concurrent)
            });

            // 4. Create shutdown channel for graceful shutdown
            let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

            // 5. Start the HTTP server
            let operator_address = billing_client.operator_address();
            let nonce_store = Arc::new(server::NonceStore::load(
                config.billing.nonce_store_path.clone(),
            ));
            let state = server::AppState {
                config: config.clone(),
                engine: engine_handle.clone(),
                billing: billing_client,
                semaphore,
                nonce_store,
                active_per_account: Arc::new(RwLock::new(HashMap::new())),
                operator_address,
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

            // 6. Watchdog loop: monitor vLLM process and respawn on crash.
            let mut _live_handle: Option<VoiceEngine> = None;
            loop {
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(10)) => {}
                    _ = tokio::signal::ctrl_c() => {
                        tracing::info!("received shutdown signal, stopping watchdog");
                        let _ = shutdown_tx.send(true);
                        engine_handle.shutdown().await;
                        if let Some(ref h) = _live_handle {
                            h.shutdown().await;
                        }
                        return;
                    }
                }

                if !engine_handle.is_healthy().await {
                    tracing::error!("vLLM health check failed — attempting respawn");

                    // Shut down the old process
                    engine_handle.shutdown().await;

                    // Attempt respawn with backoff
                    let mut respawn_delay = Duration::from_secs(5);
                    loop {
                        tracing::info!(delay_secs = respawn_delay.as_secs(), "respawning vLLM");
                        match VoiceEngine::spawn(config.clone()).await {
                            Ok(new_handle) => {
                                match new_handle.wait_ready().await {
                                    Ok(()) => {
                                        tracing::info!("vLLM respawned successfully");
                                        _live_handle = Some(new_handle);
                                        break;
                                    }
                                    Err(e) => {
                                        tracing::error!(error = %e, "respawned vLLM failed readiness");
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::error!(error = %e, "failed to respawn vLLM");
                            }
                        }
                        tokio::time::sleep(respawn_delay).await;
                        respawn_delay = (respawn_delay * 2).min(Duration::from_secs(120));
                    }
                }
            }
        });

        Ok(rx)
    }
}
