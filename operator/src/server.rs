//! OpenAI-compatible HTTP server for the vLLM-Omni TTS operator.
//!
//! All shared infrastructure (nonce store, spend-auth validation, x402 headers,
//! metrics, app state container) lives in `tangle-inference-core`. This module
//! only contains:
//!
//! * `VoiceBackend` — the backend attached to `AppState` via `AppStateBuilder`.
//! * Request/response types for the OpenAI TTS wire format.
//! * HTTP handlers that glue the shared billing flow to the vLLM-Omni subprocess.

use blueprint_sdk::std::collections::HashMap;
use blueprint_sdk::std::sync::Arc;
use blueprint_sdk::std::time::Duration;

use axum::{
    body::Body,
    extract::{DefaultBodyLimit, State},
    http::{header, HeaderMap, StatusCode},
    response::Response,
    routing::{get, post},
    Json, Router as HttpRouter,
};
use serde::{Deserialize, Serialize};
use tokio::sync::OwnedSemaphorePermit;
use tokio::task::JoinHandle;
use tower_http::cors::CorsLayer;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;

use tangle_inference_core::server::{
    error_response, extract_x402_spend_auth, payment_required, settle_billing, validate_spend_auth,
};
use tangle_inference_core::{
    detect_gpus, AppState, CostModel, CostParams, GpuInfo, PerCharCostModel, RequestGuard,
    SpendAuthPayload,
};

use crate::config::OperatorConfig;
use crate::voice_engine::VoiceEngine;

/// Backend attached to `AppState` via `AppStateBuilder::backend`. Handlers
/// retrieve it via `state.backend::<VoiceBackend>().unwrap()`.
pub struct VoiceBackend {
    pub config: Arc<OperatorConfig>,
    pub engine: Arc<VoiceEngine>,
    pub cost_model: PerCharCostModel,
}

impl VoiceBackend {
    pub fn new(config: Arc<OperatorConfig>, engine: Arc<VoiceEngine>) -> Self {
        let cost_model = PerCharCostModel {
            price_per_1k_chars: config.vllm.price_per_1k_chars,
        };
        Self {
            config,
            engine,
            cost_model,
        }
    }

    /// Calculate the cost for a request given a character count.
    pub fn calculate_cost(&self, characters: u32) -> u64 {
        let mut extra = HashMap::new();
        extra.insert("characters".to_string(), characters as u64);
        self.cost_model.calculate_cost(&CostParams {
            extra,
            ..Default::default()
        })
    }
}

/// Start the HTTP server with graceful shutdown support, returns a join handle.
pub async fn start(
    state: AppState,
    mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
) -> anyhow::Result<JoinHandle<()>> {
    let backend = state
        .backend::<VoiceBackend>()
        .ok_or_else(|| anyhow::anyhow!("AppState backend is not a VoiceBackend"))?;
    let max_request_body_bytes = state.server_config.max_request_body_bytes;
    let request_timeout_secs = state.server_config.stream_timeout_secs;
    let bind = format!("{}:{}", state.server_config.host, state.server_config.port);
    let _ = backend; // validated before spawning

    let app = HttpRouter::new()
        .route("/v1/audio/speech", post(speech_handler))
        .route("/v1/models", get(list_models))
        .route("/v1/operator", get(operator_info))
        .route("/health", get(health_check))
        .route("/health/gpu", get(gpu_health))
        .route("/metrics", get(metrics_handler))
        .layer(DefaultBodyLimit::max(max_request_body_bytes))
        .layer(TimeoutLayer::new(Duration::from_secs(request_timeout_secs)))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&bind).await?;
    tracing::info!(bind = %bind, "HTTP server listening");

    let handle = tokio::spawn(async move {
        let shutdown_signal = async move {
            let _ = shutdown_rx.wait_for(|&v| v).await;
            tracing::info!("HTTP server received shutdown signal, draining connections");
        };
        if let Err(e) = axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal)
            .await
        {
            tracing::error!(error = %e, "HTTP server error");
        }
    });

    Ok(handle)
}

// --- Request / Response types (OpenAI-compatible TTS) ---

#[derive(Debug, Deserialize)]
pub struct SpeechRequest {
    pub model: Option<String>,
    pub input: String,
    pub voice: Option<String>,
    pub response_format: Option<String>,
    pub speed: Option<f32>,

    /// ShieldedCredits spend authorization (required when billing_required is true).
    /// Can also be provided via x402 headers (X-Payment-Signature).
    pub spend_auth: Option<SpendAuthPayload>,
}

#[derive(Debug, Serialize)]
struct ModelInfo {
    id: String,
    object: String,
    owned_by: String,
}

#[derive(Debug, Serialize)]
struct ModelList {
    object: String,
    data: Vec<ModelInfo>,
}

fn content_type_for_format(format: &str) -> &'static str {
    match format {
        "mp3" => "audio/mpeg",
        "wav" => "audio/wav",
        "ogg" => "audio/ogg",
        "flac" => "audio/flac",
        "opus" => "audio/opus",
        _ => "application/octet-stream",
    }
}

// --- Handlers ---

fn backend_from(state: &AppState) -> &VoiceBackend {
    state
        .backend::<VoiceBackend>()
        .expect("AppState backend is VoiceBackend (checked in server::start)")
}

async fn speech_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(mut req): Json<SpeechRequest>,
) -> Response {
    let backend = backend_from(&state);
    let model_name = req.model.as_deref().unwrap_or(&backend.config.vllm.model);
    let mut metrics_guard = RequestGuard::new(model_name);

    // 1. Acquire semaphore permit
    let _permit: OwnedSemaphorePermit = match state.semaphore.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            return error_response(
                StatusCode::TOO_MANY_REQUESTS,
                "server at capacity".to_string(),
                "rate_limit_error",
                "too_many_requests",
            );
        }
    };

    // 2. x402 flow: if no spend_auth in body, check X-Payment-Signature header
    if req.spend_auth.is_none() {
        if let Some(x402_auth) = extract_x402_spend_auth(&headers) {
            req.spend_auth = Some(x402_auth);
        }
    }

    // 3. Enforce billing requirement — return 402 Payment Required if missing
    if state.billing_config.billing_required && req.spend_auth.is_none() {
        // Estimate for a typical 1000-character request
        let estimated = backend.calculate_cost(1000);
        return payment_required(
            &state.billing_config,
            &state.tangle_config,
            state.operator_address,
            estimated,
        );
    }

    // 4. Validate SpendAuth (signature, account info, nonce replay, etc).
    let preauth_amount: Option<u64> = if let Some(ref spend_auth) = req.spend_auth {
        match validate_spend_auth(&state, spend_auth).await {
            Ok(amt) => Some(amt),
            Err(resp) => return resp,
        }
    } else {
        None
    };

    // 5. Cost sanity check, per-account concurrency, pre-auth, nonce insert.
    if let (Some(ref spend_auth), Some(preauth)) = (&req.spend_auth, preauth_amount) {
        let character_count = req.input.len() as u32;
        let estimated_max_cost = backend.calculate_cost(character_count);
        let preauth_ceiling = estimated_max_cost.saturating_mul(3) / 2;
        if estimated_max_cost > 0 && preauth > preauth_ceiling {
            return error_response(
                StatusCode::BAD_REQUEST,
                format!(
                    "pre-auth amount ({preauth}) exceeds 1.5x estimated max cost ({estimated_max_cost}) — \
                     contract settles full pre-auth, reduce amount to avoid overcharging"
                ),
                "billing_error",
                "excessive_preauth",
            );
        }

        // 5a. Per-account concurrency limit.
        let max_per_account = state.server_config.max_per_account_requests;
        if max_per_account > 0 {
            let mut map = state.active_per_account.write().await;
            let count = map.entry(spend_auth.commitment.clone()).or_insert(0);
            if *count >= max_per_account {
                return error_response(
                    StatusCode::TOO_MANY_REQUESTS,
                    format!("account has {count} active requests (limit: {max_per_account})"),
                    "rate_limit_error",
                    "per_account_limit",
                );
            }
            *count += 1;
        }

        // 5b. Check engine health before committing gas.
        if !backend.engine.is_healthy().await {
            return error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "inference backend is unavailable — billing not initiated".to_string(),
                "upstream_error",
                "engine_unhealthy",
            );
        }

        if let Err(e) = state.billing.authorize_spend(spend_auth).await {
            tracing::error!(error = %e, "authorizeSpend failed");
            return error_response(
                StatusCode::PAYMENT_REQUIRED,
                format!("billing authorization failed: {e}"),
                "billing_error",
                "authorization_failed",
            );
        }

        // Record the nonce as used AFTER successful on-chain authorization
        let nonce_key = (spend_auth.commitment.clone(), spend_auth.nonce);
        state
            .nonce_store
            .insert(
                nonce_key,
                spend_auth.expiry,
                state.billing_config.clock_skew_tolerance_secs,
            )
            .await;
    }

    // 6. Send TTS request to voice engine
    let character_count = req.input.len() as u32;
    let response_format = req.response_format.as_deref().unwrap_or("mp3").to_string();

    let audio_bytes = match backend.engine.speech_synthesis(&req).await {
        Ok(bytes) => bytes,
        Err(e) => {
            tracing::error!(error = %e, "TTS synthesis failed");
            return error_response(
                StatusCode::BAD_GATEWAY,
                format!("upstream TTS error: {e}"),
                "upstream_error",
                "tts_error",
            );
        }
    };

    // Metered as per-char: record character count as prompt tokens for metrics.
    metrics_guard.set_tokens(character_count, 0);
    metrics_guard.set_success();

    // 7. Post-response settlement
    if let (Some(ref spend_auth), Some(preauth)) = (&req.spend_auth, preauth_amount) {
        let actual_cost = backend.calculate_cost(character_count);
        if let Err(e) = settle_billing(&state.billing, spend_auth, preauth, actual_cost).await {
            tracing::error!(error = %e, "on-chain settlement failed — manual recovery required");
        }
    }

    // 8. Return audio bytes with appropriate Content-Type
    let content_type = content_type_for_format(&response_format);
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, content_type)
        .header(header::CONTENT_LENGTH, audio_bytes.len().to_string())
        .body(Body::from(audio_bytes))
        .unwrap_or_else(|e| {
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to build audio response: {e}"),
                "internal_error",
                "response_build_failed",
            )
        })
}

async fn list_models(State(state): State<AppState>) -> Json<ModelList> {
    let backend = backend_from(&state);
    Json(ModelList {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: backend.config.vllm.model.clone(),
            object: "model".to_string(),
            owned_by: "operator".to_string(),
        }],
    })
}

/// Operator info endpoint for discovery. Returns model, pricing, GPU caps, endpoint.
async fn operator_info(State(state): State<AppState>) -> Json<serde_json::Value> {
    let backend = backend_from(&state);
    let gpu_info = detect_gpus().await.unwrap_or_default();
    Json(serde_json::json!({
        "operator": format!("{:#x}", state.operator_address),
        "model": backend.config.vllm.model,
        "pricing": {
            "price_per_1k_chars": backend.config.vllm.price_per_1k_chars,
            "currency": "tsUSD",
        },
        "gpu": {
            "count": backend.config.gpu.expected_gpu_count,
            "min_vram_mib": backend.config.gpu.min_vram_mib,
            "model": backend.config.gpu.gpu_model,
            "detected": gpu_info,
        },
        "server": {
            "max_concurrent_requests": state.server_config.max_concurrent_requests,
            "max_context_length": backend.config.vllm.max_model_len,
        },
        "billing_required": state.billing_config.billing_required,
        "payment_token": state.billing_config.payment_token_address,
    }))
}

async fn health_check(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let backend = backend_from(&state);
    let engine_healthy = backend.engine.is_healthy().await;

    if engine_healthy {
        Ok(Json(serde_json::json!({
            "status": "ok",
            "model": backend.config.vllm.model,
        })))
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}

async fn gpu_health() -> Result<Json<Vec<GpuInfo>>, (StatusCode, String)> {
    match detect_gpus().await {
        Ok(gpus) => Ok(Json(gpus)),
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string())),
    }
}

async fn metrics_handler() -> Response {
    let body = tangle_inference_core::metrics::gather();
    Response::builder()
        .status(StatusCode::OK)
        .header(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )
        .body(Body::from(body))
        .unwrap_or_else(|e| {
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to build metrics response: {e}"),
                "internal_error",
                "response_build_failed",
            )
        })
}
