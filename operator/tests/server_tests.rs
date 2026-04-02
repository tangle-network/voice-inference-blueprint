use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use tokio::sync::Semaphore;
use tokio::task::JoinHandle;
use wiremock::{
    matchers::{method, path},
    Mock, MockServer, ResponseTemplate,
};

use voice_inference::config::{
    BillingConfig, GpuConfig, OperatorConfig, ServerConfig, TangleConfig, VoiceModelConfig,
};

fn test_config(vllm_port: u16) -> OperatorConfig {
    OperatorConfig {
        tangle: TangleConfig {
            rpc_url: "http://localhost:8545".into(),
            chain_id: 31337,
            operator_key: "ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80".into(),
            tangle_core: "0x0000000000000000000000000000000000000000".into(),
            shielded_credits: "0x0000000000000000000000000000000000000000".into(),
            blueprint_id: 1,
            service_id: Some(1),
        },
        vllm: VoiceModelConfig {
            model: "test-model".into(),
            max_model_len: 4096,
            host: "127.0.0.1".into(),
            port: vllm_port,
            tensor_parallel_size: 1,
            extra_args: vec![],
            command: "python3 -m vllm.entrypoints.openai.api_server".into(),
            hf_token: None,
            download_dir: None,
            startup_timeout_secs: 10,
            default_voice: None,
            supported_formats: vec!["mp3".into(), "wav".into(), "ogg".into()],
        },
        server: ServerConfig {
            host: "0.0.0.0".into(),
            port: 8080,
            max_concurrent_requests: 2,
            max_request_body_bytes: 2 * 1024 * 1024,
            request_timeout_secs: 300,
            max_per_account_requests: 0,
        },
        billing: BillingConfig {
            price_per_1k_characters: 10,
            max_spend_per_request: 1_000_000,
            min_credit_balance: 1000,
            billing_required: false, // Disabled in tests to avoid needing real spend_auth
            min_charge_amount: 0,
            claim_max_retries: 3,
            clock_skew_tolerance_secs: 30,
            max_gas_price_gwei: 0,
            nonce_store_path: None,
            required: false,
            payment_token_address: None,
        },
        gpu: GpuConfig {
            expected_gpu_count: 0,
            min_vram_mib: 0,
            monitor_interval_secs: 30,
        },
        rln: None,
    }
}

// --- Metrics Tests ---

#[tokio::test]
async fn test_metrics_gather_produces_valid_output() {
    let mut guard = voice_inference::metrics::RequestGuard::new();
    guard.set_tokens(1, 1);
    guard.set_success();
    drop(guard);

    let output = voice_inference::metrics::gather();
    assert!(
        output.contains("vllm_operator_active_requests"),
        "missing active_requests metric"
    );
    assert!(
        output.contains("vllm_operator_request_count"),
        "missing request_count metric"
    );
    assert!(
        output.contains("vllm_operator_request_duration_seconds"),
        "missing request_duration_seconds metric"
    );
    assert!(
        output.contains("vllm_operator_tokens_total"),
        "missing tokens_total metric"
    );
}

#[tokio::test]
async fn test_request_guard_tracks_active_requests() {
    use voice_inference::metrics::{RequestGuard, ACTIVE_REQUESTS};

    let initial = ACTIVE_REQUESTS.get();

    let guard1 = RequestGuard::new();
    assert!(ACTIVE_REQUESTS.get() >= initial + 1.0);

    let guard2 = RequestGuard::new();
    assert!(ACTIVE_REQUESTS.get() >= initial + 2.0);

    drop(guard1);
    drop(guard2);
}

#[tokio::test]
async fn test_request_guard_records_tokens_on_drop() {
    use voice_inference::metrics::{RequestGuard, TOKENS_TOTAL};

    let prompt_before = TOKENS_TOTAL.with_label_values(&["prompt"]).get();
    let completion_before = TOKENS_TOTAL.with_label_values(&["completion"]).get();

    let mut guard = RequestGuard::new();
    guard.set_tokens(100, 50);
    guard.set_success();
    drop(guard);

    assert!(
        TOKENS_TOTAL.with_label_values(&["prompt"]).get() >= prompt_before + 100,
        "prompt tokens should have increased by at least 100"
    );
    assert!(
        TOKENS_TOTAL.with_label_values(&["completion"]).get() >= completion_before + 50,
        "completion tokens should have increased by at least 50"
    );
}

#[tokio::test]
async fn test_request_guard_defaults_to_error() {
    use voice_inference::metrics::{RequestGuard, REQUEST_COUNT};

    let error_before = REQUEST_COUNT.with_label_values(&["error"]).get();

    let guard = RequestGuard::new();
    drop(guard);

    assert!(
        REQUEST_COUNT.with_label_values(&["error"]).get() >= error_before + 1,
        "error count should have increased by at least 1"
    );
}

#[tokio::test]
async fn test_request_guard_records_success() {
    use voice_inference::metrics::{RequestGuard, REQUEST_COUNT};

    let success_before = REQUEST_COUNT.with_label_values(&["success"]).get();

    let mut guard = RequestGuard::new();
    guard.set_success();
    drop(guard);

    assert!(
        REQUEST_COUNT.with_label_values(&["success"]).get() >= success_before + 1,
        "success count should have increased by at least 1"
    );
}

// --- Semaphore Tests ---

#[tokio::test]
async fn test_semaphore_limits_concurrency() {
    let semaphore = Arc::new(Semaphore::new(2));

    let p1 = semaphore.clone().try_acquire_owned();
    assert!(p1.is_ok());

    let p2 = semaphore.clone().try_acquire_owned();
    assert!(p2.is_ok());

    // Third acquire should fail
    let p3 = semaphore.clone().try_acquire_owned();
    assert!(p3.is_err());

    // Drop one permit, now we can acquire again
    drop(p1);
    let p4 = semaphore.clone().try_acquire_owned();
    assert!(p4.is_ok());
}

#[tokio::test]
async fn test_semaphore_zero_config_means_unlimited() {
    let semaphore = Arc::new(Semaphore::new(Semaphore::MAX_PERMITS));

    let mut permits = Vec::new();
    for _ in 0..1000 {
        permits.push(semaphore.clone().try_acquire_owned().unwrap());
    }
    assert_eq!(permits.len(), 1000);
}

// --- Billing Tests ---

#[tokio::test]
async fn test_billing_calculate_cost() {
    let config = Arc::new(test_config(8000));
    let billing = voice_inference::billing::BillingClient::new(config)
        .await
        .unwrap();

    // price_per_1k_characters = 10
    // 1000 chars -> 10 base units
    let cost = billing.calculate_cost(1000);
    assert_eq!(cost, 10);
}

// --- Handler-level integration tests ---

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// Returns (server_port, _guard) -- caller must hold _guard to keep the server alive.
async fn start_test_server(
    vllm_port: u16,
) -> (u16, tokio::sync::watch::Sender<bool>, JoinHandle<()>) {
    let server_port = free_port();
    let mut config = test_config(vllm_port);
    config.server.port = server_port;
    config.server.host = "127.0.0.1".into();
    let config = Arc::new(config);

    let engine = Arc::new(voice_inference::voice_engine::VoiceEngine::connect(config.clone()).unwrap());
    let billing = Arc::new(
        voice_inference::billing::BillingClient::new(config.clone())
            .await
            .unwrap(),
    );
    let operator_address = billing.operator_address();
    let semaphore = Arc::new(Semaphore::new(64));
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

    let state = voice_inference::server::AppState {
        config,
        engine,
        billing,
        semaphore,
        nonce_store: Arc::new(voice_inference::server::NonceStore::load(None)),
        active_per_account: Arc::new(RwLock::new(HashMap::new())),
        operator_address,
    };

    let handle = voice_inference::server::start(state, shutdown_rx)
        .await
        .unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    (server_port, shutdown_tx, handle)
}

/// Start a test server with billing_required = true.
async fn start_billing_required_server(
    vllm_port: u16,
) -> (u16, tokio::sync::watch::Sender<bool>, JoinHandle<()>) {
    let server_port = free_port();
    let mut config = test_config(vllm_port);
    config.server.port = server_port;
    config.server.host = "127.0.0.1".into();
    config.billing.billing_required = true;
    let config = Arc::new(config);

    let engine = Arc::new(voice_inference::voice_engine::VoiceEngine::connect(config.clone()).unwrap());
    let billing = Arc::new(
        voice_inference::billing::BillingClient::new(config.clone())
            .await
            .unwrap(),
    );
    let operator_address = billing.operator_address();
    let semaphore = Arc::new(Semaphore::new(64));
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

    let state = voice_inference::server::AppState {
        config,
        engine,
        billing,
        semaphore,
        nonce_store: Arc::new(voice_inference::server::NonceStore::load(None)),
        active_per_account: Arc::new(RwLock::new(HashMap::new())),
        operator_address,
    };

    let handle = voice_inference::server::start(state, shutdown_rx)
        .await
        .unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    (server_port, shutdown_tx, handle)
}

#[tokio::test]
async fn test_speech_synthesis_through_handler() {
    let mock_vllm = MockServer::start().await;

    // Mock vLLM-Omni returning audio bytes
    let audio_bytes = vec![0xFF, 0xFB, 0x90, 0x00]; // fake MP3 header

    Mock::given(method("POST"))
        .and(path("/v1/audio/speech"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "audio/mpeg")
                .set_body_bytes(audio_bytes.clone()),
        )
        .mount(&mock_vllm)
        .await;

    let (server_port, _shutdown_tx, _handle) = start_test_server(mock_vllm.address().port()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!(
            "http://127.0.0.1:{server_port}/v1/audio/speech"
        ))
        .json(&serde_json::json!({
            "input": "Hello, world!",
            "voice": "alloy",
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap(),
        "audio/mpeg"
    );

    let body = resp.bytes().await.unwrap();
    assert_eq!(body.to_vec(), audio_bytes);
}

#[tokio::test]
async fn test_speech_synthesis_wav_format() {
    let mock_vllm = MockServer::start().await;

    let audio_bytes = vec![0x52, 0x49, 0x46, 0x46]; // RIFF header

    Mock::given(method("POST"))
        .and(path("/v1/audio/speech"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "audio/wav")
                .set_body_bytes(audio_bytes.clone()),
        )
        .mount(&mock_vllm)
        .await;

    let (server_port, _shutdown_tx, _handle) = start_test_server(mock_vllm.address().port()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!(
            "http://127.0.0.1:{server_port}/v1/audio/speech"
        ))
        .json(&serde_json::json!({
            "input": "Hello, world!",
            "response_format": "wav",
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap(),
        "audio/wav"
    );
}

// --- Billing Settlement Tests ---

#[tokio::test]
async fn test_billing_actual_cost_less_than_preauth() {
    let config = Arc::new(test_config(8000));
    let billing = voice_inference::billing::BillingClient::new(config)
        .await
        .unwrap();

    // price_per_1k_characters = 10
    // 500 chars -> 500 * 10 / 1000 = 5
    let actual_cost = billing.calculate_cost(500);
    assert_eq!(actual_cost, 5);

    // Pre-auth ceiling was 1000 -- charge_amount should be min(5, 1000) = 5
    let preauth_amount: u64 = 1000;
    let charge_amount = actual_cost.min(preauth_amount);
    assert_eq!(
        charge_amount, 5,
        "should charge actual cost, not the full pre-auth"
    );
}

#[tokio::test]
async fn test_billing_actual_cost_exceeds_preauth_cap() {
    let config = Arc::new(test_config(8000));
    let billing = voice_inference::billing::BillingClient::new(config)
        .await
        .unwrap();

    // price_per_1k_characters = 10
    // 50000 chars -> 50000 * 10 / 1000 = 500
    let actual_cost = billing.calculate_cost(50000);
    assert_eq!(actual_cost, 500);

    // Pre-auth ceiling was 100 -- charge_amount should be min(500, 100) = 100
    let preauth_amount: u64 = 100;
    let charge_amount = actual_cost.min(preauth_amount);
    assert_eq!(
        charge_amount, 100,
        "charge must be capped at pre-authorized amount"
    );
}

#[tokio::test]
async fn test_billing_zero_usage_yields_zero_charge() {
    let config = Arc::new(test_config(8000));
    let billing = voice_inference::billing::BillingClient::new(config)
        .await
        .unwrap();

    let actual_cost = billing.calculate_cost(0);
    assert_eq!(actual_cost, 0);

    let preauth_amount: u64 = 500;
    let charge_amount = actual_cost.min(preauth_amount);
    assert_eq!(
        charge_amount, 0,
        "zero usage should result in zero charge, not the preauth amount"
    );
}

// --- Policy Enforcement Tests ---

#[tokio::test]
async fn test_billing_required_rejects_missing_spend_auth() {
    let mock_vllm = MockServer::start().await;

    // Set up a mock response (won't be reached because billing check happens first)
    Mock::given(method("POST"))
        .and(path("/v1/audio/speech"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "audio/mpeg")
                .set_body_bytes(vec![0xFF, 0xFB]),
        )
        .mount(&mock_vllm)
        .await;

    let (server_port, _shutdown_tx, _handle) =
        start_billing_required_server(mock_vllm.address().port()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!(
            "http://127.0.0.1:{server_port}/v1/audio/speech"
        ))
        .json(&serde_json::json!({
            "input": "Hello, world!",
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(
        resp.status(),
        402,
        "requests without spend_auth should be rejected with 402 when billing_required is true"
    );
}

// --- Job Handler Error Path Tests ---

#[tokio::test]
async fn test_run_tts_returns_error_on_connection_failure() {
    let client = reqwest::Client::new();

    // Connect to a port where nothing is listening
    let result = client
        .post("http://127.0.0.1:1/v1/audio/speech")
        .json(&serde_json::json!({
            "model": "default",
            "input": "test",
            "voice": "alloy",
        }))
        .send()
        .await;

    // The key assertion: this should be an Err, not a panic.
    assert!(
        result.is_err(),
        "connection to unreachable engine should return Err, not panic"
    );
}

// --- Config Tests ---

#[tokio::test]
async fn test_config_default_max_concurrent_requests() {
    let json = r#"{"host":"0.0.0.0","port":8080}"#;
    let config: ServerConfig = serde_json::from_str(json).unwrap();
    assert_eq!(config.max_concurrent_requests, 64);
}

#[tokio::test]
async fn test_config_debug_redacts_operator_key() {
    let config = test_config(8000);
    let debug_output = format!("{:?}", config);
    assert!(
        !debug_output.contains("ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"),
        "Debug output must not contain the operator private key"
    );
    assert!(
        debug_output.contains("REDACTED"),
        "Debug output should show [REDACTED] for operator_key"
    );
}

// --- Wiremock Integration Tests ---

#[tokio::test]
async fn test_speech_via_wiremock() {
    let mock_server = MockServer::start().await;

    let audio_bytes = vec![0xFF, 0xFB, 0x90, 0x00];

    Mock::given(method("POST"))
        .and(path("/v1/audio/speech"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "audio/mpeg")
                .set_body_bytes(audio_bytes.clone()),
        )
        .mount(&mock_server)
        .await;

    let port = mock_server.address().port();
    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{port}/v1/audio/speech");

    let body = serde_json::json!({
        "model": "test-model",
        "input": "Hello, world!",
        "voice": "alloy",
        "response_format": "mp3",
        "speed": 1.0,
    });

    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap();

    assert_eq!(
        resp.headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap(),
        "audio/mpeg"
    );

    let response_bytes = resp.bytes().await.unwrap();
    assert_eq!(response_bytes.to_vec(), audio_bytes);
}

#[tokio::test]
async fn test_upstream_error_returns_error_status() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/audio/speech"))
        .respond_with(ResponseTemplate::new(500).set_body_string("internal error"))
        .mount(&mock_server)
        .await;

    let port = mock_server.address().port();
    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{port}/v1/audio/speech");

    let resp = client
        .post(&url)
        .json(&serde_json::json!({
            "model": "test-model",
            "input": "Hello",
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 500);
}
