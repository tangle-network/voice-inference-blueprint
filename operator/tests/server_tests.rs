use std::sync::Arc;

use tokio::sync::Semaphore;
use tokio::task::JoinHandle;
use wiremock::{
    matchers::{method, path},
    Mock, MockServer, ResponseTemplate,
};

use voice_inference::config::{
    BillingConfig, GpuConfig, OperatorConfig, ServerConfig, TangleConfig, VoiceConfig,
};
use voice_inference::server::VoiceBackend;
use voice_inference::{AppStateBuilder, BillingClient, NonceStore};

fn test_config(vllm_port: u16) -> OperatorConfig {
    OperatorConfig {
        tangle: TangleConfig {
            rpc_url: "http://localhost:8545".into(),
            chain_id: 31337,
            operator_key: "ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80".into(),
            shielded_credits: "0x0000000000000000000000000000000000000000".into(),
            blueprint_id: 1,
            service_id: Some(1),
        },
        vllm: VoiceConfig {
            model: "test-model".into(),
            max_model_len: 4096,
            host: "127.0.0.1".into(),
            port: vllm_port,
            tensor_parallel_size: 1,
            price_per_1k_chars: 10,
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
            stream_timeout_secs: 300,
            idle_chunk_timeout_secs: 30,
            max_line_buf_bytes: 1024 * 1024,
            max_per_account_requests: 0,
        },
        billing: BillingConfig {
            billing_required: false,
            max_spend_per_request: 1_000_000,
            min_credit_balance: 1000,
            min_charge_amount: 0,
            claim_max_retries: 3,
            clock_skew_tolerance_secs: 30,
            max_gas_price_gwei: 0,
            nonce_store_path: None,
            payment_token_address: None,
        },
        gpu: GpuConfig {
            expected_gpu_count: 0,
            min_vram_mib: 0,
            monitor_interval_secs: 30,
            gpu_model: None,
        },
        stt: None,
        whisper: None,
        rln: None,
    }
}

// --- Metrics Tests ---

#[tokio::test]
async fn test_metrics_gather_produces_valid_output() {
    let mut guard = voice_inference::metrics::RequestGuard::new("test-model");
    guard.set_tokens(1, 0);
    guard.set_success();
    drop(guard);

    let output = voice_inference::metrics::gather();
    assert!(
        output.contains("tangle_operator_active_requests"),
        "missing active_requests metric"
    );
    assert!(
        output.contains("tangle_operator_requests_total"),
        "missing requests_total metric"
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

    let p3 = semaphore.clone().try_acquire_owned();
    assert!(p3.is_err());

    drop(p1);
    let p4 = semaphore.clone().try_acquire_owned();
    assert!(p4.is_ok());
}

// --- Cost Model Tests ---

#[tokio::test]
async fn test_backend_calculate_cost() {
    let config = Arc::new(test_config(8000));
    let engine =
        Arc::new(voice_inference::voice_engine::VoiceEngine::connect(config.clone()).unwrap());
    let backend = VoiceBackend::new(config, engine);

    // price_per_1k_chars = 10
    // 1000 chars -> 10 base units
    assert_eq!(backend.calculate_cost(1000), 10);
    // 500 chars -> 5
    assert_eq!(backend.calculate_cost(500), 5);
    // 0 chars -> 0
    assert_eq!(backend.calculate_cost(0), 0);
    // 50000 chars -> 500
    assert_eq!(backend.calculate_cost(50000), 500);
}

#[tokio::test]
async fn test_billing_charge_capped_at_preauth() {
    let config = Arc::new(test_config(8000));
    let engine =
        Arc::new(voice_inference::voice_engine::VoiceEngine::connect(config.clone()).unwrap());
    let backend = VoiceBackend::new(config, engine);

    let actual_cost = backend.calculate_cost(50000); // 500
    let preauth_amount: u64 = 100;
    let charge_amount = actual_cost.min(preauth_amount);
    assert_eq!(
        charge_amount, 100,
        "charge must be capped at pre-authorized amount"
    );
}

#[tokio::test]
async fn test_billing_charge_uses_actual_when_below_preauth() {
    let config = Arc::new(test_config(8000));
    let engine =
        Arc::new(voice_inference::voice_engine::VoiceEngine::connect(config.clone()).unwrap());
    let backend = VoiceBackend::new(config, engine);

    let actual_cost = backend.calculate_cost(500); // 5
    let preauth_amount: u64 = 1000;
    let charge_amount = actual_cost.min(preauth_amount);
    assert_eq!(
        charge_amount, 5,
        "should charge actual cost, not the full pre-auth"
    );
}

// --- Handler-level integration tests ---

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

async fn build_test_state(config: Arc<OperatorConfig>) -> voice_inference::AppState {
    let engine =
        Arc::new(voice_inference::voice_engine::VoiceEngine::connect(config.clone()).unwrap());
    let billing = Arc::new(BillingClient::new(&config.tangle, &config.billing).unwrap());
    let operator_address = billing.operator_address();
    let nonce_store = Arc::new(NonceStore::load(None));
    let backend = VoiceBackend::new(config.clone(), engine);

    AppStateBuilder::new()
        .billing(billing)
        .nonce_store(nonce_store)
        .server_config(Arc::new(config.server.clone()))
        .billing_config(Arc::new(config.billing.clone()))
        .tangle_config(Arc::new(config.tangle.clone()))
        .operator_address(operator_address)
        .max_concurrent(64)
        .backend(backend)
        .build()
        .unwrap()
}

async fn start_test_server(
    vllm_port: u16,
) -> (u16, tokio::sync::watch::Sender<bool>, JoinHandle<()>) {
    let server_port = free_port();
    let mut config = test_config(vllm_port);
    config.server.port = server_port;
    config.server.host = "127.0.0.1".into();
    let config = Arc::new(config);

    let state = build_test_state(config).await;
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

    let handle = voice_inference::server::start(state, shutdown_rx)
        .await
        .unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    (server_port, shutdown_tx, handle)
}

async fn start_billing_required_server(
    vllm_port: u16,
) -> (u16, tokio::sync::watch::Sender<bool>, JoinHandle<()>) {
    let server_port = free_port();
    let mut config = test_config(vllm_port);
    config.server.port = server_port;
    config.server.host = "127.0.0.1".into();
    config.billing.billing_required = true;
    let config = Arc::new(config);

    let state = build_test_state(config).await;
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

    let handle = voice_inference::server::start(state, shutdown_rx)
        .await
        .unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    (server_port, shutdown_tx, handle)
}

#[tokio::test]
async fn test_speech_synthesis_through_handler() {
    let mock_vllm = MockServer::start().await;

    let audio_bytes = vec![0xFF, 0xFB, 0x90, 0x00];

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
        .post(format!("http://127.0.0.1:{server_port}/v1/audio/speech"))
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

    let audio_bytes = vec![0x52, 0x49, 0x46, 0x46];

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
        .post(format!("http://127.0.0.1:{server_port}/v1/audio/speech"))
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

#[tokio::test]
async fn test_billing_required_rejects_missing_spend_auth() {
    let mock_vllm = MockServer::start().await;

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
        .post(format!("http://127.0.0.1:{server_port}/v1/audio/speech"))
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
async fn test_upstream_error_returns_error_status() {
    let mock_vllm = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/audio/speech"))
        .respond_with(ResponseTemplate::new(500).set_body_string("internal error"))
        .mount(&mock_vllm)
        .await;

    let (server_port, _shutdown_tx, _handle) = start_test_server(mock_vllm.address().port()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://127.0.0.1:{server_port}/v1/audio/speech"))
        .json(&serde_json::json!({
            "input": "Hello",
        }))
        .send()
        .await
        .unwrap();

    // The backend returned 500; our handler maps upstream errors to 502.
    assert_eq!(resp.status(), 502);
}
