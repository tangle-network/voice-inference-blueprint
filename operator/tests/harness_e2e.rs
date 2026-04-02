//! BlueprintHarness E2E test — full BPM lifecycle with mock vLLM-Omni.
//!
//! This test exercises the complete blueprint-manager flow:
//!   1. Boot Anvil with seeded Tangle contracts (LocalTestnet)
//!   2. Wire the Router + TangleLayer into a BlueprintRunner
//!   3. Start a mock vLLM-Omni HTTP server (no GPU required)
//!   4. Submit a TTS job on-chain
//!   5. Verify the result is returned on-chain
//!   6. Shutdown cleanly
//!
//! Run:
//!   cargo test --test harness_e2e -- --nocapture
//!
//! Note: requires tnt-core contract artifacts (LocalTestnet broadcast).
//! If missing, the test is skipped gracefully.

use alloy_primitives::Bytes;
use alloy_sol_types::SolValue;
use anyhow::{Context, Result};
use axum::{
    body::Body,
    http::{header, Response, StatusCode},
    routing::post,
    Json, Router as HttpRouter,
};
use blueprint_anvil_testing_utils::{missing_tnt_core_artifacts, BlueprintHarness};
use serde_json::Value;
use std::time::Duration;
use tokio::time::timeout;
use voice_inference::{init_for_testing, router, TTSRequest, TTSResult, TTS_JOB};

const TEST_TIMEOUT: Duration = Duration::from_secs(120);
const MOCK_MODEL: &str = "test-model";

/// Start a mock vLLM-Omni server that responds to /v1/audio/speech
/// with fake audio bytes. Returns the base URL.
async fn start_mock_vllm() -> String {
    let app = HttpRouter::new().route("/v1/audio/speech", post(mock_speech));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("failed to bind mock vLLM");
    let port = listener.local_addr().unwrap().port();
    let base_url = format!("http://127.0.0.1:{port}");

    tokio::spawn(async move {
        axum::serve(listener, app).await.ok();
    });

    // Wait for listener to be ready
    tokio::time::sleep(Duration::from_millis(50)).await;
    base_url
}

async fn mock_speech(Json(body): Json<Value>) -> Response<Body> {
    let input = body["input"].as_str().unwrap_or("");
    // Generate fake audio bytes proportional to input length
    let audio_bytes: Vec<u8> = vec![0xFF, 0xFB, 0x90, 0x00]
        .into_iter()
        .cycle()
        .take(input.len().max(4))
        .collect();

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "audio/mpeg")
        .body(Body::from(audio_bytes))
        .unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_tts_job_lifecycle() -> Result<()> {
    timeout(TEST_TIMEOUT, async {
        // 1. Start mock vLLM-Omni
        let mock_url = start_mock_vllm().await;
        println!("Mock vLLM-Omni at {mock_url}");

        // 2. Boot harness (Anvil + seeded contracts + BlueprintRunner)
        let harness = match BlueprintHarness::builder(router())
            .poll_interval(Duration::from_millis(50))
            .with_pre_spawn_hook(move |_env| {
                let url = mock_url.clone();
                async move {
                    // Initialize the job handler statics to point at mock vLLM
                    init_for_testing(&url, MOCK_MODEL);
                    Ok(())
                }
            })
            .spawn()
            .await
        {
            Ok(h) => h,
            Err(err) => {
                if missing_tnt_core_artifacts(&err) {
                    eprintln!("Skipping: tnt-core artifacts not found: {err}");
                    return Ok(());
                }
                return Err(err);
            }
        };

        println!(
            "Harness ready: blueprint={}, service={}",
            harness.blueprint_id(),
            harness.service_id()
        );

        // 3. Verify service is active
        let client = harness.client();
        let service = client.get_service(harness.service_id()).await?;
        println!("Service status: {:?}", service.status);

        // 4. Submit a TTS job
        let request = TTSRequest {
            input: "Hello, world! This is a test.".to_string(),
            voice: "alloy".to_string(),
            responseFormat: "mp3".to_string(),
        };
        let payload = request.abi_encode();
        println!("Submitting TTS job ({} bytes)...", payload.len());

        let submission = harness
            .submit_job(TTS_JOB, Bytes::from(payload))
            .await
            .context("failed to submit TTS job")?;
        println!("Job submitted: call_id={}", submission.call_id);

        // 5. Wait for result
        let output = harness
            .wait_for_job_result(submission)
            .await
            .context("failed to get job result")?;
        println!("Got result ({} bytes)", output.len());

        // 6. Decode and verify
        let result =
            TTSResult::abi_decode(&output).context("failed to decode TTSResult")?;
        println!("  audioData: {} bytes", result.audioData.len());
        println!("  characterCount: {}", result.characterCount);
        println!("  format: {}", result.format);

        assert!(
            !result.audioData.is_empty(),
            "expected non-empty audio data"
        );
        assert!(result.characterCount > 0, "expected nonzero character count");
        assert_eq!(result.format, "mp3");

        println!("\n  ✓ Single TTS job lifecycle passed");

        // 7. Submit multiple sequential jobs
        let inputs = ["Hello", "Explain gravity in simple terms", "Write a haiku about the moon"];

        for input in &inputs {
            let request = TTSRequest {
                input: input.to_string(),
                voice: "alloy".to_string(),
                responseFormat: "mp3".to_string(),
            };
            let submission = harness
                .submit_job(TTS_JOB, Bytes::from(request.abi_encode()))
                .await?;
            let output = harness.wait_for_job_result(submission).await?;
            let result = TTSResult::abi_decode(&output)?;

            assert!(
                !result.audioData.is_empty(),
                "job for '{input}' returned empty audio"
            );
            println!("  ✓ Job '{input}' → {} bytes audio", result.audioData.len());
        }

        println!("  ✓ Multiple sequential TTS jobs passed");

        // 8. Shutdown
        harness.shutdown().await;
        Ok(())
    })
    .await
    .context("test timed out")?
}

// Note: Only one harness test per file due to OnceLock statics in the
// TTS handler. The init_for_testing() call binds the mock URL once
// per process. Additional tests should be added as steps within
// test_tts_job_lifecycle above.
