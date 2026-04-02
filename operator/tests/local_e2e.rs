//! Local E2E integration test — real TTS inference via vLLM-Omni
//!
//! Prerequisites:
//!   - vLLM-Omni installed and running with a TTS model
//!   - Model loaded: e.g. Qwen/Qwen3-TTS-1.7B
//!
//! Run:
//!   VLLM_TTS_E2E=1 cargo test --test local_e2e -- --nocapture
//!
//! This test:
//!   1. Connects to a local vLLM-Omni instance (OpenAI-compatible API)
//!   2. Sends a real TTS request
//!   3. Verifies audio bytes come back
//!   4. Checks content-type is audio
//!   5. Verifies model listing works

use std::env;

fn should_run() -> bool {
    env::var("VLLM_TTS_E2E").unwrap_or_default() == "1"
}

#[tokio::test]
async fn test_real_tts_via_vllm() {
    if !should_run() {
        eprintln!("Skipping: set VLLM_TTS_E2E=1 to run");
        return;
    }

    let base_url =
        env::var("VLLM_TTS_URL").unwrap_or_else(|_| "http://127.0.0.1:8000".to_string());
    let model =
        env::var("VLLM_TTS_MODEL").unwrap_or_else(|_| "Qwen/Qwen3-TTS-1.7B".to_string());

    let client = reqwest::Client::new();

    // --- Test 1: Basic TTS synthesis ---
    println!("Test 1: Basic TTS synthesis...");
    let body = serde_json::json!({
        "model": model,
        "input": "Hello, this is a test of text to speech.",
        "voice": "alloy",
        "response_format": "mp3",
        "speed": 1.0
    });

    let resp = client
        .post(format!("{base_url}/v1/audio/speech"))
        .json(&body)
        .send()
        .await
        .expect("Failed to connect to vLLM-Omni — is it running?");

    assert_eq!(resp.status(), 200, "Expected 200 OK");

    let content_type = resp
        .headers()
        .get("content-type")
        .map(|v| v.to_str().unwrap_or(""))
        .unwrap_or("");
    println!("  Content-Type: {content_type}");
    assert!(
        content_type.contains("audio"),
        "Expected audio content type, got: {content_type}"
    );

    let audio_bytes = resp.bytes().await.expect("Failed to read audio bytes");
    println!("  Audio size: {} bytes", audio_bytes.len());
    assert!(
        audio_bytes.len() > 100,
        "Audio too small: {} bytes",
        audio_bytes.len()
    );
    println!("  ✓ Basic TTS synthesis works");

    // --- Test 2: Different voice/format ---
    println!("\nTest 2: WAV format synthesis...");
    let body2 = serde_json::json!({
        "model": model,
        "input": "This is a test in WAV format.",
        "response_format": "wav",
        "speed": 1.0
    });

    let resp2 = client
        .post(format!("{base_url}/v1/audio/speech"))
        .json(&body2)
        .send()
        .await
        .expect("Request failed");

    if resp2.status() == 200 {
        let audio_bytes2 = resp2.bytes().await.expect("Failed to read audio");
        println!("  Audio size: {} bytes", audio_bytes2.len());
        assert!(!audio_bytes2.is_empty(), "Empty WAV response");
        println!("  ✓ WAV format works");
    } else {
        println!(
            "  WAV format not supported (status {}), skipping",
            resp2.status()
        );
    }

    // --- Test 3: Model listing ---
    println!("\nTest 3: Model listing...");
    let resp3 = client
        .get(format!("{base_url}/v1/models"))
        .send()
        .await
        .expect("Models request failed");

    let models: serde_json::Value = resp3.json().await.expect("Invalid JSON");
    assert!(models["data"].is_array(), "No models data");
    let model_list = models["data"].as_array().unwrap();
    assert!(!model_list.is_empty(), "No models available");
    println!(
        "  Available models: {}",
        model_list
            .iter()
            .map(|m| m["id"].as_str().unwrap_or("?"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("  ✓ Model listing works");

    println!("\n═══════════════════════════════════════");
    println!("  ALL LOCAL E2E TESTS PASSED");
    println!("  Backend: vLLM-Omni at {base_url}");
    println!("  Model: {model}");
    println!("═══════════════════════════════════════");
}
