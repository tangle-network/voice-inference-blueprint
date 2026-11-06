use std::{process::Command, sync::Arc};
use voice_inference::{config::OperatorConfig, server::SpeechRequest, speech::SpeechClient};

#[tokio::test]
async fn speech_endpoint_uses_core_address() {
    if std::env::var_os("VOICE_ENDPOINT_CHILD").is_none() {
        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "speech_endpoint_uses_core_address",
                "--nocapture",
            ])
            .env("VOICE_ENDPOINT_CHILD", "1")
            .env("NO_PROXY", "*")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }

    let listener = tokio::net::TcpListener::bind("[::1]:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let peer = tokio::spawn(async move {
        axum::serve(
            listener,
            axum::Router::new()
                .route("/health", axum::routing::get(|| async { "ok" }))
                .route(
                    "/v1/audio/speech",
                    axum::routing::post(
                        |axum::Json(body): axum::Json<serde_json::Value>| async move {
                            assert_eq!(body["input"], "Read this over IPv6");
                            assert_eq!(body["voice"], "alloy");
                            assert_eq!(body["response_format"], "wav");
                            b"RIFF-public-http-audio".as_slice()
                        },
                    ),
                ),
        )
        .await
        .unwrap();
    });
    let mut config: OperatorConfig =
        serde_json::from_str(include_str!("../../deploy/config.example.json")).unwrap();
    config.vllm.engine.host = "::1".into();
    config.vllm.engine.port = port;
    let speech = SpeechClient::connect(Arc::new(config)).unwrap();
    assert!(speech.is_healthy().await, "core must reach the IPv6 peer");
    let audio = speech
        .speech_synthesis(&SpeechRequest {
            model: None,
            input: "Read this over IPv6".into(),
            voice: Some("alloy".into()),
            response_format: Some("wav".into()),
            speed: None,
            spend_auth: None,
        })
        .await
        .expect("speech must use the same reachable address as core health");
    assert_eq!(audio, b"RIFF-public-http-audio");
    peer.abort();
}
