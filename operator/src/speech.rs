use crate::{config::OperatorConfig, server::SpeechRequest};
use std::sync::Arc;
use tangle_inference_core::engine::EngineProcess;

/// Speech request adapter; serving process ownership stays in core.
pub struct SpeechClient {
    process: EngineProcess,
    config: Arc<OperatorConfig>,
}
impl SpeechClient {
    pub fn connect(config: Arc<OperatorConfig>) -> anyhow::Result<Self> {
        Ok(Self {
            process: EngineProcess::connect(config.vllm.engine.clone())?,
            config,
        })
    }
    pub async fn spawn(config: Arc<OperatorConfig>) -> anyhow::Result<Self> {
        let mut engine = config.vllm.engine.clone();
        engine
            .extra_args
            .extend(["--served-model-name".into(), engine.model.clone()]);
        Ok(Self {
            process: EngineProcess::spawn(engine).await?,
            config,
        })
    }
    pub async fn wait_ready(&self) -> anyhow::Result<()> {
        self.process.wait_ready().await
    }
    pub async fn is_healthy(&self) -> bool {
        self.process.is_healthy().await
    }
    pub async fn shutdown(&self) {
        self.process.shutdown().await;
    }
    pub async fn supervise(
        &self,
        shutdown: tokio::sync::watch::Receiver<bool>,
    ) -> anyhow::Result<()> {
        self.process.supervise(shutdown).await
    }
    fn speech_url(&self) -> String {
        format!(
            "http://{}:{}/v1/audio/speech",
            self.config.vllm.host, self.config.vllm.port
        )
    }

    /// Proxy a speech synthesis request to vLLM-Omni, returning raw audio bytes.
    pub async fn speech_synthesis(&self, req: &SpeechRequest) -> anyhow::Result<Vec<u8>> {
        let body = serde_json::json!({
            "model": req.model.as_deref().unwrap_or(&self.config.vllm.model),
            "input": req.input,
            "voice": req.voice,
            "response_format": req.response_format.as_deref().unwrap_or("mp3"),
            "speed": req.speed.unwrap_or(1.0),
        });

        let resp = self
            .process
            .client()
            .post(self.speech_url())
            .json(&body)
            .send()
            .await?
            .error_for_status()?;

        let bytes = resp.bytes().await?;
        Ok(bytes.to_vec())
    }
}
