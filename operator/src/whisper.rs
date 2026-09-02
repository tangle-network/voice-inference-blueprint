//! Whisper STT backend — subprocess or HTTP server mode.
//!
//! Subprocess mode: shells out to `insanely-fast-whisper` per request.
//! Server mode: POSTs audio to a pre-running whisper HTTP endpoint.

use serde::{Deserialize, Serialize};
use tokio::process::Command;

use crate::config::WhisperConfig;

/// Transcription result returned to the HTTP handler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResult {
    pub text: String,
    pub language: String,
    pub duration: f64,
}

/// Output chunk from insanely-fast-whisper JSON output.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct WhisperChunk {
    text: String,
    timestamp: (f64, Option<f64>),
}

/// Top-level output from insanely-fast-whisper `--output-format json`.
#[derive(Debug, Deserialize)]
struct WhisperOutput {
    text: String,
    chunks: Vec<WhisperChunk>,
}

pub struct WhisperClient {
    config: WhisperConfig,
    http: reqwest::Client,
}

impl WhisperClient {
    pub fn new(config: WhisperConfig) -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .expect("failed to build reqwest client");
        Self { config, http }
    }

    /// Transcribe raw audio bytes. Returns text, detected language, and duration.
    pub async fn transcribe(
        &self,
        audio_bytes: &[u8],
        language: Option<&str>,
    ) -> Result<TranscriptionResult, anyhow::Error> {
        match self.config.mode.as_str() {
            "subprocess" => self.transcribe_subprocess(audio_bytes, language).await,
            "server" => self.transcribe_server(audio_bytes, language).await,
            other => anyhow::bail!("unknown whisper mode: {other}"),
        }
    }

    async fn transcribe_subprocess(
        &self,
        audio_bytes: &[u8],
        language: Option<&str>,
    ) -> Result<TranscriptionResult, anyhow::Error> {
        // Write audio to a temp file (insanely-fast-whisper requires a file path).
        let tmp_dir = tempfile::tempdir()?;
        let audio_path = tmp_dir.path().join("input.wav");
        let output_path = tmp_dir.path().join("output.json");
        tokio::fs::write(&audio_path, audio_bytes).await?;

        let mut cmd = Command::new("insanely-fast-whisper");
        cmd.arg("--file-name")
            .arg(&audio_path)
            .arg("--model-name")
            .arg(&self.config.model)
            .arg("--task")
            .arg("transcribe")
            .arg("--transcript-path")
            .arg(&output_path);

        let lang = language.unwrap_or("en");
        cmd.arg("--language").arg(lang);

        cmd.stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        let output = cmd.output().await?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!(
                "insanely-fast-whisper failed (exit {}): {stderr}",
                output.status
            );
        }

        let raw = tokio::fs::read_to_string(&output_path).await?;
        let parsed: WhisperOutput = serde_json::from_str(&raw)?;

        // Derive duration from the last chunk's end timestamp, falling back to
        // a rough estimate from file size (16kHz, 16-bit mono = 32000 bytes/s).
        let duration = parsed
            .chunks
            .last()
            .and_then(|c| c.timestamp.1)
            .unwrap_or_else(|| audio_bytes.len() as f64 / 32000.0);

        Ok(TranscriptionResult {
            text: parsed.text.trim().to_string(),
            language: lang.to_string(),
            duration,
        })
    }

    async fn transcribe_server(
        &self,
        audio_bytes: &[u8],
        language: Option<&str>,
    ) -> Result<TranscriptionResult, anyhow::Error> {
        let endpoint = self
            .config
            .endpoint
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("whisper server mode requires endpoint"))?;

        let url = format!("{}/v1/audio/transcriptions", endpoint.trim_end_matches('/'));

        let file_part = reqwest::multipart::Part::bytes(audio_bytes.to_vec())
            .file_name("audio.wav")
            .mime_str("audio/wav")?;

        let mut form = reqwest::multipart::Form::new()
            .part("file", file_part)
            .text("model", self.config.model.clone());

        if let Some(lang) = language {
            form = form.text("language", lang.to_string());
        }

        let resp = self.http.post(&url).multipart(form).send().await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("whisper server returned {status}: {body}");
        }

        let result: TranscriptionResult = resp.json().await?;
        Ok(result)
    }
}
