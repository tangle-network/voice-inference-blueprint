//! Multi-backend STT engine — unified interface over Whisper subprocess,
//! Whisper server, and generic HTTP STT servers (Faster Whisper, Parakeet,
//! SenseVoice, Moonshine).

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::config::{SttConfig, WhisperConfig};
use crate::whisper::WhisperClient;

/// Transcription result returned to the HTTP handler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResult {
    pub text: String,
    pub language: String,
    pub duration: f64,
}

/// Unified STT engine supporting multiple backends.
pub enum SttEngine {
    /// Original Whisper backend (subprocess or server mode).
    Whisper(WhisperClient),
    /// Generic HTTP STT server (Faster Whisper, Parakeet, SenseVoice, Moonshine).
    Server {
        endpoint: String,
        backend: String,
        model: String,
        http: reqwest::Client,
    },
}

impl SttEngine {
    /// Build an SttEngine from the resolved SttConfig.
    pub fn from_config(config: &SttConfig) -> Result<Self, anyhow::Error> {
        match config.backend.as_str() {
            "whisper" => {
                let whisper_config = WhisperConfig {
                    model: config.model.clone(),
                    mode: config.mode.clone(),
                    endpoint: config.endpoint.clone(),
                    price_per_audio_second: config.price_per_audio_second,
                };
                Ok(Self::Whisper(WhisperClient::new(whisper_config)))
            }
            "faster-whisper" | "parakeet" | "sensevoice" | "moonshine" => {
                let endpoint = config
                    .endpoint
                    .as_ref()
                    .ok_or_else(|| {
                        anyhow::anyhow!("{} backend requires an endpoint URL", config.backend)
                    })?
                    .clone();
                let http = reqwest::Client::builder()
                    .timeout(Duration::from_secs(60))
                    .build()?;
                Ok(Self::Server {
                    endpoint,
                    backend: config.backend.clone(),
                    model: config.model.clone(),
                    http,
                })
            }
            other => anyhow::bail!("unknown STT backend: {other}"),
        }
    }

    /// Transcribe raw audio bytes. Returns text, detected language, and duration.
    pub async fn transcribe(
        &self,
        audio_bytes: &[u8],
        language: Option<&str>,
    ) -> Result<TranscriptionResult, anyhow::Error> {
        match self {
            Self::Whisper(client) => {
                let result = client.transcribe(audio_bytes, language).await?;
                Ok(TranscriptionResult {
                    text: result.text,
                    language: result.language,
                    duration: result.duration,
                })
            }
            Self::Server {
                endpoint,
                backend,
                model,
                http,
            } => {
                let url = format!(
                    "{}/v1/audio/transcriptions",
                    endpoint.trim_end_matches('/')
                );

                let file_part = reqwest::multipart::Part::bytes(audio_bytes.to_vec())
                    .file_name("audio.wav")
                    .mime_str("audio/wav")?;

                let mut form = reqwest::multipart::Form::new()
                    .part("file", file_part)
                    .text("model", model.clone());

                if let Some(lang) = language {
                    form = form.text("language", lang.to_string());
                }

                let resp = http.post(&url).multipart(form).send().await?;

                if !resp.status().is_success() {
                    let status = resp.status();
                    let body = resp.text().await.unwrap_or_default();
                    anyhow::bail!("{backend} transcription failed ({status}): {body}");
                }

                // Accept both the OpenAI-compatible format (text, language, duration)
                // and a simpler JSON format (text, language, duration_s).
                let result: serde_json::Value = resp.json().await?;
                let text = result["text"]
                    .as_str()
                    .unwrap_or("")
                    .trim()
                    .to_string();
                let lang = result["language"]
                    .as_str()
                    .unwrap_or(language.unwrap_or("en"))
                    .to_string();
                let duration = result["duration"]
                    .as_f64()
                    .or_else(|| result["duration_s"].as_f64())
                    .unwrap_or(0.0);

                Ok(TranscriptionResult {
                    text,
                    language: lang,
                    duration,
                })
            }
        }
    }
}
