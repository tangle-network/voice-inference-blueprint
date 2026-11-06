use alloy_sol_types::SolValue;
use voice_inference::config::OperatorConfig;
use voice_inference::VoiceInferenceServer;

fn registration_payload(config: &OperatorConfig) -> Vec<u8> {
    (
        config.vllm.model.clone(),
        config.gpu.expected_gpu_count,
        config.gpu.min_vram_mib,
        config
            .gpu
            .gpu_model
            .clone()
            .unwrap_or_else(|| "unknown".into()),
        format!("http://{}:{}", config.server.host, config.server.port),
    )
        .abi_encode()
}

tangle_inference_core::run_operator!(
    OperatorConfig::load(None),
    registration_payload,
    |_| voice_inference::router(),
    |config| VoiceInferenceServer { config },
    no_qos
);
