use std::path::PathBuf;

fn main() {
    println!("cargo::rerun-if-changed=src");

    let blueprint_metadata = serde_json::json!({
        "name": "voice-inference",
        "description": "Voice synthesis and transcription operator (TTS/STT) via Tangle",
        "version": env!("CARGO_PKG_VERSION"),
        "manager": {
            "Evm": "VoiceBSM"
        },
        "master_revision": "Latest",
        "jobs": [
            {
                "name": "tts",
                "job_index": 0,
                "description": "Text-to-speech synthesis (input text → audio bytes)",
                "inputs": ["(string,string,string)"],
                "outputs": ["(bytes,uint32,string)"],
                "required_results": 1,
                "execution": "local"
            }
        ]
    });

    let json = serde_json::to_string_pretty(&blueprint_metadata).unwrap();
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir.parent().expect("workspace root");
    std::fs::write(workspace_root.join("blueprint.json"), json.as_bytes()).unwrap();
}
