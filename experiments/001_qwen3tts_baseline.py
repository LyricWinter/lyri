"""
Experiment 001: Qwen3-TTS Baseline + Multi-Speaker Probe

Goals:
1. Verify Qwen3-TTS works on Modal (download weights to volume, run inference)
2. Test single-speaker baseline quality with voice clone + voice design
3. Probe: what happens when we try to switch speakers mid-sequence?

Expected cost: ~$1-3 (model download + a few minutes of A100 inference)
"""

import modal

app = modal.App("lyri-exp-001")

# Persistent volume for model weights — survives across runs
model_volume = modal.Volume.from_name("lyri-models", create_if_missing=True)
output_volume = modal.Volume.from_name("lyri-outputs", create_if_missing=True)

MODEL_DIR = "/models"
OUTPUT_DIR = "/outputs"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "qwen-tts",
        "soundfile",
        "numpy",
        "torch",
        "huggingface_hub[cli]",
    )
    .pip_install("flash-attn", extra_options="--no-build-isolation")
)


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={MODEL_DIR: model_volume, OUTPUT_DIR: output_volume},
    timeout=600,
)
def download_models():
    """Download Qwen3-TTS models to persistent volume."""
    from huggingface_hub import snapshot_download
    import os

    models = [
        "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",           # Voice clone
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",    # Voice design
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",    # Pre-built voices
    ]

    for model_id in models:
        local_dir = os.path.join(MODEL_DIR, model_id.split("/")[-1])
        if os.path.exists(local_dir) and os.listdir(local_dir):
            print(f"✓ {model_id} already cached")
            continue
        print(f"⬇ Downloading {model_id}...")
        snapshot_download(repo_id=model_id, local_dir=local_dir)
        print(f"✓ {model_id} downloaded")

    model_volume.commit()
    print("All models cached in volume.")


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={MODEL_DIR: model_volume, OUTPUT_DIR: output_volume},
    timeout=300,
)
def test_single_speaker():
    """Baseline: single speaker, custom voice + voice design + voice clone."""
    import torch
    import soundfile as sf
    import os

    from qwen_tts import Qwen3TTSModel

    results = {}

    # --- Test 1: CustomVoice (pre-built speaker) ---
    print("=== Test 1: CustomVoice ===")
    model = Qwen3TTSModel.from_pretrained(
        os.path.join(MODEL_DIR, "Qwen3-TTS-12Hz-1.7B-CustomVoice"),
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    wavs, sr = model.generate_custom_voice(
        text="The future belongs to those who believe in the beauty of their dreams.",
        language="English",
        speaker="Ryan",
        instruct="Speak with warm confidence, like telling a friend something important.",
    )
    path = os.path.join(OUTPUT_DIR, "001_custom_voice_ryan.wav")
    sf.write(path, wavs[0], sr)
    results["custom_voice"] = {"path": path, "samples": len(wavs[0]), "sr": sr}
    print(f"  ✓ Saved {path} ({len(wavs[0])/sr:.1f}s)")

    del model
    torch.cuda.empty_cache()

    # --- Test 2: VoiceDesign (voice from description) ---
    print("\n=== Test 2: VoiceDesign ===")
    model = Qwen3TTSModel.from_pretrained(
        os.path.join(MODEL_DIR, "Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    wavs, sr = model.generate_voice_design(
        text="Look, I told you this was going to happen. But did anyone listen? No. They never do.",
        language="English",
        instruct="Gruff older male voice, deep baritone, slightly world-weary but not without humor. Like a retired detective.",
    )
    path = os.path.join(OUTPUT_DIR, "001_voice_design_detective.wav")
    sf.write(path, wavs[0], sr)
    results["voice_design"] = {"path": path, "samples": len(wavs[0]), "sr": sr}
    print(f"  ✓ Saved {path} ({len(wavs[0])/sr:.1f}s)")

    del model
    torch.cuda.empty_cache()

    # --- Test 3: Batch with different speakers (CustomVoice) ---
    print("\n=== Test 3: Batch - Two speakers ===")
    model = Qwen3TTSModel.from_pretrained(
        os.path.join(MODEL_DIR, "Qwen3-TTS-12Hz-1.7B-CustomVoice"),
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    wavs, sr = model.generate_custom_voice(
        text=[
            "Are you sure about this? It seems risky.",
            "Trust me. I've run the numbers. We're going to be fine.",
        ],
        language=["English", "English"],
        speaker=["Vivian", "Ryan"],
        instruct=["Worried and uncertain.", "Confident and reassuring."],
    )

    for i, w in enumerate(wavs):
        path = os.path.join(OUTPUT_DIR, f"001_batch_speaker_{i}.wav")
        sf.write(path, w, sr)
        results[f"batch_{i}"] = {"path": path, "samples": len(w), "sr": sr}
        print(f"  ✓ Saved {path} ({len(w)/sr:.1f}s)")

    del model
    torch.cuda.empty_cache()

    output_volume.commit()
    print(f"\nResults: {results}")
    return results


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={MODEL_DIR: model_volume, OUTPUT_DIR: output_volume},
    timeout=300,
)
def probe_multi_speaker():
    """
    Probe: Can we get multi-speaker from a single generation call?

    Approaches to try:
    1. Concatenate two speakers' text in a single string with speaker tags
    2. Feed two voice_clone_prompts sequentially
    3. Look at what the model actually sees at the token level
    """
    import torch
    import soundfile as sf
    import numpy as np
    import os

    from qwen_tts import Qwen3TTSModel

    print("=== Probe: Multi-speaker in single generation ===")

    model = Qwen3TTSModel.from_pretrained(
        os.path.join(MODEL_DIR, "Qwen3-TTS-12Hz-1.7B-CustomVoice"),
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # --- Probe 1: Speaker tag in text ---
    # What happens if we put role markers in the text?
    print("\n--- Probe 1: Speaker tags in text ---")
    tagged_text = (
        '[Ryan]: "Are you ready for this?" '
        '[Vivian]: "I was born ready." '
        '[Ryan]: "Then let\'s go."'
    )
    wavs, sr = model.generate_custom_voice(
        text=tagged_text,
        language="English",
        speaker="Ryan",  # Only one speaker param allowed
    )
    path = os.path.join(OUTPUT_DIR, "001_probe_tagged_text.wav")
    sf.write(path, wavs[0], sr)
    print(f"  Saved {path} ({len(wavs[0])/sr:.1f}s) — does it switch voices?")

    # --- Probe 2: Concatenate batch outputs ---
    # Generate each speaker separately, measure the gap vs stitched
    print("\n--- Probe 2: Sequential generation for comparison ---")
    line_a, sr = model.generate_custom_voice(
        text="Are you ready for this?",
        language="English",
        speaker="Ryan",
        instruct="Excited, anticipatory.",
    )
    line_b, sr = model.generate_custom_voice(
        text="I was born ready.",
        language="English",
        speaker="Vivian",
        instruct="Confident, slightly playful.",
    )
    line_c, sr = model.generate_custom_voice(
        text="Then let's go.",
        language="English",
        speaker="Ryan",
        instruct="Determined.",
    )

    # Simple concatenation (no crossfade)
    stitched = np.concatenate([line_a[0], line_b[0], line_c[0]])
    path = os.path.join(OUTPUT_DIR, "001_probe_stitched.wav")
    sf.write(path, stitched, sr)
    print(f"  Saved {path} ({len(stitched)/sr:.1f}s) — baseline stitched quality")

    # --- Probe 3: Inspect tokenizer behavior ---
    print("\n--- Probe 3: Token-level inspection ---")
    print(f"  Model type: {type(model)}")
    print(f"  Model config keys: {list(model.config.__dict__.keys()) if hasattr(model, 'config') else 'N/A'}")

    # Check what methods/attributes are available
    public_methods = [m for m in dir(model) if not m.startswith('_')]
    print(f"  Public methods: {public_methods}")

    del model
    torch.cuda.empty_cache()
    output_volume.commit()

    print("\n=== Probe complete. Listen to outputs and compare. ===")


@app.local_entrypoint()
def main():
    import sys

    if len(sys.argv) < 2 or sys.argv[1] == "download":
        print("Step 1: Downloading models...")
        download_models.remote()
    elif sys.argv[1] == "baseline":
        print("Step 2: Running single-speaker baselines...")
        results = test_single_speaker.remote()
        print(f"Results: {results}")
    elif sys.argv[1] == "probe":
        print("Step 3: Probing multi-speaker behavior...")
        probe_multi_speaker.remote()
    elif sys.argv[1] == "all":
        print("Running all experiments...")
        download_models.remote()
        test_single_speaker.remote()
        probe_multi_speaker.remote()
    else:
        print(f"Unknown command: {sys.argv[1]}")
        print("Usage: modal run 001_qwen3tts_baseline.py [download|baseline|probe|all]")
