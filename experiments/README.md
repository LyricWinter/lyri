# Experiments

## 001: Qwen3-TTS Baseline + Multi-Speaker Probe

**Status:** Written, not yet run
**Estimated cost:** ~$1-3

**What it does:**
1. Downloads Qwen3-TTS models (Base, CustomVoice, VoiceDesign) to a persistent Modal volume
2. Tests single-speaker generation quality across all three model variants
3. Probes multi-speaker behavior:
   - Can speaker tags in text trigger voice switching?
   - What does stitched multi-speaker audio sound like as a baseline?
   - What does the model expose at the token/method level?

**How to run:**
```bash
cd experiments

# Step 1: Download models (~10 min first time, cached after)
modal run 001_qwen3tts_baseline.py download

# Step 2: Single-speaker baselines
modal run 001_qwen3tts_baseline.py baseline

# Step 3: Multi-speaker probing
modal run 001_qwen3tts_baseline.py probe

# Or run everything
modal run 001_qwen3tts_baseline.py all
```

**Output:** WAV files saved to `lyri-outputs` Modal volume.
