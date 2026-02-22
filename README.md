# lyri

Single-shot multi-character prosody-aware TTS.

One model. One pass. All characters.

## Goal

Given a story and a set of character definitions (name, description, voice sample or description), generate a complete multi-voice audio output in a single forward pass — with natural prosody, emotion, and speaker transitions.

## Status

**Research & experimentation phase.** Probing open-source TTS models to find the shortest path to single-pass multi-speaker generation.

## Approach

Start from pre-trained open-source models (Qwen3-TTS, CosyVoice, F5-TTS) and investigate:

1. Can existing models be pushed toward multi-speaker in a single generation?
2. What architectural modifications are needed?
3. What training data and compute is required?
4. What's the minimum viable path to a working prototype?

All findings documented in `experiments/`.

## Input Format (Target)

```
Story: <full text with dialogue>

Characters:
  - Name: Sam
    Match: Sam, Samuel
    Description: Technological expert, confident
    Voice: <sam.wav>  OR  "male, 40s, deep, authoritative"

  - Name: Lina  
    Match: Lina
    Description: Young activist, sharp
    Voice: "female, early 20s, fast-paced, determined"

  - Name: Narrator
    Voice: <narrator.wav>
```

## Output

A single audio file with distinct voices per character, natural transitions, and prosody that matches the emotional context of the story.

## Setup

```bash
# Clone
git clone https://github.com/LyricWinter/lyri.git
cd lyri

# Configure secrets
cp .env.template .env.local
# Edit .env.local with your HuggingFace token
```

## Project Structure

```
lyri/
├── .secrets/          # Gitignored — HF token, API keys
├── experiments/       # Findings, notebooks, scripts
├── src/               # Core source code
└── research/          # Papers, notes, references
```

## License

MIT
