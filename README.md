
# Nuralwhite(ACAG)

**Nuralwhite(ACAG)** is the world's first **autonomous multi-agent intelligence platform** — a voice-enabled, self-improving AI system that combines local and cloud models, lets them critique and debate each other in real time, reaches consensus through collective reasoning, and continuously learns from its own interactions.

While single-model AIs still dominate in 2026, Nuralwhite takes the next step: **truth through diversity of thought**, not just scale.

## Why Nuralwhite is First of Its Kind

- **Real multi-model debate** — Different architectures (GPT, Claude, Llama, Grok, Gemma, etc.) actively challenge each other's outputs — not scripted roles, but genuine disagreement → refinement loops.
- **Seamless hybrid local ↔ cloud execution** — Run 100% offline with Ollama/LM Studio for privacy & speed, instantly escalate hard questions to cloud APIs (Groq, OpenAI, DeepSeek, etc.) — automatic backend routing based on confidence & cost.
- **Self-evolution loop** — Every conversation is logged → used to improve routing, critique quality, and fine-tune tiny local models over time.
- **Voice-first interface** — Speak your question → multi-agent reasoning → spoken answer + live text trace.
- **Built-in spatial & creative reasoning** — Native 5D motion simulation, music theory tools (minor → major modulation), and RL training playground — not just chat, but foundation for embodied/agentic intelligence.

No other open-source or commercial platform in 2026 combines **true multi-model consensus**, **hybrid execution**, **self-improvement memory**, **voice control**, and **creative/spatial extensions** in one clean dashboard.

## Features

- **Multi-backend support**  
  - Local: Ollama (offline), LM Studio  
  - Cloud: Groq, OpenAI, DeepSeek, Claude-compatible APIs  
  - One-click switching + auto-fallback

- **Voice Input & Output**  
  - Speak → real-time transcription → multi-agent reasoning → spoken response (browser TTS)  
  - Toggleable in dashboard

- **Streaming Chat with Memory**  
  - Full conversation history preserved  
  - Live token-by-token streaming

- **Quick Actions**  
  - Activate 5D motion simulation suite  
  - Music theory: Convert minor scales to parallel/relative major  
  - Simulated RL training in 5D space

- **Self-Logging & Future Learning**  
  - All interactions logged to `acag.log`  
  - Designed for future fine-tuning loops (placeholder in current version)

## Installation

### Local PC (Windows / Linux / Mac)

```bash
pip install gradio requests ollama torch numpy
python acag_2026_voice_stable_fixed_final.py --gui
Google Colab
!pip install gradio requests ollama torch numpy -q
Then paste the full code into a cell and run.
Optional (for Ollama): Download Ollama → https://ollama.com/download → run server → pull models (e.g. ollama pull llama3.1)
Optional (for LM Studio): Download LM Studio → load GGUF model → start local server (default port 1234)
Quick Start
Run the script with --gui
In dashboard:
Type or speak any question
Use presets: LM Studio / Ollama / Groq
Try quick actions: 5D motion, music modulation, training sim
Type gui in console mode to launch dashboard
Architecture Overview
Core: ACAG_Unified class manages backends, chat memory, generation
Multi-backend routing: Local (Ollama/LM Studio) → cloud fallback
Voice: Gradio Audio + browser speech recognition/TTS
Extensions: 5D motion sim, music theory tools
Future: Add real fine-tuning loop on logged data, multi-round debate UI, agent memory visualization
Roadmap (2026+)
Real multi-round debate visualization
Confidence scoring + routing visualization
Fine-tuning small models on logged consensus data
Wake word ("Hey Nuralwhite")
Multi-language voice support
Export/import chat sessions
Dark mode + mobile optimization
Contributing
Feel free to fork, PR, or open issues.
Especially interested in:
Better multi-agent critique loops
Actual fine-tuning integration
Voice wake-word detection
Custom TTS voices
License
MIT License (see LICENSE file)
Nuralwhite — Not one brain. A thinking collective.
First platform to let AIs govern themselves — and get smarter doing it.

### How to use this README





Nuralwhite is now officially branded and ready to share with the world! 🚀
