# Multimodal AI Desktop Control System (NovaControl)

A real-time **multimodal desktop controller** that fuses **hand gestures + eye direction + voice commands** into a single **intent engine** to safely execute **OS-level actions** (mouse, click, scroll). Includes a **top-right camera preview overlay** that visualizes tracking + current intent for a clean demo.

## Why it’s different (not a tutorial clone)
- **Event-driven architecture** (modules publish events; intent engine decides)
- **Intent fusion** (gesture + gaze + voice → intent)
- **Safety gating** (armed mode, confirmations, rate limits, failsafes)
- **Real-time metrics** (latency + event logs)
- **Demo-ready UI** (PiP camera overlay proves it’s live)

---

## Features (planned / in-progress)
- Vision: MediaPipe **Hand Tracking** + gesture classification
- Vision: MediaPipe **Face Mesh** → gaze direction (L/C/R, optional U/D)
- Audio: Push-to-talk speech-to-text (Whisper / faster-whisper)
- Intent Engine: fuses inputs into high-level intents (click/drag/scroll/move)
- SafetyGuard: arming mode, confidence thresholds, cooldowns, emergency stop
- Action Engine: OS mouse + scroll + click
- Overlay UI: top-right preview box with landmarks + labels + system status
- Logging: session logs + latency metrics

---

## Repo Structure
```
NovaControl/
  core/        # event bus, config, main orchestration
  vision/      # hand tracker, gaze tracker
  audio/       # speech capture + ASR
  intent/      # intent fusion + state machine
  actions/     # OS controls + safety guard
  ui/          # preview overlay
  logs/        # session logs, metrics
  scripts/     # utilities (calibration, diagnostics)
```

---

## Safety Notice (read this)
This project can move/click your mouse and trigger actions on your computer.

**Safety rules when testing:**
- Close important apps/tabs.
- Use it in a safe environment (empty desktop).
- Keep your hands near the **Emergency Stop key**.
- Default mode should be **SAFE** (no actions) until explicitly ARMED.

---

## Requirements
- Python 3.10+ recommended
- Webcam
- OS: Windows / Linux / macOS (OS control may vary slightly by platform)

### Dependencies (core)
- opencv-python
- mediapipe
- numpy
- pynput (or pyautogui)

### Optional (voice)
- faster-whisper (recommended on CPU)
- sounddevice / pyaudio (audio capture)
- webrtcvad (optional VAD)

---

## Setup
### 1) Create a virtual environment
**Windows (PowerShell)**
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS/Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -U pip
pip install -r requirements.txt
```

Example `requirements.txt`:
```txt
opencv-python
mediapipe
numpy
pynput
# Optional voice:
# faster-whisper
# sounddevice
# webrtcvad
```

---

## Quickstart
### Run the main app
```bash
python -m core.main
```

You should see:
- A live camera preview with landmarks (PiP overlay)
- Status labels: Gesture / Gaze / Mode / Intent
- (Later) the cursor should move only when ARMED

---

## Default Controls (planned)
- **Hold Space** → ARM (allow actions)
- **Release Space** → SAFE (no actions)
- **Esc** → Emergency stop (kills actions immediately)
- **Open palm gesture** → Cancel / stop actions (soft stop)
- Voice “stop” → Cancel / disarm (soft stop)

> Controls are configurable in `core/config.py`.

---

## Architecture Overview
### Event-driven pipeline
1. Vision module reads camera frames → publishes:
   - `gesture` events (PINCH/POINT/FIST/OPEN_PALM)
   - `gaze` events (LEFT/CENTER/RIGHT)
2. Audio module (push-to-talk) → publishes `voice` events
3. Intent engine consumes the latest events → emits high-level intents
4. Safety guard approves/blocks intents
5. Action engine executes approved intents (mouse/scroll/click)
6. Logger records events + latencies

### Event schema (example)
```json
{
  "ts": 1733160000.123,
  "type": "gesture",
  "name": "PINCH",
  "confidence": 0.93,
  "meta": { "hand": "right" }
}
```

### Intent schema (example)
```json
{
  "ts": 1733160000.456,
  "type": "intent",
  "name": "CLICK",
  "confidence": 0.88,
  "meta": { "button": "left" }
}
```

---

## Logging & Metrics
Outputs to `logs/` per session:
- `session_YYYYMMDD_HHMM.jsonl` (events + intents)
- `metrics_YYYYMMDD_HHMM.csv` (latency, fps, block rates)

Suggested KPIs:
- Vision latency (ms)
- Intent decision latency (ms)
- Effective FPS
- Blocked intent rate (safety)
- False trigger counts

---

## Roadmap
See `TODO.md` for the full checklist. High-level milestones:
- [ ] Camera preview + hand landmarks overlay
- [ ] Gesture classification + smoothing
- [ ] Cursor movement (pointing) + pinch click candidate
- [ ] SafetyGuard + arming mode + emergency stop
- [ ] Voice push-to-talk + command parsing
- [ ] Input fusion (gesture + voice confirmation)
- [ ] Gaze direction + calibration
- [ ] Metrics + demo polish + profiles

---

## Demo Video (recommended)
Record a 60–90s clip:
1. Show PiP overlay with landmarks + labels
2. Arm mode on/off
3. Move cursor with pointing
4. Pinch + “click” voice confirmation
5. Scroll and stop/cancel

OBS easiest:
- Desktop capture
- Webcam overlay (top-right)
- Optional: console window showing status

---

## Resume-ready description (copy/paste)
**Multimodal AI Desktop Control System**
- Built a real-time multimodal controller integrating computer vision (hand + gaze) and speech recognition into an event-driven intent engine for OS-level control.
- Implemented safety-gated action execution with arming mode, confidence thresholds, and rate limiting to prevent unintended interactions.
- Instrumented latency and session logging to measure performance and reliability on consumer CPU hardware.

---

## License
Copyright <2025> <Psychy>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 
