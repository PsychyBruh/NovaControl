---

# `TODO.md` (full checklist)

```md
# TODO - Multimodal AI Desktop Control System

## 0) Repo + Basics
- [x] Create repo + folders (core/ vision/ audio/ intent/ actions/ ui/ logs/ scripts/)
- [x] Add README.md + TODO.md
- [x] Add requirements.txt
- [x] Add .gitignore (Python, venv, logs/)
- [x] Add core/config.py (all tunables in one place)

---

## 1) Core Framework (Event-Driven)
- [x] Implement `core/event_bus.py`
  - [x] Async queue
  - [x] Publish/subscribe helpers
  - [x] Latest-state cache (latest gesture/gaze/voice)
- [x] Implement `core/main.py` orchestrator
  - [x] Start/stop lifecycle
  - [x] Graceful shutdown
  - [x] Keyboard hook for arm/disarm + emergency stop

---

## 2) UI Overlay (PiP Preview)
- [x] Create `ui/preview_overlay.py`
  - [x] Webcam frame display
  - [x] Draw hand landmarks
  - [x] Draw gaze direction indicator
  - [x] Print labels: Gesture, Gaze, Mode, Intent, Confidence
  - [x] Optional: show FPS + vision latency
- [x] Add status bar text formatting (clean + demo-friendly)

---

## 3) Vision: Hand Tracking + Gestures
- [x] Create `vision/hand_tracker.py`
  - [x] MediaPipe Hands pipeline
  - [x] Normalize landmarks
  - [x] Gesture classification (V1)
    - [x] OPEN_PALM
    - [x] FIST
    - [x] POINT (index extended)
    - [x] PINCH (thumb-index distance threshold)
  - [x] Confidence scoring
  - [x] Temporal stability (must persist N frames)
  - [x] Smoothing (EMA) for cursor control points
- [x] Publish events:
  - [x] gesture events
  - [x] pointing vector / target coordinate events

---

## 4) Vision: Gaze Direction (Robust, Simple)
- [x] Create `vision/gaze_tracker.py`
  - [x] MediaPipe Face Mesh pipeline
  - [x] Compute gaze direction (LEFT/CENTER/RIGHT) using landmark geometry
  - [x] Add smoothing (avoid flicker)
  - [x] Handle no face detected -> SAFE mode event
- [x] Add `scripts/calibrate_gaze.py`
  - [x] 2-3s neutral center calibration
  - [x] Save calibration to config or file

---

## 5) Actions: OS Control (Mouse/Scroll/Click)
- [x] Create `actions/mouse_controller.py`
  - [x] Move mouse absolute/relative
  - [x] Left click / right click / double click
  - [x] Scroll up/down
  - [x] Drag start/stop (click-hold)
- [ ] Add platform notes (Windows/macOS/Linux quirks)

---

## 6) SafetyGuard (Must-Have)
- [x] Create `actions/safety_guard.py`
  - [x] ARMED mode required for OS actions
  - [x] Emergency stop key (hard stop)
  - [x] Confidence thresholds
  - [x] Rate limiting (click cooldown, scroll burst limit)
  - [x] Auto-disarm if tracking lost
  - [x] Open palm / voice stop cancels queued intents
- [ ] Add Two-signal confirmation (V1 rule)
  - [ ] Click requires PINCH + voice click OR PINCH held > X ms while armed
- [x] Add safe mode default on startup

---

## 7) Intent Engine (The Brain)
- [x] Create `intent/intent_engine.py`
  - [x] Maintains state machine:
    - [x] SAFE
    - [x] ARMED
    - [x] CURSOR_MOVE
    - [x] DRAG_MODE
  - [x] Fusion rules:
    - [x] POINT -> move cursor
    - [x] PINCH -> click candidate
    - [ ] FIST + vertical motion -> scroll
    - [x] Long PINCH -> drag
    - [x] OPEN_PALM -> cancel
  - [x] Emits intent events with confidence + metadata
- [x] Hook to SafetyGuard -> approved intents only

---

## 8) Audio: Push-to-Talk Speech-to-Text (Optional but Strong)
- [ ] Create `audio/speech_listener.py`
  - [ ] Push-to-talk hotkey (configurable)
  - [ ] Record short audio clip
  - [ ] Transcribe with faster-whisper
  - [ ] Publish voice event: `voice_text`
- [ ] Create `audio/command_parser.py`
  - [ ] Map phrases + canonical commands:
    - [ ] click / double click / right click
    - [ ] scroll up/down
    - [ ] stop/cancel
    - [ ] open <app>
- [ ] Add VAD (optional) to auto-stop recording on silence

---

## 9) Gaze + Gesture Smart Behaviors (Polish)
- [ ] Gaze bias: cursor moves faster toward where you look
- [ ] Multi-monitor selection:
  - [ ] Look-left selects left monitor
  - [ ] Look-right selects right monitor
- [ ] Attention safety:
  - [ ] If gaze not detected -> auto SAFE
- [ ] Per-app profiles:
  - [ ] Coding mode vs Gaming mode

---

## 10) Logging + Metrics (Resume Sauce)
- [ ] Create `core/logger.py`
  - [ ] JSONL event logs
  - [ ] Metrics CSV
- [ ] Measure:
  - [ ] FPS
  - [ ] vision latency
  - [ ] intent latency
  - [ ] approved vs blocked intents
- [ ] Add `scripts/summarize_logs.py` to print session summary

---

## 11) Demo & Packaging
- [ ] Add `scripts/demo_mode.py`
  - [ ] big ARMED banner
  - [ ] shows intent pending clearly
- [ ] Record 60-90s demo video
- [ ] Add GIF to README
- [ ] Add How it works diagram in README
- [ ] Add resume bullets section

---

## 12) Stretch Goals (Elite)
- [ ] Custom gesture training (user-recorded gestures)
- [ ] LLM intent parsing for natural language commands (SAFE-gated)
- [ ] Plugin system for new actions
- [ ] GUI settings panel
- [ ] Installer / one-click run script
```
