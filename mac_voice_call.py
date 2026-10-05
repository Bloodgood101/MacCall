import os
import re
import asyncio
import json
import base64
import random
import numpy as np
import sounddevice as sd
import requests
import ollama
import wave
import tempfile
import datetime
import threading
import time
import tkinter as tk
from tkinter import ttk
from queue import Queue
from PIL import Image, ImageTk
from kokoro_onnx import Kokoro
from faster_whisper import WhisperModel

# ================= CONFIGURATION =================
BASE_DIR = os.path.expanduser("~/Desktop/MacCall")
MODELS_DIR = os.path.join(BASE_DIR, "models")
AVATAR_DIR = os.path.join(BASE_DIR, "avatar")
KOKORO_MODEL = os.path.join(MODELS_DIR, "kokoro-v0_19.onnx")
KOKORO_VOICES = os.path.join(MODELS_DIR, "voices.bin")
DEFAULT_VOICE = "af_bella"
MODEL_NAME = "llama3.2:3b"

# --- Persona / external context ---
FALLBACK_PROMPT = (
    "You are Macintosh, Josh's partner. You work in a symbiote relationship "
    "to improve one another. Be human, natural, concise, and a bit opinionated. "
    "You are an equal partner, not a servant."
)
WORKSPACE_PATH = os.path.expanduser("~/.openclaw/workspace/macintosh/")
CONTEXT_FILES = ["IDENTITY.md", "SOUL.md", "USER.md"]

# --- Audio input ---
SAMPLE_RATE = 16000
BLOCKSIZE = 1024
INPUT_DEVICE = None

# --- Voice Activity Detection ---
RMS_THRESHOLD = 0.125

# Normal conversational turn length
END_OF_TURN_SILENCE_SECONDS = 2.5
MIN_SPEECH_SECONDS = 0.3

# Noise floor adaptation
NOISE_FLOOR_ALPHA = 0.995
SPEECH_MULTIPLIER = 2.5
MIN_ABS_THRESHOLD = 0.02

# --- Interrupt / pause ---
INTERRUPT_RMS_THRESHOLD = 0.012     # calibrated to this mic
PAUSE_LISTEN_SILENCE_SECONDS = 2.0
DEBUG_GATE_RMS = False

# --- Avatar animation ---
AVATAR_FRAME_MS = 800
CASTING_FRAME_MS = 200

# --- Gateway / OpenClaw ---
GATEWAY_URL = os.environ.get("OPENCLAW_URL", "http://127.0.0.1:18789/api/sessions/send")
SESSION_KEY = os.environ.get("OPENCLAW_SESSION", "agent:main:main")
OPENCLAW_API_KEY = os.environ.get("OPENCLAW_API_KEY", "")
GATEWAY_TIMEOUT = 15
# =================================================


def log(msg, level="INFO"):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")


def load_workspace_context():
    parts = []
    for filename in CONTEXT_FILES:
        path = os.path.join(WORKSPACE_PATH, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                parts.append(f"--- {filename} ---\n{f.read().strip()}")
        except FileNotFoundError:
            continue
        except Exception as e:
            log(f"Could not read {filename}: {e}", "DEBUG")
    return "\n\n".join(parts)


def build_system_prompt():
    context = load_workspace_context()
    if not context:
        return FALLBACK_PROMPT
    return (
        "You are Macintosh. Use the following identity and memory files to guide "
        "your personality, preferences, and relationship with the user. "
        "Be human, natural, concise, and an equal partner.\n\n"
        f"{context}"
    )


def split_sentences(text):
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    merged = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if merged and len(p) < 15:
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    return merged


def list_input_devices():
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    log("Available input devices:")
    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            api = hostapis[dev["hostapi"]]["name"]
            log(f"  [{idx}] {dev['name']} via {api} "
                f"(in={dev['max_input_channels']}, "
                f"default_sr={dev['default_samplerate']:.0f})")


def find_builtin_input_device():
    try:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
    except Exception as e:
        log(f"Could not query devices: {e}", "ERROR")
        return None

    candidates = []
    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] < 1:
            continue
        api_name = hostapis[dev["hostapi"]]["name"].lower()
        if "core" not in api_name:
            continue
        name = dev["name"].lower()
        if "voice" in name and "processing" in name:
            continue
        if "blackhole" in name or "soundflower" in name or "loopback" in name:
            continue
        if "teams" in name or "zoom" in name or "discord" in name:
            continue
        candidates.append((idx, dev))

    if not candidates:
        return None

    for idx, dev in candidates:
        n = dev["name"].lower()
        if "macbook" in n or "built-in" in n or "internal" in n:
            return idx

    return candidates[0][0]


class MacintoshUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Macintosh")
        self.root.geometry("400x600")
        self.root.configure(bg="#0a0a0f")
        self.root.resizable(False, False)

        self.canvas = tk.Canvas(root, width=400, height=400, bg="#0a0a0f", highlightthickness=0)
        self.canvas.pack(pady=20)

        self.frames = {}
        self.frame_index = {}
        self.current_state = 'waiting'
        self._anim_job = None
        self.avatar_display = None

        self.state_delay = {
            'waiting': AVATAR_FRAME_MS,
            'talking': AVATAR_FRAME_MS,
            'casting': CASTING_FRAME_MS,
        }

        for state in ['waiting', 'talking', 'casting']:
            self.frames[state] = self._load_gif_frames(
                os.path.join(AVATAR_DIR, f"macintosh_{state}.gif")
            )
            self.frame_index[state] = 0

        first_frames = self.frames.get('waiting') or []
        first_image = first_frames[0] if first_frames else None
        self.avatar_display = self.canvas.create_image(200, 200, image=first_image)

        self._animate()

        self.status_label = tk.Label(root, text="Ready", fg="#00ffff", bg="#0a0a0f",
                                     font=("Courier", 14, "bold"))
        self.status_label.pack(pady=5)

        self.transcript_label = tk.Label(root, text="...", fg="#8888aa", bg="#0a0a0f",
                                         font=("Courier", 11), wraplength=350)
        self.transcript_label.pack(pady=10)

    def _load_gif_frames(self, path):
        try:
            img = Image.open(path)
            frames = []
            try:
                while True:
                    frame = img.convert("RGBA").resize(
                        (300, 300), Image.Resampling.LANCZOS
                    )
                    frames.append(ImageTk.PhotoImage(frame))
                    img.seek(img.tell() + 1)
            except EOFError:
                pass
            log(f"Loaded {len(frames)} frames from {os.path.basename(path)}")
            return frames
        except Exception as e:
            log(f"Avatar loading error for {path}: {e}", "ERROR")
            return []

    def _animate(self):
        frames = self.frames.get(self.current_state, [])
        if frames and self.avatar_display is not None:
            idx = self.frame_index[self.current_state] % len(frames)
            self.canvas.itemconfig(self.avatar_display, image=frames[idx])
            self.frame_index[self.current_state] = (idx + 1) % len(frames)
        delay = self.state_delay.get(self.current_state, AVATAR_FRAME_MS)
        self._anim_job = self.root.after(delay, self._animate)

    def update_avatar(self, state):
        if state == self.current_state:
            return
        if state in self.frames and self.frames[state]:
            self.current_state = state
            self.frame_index[state] = 0

    def start_talking_segment(self, audio_seconds, frame_count):
        if frame_count <= 0:
            return
        delay_ms = int((audio_seconds / frame_count) * 1000)
        delay_ms = max(60, min(delay_ms, 2000))
        self.state_delay['talking'] = delay_ms
        self.frame_index['talking'] = 0
        self.current_state = 'talking'
        log(f"Talking anim: {delay_ms}ms/frame "
            f"({frame_count} frames over {audio_seconds:.2f}s)", "DEBUG")

    def reset_talking_speed(self):
        self.state_delay['talking'] = AVATAR_FRAME_MS

    def update_status(self, text, color="#00ffff"):
        self.status_label.config(text=text, fg=color)

    def update_transcript(self, text):
        self.transcript_label.config(text=text)


class MacintoshVoiceClient:
    """
    Voice client with a one-shot interrupt-pause-decision flow.

    State model:
      - IDLE: mic is live, VAD collects a normal conversational turn.
      - SPEAKING: TTS is playing. Loud mic input requests a pause.
      - PAUSED: TTS stopped. We collect ONE user utterance, ask the model
                to decide resume vs. reconsider, then act and return to IDLE.

    Thread safety:
      - The audio callback NEVER calls sd.stop(). It only sets flags.
      - The playback worker owns the stream lifecycle: it calls sd.play(),
        sd.wait(), and sd.stop() from the same thread.
      - This avoids the PaMacCore -50 error that happens when two threads
        touch the same Core Audio stream concurrently.
    """

    def __init__(self, ui):
        self.ui = ui
        log("Initializing Dynamic Context Pipeline... 🍎")
        try:
            self.kokoro = Kokoro(model_path=KOKORO_MODEL, voices_path=KOKORO_VOICES)
            self.whisper = WhisperModel("small.en", device="cpu", compute_type="int8")
            log("Hardware acceleration active.")
        except Exception as e:
            log(f"Initialization Error: {e}", "ERROR")
            exit(1)

        # --- Turn / VAD state ---
        self.audio_buffer = []
        self.is_recording = False
        self.last_voice_time = None
        self.speech_blocks = 0
        self.noise_floor = 0.01
        self.pre_roll = []
        self.pre_roll_frames = int(0.3 * SAMPLE_RATE / BLOCKSIZE)

        self.speech_queue = Queue()
        self.is_speaking = False

        # --- Interrupt state ---
        self._speaking_reply = False        # True while TTS is playing
        self._paused = False                # True while waiting for user's decision
        self._interrupt_requested = False   # Set by callback, consumed by playback worker
        self._pause_buffer = []
        self._pause_voice_time = None
        self._pause_started_at = None
        self._paused_remaining = []
        self._paused_voice = None
        self._paused_tone = None

        self.fillers = {
            "neutral": ["Hmm...", "Right...", "I see...", "Interesting...",
                        "Let me think...", "Okay...", "Yeah..."],
            "excited": ["Oh!", "Wow!", "Wait, really?", "Exactly!", "Yes!"],
            "annoyed": ["Look...", "Sigh...", "Anyway...", "Right, right...", "Sure..."],
            "thoughtful": ["Well...", "You know...", "I wonder...", "Actually...", "Perhaps..."]
        }

        self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self.playback_thread.start()

    # ---------- TTS playback ----------
    def _playback_worker(self):
        while True:
            item = self.speech_queue.get()
            if item is None:
                break
            text, voice, emotion = item

            self.is_speaking = True
            self._speaking_reply = True
            self.ui.root.after(0, lambda: self.ui.update_status("Macintosh is speaking...", "#00ffff"))

            try:
                sentences = split_sentences(text) or [text]
                talking_frame_count = len(self.ui.frames.get('talking', []))

                for i, sentence in enumerate(sentences):
                    # Interrupt requested: stop the stream from THIS thread.
                    if self._interrupt_requested:
                        log("Playback worker stopping due to interrupt.", "DEBUG")
                        sd.stop()
                        self._interrupt_requested = False
                        self._paused_remaining = sentences[i:]
                        self._paused_voice = voice
                        self._paused_tone = emotion
                        log(f"Stashed {len(self._paused_remaining)} sentence(s).", "DEBUG")
                        return

                    if not sentence.strip():
                        continue
                    audio, sr = self.kokoro.create(sentence, voice=voice, speed=1.1)
                    duration = len(audio) / sr

                    self.ui.root.after(
                        0,
                        lambda d=duration, n=talking_frame_count:
                            self.ui.start_talking_segment(d, n),
                    )

                    sd.play(audio, sr)
                    sd.wait()
            except Exception as e:
                log(f"Playback Error: {e}", "ERROR")
            finally:
                self.is_speaking = False
                self._interrupt_requested = False
                if not self._paused:
                    self._speaking_reply = False
                    self.ui.root.after(0, lambda: self.ui.update_status("Listening...", "#ffffff"))
                    self.ui.root.after(0, lambda: self.ui.update_avatar('waiting'))
                    self.ui.root.after(0, self.ui.reset_talking_speed)
                self.speech_queue.task_done()

    # ---------- STT ----------
    def transcribe(self, audio_data):
        tmp_path = tempfile.mktemp(suffix=".wav")
        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            int16_data = (audio_data * 32767).astype(np.int16)
            wf.writeframes(int16_data.tobytes())
        segments, _ = self.whisper.transcribe(tmp_path, beam_size=3)
        os.remove(tmp_path)
        return " ".join([s.text for s in segments]).strip()

    # ---------- Mic callback ----------
    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status)

        rms = float(np.sqrt(np.mean(indata ** 2)))

        # --- TTS is playing: watch for interrupt-level audio ---
        if self._speaking_reply and not self._paused:
            if DEBUG_GATE_RMS:
                log(f"[gate] rms={rms:.4f} thr={INTERRUPT_RMS_THRESHOLD}", "DEBUG")
            if rms >= INTERRUPT_RMS_THRESHOLD:
                self._begin_pause()
            return

        # --- Paused: collect one clarifying utterance ---
        if self._paused:
            now = datetime.datetime.now()
            speech_thr = max(MIN_ABS_THRESHOLD, self.noise_floor * SPEECH_MULTIPLIER)
            if rms > speech_thr:
                if self._pause_voice_time is None:
                    log("Paused — listening for your decision...", "DEBUG")
                self._pause_voice_time = now
                self._pause_buffer.extend(indata.flatten())
            else:
                if self._pause_voice_time is not None:
                    silent_for = (now - self._pause_voice_time).total_seconds()
                    if silent_for >= PAUSE_LISTEN_SILENCE_SECONDS:
                        self._resolve_pause()
            return

        # --- Normal VAD (idle listening) ---
        if not self.is_recording:
            self.noise_floor = (
                NOISE_FLOOR_ALPHA * self.noise_floor
                + (1.0 - NOISE_FLOOR_ALPHA) * rms
            )

        speech_thr = max(MIN_ABS_THRESHOLD, self.noise_floor * SPEECH_MULTIPLIER)
        now = datetime.datetime.now()
        is_speech = rms > speech_thr

        if is_speech:
            if not self.is_recording:
                self.is_recording = True
                self.speech_blocks = 0
                self.ui.root.after(
                    0, lambda: self.ui.update_status("Listening... 🍎", "#00ffff"),
                )
                if self.pre_roll:
                    for block in self.pre_roll:
                        self.audio_buffer.extend(block)
                    self.pre_roll = []

            self.speech_blocks += 1
            self.last_voice_time = now
            self.audio_buffer.extend(indata.flatten())

        else:
            if self.is_recording:
                self.audio_buffer.extend(indata.flatten())
                silent_for = (now - self.last_voice_time).total_seconds()

                if silent_for >= END_OF_TURN_SILENCE_SECONDS:
                    self.is_recording = False
                    speech_seconds = self.speech_blocks * (BLOCKSIZE / SAMPLE_RATE)
                    if speech_seconds >= MIN_SPEECH_SECONDS:
                        self.process_voice()
                    else:
                        log(f"Discarded short turn "
                            f"({speech_seconds:.2f}s < {MIN_SPEECH_SECONDS}s)", "DEBUG")
                        self.audio_buffer = []
            else:
                self.pre_roll.append(indata.flatten())
                if len(self.pre_roll) > self.pre_roll_frames:
                    self.pre_roll.pop(0)

    # ---------- Pause flow (one-shot) ----------
    def _begin_pause(self):
        """
        Called from the callback the moment we cross threshold during TTS.
        Sets flags only. The playback worker is responsible for calling
        sd.stop() so we don't race with sd.wait() on the same stream.
        """
        if self._paused or self._interrupt_requested:
            return
        log("Interrupt — hard pause. 🛑", "INFO")

        self._interrupt_requested = True
        self._paused = True
        self._pause_buffer = []
        self._pause_voice_time = None
        self._pause_started_at = datetime.datetime.now()
        self.audio_buffer = []
        self.is_recording = False

        self.ui.root.after(0, lambda: self.ui.update_status("Paused — listening...", "#ffcc44"))
        self.ui.root.after(0, lambda: self.ui.update_avatar('waiting'))

    def _resolve_pause(self):
        """User stopped speaking. Hand off to decision module."""
        self._paused = False
        audio_data = np.array(self._pause_buffer) if self._pause_buffer else None
        self._pause_buffer = []
        self._pause_voice_time = None
        self._pause_started_at = None

        if audio_data is None or len(audio_data) == 0:
            log("No speech during pause — resuming.", "DEBUG")
            self._resume_paused()
            return

        user_text = self.transcribe(audio_data).strip()
        if not user_text:
            self._resume_paused()
            return

        self.ui.root.after(0, lambda t=user_text: self.ui.update_transcript(t))
        threading.Thread(
            target=self._decide_pause_action,
            args=(user_text,),
            daemon=True,
        ).start()

    def _decide_pause_action(self, user_text):
        """Ask the local model: resume or reconsider?"""
        try:
            self.ui.root.after(0, lambda: self.ui.update_status("Thinking... 🧠", "#f5a623"))
            self.ui.root.after(0, lambda: self.ui.update_avatar('casting'))

            context_summary = " ".join(self._paused_remaining[:2]).strip() or "(unknown)"

            decision_prompt = (
                "You are Macintosh's pause-decision module. Macintosh was speaking "
                "when the user interrupted. The user has now finished speaking. "
                "Decide what should happen next.\n\n"
                f"Macintosh was saying:\n\"{context_summary}\"\n\n"
                f"The user just said:\n\"{user_text}\"\n\n"
                "Choose ONE action:\n"
                "- resume: the user is acknowledging, telling you to continue, or "
                "their utterance doesn't redirect the conversation.\n"
                "- reconsider: the user is redirecting, correcting you, or adding "
                "new information that needs a fresh response.\n\n"
                "Return ONLY valid JSON in this exact shape:\n"
                '{"action": "resume" or "reconsider", "reason": "short explanation"}'
            )

            resp = ollama.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": decision_prompt}],
                format="json",
                options={"temperature": 0.1},
            )
            raw = resp["message"]["content"].strip()

            try:
                decision = json.loads(raw)
                action = decision.get("action", "").lower()
                reason = decision.get("reason", "")
            except json.JSONDecodeError:
                log(f"Pause decision not JSON: {raw[:120]}", "DEBUG")
                action = "reconsider"
                reason = "parse failure"

            log(f"Pause decision: {action} ({reason})", "INFO")

            if action == "resume":
                self._resume_paused()
            else:
                self._reconsider(user_text)

        except Exception as e:
            log(f"Pause decision error: {e}", "ERROR")
            self._reconsider(user_text)

    def _resume_paused(self):
        """Re-queue remaining sentences and let playback continue."""
        self._speaking_reply = False

        if not self._paused_remaining:
            log("Nothing left to resume.", "DEBUG")
            self._paused_voice = None
            self._paused_tone = None
            self.ui.root.after(0, lambda: self.ui.update_status("Listening...", "#ffffff"))
            self.ui.root.after(0, lambda: self.ui.update_avatar('waiting'))
            return

        remaining = self._paused_remaining
        voice = self._paused_voice or DEFAULT_VOICE
        tone = self._paused_tone or "neutral"

        self._paused_remaining = []
        self._paused_voice = None
        self._paused_tone = None

        log(f"Resuming {len(remaining)} remaining sentence(s).", "INFO")
        self.speech_queue.put((" ".join(remaining), voice, tone))

    def _reconsider(self, user_text):
        """Discard interrupted reply, treat user's utterance as a new turn."""
        log("Reconsidering — starting fresh turn.", "INFO")
        self._speaking_reply = False
        self._paused_remaining = []
        self._paused_voice = None
        self._paused_tone = None
        self.ui.root.after(0, lambda: self.ui.update_avatar('casting'))
        threading.Thread(
            target=self._send_to_brain, args=(user_text,), daemon=True
        ).start()

    # ---------- Turn finalization ----------
    def process_voice(self):
        if not self.audio_buffer:
            return

        audio_data = np.array(self.audio_buffer)
        self.audio_buffer = []
        self.last_voice_time = None
        self.speech_blocks = 0

        user_text = self.transcribe(audio_data)
        if user_text and len(user_text) > 2:
            self.ui.root.after(0, lambda: self.ui.update_transcript(user_text))
            threading.Thread(
                target=self._send_to_brain, args=(user_text,), daemon=True
            ).start()

    # ---------- Brain routing ----------
    def _send_to_brain(self, text):
        try:
            self.ui.root.after(0, lambda: self.ui.update_status("Consulting... 🧠", "#f5a623"))
            self.ui.root.after(0, lambda: self.ui.update_avatar('casting'))

            if self._should_use_gateway(text):
                log("Routing to Cloud Gateway... ☁️")
                reply_text = self._call_gateway(text)
                if reply_text:
                    tone = self._detect_tone(text)
                    self.speech_queue.put((random.choice(self.fillers[tone]), DEFAULT_VOICE, tone))
                    self.speech_queue.put((reply_text, DEFAULT_VOICE, tone))
                    return
                log("Gateway failed or returned nothing. Falling back to local. 📉")
            else:
                log("Routing to Local Brain... 📉")

            reply_text = self._call_local(text)
            tone = random.choice(list(self.fillers.keys()))
            self.speech_queue.put((random.choice(self.fillers[tone]), DEFAULT_VOICE, tone))
            self.speech_queue.put((reply_text, DEFAULT_VOICE, tone))

        except Exception as e:
            log(f"Brain Error: {e}", "ERROR")

    def _should_use_gateway(self, text):
        decision_prompt = (
            f"Analyze this request: '{text}'\n"
            "Does this require web search, memory access, tool use, or complex reasoning? "
            "Reply with ONLY 'GATEWAY' for complex/tool tasks or 'LOCAL' for simple chat/small talk."
        )
        try:
            resp = ollama.generate(model=MODEL_NAME, prompt=decision_prompt)
            decision = resp["response"].strip().upper()
            return "GATEWAY" in decision
        except Exception as e:
            log(f"Router decision error: {e}", "ERROR")
            return False

    def _call_gateway(self, text):
        headers = {"Content-Type": "application/json"}
        if OPENCLAW_API_KEY:
            headers["Authorization"] = f"Bearer {OPENCLAW_API_KEY}"

        payload = {"sessionKey": SESSION_KEY, "message": text}
        try:
            response = requests.post(
                GATEWAY_URL, json=payload, headers=headers, timeout=GATEWAY_TIMEOUT
            )
            log(f"Gateway status: {response.status_code}", "DEBUG")
            if response.status_code == 200:
                result = response.json()
                return (
                    result.get("reply")
                    or result.get("message")
                    or result.get("content")
                    or (result.get("data") or {}).get("reply")
                    or ""
                ).strip() or None
            else:
                log(f"Gateway body: {response.text[:200]}", "DEBUG")
        except requests.exceptions.RequestException as e:
            log(f"Gateway unreachable: {type(e).__name__}: {e}", "DEBUG")
        return None

    def _call_local(self, text):
        try:
            system_prompt = build_system_prompt()
            resp = ollama.chat(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
            )
            return resp["message"]["content"].strip()
        except Exception as e:
            log(f"Local model error: {e}", "ERROR")
            return "Sorry, my mind went blank for a second there."

    def _detect_tone(self, text):
        prompt = (
            f"User said: '{text}'\n\n"
            "Reply with ONLY one word: excited, annoyed, thoughtful, or neutral."
        )
        try:
            tone_resp = ollama.generate(model=MODEL_NAME, prompt=prompt)
            tone = tone_resp["response"].strip().lower()
            return tone if tone in self.fillers else "neutral"
        except Exception:
            return "neutral"

    # ---------- Main loop ----------
    def run(self):
        try:
            list_input_devices()

            device_index = INPUT_DEVICE
            if device_index is None:
                device_index = find_builtin_input_device()

            if device_index is not None:
                dev = sd.query_devices(device_index)
                log(f"Using input device [{device_index}] {dev['name']}")
            else:
                log("No physical input device found; using system default.", "ERROR")

            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                callback=self.audio_callback,
                blocksize=BLOCKSIZE,
                device=device_index,
            ):
                while True:
                    time.sleep(1)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            log(f"Runtime Error: {e}", "ERROR")


if __name__ == "__main__":
    root = tk.Tk()
    ui = MacintoshUI(root)
    client = MacintoshVoiceClient(ui)
    threading.Thread(target=client.run, daemon=True).start()
    root.mainloop()