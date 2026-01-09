from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
import os
import json
import logging
import threading
import time
import aiohttp
import websockets
from websockets.exceptions import ConnectionClosed
import base64

from dotenv import load_dotenv
from groq import Groq

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VeloxBackend")

# Load Env
load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GLADIA_API_KEY = os.getenv("GLADIA_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
TTS_RATE = 24000
SILENCE_TIMEOUT = 0.8

# --- Shared Logic (Ported from Main Test) ---

class TaskStateManager:
    """Manages the conversation state and handles interruptions."""
    def __init__(self):
        self.is_user_speaking_now = False
        self.is_ai_speaking = False
        self.is_thinking = False
        
        # Reference to the current LLM task to allow cancellation
        self.llm_task: asyncio.Task = None
        self.interrupt_signal = threading.Event()
        self.was_interrupted = False # Context flag

    async def handle_interruption(self, websocket: WebSocket):
        """Cancels current tasks and notifies frontend."""
        if self.is_ai_speaking or self.is_thinking:
            logger.info("⚡ INTERRUPTION TRIGGERED")
            
            # 1. Cancel LLM Task
            if self.llm_task and not self.llm_task.done():
                self.llm_task.cancel()
                try:
                    await self.llm_task
                except asyncio.CancelledError:
                    pass
                self.llm_task = None

            # 2. Reset Flags
            self.is_ai_speaking = False
            self.is_thinking = False
            self.was_interrupted = True # Flag interruption
            self.interrupt_signal.set() # Signal to TTS loops to abort
            
            # 3. Notify Frontend to Stop Audio
            await websocket.send_json({"type": "control", "action": "interrupt"})

    async def schedule_llm_task(self, coro):
        """Safely replaces the current LLM task with a new one."""
        # 1. Cancel existing
        if self.llm_task and not self.llm_task.done():
            self.llm_task.cancel()
            try:
                await self.llm_task
            except asyncio.CancelledError:
                pass
        
        # 2. Reset Signal for new task
        self.interrupt_signal.clear()
        
        # 3. Start New
        self.llm_task = asyncio.create_task(coro)

class SilenceManager:
    """Manages the user state (Speaking/Silent) to trigger the AI."""
    def __init__(self):
        self.last_activity = time.time()
        self.buffer = []
        self.lock = threading.Lock()
        self.is_user_speaking = False

    def update_activity(self):
        self.last_activity = time.time()
        self.is_user_speaking = True

    def add_text(self, text):
        with self.lock:
            self.buffer.append(text)

    def flush(self):
        with self.lock:
            if self.buffer:
                full_text = " ".join(self.buffer)
                self.buffer = []
                self.is_user_speaking = False
                return full_text.strip()
            return None



# ... (Previous imports)
import subprocess

# ... (Previous constants)
TTS_RATE_DEEPGRAM = 24000
TTS_RATE_PIPER = 22050

import glob
import zipfile
import io
import shutil

# Piper Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPER_DIR = os.path.join(BASE_DIR, "piper")
PIPER_BINARY = os.path.join(PIPER_DIR, "piper.exe")
PIPER_VOICES_DIR = os.path.join(PIPER_DIR, "voices")
ESPEAK_DATA = os.path.join(PIPER_DIR, "espeak-ng-data")

PIPER_READY = False # Global Flag

# Voice Catalog (Source: Hugging Face)
VOICE_CATALOG = {
    "en_US-amy-medium": {
        "name": "Amy (US) - Medium",
        "locale": "usa",
        "file": "en_US-amy-medium.onnx",
        "url_model": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx?download=true",
        "url_config": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx.json?download=true"
    },
    "en_US-hfc_female-medium": {
        "name": "HFC Female (US) - Medium",
        "locale": "usa",
        "file": "en_US-hfc_female-medium.onnx",
        "url_model": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/hfc_female/medium/en_US-hfc_female-medium.onnx?download=true",
        "url_config": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/hfc_female/medium/en_US-hfc_female-medium.onnx.json?download=true"
    },
    "en_GB-semaine-medium": {
        "name": "Semaine (UK) - Medium",
        "locale": "britain", # "britain" specific folder as requested
        "file": "en_GB-semaine-medium.onnx",
        "url_model": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/semaine/medium/en_GB-semaine-medium.onnx?download=true",
        "url_config": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/semaine/medium/en_GB-semaine-medium.onnx.json?download=true"
    },
    "en_GB-cori-high": {
        "name": "Cori (UK) - High",
        "locale": "britain",
        "file": "en_GB-cori-high.onnx",
        "url_model": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/cori/high/en_GB-cori-high.onnx?download=true",
        "url_config": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/cori/high/en_GB-cori-high.onnx.json?download=true"
    }
}

async def download_file(url, dest_path):
    """Downloads a file from URL to dest_path."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    with open(dest_path, 'wb') as f:
                        f.write(await resp.read())
                    return True
                else:
                    logger.error(f"Failed to download {url}: {resp.status}")
                    return False
    except Exception as e:
        logger.error(f"Download Error for {url}: {e}")
        return False

async def ensure_voice_model(voice_id):
    """Ensures the voice model exists locally. Downloads if missing."""
    if voice_id not in VOICE_CATALOG:
        return None

    info = VOICE_CATALOG[voice_id]
    
    # Construct paths
    # Ensure subdirectory exists (e.g., piper/voices/usa)
    locale_dir = os.path.join(PIPER_VOICES_DIR, info["locale"])
    os.makedirs(locale_dir, exist_ok=True)
    
    model_path = os.path.join(locale_dir, info["file"])
    config_path = model_path + ".json"

    # Check existence
    if os.path.exists(model_path) and os.path.exists(config_path):
        return model_path

    logger.info(f"Downloading missing voice: {info['name']}...")
    
    # Download Model
    if not await download_file(info["url_model"], model_path):
        return None
    
    # Download Config
    if not await download_file(info["url_config"], config_path):
        return None
        
    logger.info(f"Voice {info['name']} ready.")
    return model_path


async def install_piper():
    """Background task to download and set up Piper dependencies."""
    global PIPER_READY
    
    # 1. Check if valid
    if os.path.exists(PIPER_BINARY) and os.path.exists(ESPEAK_DATA):
        logger.info("Piper dependencies found locally.")
        PIPER_READY = True
        return

    logger.info("Piper dependencies missing. Starting background download...")
    
    url = "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_windows_amd64.zip"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.read()
                    
                    # Extract
                    logger.info("Download complete. Extracting Piper...")
                    with zipfile.ZipFile(io.BytesIO(data)) as z:
                        # The zip contains a root folder 'piper/'. We want contents in 'backend/piper/'
                        # So we extract to 'backend/' (BASE_DIR)
                        z.extractall(BASE_DIR)
                        
                    logger.info("Piper installation complete!")
                    PIPER_READY = True
                else:
                    logger.error(f"Failed to download Piper: HTTP {resp.status}")
    except Exception as e:
        logger.error(f"Piper Installation Error: {e}")

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(install_piper())


def get_piper_voices():
    """Returns the static catalog of voices."""
    voices = []
    for voice_id, info in VOICE_CATALOG.items():
        # Ideally we check if installed, but for listing we verify availability in catalog
        # Construct expected path
        path = os.path.join(PIPER_VOICES_DIR, info["locale"], info["file"])
        
        voices.append({
            "id": voice_id,
            "name": info["name"],
            "path": path # This is where it *should* be
        })
    return voices

@app.get("/voices")
def list_voices():
    return get_piper_voices()

# ... (Shared Logic SilenceManager - Unchanged)

async def run_piper_tts(text: str, websocket: WebSocket, model_path: str, length_scale: float = 1.0):
    """Streams text to Piper binary and audio to WebSocket."""
    if not text.strip(): return
    
    # Notify Client: AI is Speaking (Text)
    try:
        await websocket.send_json({"type": "transcript", "role": "assistant", "content": text})
    except (WebSocketDisconnect, RuntimeError):
        return

    if not PIPER_READY:
        logger.warning("Piper is not ready yet (Installing/Missing).")
        # Optional: Send a status message to frontend?
        # For now, we just skip audio generation, or maybe send a placeholder "One moment..." logic?
        # Let's just return to avoid crashing.
        return

    if not os.path.exists(PIPER_BINARY):
        logger.error(f"Piper binary not found at {PIPER_BINARY} despite READY flag.")
        return

    # ESPEAK_DATA already defined globally
    
#    logger.info(f"Piper: Generating audio for '{text[:20]}...' using model: {model_path}")
    
    command = [
        PIPER_BINARY,
        "--model", model_path,
        "--espeak_data", ESPEAK_DATA,
        "--length_scale", str(length_scale),
        "--output-raw",
    ]

    start_time = time.time()
    total_bytes = 0

    try:
#        logger.info(f"Piper Command: {' '.join(command)}")
        # Create subprocess
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Write text to stdin
        if process.stdin:
            process.stdin.write(text.encode('utf-8'))
            await process.stdin.drain()
            process.stdin.close()

        # Read stdout stream
        while True:
            chunk = await process.stdout.read(1024)
            if not chunk:
                break
            try:
                await websocket.send_bytes(chunk)
                total_bytes += len(chunk)
            except (WebSocketDisconnect, RuntimeError):
                process.terminate() # Kill process if client gone
                return

        await process.wait()
        
        if process.returncode != 0:
            err = await process.stderr.read()
            logger.error(f"Piper Error: {err.decode()}")
        
        # SIMULATION: Wait for audio duration to prevent premature turn-taking
        # Piper generates fast, but we want to block logical flow until audio "finishes" playing
        # Duration = Bytes / (Rate * Channels * BytesPerSample)
        # Piper output is 16-bit (2 bytes) Mono (1 channel)
        audio_duration = total_bytes / (TTS_RATE_PIPER * 1 * 2)
        elapsed = time.time() - start_time
        remaining = audio_duration - elapsed
        
        if remaining > 0:
            # logger.info(f"Piper: Sleeping {remaining:.2f}s to simulate playback")
            await asyncio.sleep(remaining)

    except Exception as e:
        logger.error(f"Piper Exception: {e}")


async def run_llm_and_tts(text: str, websocket: WebSocket, tts_provider: str, tts_voice_path: str, length_scale: float, task_manager: TaskStateManager, history: list):
    """Pipeline: Text -> Groq (Stream) -> Sentence Buffer -> TTS (Stream) -> WebSocket"""
    
    # 1. Update History with User Input
    history.append({"role": "user", "content": text})
    
    task_manager.is_thinking = True
    task_manager.interrupt_signal.clear()
    try:
        await websocket.send_json({"type": "status", "content": "thinking"})

        groq = Groq(api_key=GROQ_API_KEY)

        # 2. Prepare Messages (System + Last 25 Context)
        # 2. Prepare Messages (System + Last 25 Context)
        system_msg = {"role": "system", "content": "You are a helpful voice assistant. Keep answers concise (1-2 sentences) for voice output. If the user interrupts you (marked as '[User interrupted you]'), handle it naturally and stop your previous train of thought. Do not apologize for being interrupted in every turn, just move on."}
        
        # Contextual Barge-In
        if task_manager.was_interrupted:
            text += " [User interrupted you]"
            task_manager.was_interrupted = False
        
        logger.info(f"LLM Input: {text}") # Debug Log

        # Limit context to last 25 messages
        context_messages = history[-25:]
        messages = [system_msg] + context_messages

        try:
            stream = groq.chat.completions.create(
                messages=messages,
                model="llama-3.1-8b-instant",
                stream=True
            )

            sentence_buffer = ""
            full_response = "" # Track full response for history
            punctuation = {'.', '?', '!', ':', ';'} 
            
            # Reuse session for efficiency
            async with aiohttp.ClientSession() as session:

                async def speak(text_chunk):
                    if not text_chunk.strip(): return
                    if task_manager.interrupt_signal.is_set(): return # Abort if interrupted

                    task_manager.is_ai_speaking = True
                    task_manager.is_thinking = False # Thinking phase done
                    
                    if tts_provider == "piper":
                        await run_piper_tts(text_chunk, websocket, tts_voice_path, length_scale)
                    else:
                        # Deepgram Logic
                        await websocket.send_json({"type": "transcript", "role": "assistant", "content": text_chunk})
                        
                        url = f"https://api.deepgram.com/v1/speak?model=aura-asteria-en&encoding=linear16&sample_rate={TTS_RATE_DEEPGRAM}&container=none"
                        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "application/json"}
                        payload = {"text": text_chunk}

                        start_time = time.time()
                        total_bytes = 0

                        try:
                            async with session.post(url, headers=headers, json=payload) as response:
                                if response.status == 200:
                                    async for chunk in response.content.iter_chunked(8192):
                                        if task_manager.interrupt_signal.is_set(): break
                                        if chunk: 
                                            await websocket.send_bytes(chunk)
                                            total_bytes += len(chunk)
                                else:
                                    logger.error(f"TTS Error: {await response.text()}")
                            
                            # Sync State with Audio Duration
                            if total_bytes > 0:
                                audio_duration = total_bytes / (TTS_RATE_DEEPGRAM * 1 * 2) # Rate * Channels * BytesPerSample
                                elapsed = time.time() - start_time
                                remaining = audio_duration - elapsed
                                if remaining > 0:
                                    await asyncio.sleep(remaining)

                        except Exception as e:
                            logger.error(f"TTS Exception: {e}")

                for chunk in stream:
                    if task_manager.interrupt_signal.is_set(): break
                    token = chunk.choices[0].delta.content
                    if token:
                        sentence_buffer += token
                        full_response += token
                        if any(p in token for p in punctuation):
                            await speak(sentence_buffer)
                            sentence_buffer = ""
                
                if sentence_buffer and not task_manager.interrupt_signal.is_set():
                    await speak(sentence_buffer)
            
            # 3. Update History with Assistant Response
            if full_response.strip():
                history.append({"role": "assistant", "content": full_response})

            try:
                await websocket.send_json({"type": "status", "content": "listening"})
            except (WebSocketDisconnect, RuntimeError):
                pass

        except (WebSocketDisconnect, RuntimeError):
            # Normal client disconnect
            pass
        except Exception as e:
            if "Cancel" not in str(e): # Ignore cancellation errors
                 logger.error(f"LLM Error: {e}")
                 try:
                    await websocket.send_json({"type": "error", "content": str(e)})
                 except (WebSocketDisconnect, RuntimeError):
                    pass

    finally:
        task_manager.is_thinking = False
        task_manager.is_ai_speaking = False
        await asyncio.sleep(0.1)


# --- Security --- (Unchanged)
def verify_token(token: str):
    if token == "velox-secret-123":
        return True
    return True 

# --- WebSocket Endpoint ---

@app.websocket("/ws/stream")
async def audio_stream(websocket: WebSocket, token: str = Query(None), tts_provider: str = Query("deepgram"), tts_voice: str = Query(None), tts_speed: str = Query("normal"), stt_provider: str = Query("deepgram"), stt_language: str = Query("english")):
    # 1. Verification
    if not verify_token(token):
        logger.warning(f"Unauthorized connection attempt with token: {token}")
        await websocket.close(code=4003)
        return

    await websocket.accept()
    logger.info(f"Client connected [TTS: {tts_provider} | Speed: {tts_speed}]")

    # 2. Send Configuration
    sample_rate = TTS_RATE_PIPER if tts_provider == "piper" else TTS_RATE_DEEPGRAM
    await websocket.send_json({"type": "config", "sample_rate": sample_rate})
    
    # Resolve TTS Speed to Scale
    # Piper: Scale < 1.0 is FASTER, Scale > 1.0 is SLOWER
    length_scale = 1.0
    if tts_speed == "super-fast":
        length_scale = 0.6
    elif tts_speed == "fast":
        length_scale = 0.8
    elif tts_speed == "slow":
        length_scale = 1.2
    elif tts_speed == "super-slow":
        length_scale = 1.4


    silence_manager = SilenceManager()
    task_manager = TaskStateManager() # NEW: State Manager
    conversation_history = []
    
    # Resolve Voice Path for Piper
    piper_voice_path = None
    if tts_provider == "piper":
        if not PIPER_READY:
             await websocket.send_json({"type": "status", "content": "Installing TTS Engine..."})
             
             # Wait for startup task? Or just error. 
             # For simplicity, we error if critical binaries are missing, but voices we download.
             # Ideally PIPER_READY covers binaries.
        
        # Determine voice ID
        target_voice = tts_voice if tts_voice else "en_US-amy-medium" # Default
        
        # Ensure Voice Model (Download if needed)
        # Notify user if downloading
        if target_voice in VOICE_CATALOG:
             path_check = os.path.join(PIPER_VOICES_DIR, VOICE_CATALOG[target_voice]["locale"], VOICE_CATALOG[target_voice]["file"])
             if not os.path.exists(path_check):
                 await websocket.send_json({"type": "status", "content": f"Downloading Voice: {VOICE_CATALOG[target_voice]['name']}..."})

        piper_voice_path = await ensure_voice_model(target_voice)
        
        if not piper_voice_path:
            logger.error(f"Failed to prepare Piper voice: {target_voice}")
            # Fallback to ANY existing voice? Or fail.
            # Try finding one from get_piper_voices as fallback
            fallback_list = get_piper_voices()
            for v in fallback_list:
                if os.path.exists(v["path"]):
                    piper_voice_path = v["path"]
                    break
        
        if not piper_voice_path:
             logger.error("No usable Piper voices found.")
             await websocket.close(code=4002)
             return

    
    # 3. STT Setup (Gladia or Deepgram)
    
    if stt_provider == "gladia":
        if not GLADIA_API_KEY:
            logger.error("Gladia API Key missing")
            await websocket.close(code=4000)
            return

        gladia_url = "wss://api.gladia.io/audio/text/audio-transcription"
        
        # Retry Logic for Gladia
        for attempt in range(1, 4):
            try:
                async with websockets.connect(gladia_url) as gladia_ws:
                    # Send Config
                    config = {
                        "x_gladia_key": GLADIA_API_KEY,
                        "sample_rate": 16000,
                        "encoding": "wav",
                        "language_behaviour": "manual",
                        "language": stt_language,
                        "frames_format": "base64",
                        "endpointing": 100, # Native Endpointing (ms)
                    }
                    await gladia_ws.send(json.dumps(config))
                    logger.info("Connected to Gladia Utils")

                    # Keep Alive Task (Gladia needs activity)
                    async def keep_alive():
                        while True:
                            await asyncio.sleep(5)
                            try:
                                await gladia_ws.send(json.dumps({"type": "keep_alive"})) 
                            except Exception:
                                break
                    
                    keep_alive_task = asyncio.create_task(keep_alive())

                    # Receive Task for Gladia
                    async def receive_gladia():
                        try:
                            async for message in gladia_ws:
                                data = json.loads(message)
                                if "transcription" in data and data["transcription"]:
                                    text = data["transcription"]
                                    is_final = data.get("type") == "final"
                                    
                                    silence_manager.update_activity()
                                    if is_final:
                                        logger.info(f"User (Gladia Final): {text}")
                                        silence_manager.add_text(text)
                                        await websocket.send_json({"type": "transcript", "role": "user", "content": text})
                                        
                                        # Trigger LLM Immediately on Final (Endpointing)
                                        await task_manager.schedule_llm_task(
                                            run_llm_and_tts(text, websocket, tts_provider, piper_voice_path, length_scale, task_manager, conversation_history)
                                        )
                                    else:
                                        print(f"User (Gladia Partial): {text}") # Print partials
                                        
                                        # Transcript-based Interruption (Partial or Final)
                                        if text and (task_manager.is_ai_speaking or task_manager.is_thinking):
                                            await task_manager.handle_interruption(websocket)
                        except ConnectionClosed:
                            logger.warning("Gladia Connection Closed (Normal)")
                        except Exception as e:
                            logger.error(f"Gladia Receive Error: {e}")

                    gladia_receiver = asyncio.create_task(receive_gladia())
                    
                    try:
                        while True:
                            # Receive Audio from Client
                            data = await websocket.receive_bytes()
                            
                            # Send to Gladia
                            base64_data = base64.b64encode(data).decode("utf-8")
                            await gladia_ws.send(json.dumps({"frames": base64_data}))
                    
                    except WebSocketDisconnect:
                        logger.info("Client disconnected")
                    finally:
                        gladia_receiver.cancel()
                        keep_alive_task.cancel()
                        await gladia_ws.close() # Explicit close for Max Sessions Fix
                    
                    # If we exit loop normally (client disconnect), break retry
                    break

            except Exception as e:
                logger.warning(f"Gladia Connection Attempt {attempt} failed: {e}")
                if attempt < 3:
                     # Wait for previous session to clear (Max sessions 4129)
                    await asyncio.sleep(2)
                else:
                    logger.error("Gladia Connection Failed after 3 attempts")
                    await websocket.close()
                    return

        return

    # Default: Deepgram Logic (Manual WebSocket)
    deepgram_url = (
        f"wss://api.deepgram.com/v1/listen?"
        f"model=nova-2&language=en-US&smart_format=true&encoding=linear16&sample_rate=16000&channels=1"
        f"&interim_results=true&utterance_end_ms=1000&vad_events=true&endpointing=100"
    )
    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}

    # Retry Logic for Deepgram Connection
    connected = False
    for attempt in range(1, 4):
        try:
            async with websockets.connect(deepgram_url, additional_headers=headers) as dg_ws:
                logger.info("Connected to Deepgram (Manual WS)")
                connected = True

                # 1. Keep Alive Task
                async def keep_alive():
                    while True:
                        await asyncio.sleep(5)
                        try:
                            await dg_ws.send(json.dumps({"type": "KeepAlive"}))
                        except Exception:
                            break
                
                keep_alive_task = asyncio.create_task(keep_alive())

                # 2. Receive Task
                async def receive_deepgram():
                    try:
                        async for message in dg_ws:
                            msg = json.loads(message)
                            msg_type = msg.get("type")

                            if msg_type == "Results":
                                if "channel" in msg and "alternatives" in msg["channel"]:
                                    alt = msg["channel"]["alternatives"][0]
                                    sentence = alt.get("transcript", "")
                                    if not sentence: continue

                                    silence_manager.update_activity()
                                    is_final = msg.get("is_final")

                                    if is_final:
                                        logger.info(f"User (Deepgram Final): {sentence}")
                                        silence_manager.add_text(sentence)
                                        await websocket.send_json({"type": "transcript", "role": "user", "content": sentence})
                                        
                                        # Trigger LLM Immediately on Final (Endpointing Fallback)
                                        # Since we use endpointing=100, is_final means 100ms silence.
                                        # We treat this as a turn completion.
                                        full_text = silence_manager.flush()
                                        if full_text:
                                            print("Deepgram Final -> Triggering LLM")
                                            await task_manager.schedule_llm_task(
                                                run_llm_and_tts(full_text, websocket, tts_provider, piper_voice_path, length_scale, task_manager, conversation_history)
                                            )
                                    else:
                                        print(f"User (Deepgram Partial): {sentence}")

                                    # Transcript-based Interruption
                                    if sentence and (task_manager.is_ai_speaking or task_manager.is_thinking):
                                        await task_manager.handle_interruption(websocket)

                            elif msg_type == "UtteranceEnd":
                                # Native Endpointing Trigger
                                text = silence_manager.flush() # This cleans the buffer
                                if text:
                                    logger.info("Deepgram UtteranceEnd -> Triggering LLM")
                                    await task_manager.schedule_llm_task(
                                        run_llm_and_tts(text, websocket, tts_provider, piper_voice_path, length_scale, task_manager, conversation_history)
                                    )

                    except Exception as e:
                        logger.error(f"Deepgram Receiver Error: {e}")

                receiver_task = asyncio.create_task(receive_deepgram())

                # 3. Main Loop
                try:
                    while True:
                        # Receive Audio from Client
                        try:
                            data = await asyncio.wait_for(websocket.receive_bytes(), timeout=0.1)
                            
                            await dg_ws.send(data)

                        except asyncio.TimeoutError:
                            continue 
                
                except WebSocketDisconnect:
                    logger.info("Client disconnected")
                    return # Exit cleanly on disconnect
                finally:
                    keep_alive_task.cancel()
                    receiver_task.cancel()
                
                # If we exit the context manager safely (unlikely unless return/break), we break retry
                break 

        except Exception as e:
            logger.warning(f"Deepgram Connection Attempt {attempt} failed: {e}")
            if attempt < 3:
                await asyncio.sleep(attempt * 1) # Exponential backoff: 1s, 2s
            else:
                logger.error("Deepgram Connection Failed after 3 attempts")
                await websocket.close()
                return


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
