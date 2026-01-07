from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
import os
import json
import logging
import threading
import time
import time
import aiohttp
import websockets
import base64
import struct
import math
import numpy as np
import onnxruntime as ort
from dotenv import load_dotenv

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
)
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
    """Manages the conversation state and handles interruptions using Silero VAD."""
    def __init__(self):
        self.is_user_speaking_now = False
        self.is_ai_speaking = False
        self.is_thinking = False
        
        # Reference to the current LLM task to allow cancellation
        self.llm_task: asyncio.Task = None
        self.interrupt_signal = threading.Event()
        self.was_interrupted = False # Context flag


        # VAD State
        self.vad_model_path = "silero_vad.onnx"
        self.vad_session = ort.InferenceSession(self.vad_model_path)
        
        # Silero VAD internal state (h, c) - Shape (2, 1, 64)
        self.vad_h = np.zeros((2, 1, 64), dtype=np.float32)
        self.vad_c = np.zeros((2, 1, 64), dtype=np.float32)
        
        self.vad_buffer = bytearray()
        self.vad_window_size = 512 # Silero expects specific chunk sizes (512 is common for 16k)
        
    def check_silero_vad(self, audio_chunk: bytes, threshold=0.5) -> bool:
        """accumulates audio and runs Silero VAD inference."""
        self.vad_buffer.extend(audio_chunk)
        
        triggered = False
        
        while len(self.vad_buffer) >= self.vad_window_size * 2: # 2 bytes per sample
            chunk_bytes = self.vad_buffer[:self.vad_window_size * 2]
            self.vad_buffer = self.vad_buffer[self.vad_window_size * 2:]
            
            # Convert PCM16 -> Float32
            # equivalent to: x = torch.from_numpy(np.frombuffer(audio_chunk, dtype=np.int16)).float() / 32768.0
            input_tensor = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            input_tensor = input_tensor.reshape(1, -1) # Add batch dimension (1, 512)
            
            # Run Inference
            # input size: [1, 512], h: [2, 1, 64], c: [2, 1, 64], sr: [1]
            ort_inputs = {
                "input": input_tensor,
                "state": self.vad_h,
                "context": self.vad_c, # Note: some versions name it 'state' and 'context', check model inputs? 
                # Usually v4 takes input, state, sr. v5 might differ. 
                # Let's assume standard sequence for v4: input, state, sr_tensor
            }
            
            # Specific handling for Silero v4 ONNX inputs
            # Inputs: input (1, N), state (2, 1, 64), sr (1)
            # Outputs: output (1, 1), state (2, 1, 64)
            
            # Prepare SR tensor
            sr_tensor = np.array([16000], dtype=np.int64)
            
            ort_inputs = {
                "input": input_tensor,
                "state": np.concatenate((self.vad_h, self.vad_c), axis=0), # Usually joined? No, v4 is likely just 'state'
                "sr": sr_tensor
            }
            
            # WAIT: Silero V4 often takes 'input', 'state' (2,1,128) or similar.
            # Let's inspect, or simplest is to try catch. But 'state' usually concatenates h and c?
            # Actually, standard V4 ONNX signature:
            # Input: 'input' (Batch, Time), 'state' (2, Batch, 64), 'sr' (1)
            # Output: 'output', 'stateN'
            
            # Let's try standard V4 construction
            # We treat h and c together as 'state'
            context = np.zeros((2, 1, 64), dtype=np.float32) # Dummy init if needed, but we persist
            
            # For first run allow zero state
            current_state = np.concatenate((self.vad_h, self.vad_c), axis=0) if self.vad_h is not None else np.zeros((2, 1, 128), dtype=np.float32)
            
            # Wait, easier approach: Re-check model inputs if possible. 
            # Or use 'silero-vad' pip logic? 
            # I will assume standard signature: input, state, sr.
            
            # We track 'vad_state' as (2, 1, 128) for simplicity if that's what it wants.
            # Actually, let's keep it simple: Start with zeros (2, 1, 128).
            
            # Silero V5 Input Names: input, sr, h, c
            ort_inputs = {
                "input": input_tensor,
                "sr": sr_tensor,
                "h": self.vad_h,
                "c": self.vad_c
            }
            
            output, new_h, new_c = self.vad_session.run(None, ort_inputs)
            self.vad_h = new_h
            self.vad_c = new_c
            
            prob = output[0][0]
            if prob > threshold:
                triggered = True
                
        return triggered

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

    def get_if_silence(self):
        if not self.is_user_speaking:
            return None
        
        if time.time() - self.last_activity > SILENCE_TIMEOUT:
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

# Piper Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPER_BINARY = os.path.join(BASE_DIR, "piper", "piper.exe")
PIPER_VOICES_DIR = os.path.join(BASE_DIR, "piper", "voices")

def get_piper_voices():
    """Recursively finds all .onnx voice models in the voices directory."""
    voices = []
    # Recursive glob for .onnx files
    pattern = os.path.join(PIPER_VOICES_DIR, "**", "*.onnx")
    for file_path in glob.glob(pattern, recursive=True):
        # Create readable ID/Name
        # ID = Relative path from voices dir (e.g., "usa\en_US-amy-medium.onnx")
        rel_path = os.path.relpath(file_path, PIPER_VOICES_DIR)
        
        # Name = Filename without extension (e.g., "en_US-amy-medium")
        name = os.path.splitext(os.path.basename(file_path))[0]
        
        voices.append({
            "id": rel_path,
            "name": name,
            "path": file_path
        })
    return voices

@app.get("/voices")
def list_voices():
    return get_piper_voices()

# ... (Shared Logic SilenceManager - Unchanged)

async def run_piper_tts(text: str, websocket: WebSocket, model_path: str):
    """Streams text to Piper binary and audio to WebSocket."""
    if not text.strip(): return
    
    # Notify Client: AI is Speaking (Text)
    await websocket.send_json({"type": "transcript", "role": "assistant", "content": text})

    if not os.path.exists(PIPER_BINARY):
        logger.error(f"Piper binary not found at {PIPER_BINARY}")
        return

    ESPEAK_DATA = os.path.join(BASE_DIR, "piper", "espeak-ng-data")
    
    logger.info(f"Piper: Generating audio for '{text[:20]}...' using model: {model_path}")
    
    command = [
        PIPER_BINARY,
        "--model", model_path,
        "--espeak_data", ESPEAK_DATA,
        "--output-raw",
    ]

    start_time = time.time()
    total_bytes = 0

    try:
        logger.info(f"Piper Command: {' '.join(command)}")
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
            await websocket.send_bytes(chunk)
            total_bytes += len(chunk)

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


async def run_llm_and_tts(text: str, websocket: WebSocket, tts_provider: str, tts_voice_path: str, task_manager: TaskStateManager, history: list):
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
                        await run_piper_tts(text_chunk, websocket, tts_voice_path)
                    else:
                        # Deepgram Logic
                        await websocket.send_json({"type": "transcript", "role": "assistant", "content": text_chunk})
                        
                        url = f"https://api.deepgram.com/v1/speak?model=aura-asteria-en&encoding=linear16&sample_rate={TTS_RATE_DEEPGRAM}&container=none"
                        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "application/json"}
                        payload = {"text": text_chunk}

                        try:
                            async with session.post(url, headers=headers, json=payload) as response:
                                if response.status == 200:
                                    async for chunk in response.content.iter_chunked(8192):
                                        if task_manager.interrupt_signal.is_set(): break
                                        if chunk: 
                                            await websocket.send_bytes(chunk)
                                else:
                                    logger.error(f"TTS Error: {await response.text()}")
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

            await websocket.send_json({"type": "status", "content": "listening"})

        except Exception as e:
            if "Cancel" not in str(e): # Ignore cancellation errors
                 logger.error(f"LLM Error: {e}")
                 await websocket.send_json({"type": "error", "content": str(e)})

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
async def audio_stream(websocket: WebSocket, token: str = Query(None), tts_provider: str = Query("deepgram"), tts_voice: str = Query(None), stt_provider: str = Query("deepgram"), stt_language: str = Query("english")):
    # 1. Verification
    if not verify_token(token):
        logger.warning(f"Unauthorized connection attempt with token: {token}")
        await websocket.close(code=4003)
        return

    await websocket.accept()
    logger.info(f"Client connected [TTS: {tts_provider}]")

    # 2. Send Configuration
    sample_rate = TTS_RATE_PIPER if tts_provider == "piper" else TTS_RATE_DEEPGRAM
    await websocket.send_json({"type": "config", "sample_rate": sample_rate})

    silence_manager = SilenceManager()
    task_manager = TaskStateManager() # NEW: State Manager
    conversation_history = []
    
    # Resolve Voice Path for Piper
    piper_voice_path = None
    if tts_provider == "piper":
        voices = get_piper_voices()
        if tts_voice:
             # Find matching voice by ID
             found = next((v for v in voices if v["id"] == tts_voice), None)
             if found:
                 piper_voice_path = found["path"]
        
        if not piper_voice_path and voices:
            # Default to first
            piper_voice_path = voices[0]["path"]
            logger.info(f"Defaulting to Piper Voice: {voices[0]['name']}")
            
        if not piper_voice_path:
            logger.error("No Piper voices found!")
            await websocket.close(code=4002) # Custom code for misconfig
            return
    
    # 3. STT Setup (Gladia or Deepgram)
    
    if stt_provider == "gladia":
        if not GLADIA_API_KEY:
            logger.error("Gladia API Key missing")
            await websocket.close(code=4000)
            return

        gladia_url = "wss://api.gladia.io/audio/text/audio-transcription"
        
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
                    "endpointing": 800, # Native Endpointing (ms)
                }
                await gladia_ws.send(json.dumps(config))
                logger.info("Connected to Gladia Utils")

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
                                        run_llm_and_tts(text, websocket, tts_provider, piper_voice_path, task_manager, conversation_history)
                                    )
                                else:
                                    print(f"User (Gladia Partial): {text}") # Print partials
                                    
                                    # Transcript-based Interruption (Partial or Final)
                                    if text and (task_manager.is_ai_speaking or task_manager.is_thinking):
                                        await task_manager.handle_interruption(websocket)
                    except Exception as e:
                        logger.error(f"Gladia Receive Error: {e}")

                gladia_receiver = asyncio.create_task(receive_gladia())
                
                # silence_checker removed (Native Endpointing)
                
                try:
                    while True:
                        # Receive Audio from Client
                        data = await websocket.receive_bytes()
                        
                        # VAD Check (Silero REMOVED - using Transcript VAD)
                        # if task_manager.check_silero_vad(data):
                        #    await task_manager.handle_interruption(websocket)

                        # Send to Gladia
                        base64_data = base64.b64encode(data).decode("utf-8")
                        await gladia_ws.send(json.dumps({"frames": base64_data}))
                
                except WebSocketDisconnect:
                    logger.info("Client disconnected")
                finally:
                    gladia_receiver.cancel()
                    # checker_task.cancel() # Removed

        except Exception as e:
             logger.error(f"Gladia Connection Error: {e}")
             await websocket.close()
             return

        return

    # Default: Deepgram Logic (Manual WebSocket)
    deepgram_url = (
        f"wss://api.deepgram.com/v1/listen?"
        f"model=nova-2&language=en-US&smart_format=true&encoding=linear16&sample_rate=16000&channels=1"
        f"&interim_results=true&utterance_end_ms=1000&vad_events=true"
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
                                    else:
                                        print(f"User (Deepgram Partial): {sentence}")

                                    # Transcript-based Interruption
                                    if sentence and (task_manager.is_ai_speaking or task_manager.is_thinking):
                                        await task_manager.handle_interruption(websocket)

                            elif msg_type == "UtteranceEnd":
                                # Native Endpointing Trigger
                                text = silence_manager.get_if_silence() # This cleans the buffer
                                if text:
                                    logger.info("Deepgram UtteranceEnd -> Triggering LLM")
                                    await task_manager.schedule_llm_task(
                                        run_llm_and_tts(text, websocket, tts_provider, piper_voice_path, task_manager, conversation_history)
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
                            
                            # VAD CHECK (Silero REMOVED - using Transcript VAD)
                            # if task_manager.check_silero_vad(data):
                            #      await task_manager.handle_interruption(websocket)

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
