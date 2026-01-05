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
SILENCE_TIMEOUT = 1.0

# --- Shared Logic (Ported from Main Test) ---

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

# Piper Configuration (Hardcoded for Demo)
PIPER_PATH = r"d:\piper\piper.exe"
PIPER_MODEL_PATH = r"d:\piper\en_US-hfc_female-medium.onnx"

# ... (Shared Logic SilenceManager - Unchanged)

async def run_piper_tts(text: str, websocket: WebSocket):
    """Streams text to Piper binary and audio to WebSocket."""
    if not text.strip(): return
    
    # Notify Client: AI is Speaking (Text)
    await websocket.send_json({"type": "transcript", "role": "assistant", "content": text})

    command = [
        PIPER_PATH,
        "--model", PIPER_MODEL_PATH,
        "--output-raw",
    ]

    start_time = time.time()
    total_bytes = 0

    try:
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


async def run_llm_and_tts(text: str, websocket: WebSocket, tts_provider: str, state: dict, history: list):
    """Pipeline: Text -> Groq (Stream) -> Sentence Buffer -> TTS (Stream) -> WebSocket"""
    logger.info(f"Processing: {text} [TTS: {tts_provider}]")
    
    # 1. Update History with User Input
    history.append({"role": "user", "content": text})

    # state["is_speaking"] = True # Gate Removed for Full Duplex
    try:
        await websocket.send_json({"type": "status", "content": "thinking"})

        groq = Groq(api_key=GROQ_API_KEY)

        # 2. Prepare Messages (System + Last 25 Context)
        system_msg = {"role": "system", "content": "You are a helpful voice assistant. Keep answers concise (1-2 sentences) for voice output."}
        
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

            async def speak(text_chunk):
                if not text_chunk.strip(): return
                
                if tts_provider == "piper":
                    await run_piper_tts(text_chunk, websocket)
                else:
                    # Deepgram Logic
                    await websocket.send_json({"type": "transcript", "role": "assistant", "content": text_chunk})
                    
                    url = f"https://api.deepgram.com/v1/speak?model=aura-asteria-en&encoding=linear16&sample_rate={TTS_RATE_DEEPGRAM}&container=none"
                    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "application/json"}
                    payload = {"text": text_chunk}

                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.post(url, headers=headers, json=payload) as response:
                                if response.status == 200:
                                    async for chunk in response.content.iter_chunked(1024):
                                        if chunk: 
                                            await websocket.send_bytes(chunk)
                                else:
                                    logger.error(f"TTS Error: {await response.text()}")
                    except Exception as e:
                        logger.error(f"TTS Exception: {e}")

            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    sentence_buffer += token
                    full_response += token
                    if any(p in token for p in punctuation):
                        await speak(sentence_buffer)
                        sentence_buffer = ""
            
            if sentence_buffer:
                await speak(sentence_buffer)
            
            # 3. Update History with Assistant Response
            if full_response.strip():
                history.append({"role": "assistant", "content": full_response})

            await websocket.send_json({"type": "status", "content": "listening"})

        except Exception as e:
            logger.error(f"LLM Error: {e}")
            await websocket.send_json({"type": "error", "content": str(e)})

    finally:
        # VERY IMPORTANT: Release the gate
        # Add a small buffer to ensure echo dies down?
        await asyncio.sleep(0.5) 
        # state["is_speaking"] = False


# --- Security --- (Unchanged)
def verify_token(token: str):
    if token == "velox-secret-123":
        return True
    return True 

# --- WebSocket Endpoint ---

@app.websocket("/ws/stream")
async def audio_stream(websocket: WebSocket, token: str = Query(None), tts_provider: str = Query("deepgram"), stt_provider: str = Query("deepgram"), stt_language: str = Query("english")):
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
    server_state = {"is_speaking": False}
    conversation_history = []
    
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
                                    logger.info(f"User (Gladia): {text}")
                                    silence_manager.add_text(text)
                                    await websocket.send_json({"type": "transcript", "role": "user", "content": text})
                    except Exception as e:
                        logger.error(f"Gladia Receive Error: {e}")

                gladia_receiver = asyncio.create_task(receive_gladia())
                
                # Main Loop (Sending to Gladia)
                async def silence_checker():
                    while True:
                        await asyncio.sleep(0.1)
                        text = silence_manager.get_if_silence()
                        if text:
                             await run_llm_and_tts(text, websocket, tts_provider, server_state, conversation_history)

                checker_task = asyncio.create_task(silence_checker())
                
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
                    checker_task.cancel()

        except Exception as e:
             logger.error(f"Gladia Connection Error: {e}")
             await websocket.close()
             return

        return

    # Default: Deepgram Logic
    try:
        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram = DeepgramClient(DEEPGRAM_API_KEY, config)
        dg_connection = deepgram.listen.asyncwebsocket.v("1")

        async def on_message(self, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            if not sentence: return
            
            # Double check: We want to process EVERYTHING now for full duplex.
            # if server_state["is_speaking"] and tts_provider == "piper":
            #      return

            silence_manager.update_activity()
            
            if result.is_final:
                logger.info(f"User: {sentence}")
                silence_manager.add_text(sentence)
                await websocket.send_json({"type": "transcript", "role": "user", "content": sentence})

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        
        options = LiveOptions(
            model="nova-2", 
            language="en-US", 
            smart_format=True, 
            encoding="linear16", 
            channels=1, 
            sample_rate=16000,
            interim_results=True
        )

        # Connect with Retry Logic
        connected = False
        for attempt in range(1, 4):
            try:
                logger.info(f"Connecting to Deepgram (Attempt {attempt}/3)...")
                if await dg_connection.start(options) is True:
                    connected = True
                    logger.info("Deepgram connected successfully.")
                    break
            except Exception as e:
                logger.warning(f"Deepgram connection attempt {attempt} failed: {e}")
            
            if attempt < 3:
                wait_time = attempt * 1
                logger.info(f"Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)

        if not connected:
            logger.error("Failed to connect to Deepgram after 3 attempts")
            await websocket.close()
            return

        # 4. Main Loop
        async def silence_checker():
            while True:
                await asyncio.sleep(0.1)
                text = silence_manager.get_if_silence()
                if text:
                     await run_llm_and_tts(text, websocket, tts_provider, server_state, conversation_history)

        checker_task = asyncio.create_task(silence_checker())

        try:
            while True:
                # Receive Audio from Client
                try:
                    data = await asyncio.wait_for(websocket.receive_bytes(), timeout=0.1)
                    
                    # GATE REMOVED: Full Duplex Mode (Barge-In compliant)
                    # We send all audio to Deepgram. Browser AEC should handle echo.
                    # Future VAD will handle logical interruption.
                    await dg_connection.send(data)

                except asyncio.TimeoutError:
                    continue 
        
        except WebSocketDisconnect:
            logger.info("Client disconnected")
        except Exception as e:
            logger.error(f"Connection error: {e}")
        finally:
            checker_task.cancel()
            await dg_connection.finish()

    except Exception as e:
        logger.error(f"Deepgram setup error: {e}")
        # await websocket.close() # Avoid closing if connection failed earlier, but safe to keep


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
