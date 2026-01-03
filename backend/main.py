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

        await process.wait()
        
        if process.returncode != 0:
            err = await process.stderr.read()
            logger.error(f"Piper Error: {err.decode()}")

    except Exception as e:
        logger.error(f"Piper Exception: {e}")


async def run_llm_and_tts(text: str, websocket: WebSocket, tts_provider: str):
    """Pipeline: Text -> Groq (Stream) -> Sentence Buffer -> TTS (Stream) -> WebSocket"""
    logger.info(f"Processing: {text} [TTS: {tts_provider}]")
    
    await websocket.send_json({"type": "status", "content": "thinking"})

    groq = Groq(api_key=GROQ_API_KEY)

    messages = [
        {"role": "system", "content": "You are a helpful voice assistant. Keep answers concise (1-2 sentences) for voice output."},
        {"role": "user", "content": text}
    ]

    try:
        stream = groq.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant",
            stream=True
        )

        sentence_buffer = ""
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
                if any(p in token for p in punctuation):
                    await speak(sentence_buffer)
                    sentence_buffer = ""
        
        if sentence_buffer:
            await speak(sentence_buffer)
            
        await websocket.send_json({"type": "status", "content": "listening"})

    except Exception as e:
        logger.error(f"LLM Error: {e}")
        await websocket.send_json({"type": "error", "content": str(e)})


# --- Security --- (Unchanged)
def verify_token(token: str):
    if token == "velox-secret-123":
        return True
    return True 

# --- WebSocket Endpoint ---

@app.websocket("/ws/stream")
async def audio_stream(websocket: WebSocket, token: str = Query(None), tts_provider: str = Query("deepgram")):
    # 1. Verification
    if not verify_token(token):
        logger.warning(f"Unauthorized connection attempt with token: {token}")
        await websocket.close(code=4003)
        return

    await websocket.accept()
    logger.info(f"Client connected [TTS: {tts_provider}]")

    # 2. Send Configuration to Client
    # Tell frontend which sample rate to use for playback
    sample_rate = TTS_RATE_PIPER if tts_provider == "piper" else TTS_RATE_DEEPGRAM
    await websocket.send_json({"type": "config", "sample_rate": sample_rate})

    silence_manager = SilenceManager()
    
    # 3. Setup Deepgram STT
    try:
        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram = DeepgramClient(DEEPGRAM_API_KEY, config)
        dg_connection = deepgram.listen.asyncwebsocket.v("1")

        async def on_message(self, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            if not sentence: return
            
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

        if await dg_connection.start(options) is False:
            logger.error("Failed to connect to Deepgram")
            await websocket.close()
            return

        # 4. Main Loop
        async def silence_checker():
            while True:
                await asyncio.sleep(0.1)
                text = silence_manager.get_if_silence()
                if text:
                     await run_llm_and_tts(text, websocket, tts_provider)

        checker_task = asyncio.create_task(silence_checker())

        try:
            while True:
                # Receive Audio from Client
                try:
                    data = await asyncio.wait_for(websocket.receive_bytes(), timeout=0.1)
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
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
