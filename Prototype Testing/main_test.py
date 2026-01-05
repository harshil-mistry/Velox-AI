import asyncio
import pyaudio
import os
import threading
import queue
import time
import requests
from dotenv import load_dotenv
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
)
from groq import Groq

# Load environment variables
load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Configuration
MIC_RATE = 16000
MIC_CHUNK = 1024
TTS_RATE = 24000  # Aura Asteria
SILENCE_TIMEOUT = 1.0  # Seconds

class AsyncMic:
    """Captures audio in a separate thread and puts it in an async queue."""
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=MIC_RATE,
            input=True,
            frames_per_buffer=MIC_CHUNK,
        )
        self.running = False
        self.is_muted = False
        self.queue = asyncio.Queue()
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def _record_loop(self, loop_instance):
        while self.running:
            try:
                data = self.stream.read(MIC_CHUNK, exception_on_overflow=False)
                # Only enqueue if not muted
                if not self.is_muted:
                    loop_instance.call_soon_threadsafe(self.queue.put_nowait, data)
            except Exception as e:
                print(f"Mic Error: {e}")
                break

    def start(self):
        self.running = True
        self.main_loop = asyncio.get_event_loop()
        self.thread = threading.Thread(target=self._record_loop, args=(self.main_loop,))
        self.thread.start()
        print("\n🔴 Mic Active")

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    async def get_audio_chunk(self):
        return await self.queue.get()
        
    def flush(self):
        """Empties the queue."""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass

class AudioPlayer:
    """Plays raw PCM audio chunks."""
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=TTS_RATE,
            output=True
        )

    def play_chunk(self, chunk):
        self.stream.write(chunk)

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

class SilenceManager:
    """Manages the user state (Speaking/Silent) to trigger the AI."""
    def __init__(self):
        self.last_activity = time.time()
        self.buffer = []
        self.lock = threading.Lock()
        self.is_user_speaking = False

    def update_activity(self):
        """Call this whenever user makes a sound (interim or final)."""
        self.last_activity = time.time()
        self.is_user_speaking = True

    def add_text(self, text):
        """Add finalized text to the buffer."""
        with self.lock:
            self.buffer.append(text)

    def get_if_silence(self):
        """Returns the full text if silence timeout is reached, else None."""
        if not self.is_user_speaking:
            return None
            
        if time.time() - self.last_activity > SILENCE_TIMEOUT:
            with self.lock:
                if self.buffer:
                    full_text = " ".join(self.buffer)
                    self.buffer = [] # Clear buffer
                    self.is_user_speaking = False # Reset flag so we don't trigger again until new speech
                    return full_text.strip()
        return None

async def run_llm_and_tts(text):
    """Pipeline: Text -> Groq (Stream) -> Sentence Buffer -> TTS (Stream) -> Audio"""
    print(f"\n[User Final]: {text}")
    print("🤖 AI Thinking...")
    
    groq = Groq(api_key=GROQ_API_KEY)
    player = AudioPlayer()

    # Context setup
    messages = [
        {"role": "system", "content": "You are a helpful voice assistant. Keep answers concise (1-2 sentences) for voice output."},
        {"role": "user", "content": text}
    ]

    # Groq Streaming
    stream = groq.chat.completions.create(
        messages=messages,
        model="llama-3.1-8b-instant",
        stream=True
    )

    sentence_buffer = ""
    # Simple punctuation splitting for "streaming" TTS feel
    punctuation = {'.', '?', '!', ':', ';'} 

    def speak(text_chunk):
        if not text_chunk.strip(): return
        
        # Deepgram Aura TTS (Raw Streaming)
        url = f"https://api.deepgram.com/v1/speak?model=aura-asteria-en&encoding=linear16&sample_rate={TTS_RATE}&container=none"
        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "application/json"}
        payload = {"text": text_chunk}

        try:
            with requests.post(url, headers=headers, json=payload, stream=True) as response:
                if response.status_code == 200:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk: player.play_chunk(chunk)
                else:
                    print(f"TTS Error: {response.text}")
        except Exception as e:
            print(f"TTS Exception: {e}")

    print("🔊 AI Speaking: ", end="", flush=True)

    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            print(token, end="", flush=True)
            sentence_buffer += token
            
            # Check if we have a full sentence/clause
            if any(p in token for p in punctuation):
                speak(sentence_buffer)
                sentence_buffer = ""
    
    # Speak any remaining text
    if sentence_buffer:
        speak(sentence_buffer)
    
    print("\n✅ Done.")
    player.close()

async def main():
    if not DEEPGRAM_API_KEY or not GROQ_API_KEY:
        print("Error: Missing API Keys in .env")
        return

    # Initialize Components
    mic = AsyncMic()
    silence_manager = SilenceManager()
    
    # Deepgram STT Setup
    deepgram = DeepgramClient(DEEPGRAM_API_KEY)
    dg_connection = deepgram.listen.asyncwebsocket.v("1")

    async def on_message(self, result, **kwargs):
        sentence = result.channel.alternatives[0].transcript
        if not sentence: return

        # Reset silence timer on any speech
        silence_manager.update_activity()

        if result.is_final:
            # print(f" [Buffer]: {sentence}")
            silence_manager.add_text(sentence)

    dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
    
    options = LiveOptions(
        model="nova-2", 
        language="en-US", 
        smart_format=True, 
        encoding="linear16", 
        channels=1, 
        sample_rate=MIC_RATE,
        interim_results=True # Critical for VAD resetting
    )

    if await dg_connection.start(options) is False:
        print("Failed to connect to Deepgram STT")
        return

    mic.start()
    
    print("\n💬 System Ready. Speak now! (Waiting for 1s silence to reply)")

    # Main Event Loop
    try:
        while True:
            # 1. Process Audio
            data = await mic.get_audio_chunk()
            await dg_connection.send(data)

            # 2. Check Silence Trigger
            user_text_block = silence_manager.get_if_silence()
            if user_text_block:
                # Pause Mic processing during AI response to prevent self-interruption (Echo)
                print("\n[State] Muting Mic to prevent Echo...")
                mic.is_muted = True
                
                await run_llm_and_tts(user_text_block)
                
                # Flush the buffer to remove any echo recorded during the very end
                mic.flush()
                mic.is_muted = False
                print("[State] Mic Unmuted. Listening...")
                
                # Reset activity so we don't trigger immediately again
                # silence_manager.update_activity() <--- REMOVED: We want to wait for REAL user speech
                print("\n💬 Listening again...")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        mic.stop()
        await dg_connection.finish()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
