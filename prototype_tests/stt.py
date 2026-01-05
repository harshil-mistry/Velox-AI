import asyncio
import pyaudio
import os
import threading
import queue
from dotenv import load_dotenv
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
)

load_dotenv()

API_KEY = os.getenv("DEEPGRAM_API_KEY")
RATE = 16000 
CHUNK = 1024

class AsyncMic:
    """
    Captures audio in a separate thread so it doesn't block the AsyncIO loop.
    Puts raw bytes into a thread-safe Queue.
    """
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )
        self.running = False
        self.queue = asyncio.Queue() # Bridge between Audio Thread and Async Loop
        self.loop = asyncio.get_event_loop()

    def _record_loop(self):
        """This runs in a background thread."""
        while self.running:
            try:
                # Read raw data (Blocking operation)
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                # Safely put data into the Async Queue from this thread
                self.loop.call_soon_threadsafe(self.queue.put_nowait, data)
            except Exception as e:
                print(f"Mic Error: {e}")
                break

    def start(self):
        self.running = True
        # Start the recording in a separate thread
        self.thread = threading.Thread(target=self._record_loop)
        self.thread.start()
        print("\n🔴 Recording... (Speak into your microphone)")

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        
    async def get_audio_chunk(self):
        """Async method to get the next audio chunk."""
        return await self.queue.get()

async def main():
    if not API_KEY:
        print("Please set DEEPGRAM_API_KEY in your .env file")
        return

    # 1. Initialize Deepgram
    config = DeepgramClientOptions(options={"keepalive": "true"})
    deepgram = DeepgramClient(API_KEY, config)
    dg_connection = deepgram.listen.asyncwebsocket.v("1")

    # 2. Event Handler (Prints Transcripts)
# 2. Event Handler (Prints Transcripts)
    async def on_message(self, result, **kwargs):
        sentence = result.channel.alternatives[0].transcript
        
        # ONLY print if the sentence is finished
        if len(sentence) > 0 and result.is_final:
            print(f"User: {sentence}")

    dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

    # 3. Connect to WebSocket
    options = LiveOptions(
        model="nova-2", 
        language="en-US", 
        smart_format=True, 
        encoding="linear16", 
        channels=1, 
        sample_rate=RATE,
        interim_results=True
    )
    
    print("Connecting to Deepgram...")
    if await dg_connection.start(options) is False:
        print("Failed to connect to Deepgram")
        return

    # 4. Start Mic in Thread
    mic = AsyncMic()
    mic.start()

    # 5. The Main Loop (Non-Blocking)
    try:
        while True:
            # Wait for next audio chunk from the queue
            data = await mic.get_audio_chunk()
            
            # Send it to Deepgram (Now we can properly await this!)
            await dg_connection.send(data)
            
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