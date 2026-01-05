import asyncio
import pyaudio
import websockets
import json
import base64
import os

# === Configuration ===
# Get API Key from https://app.gladia.io/
GLADIA_API_KEY = os.getenv("GLADIA_API_KEY") 

# Audio Configuration
SAMPLE_RATE = 16000
FRAMES_PER_BUFFER = 3200 # 200ms

async def send_receive(api_key):
    if not api_key:
        print("Error: GLADIA_API_KEY environment variable is not set.")
        return

    print(f"Connecting to Gladia...")
    
    # Gladia V2 Endpoint
    url = "wss://api.gladia.io/audio/text/audio-transcription"
    
    async with websockets.connect(url) as ws:
        print("✅ Connected to Gladia.")
        
        # 1. Send Configuration Message
        config = {
            "x_gladia_key": api_key,
            "sample_rate": SAMPLE_RATE,
            "encoding": "wav", # or "pcm" if you send raw bytes without headers? 
                               # Gladia supports WAV/PCM. Let's send JSON with base64.
            "language_behaviour": "automatic single language", 
            # "language": "en", # Optional: force english for speed
            "frames_format": "base64",
        }
        await ws.send(json.dumps(config))

        print("🎤 Listening... (Press Ctrl+C to stop)")

        # Initialize PyAudio
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=FRAMES_PER_BUFFER
        )

        async def send_audio():
            while True:
                try:
                    data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
                    base64_data = base64.b64encode(data).decode("utf-8")
                    json_data = json.dumps({"frames": base64_data})
                    await ws.send(json_data)
                    await asyncio.sleep(0.01)
                except Exception as e:
                    print(f"Audio Error: {e}")
                    break

        async def receive_transcripts():
            while True:
                try:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    
                    if "transcription" in data:
                        text = data["transcription"]
                        is_final = data.get("type") == "final"
                        
                        if is_final:
                            # Clean print for final
                            print(f"\rUser [Final]: {text}          ")
                        else:
                            # Overwrite for partial
                            print(f"\rUser [Partial]: {text}", end="", flush=True)
                    
                    if "error" in data:
                        print(f"\nGladia Error: {data['error']}")

                except websockets.exceptions.ConnectionClosed:
                    print("\nConnection Closed.")
                    break
                except Exception as e:
                    print(f"\nReceive Error: {e}")
                    break

        # Run Send/Receive concurrently
        send_task = asyncio.create_task(send_audio())
        receive_task = asyncio.create_task(receive_transcripts())

        try:
            await asyncio.gather(send_task, receive_task)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

if __name__ == "__main__":
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')

    if not GLADIA_API_KEY:
        GLADIA_API_KEY = input("Enter your Gladia API Key: ").strip()
    
    try:
        asyncio.run(send_receive(GLADIA_API_KEY))
    except KeyboardInterrupt:
        print("\nExiting...")
