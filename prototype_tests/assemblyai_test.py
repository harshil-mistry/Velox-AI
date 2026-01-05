import asyncio
import pyaudio
import websockets
import json
import base64
import os

# === Configuration ===
# You need an API Key from https://www.assemblyai.com/
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY") 

# Audio Configuration
SAMPLE_RATE = 16000
FRAMES_PER_BUFFER = 3200 # 200ms

async def send_receive(api_key):
    if not api_key:
        print("Error: ASSEMBLYAI_API_KEY environment variable is not set.")
        return

    print(f"Connecting to AssemblyAI...")
    
    auth_header = {"Authorization": api_key}
    url = f"wss://api.assemblyai.com/v2/realtime/ws?sample_rate={SAMPLE_RATE}"
    
    # 'additional_headers' is for websockets v14+, 'extra_headers' for older.
    # We'll try 'additional_headers' as the traceback indicates newer version.
    async with websockets.connect(url, additional_headers=auth_header) as ws:
        print("✅ Connected to AssemblyAI.")
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
                    json_data = json.dumps({"audio_data": base64_data})
                    await ws.send(json_data)
                    await asyncio.sleep(0.01) # Yield control
                except Exception as e:
                    print(f"Audio Error: {e}")
                    break

        async def receive_transcripts():
            while True:
                try:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    
                    if "text" in data:
                        text = data["text"]
                        
                        # AssemblyAI sends "Partial" and "Final" results
                        is_final = data.get("message_type") == "FinalTranscript"
                        
                        if is_final:
                            print(f"\rUser [Final]: {text}          ")
                        else:
                            # Overwrite line for partials
                            print(f"\rUser [Partial]: {text}", end="", flush=True)

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

    if not ASSEMBLYAI_API_KEY:
        ASSEMBLYAI_API_KEY = input("Enter your AssemblyAI API Key: ").strip()
    
    try:
        asyncio.run(send_receive(ASSEMBLYAI_API_KEY))
    except KeyboardInterrupt:
        print("\nExiting...")
