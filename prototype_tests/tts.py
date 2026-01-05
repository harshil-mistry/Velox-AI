import pyaudio
import os
import requests
import time
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("DEEPGRAM_API_KEY")

# CONFIGURATION
# 24kHz is the native rate for Aura (High Quality + Low Latency)
SAMPLE_RATE = 24000 
CHUNK_SIZE = 1024

def play_stream_raw():
    if not API_KEY:
        print("Please set DEEPGRAM_API_KEY in your .env file")
        return

    text_input = ("Hello! This is Velox AI. "
    "I am currently streaming this audio directly to your speakers using raw binary data. "
    "This reduces latency because we don't have to wait for the whole file to generate.")
    
    # 1. Construct the URL manually (Bypassing SDK)
    # container=none is CRITICAL. It removes the WAV header so we can stream raw bytes.
    url = f"https://api.deepgram.com/v1/speak?model=aura-asteria-en&encoding=linear16&sample_rate={SAMPLE_RATE}&container=none"
    
    headers = {
        "Authorization": f"Token {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {"text": text_input}

    print(f"Requesting TTS: '{text_input}'...")
    start_time = time.time()

    # 2. Make the Request with stream=True
    # This tells Python: "Don't download the whole file. Give me bytes as they arrive."
    with requests.post(url, headers=headers, json=payload, stream=True) as response:
        if response.status_code != 200:
            print(f"Error: {response.text}")
            return

        # 3. Setup Audio Output
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16, # Raw PCM 16-bit
            channels=1,
            rate=SAMPLE_RATE,
            output=True
        )

        first_byte_received = False
        
        # 4. Stream and Play
        try:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    if not first_byte_received:
                        first_byte_received = True
                        latency = (time.time() - start_time) * 1000
                        print(f"\n⚡ Latency (Time to First Byte): {latency:.2f}ms")
                    
                    stream.write(chunk)
                    
        except Exception as e:
            print(f"Stream Error: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            print("\n✅ Playback Finished")

if __name__ == "__main__":
    play_stream_raw()