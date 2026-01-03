import subprocess
import pyaudio
import os
import sys

# CONFIGURATION
# 1. Path to the Piper Binary (You must download this from https://github.com/rhasspy/piper/releases)
#    On Windows, it's usually 'piper.exe' inside the folder.
PIPER_PATH = r"d:\piper\piper.exe" 

# 2. Path to the Voice Model (Download .onnx and .json from https://github.com/rhasspy/piper/issues/4)
#    Example: 'en_US-amy-medium.onnx'
MODEL_PATH = r"d:\piper\en_US-hfc_female-medium.onnx"

# Audio Settings (Piper standard is usually 22050Hz or 16000Hz depending on model)
# Amy-Medium is 22050Hz.
SAMPLE_RATE = 22050 

def verify_files():
    if not os.path.exists(PIPER_PATH):
        print(f"❌ Error: Piper binary not found at: {PIPER_PATH}")
        print("-> Please download it from: https://github.com/rhasspy/piper/releases")
        return False
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at: {MODEL_PATH}")
        print("-> Please download a model (onnx + json) and set MODEL_PATH.")
        return False
    return True

def speak(text):
    print(f"🗣️ Speaking: '{text}'")
    
    # 1. Setup PyAudio Stream
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16, # Piper outputs 16-bit PCM
        channels=1,             # Mono
        rate=SAMPLE_RATE,
        output=True
    )

    # 2. Build the Command
    # echo "text" | piper --model ... --output-raw
    command = [
        PIPER_PATH,
        "--model", MODEL_PATH,
        "--output-raw", # Important: Output raw binary to stdout (no wav headers)
    ]

    try:
        # 3. Open Subprocess
        # We pass text via stdin, and read audio from stdout
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE # Hide logs or capture them
        )

        # 4. Send Text and Stream Output
        # We need to write text to stdin. 
        # Note: communicating relies on the OS pipe buffer. For short text, .communicate() is fine.
        # For streaming *long* text, we'd write to stdin in a thread.
        
        # Here we write the input text (encoded) and close the stdin so Piper knows to start.
        process.stdin.write(text.encode('utf-8'))
        process.stdin.close()

        # 5. Read and Play Audio Configs
        while True:
            # Read a chunk of data (e.g., 1024 bytes) from Piper's stdout
            data = process.stdout.read(1024)
            if not data:
                break # End of stream
            
            # Play the chunk
            stream.write(data)
        
        # Check for errors
        process.wait()
        if process.returncode != 0:
            err = process.stderr.read().decode()
            print(f"Piper Error: {err}")

    except Exception as e:
        print(f"Exception: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("✅ Done")

if __name__ == "__main__":
    if verify_files():
        text = "Hello! This is Velox AI. I am currently streaming this audio directly to your speakers using raw binary data. This reduces latency because we don't have to wait for the whole file to generate."
        speak(text)
