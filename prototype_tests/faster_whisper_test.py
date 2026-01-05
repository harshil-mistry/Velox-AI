import asyncio
import pyaudio
import numpy as np
import time
import os
from faster_whisper import WhisperModel

# === Configuration ===
MODEL_SIZE = "large-v2.en"  # options: tiny, base, small, medium, large-v2
COMPUTE_TYPE = "int8"   # Use "float16" if you have a GPU, "int8" for CPU
# VAD Config
VAD_THRESHOLD = 0.5     # Simple energy threshold for this demo (Faster Whisper has internal VAD too)

# Audio Config
RATE = 16000
CHUNK = 1024
CHANNELS = 1
FORMAT = pyaudio.paInt16

class RealTimeTranscriber:
    def __init__(self):
        print(f"Loading Whisper Model ({MODEL_SIZE})...")
        self.model = WhisperModel(MODEL_SIZE, device="cpu", compute_type=COMPUTE_TYPE)
        print("Model Loaded.")
        
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        self.running = True
        self.buffer = np.array([], dtype=np.float32)

    def process_audio(self):
        print("🎤 Listening... (Press Ctrl+C to stop)")
        try:
            while self.running:
                # 1. Read Raw Audio
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                # Convert to float32 for Whisper
                audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # 2. Accumulate Buffer (Testing Strategy: Accumulate ~1-2s then transcribe)
                # Real-time streaming with purely local Whisper is tricky because it processes chunks.
                # Here we simulate a "rolling buffer" or just accumulate until silence.
                
                self.buffer = np.concatenate((self.buffer, audio_chunk))

                # If buffer is long enough (e.g., 2 seconds), try to transcribe
                if len(self.buffer) > RATE * 2: 
                    # 3. Transcribe
                    segments, info = self.model.transcribe(
                        self.buffer, 
                        beam_size=5,
                        language="en",
                        vad_filter=True, # Built-in VAD to ignore silence
                        vad_parameters=dict(min_silence_duration_ms=500)
                    )

                    text_output = ""
                    for segment in segments:
                        text_output += segment.text
                    
                    if text_output.strip():
                        print(f"User: {text_output.strip()}")
                        # Clear buffer only if we got a finalized sentence? 
                        # Ideally, faster-whisper is for full files or larger chunks. 
                        # For true streaming, we'd keep the last part of the audio.
                        self.buffer = np.array([], dtype=np.float32) 
                    else:
                        # If silence/noise, just keep last 1s context to avoid cutting words
                        # self.buffer = self.buffer[-RATE:] 
                        self.buffer = np.array([], dtype=np.float32) # Reset for demo simplicity

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()

if __name__ == "__main__":
    transcriber = RealTimeTranscriber()
    transcriber.process_audio()
