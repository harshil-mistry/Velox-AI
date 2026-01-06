import struct
import math

class TaskStateManager:
    """Manages the conversation state and handles interruptions."""
    def __init__(self):
        self.is_user_speaking_now = False
        self.is_ai_speaking = False
        self.is_thinking = False
        
        # Reference to the current LLM task to allow cancellation
        self.llm_task: asyncio.Task = None
        
        # TTS Queue Cleared Flag
        self.interrupt_signal = threading.Event()
        
    def check_energy_vad(self, audio_chunk: bytes, threshold=0.01) -> bool:
        """Simple RMS-based VAD for barge-in detection."""
        if not audio_chunk: return False
        
        try:
            # Assume 16-bit PCM (2 bytes per sample)
            count = len(audio_chunk) // 2
            format_str = f"<{count}h" 
            shorts = struct.unpack(format_str, audio_chunk)
            
            sum_squares = sum(s**2 for s in shorts)
            rms = math.sqrt(sum_squares / count) / 32768.0
            
            return rms > threshold
        except Exception:
            return False

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
            self.interrupt_signal.set() # Signal to TTS loops to abort
            
            # 3. Notify Frontend to Stop Audio
            await websocket.send_json({"type": "control", "action": "interrupt"})
