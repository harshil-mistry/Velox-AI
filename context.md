### **System Context: Velox AI Platform Architecture**

**1. Project Definition & Goal**
**Velox AI** is a high-performance, full-duplex **Conversational AI Orchestration Platform** designed for real-time sales, support, and enquiry automation.
* **Core Objective:** To replace standard "turn-based" chatbots with a **sub-500ms latency Voice Agent** that feels indistinguishable from a human operator.
* **Key Differentiator:** The system utilizes a **Streaming-First Architecture** where audio, text, and logic are processed in continuous binary streams rather than discrete file uploads, enabling "Interruption Handling" (Barge-in) and near-instant responses.

---

**2. The "3-Pillar" Ecosystem**
The platform is distributed across three distinct interfaces:
1.  **The Neural Backend (The Brain):** A Python-based **FastAPI & WebSocket Server** that acts as the central router for all intelligence. It manages state, connects to AI models, and handles binary data traffic.
2.  **The Web Dashboard (Command Center):** A **Next.js** application where businesses create agents, define system personas, and view call analytics. It features a custom **AudioWorklet** implementation for web-based calling.
3.  **The Mobile App (The Field Unit):** A **Flutter** application for Android/iOS providing native performance, utilizing hardware echo cancellation (AEC) and background connectivity for on-the-go voice interactions.

---

**3. The Core "Streaming Pipeline" (Technical Architecture)**
The system rejects standard REST API patterns in favor of a **bi-directional WebSocket pipeline** to minimize latency. The data flow is circular and continuous:

**Phase A: Input & Transduction (The Ear)**
* **User Audio:** The client (Web/Mobile) captures raw microphone data in **16-bit PCM (Linear16) at 16,000Hz**.
* **Chunking:** Audio is not saved to a file. It is sliced into small **chunks (frames)** and blasted directly over the WebSocket to the Backend.
* **Real-Time STT:** The Backend passes these binary chunks instantly to **Deepgram Nova-2** (via WebSocket).
* **Outcome:** Transcripts are generated *while the user is still speaking*. The system does not wait for silence to begin processing.

**Phase B: Intelligence & Generation (The Brain)**
* **LLM Trigger:** Once a user's utterance is finalized (detected via semantic pause or VAD), the transcript is sent to **Groq (Llama 3.1-8b-instant)**.
* **Token Streaming:** The LLM does not generate a full paragraph. It streams **tokens** one by one.
* **Sentence Buffering Logic:** A specialized "Aggregator" collects these tokens in a buffer. It monitors for punctuation marks (`.`, `?`, `!`).
    * *Why?* Sending half-words to TTS sounds glitchy. We wait for a full semantic unit (a sentence) before converting to audio.

**Phase C: Synthesis & Playback (The Voice)**
* **TTS Injection:** As soon as a sentence is formed, it is pushed to the **TTS Engine** (Cartesia Sonic or ElevenLabs Turbo v2.5).
* **Audio Streaming:** The TTS provider returns **Raw Audio Bytes** (not MP3s) immediately.
* **Playback:** The Backend pushes these bytes to the Client's audio queue. The Client plays them instantly, creating a seamless "Talking" effect even if the LLM is still thinking about the second sentence.

---

**4. The Interruption Engine (VAD System)**
To simulate human conversation, the system must handle interruptions naturally.
* **Mechanism:** A **Voice Activity Detector (VAD)** runs continuously on the input stream.
* **Logic:** If the user starts speaking *while* the AI is playing audio:
    1.  **Detection:** VAD flags "Active Speech."
    2.  **Signal:** A `CLEAR_QUEUE` event is fired.
    3.  **Action:** The Backend kills the current TTS stream and tells the Client to **flush its audio buffer** immediately.
    4.  **Result:** The AI stops talking instantly (latency < 200ms), acknowledging the user's interruption without awkward overlap.

---

**5. Data & State Management**
* **Hot Storage (Session):** The active conversation history is held in a **Python Deque (In-Memory)** for 0ms retrieval latency during the call.
* **Cold Storage (Persistence):** After the socket closes, logs and metadata are dumped into **PostgreSQL** for analysis and history tracking.