import time
import os
from groq import Groq

# 1. SETUP: Replace with your actual API Key or set it in your environment
# Get a free key here: https://console.groq.com/keys
api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=api_key)

# 2. SIMULATION: Creating a "Heavy" Prompt
# This mimics the "System Prompt + PDF Content" you will send in Velox AI.
# We make it long to test if the input size slows down the output start time.
system_context = """
You are Sarah, a senior sales representative for 'Velox Dental Solutions'.
Your tone is professional, warm, and empathetic.
You are speaking to a potential client over the phone.

CONTEXT KNOWLEDGE BASE:
- We offer three plans: Basic ($29/mo), Pro ($99/mo), and Enterprise (Custom).
- The Basic plan includes appointment scheduling and email reminders.
- The Pro plan adds SMS reminders, 2-way chat, and insurance verification.
- The Enterprise plan includes dedicated support and API access.
- We have a 14-day free trial. No credit card required.
- Integration takes about 5 minutes.
- We support integration with Dentrix, EagleSoft, and OpenDental.

INSTRUCTIONS:
- Answer the user's question concisely.
- Do not use markdown (bold/italics) because this is for voice (TTS).
- End with a polite closing or a follow-up question.
"""

user_query = "Hi, I use EagleSoft and I'm looking for a way to automate my appointment reminders. Can you tell me which plan I need and how much it costs?"

print(f"--- Starting Simulation ---")
print(f"Model: llama-3.1-8b-instant")
print(f"Input Context Length: ~{len(system_context.split())} words")
print(f"Waiting for stream...\n")

# 3. EXECUTION
start_time = time.time()
first_token_time = None
token_count = 0

stream = client.chat.completions.create(
    messages=[
        {"role": "system", "content": system_context},
        {"role": "user", "content": user_query}
    ],
    model="llama-3.1-8b-instant", # Extremely fast model
    stream=True,            # CRITICAL: Enable streaming
)

# 4. STREAMING LOOP
print("--- AI RESPONSE (Streaming) ---")
for chunk in stream:
    # Get the token text
    token = chunk.choices[0].delta.content
    
    if token:
        # Measure TTFT (Time to First Token)
        if first_token_time is None:
            first_token_time = time.time()
            ttft_ms = (first_token_time - start_time) * 1000
            print(f"\n[FIRST TOKEN RECEIVED IN {ttft_ms:.2f}ms]\n")
        
        # Print token immediately (as requested) to visualize flow
        # end="" prevents extra newlines, flush=True forces print immediately
        print(token, end="", flush=True) 
        token_count += 1

# 5. FINAL STATS
end_time = time.time()
total_duration = end_time - start_time
tokens_per_sec = token_count / (end_time - first_token_time) if first_token_time else 0

print(f"\n\n--- PERFORMANCE STATS ---")
print(f"1. Time to First Token (Latency): {ttft_ms:.2f} ms")
print(f"2. Total Generation Time:         {total_duration:.2f} seconds")
print(f"3. Tokens Generated:              {token_count}")
print(f"4. Speed:                         {tokens_per_sec:.2f} tokens/sec")

if ttft_ms < 500:
    print("\n✅ VERDICT: Excellent for Real-Time Voice (Sub-500ms)")
else:
    print("\n⚠️ VERDICT: Too slow for seamless Voice Interaction")