import aiohttp
import json
import logging

logger = logging.getLogger("VeloxLLM")

async def stream_llm_response(messages: list, api_key: str, base_url: str, model: str, temperature: float = 0.7):
    """
    Generic generator to stream text chunks from an OpenAI-compatible API.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Ensure URL ends correctly for chat completions
    # Assumption: base_url is like "https://api.groq.com/openai/v1"
    url = f"{base_url.rstrip('/')}/chat/completions"
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": temperature
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    text = await response.text()
                    logger.error(f"LLM API Error ({response.status}): {text}")
                    # Yielding error as text for simple frontend feedback, or raise exception?
                    # For now, log and return.
                    yield f" [Error: LLM API returned {response.status}]"
                    return

                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0]["delta"]
                                if "content" in delta and delta["content"]:
                                    yield delta["content"]
                        except json.JSONDecodeError:
                            continue
    except Exception as e:
        logger.error(f"LLM Stream Exception: {e}")
        yield " [System Error: Failed to stream from LLM provider]"
