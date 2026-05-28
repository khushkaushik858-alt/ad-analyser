import base64
import httpx
import os
from dotenv import load_dotenv
from prompt import AD_ANALYSIS_PROMPT

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

async def analyze_ad(video_bytes: bytes, filename: str) -> str:
    video_b64 = base64.b64encode(video_bytes).decode()

    payload = {
        "model": "google/gemini-2.0-flash-001",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:video/mp4;base64,{video_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": AD_ANALYSIS_PROMPT
                    }
                ]
            }
        ],
        "max_tokens": 2000
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "Ad Analyser"
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload,
            headers=headers
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]