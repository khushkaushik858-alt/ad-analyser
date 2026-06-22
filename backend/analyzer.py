import re
import httpx
import os
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
from prompt import AD_ANALYSIS_PROMPT

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)


def clean_model_output(text: str) -> str:
    # Remove internal model tokens like <start_of_description>, <0.0 - 0.0> etc
    text = re.sub(r'<[^>]+>', '', text)
    # Remove frame description lines like **Frame 1 (00:00):**
    text = re.sub(r'\*\*Frame \d+[^*]*\*\*:?', '', text)
    # Remove unbolded frame lines like "Frame 2: The woman..."
    text = re.sub(r'Frame \d+[^\n]*\n', '', text)
    # Remove timestamp lines like <0.0 - 0.0> or <00:01>
    text = re.sub(r'\d+:\d+\s*[-–]\s*\d+:\d+', '', text)
    # Collapse excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


async def upload_to_cloudinary(video_bytes: bytes, filename: str) -> str:
    print("Uploading video to Cloudinary...")
    result = cloudinary.uploader.upload(
        video_bytes,
        resource_type="video",
        public_id=f"ad_analyser/{filename}",
        overwrite=True
    )
    url = result["secure_url"]
    print(f"Cloudinary URL: {url}")
    return url


async def delete_from_cloudinary(filename: str):
    try:
        cloudinary.uploader.destroy(
            f"ad_analyser/{filename}",
            resource_type="video"
        )
        print("Deleted video from Cloudinary")
    except Exception as e:
        print(f"Cloudinary delete failed (non-critical): {e}")


async def analyze_ad(video_bytes: bytes, filename: str) -> str:
    video_url = await upload_to_cloudinary(video_bytes, filename)

    try:
        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": AD_ANALYSIS_PROMPT
                        },
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": video_url
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 4096,
            "temperature": 0.3,
            "chat_template_kwargs": {"enable_thinking": False},
            "mm_processor_kwargs": {"use_audio_in_video": True},
            "media_io_kwargs": {"video": {"fps": 1.0, "num_frames": -1}},
            "stream": False
        }

        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        print("Sending video URL to NVIDIA NIM...")

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(NIM_URL, json=payload, headers=headers)

            print("Status:", response.status_code)
            if response.status_code != 200:
                print("Error:", response.text)
                response.raise_for_status()

            data = response.json()
            message = data["choices"][0]["message"]
            content = message.get("content", "")

            if not content:
                raise ValueError("Model returned empty content.")

            # Clean internal tokens before returning
            content = clean_model_output(content)

            return content

    finally:
        await delete_from_cloudinary(filename)