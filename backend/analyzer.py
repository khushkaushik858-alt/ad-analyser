import base64
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
    # Step 1 — upload to Cloudinary to get public URL
    video_url = await upload_to_cloudinary(video_bytes, filename)

    try:
        # Step 2 — send public URL to NVIDIA NIM
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

        print(f"Sending video URL to NVIDIA NIM...")

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

            return content

    finally:
        # Step 3 — always delete from Cloudinary after analysis
        await delete_from_cloudinary(filename)