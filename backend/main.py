from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from analyzer import analyze_ad

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://frontend-six-tau-76.vercel.app",
        "https://frontend-e7gw0u5sl-khushkaushik858-7069s-projects.vercel.app",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze(video: UploadFile = File(...)):
    if not video.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only MP4 files supported")

    video_bytes = await video.read()

    if len(video_bytes) > 100_000_000:
        raise HTTPException(status_code=400, detail="File too large. Keep under 100MB.")

    result = await analyze_ad(video_bytes, video.filename)
    return {"analysis": result}

@app.get("/health")
def health():
    return {"status": "ok"}