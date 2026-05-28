from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from analyzer import analyze_ad

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze(video: UploadFile = File(...)):
    if not video.filename.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only MP4 files supported")
    
    video_bytes = await video.read()
    
    # Warn if file is large (base64 inflates ~33%)
    if len(video_bytes) > 8_000_000:
        raise HTTPException(status_code=400, detail="File too large. Keep under 8MB.")
    
    result = await analyze_ad(video_bytes, video.filename)
    return {"analysis": result}

@app.get("/health")
def health():
    return {"status": "ok"}