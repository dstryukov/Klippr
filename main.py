from fastapi import FastAPI

app = FastAPI(
    title="Klippr API",
    description="AI Service for cutting long videos into vertical clips (Reels/Shorts)",
    version="0.1.0",
)


@app.get("/health")
async def health_check():
    return {"status": "ok"}
