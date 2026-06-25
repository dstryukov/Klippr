import os
import uuid
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, HttpUrl
from typing import List

from core.ingestion import VideoIngestor
from core.analyzer import HighlightAnalyzer
from core.renderer import VerticalRenderer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Klippr API",
    description="AI Service for cutting long videos into vertical clips (Reels/Shorts)",
    version="0.1.0",
)

class VideoRequest(BaseModel):
    url: HttpUrl
    num_clips: int = Field(default=3, ge=1, le=10)

class ClipResponse(BaseModel):
    title: str
    reason: str
    file_path: str

class ProcessResponse(BaseModel):
    job_id: str
    clips: List[ClipResponse]

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/generate", response_model=ProcessResponse)
async def generate_clips(req: VideoRequest):
    # Generate a unique job ID to separate temporary files and outputs
    job_id = str(uuid.uuid4())[:8]
    logger.info(f"Starting job {job_id} for URL: {req.url}")
    
    # Define job-specific directories
    temp_dir = os.path.join("tmp", job_id)
    out_dir = os.path.join("output", job_id)
    
    try:
        # Step 1: Ingestion
        logger.info(f"[{job_id}] STEP 1: Ingestion")
        ingestor = VideoIngestor(temp_dir=temp_dir)
        video_path = ingestor.download_video(str(req.url))
        audio_path = ingestor.extract_audio(video_path)
        transcript = ingestor.transcribe(audio_path)
        
        # Step 2: Analysis
        logger.info(f"[{job_id}] STEP 2: Analysis")
        analyzer = HighlightAnalyzer()
        highlights = analyzer.find_highlights(transcript, num_clips=req.num_clips)
        
        # Snap boundaries to silence to avoid cutting words
        highlights = analyzer.snap_to_silence(highlights, audio_path, transcript)
        
        # Step 3: Rendering
        logger.info(f"[{job_id}] STEP 3: Rendering")
        renderer = VerticalRenderer(output_dir=out_dir)
        generated_clips = []
        
        for i, hl in enumerate(highlights):
            clip_filename = f"clip_{i+1}.mp4"
            out_path = os.path.join(out_dir, clip_filename)
            
            renderer.render_clip(video_path, hl, out_path, transcript=transcript)
            
            generated_clips.append(ClipResponse(
                title=hl.get("title", f"Clip {i+1}"),
                reason=hl.get("reason", ""),
                file_path=out_path
            ))
            
        logger.info(f"[{job_id}] Job completed successfully. Generated {len(generated_clips)} clips.")
        return ProcessResponse(job_id=job_id, clips=generated_clips)
        
    except Exception as e:
        logger.exception(f"[{job_id}] Job failed")
        raise HTTPException(status_code=500, detail=str(e))
