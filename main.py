import logging
import os
from pathlib import Path
from typing import Any, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, HttpUrl

from config import settings
from core.analyzer import HighlightAnalyzer
from core.ingestion import VideoIngestor
from core.jobs import job_manager
from core.projects import (
    add_clip,
    clips_dir,
    create_project,
    list_projects,
    load_candidates,
    load_project,
    load_transcript,
    save_candidates,
    save_project,
    save_transcript,
    tmp_dir,
)
from core.renderer import VerticalRenderer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Klippr API",
    description="AI service for cutting long videos into vertical clips",
    version="0.2.0",
)


class ProjectCreateRequest(BaseModel):
    name: str = Field(default="New shorts project")
    source_url: str = Field(default="")


class ProjectUpdateRequest(BaseModel):
    name: str | None = None
    source_url: str | None = None
    notes: str | None = None


class AnalyzeRequest(BaseModel):
    candidate_count: int = Field(default=12, ge=1, le=30)


class RenderRequest(BaseModel):
    candidate_ids: list[str] = Field(default_factory=list)


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


def project_payload(project_id: str) -> dict[str, Any]:
    project = load_project(project_id)
    return {
        "project": project,
        "candidates": load_candidates(project_id),
        "transcript_count": len(load_transcript(project_id)),
    }


def _analyze_project_job(job_id: str, project_id: str, candidate_count: int) -> dict[str, Any]:
    project = load_project(project_id)
    if not project.get("source_url"):
        raise ValueError("Project source_url is empty")

    project["status"] = "analyzing"
    save_project(project)

    job_manager.set_progress(job_id, 5, "Initializing downloader")
    ingestor = VideoIngestor(temp_dir=str(tmp_dir(project_id)))

    job_manager.set_progress(job_id, 15, "Downloading video")
    video_path = ingestor.download_video(project["source_url"])
    project["video_path"] = video_path
    save_project(project)

    job_manager.set_progress(job_id, 30, "Extracting audio")
    audio_path = ingestor.extract_audio(video_path)
    project["audio_path"] = audio_path
    save_project(project)

    job_manager.set_progress(job_id, 45, "Transcribing audio with word timestamps")
    transcript = ingestor.transcribe(audio_path)
    project["transcript_path"] = save_transcript(project_id, transcript)
    save_project(project)

    job_manager.set_progress(job_id, 70, "Finding AI highlight candidates")
    analyzer = HighlightAnalyzer()
    candidates = analyzer.find_highlight_candidates(transcript, num_candidates=candidate_count)

    job_manager.set_progress(job_id, 85, "Snapping candidates to silence")
    candidates = analyzer.snap_to_silence(candidates, audio_path, transcript)
    project["candidates_path"] = save_candidates(project_id, candidates)
    project["selected_candidate_ids"] = [c.get("id") for c in candidates[: int(getattr(settings, "NUM_CLIPS", 3))] if c.get("id")]
    project["status"] = "candidates_ready"
    save_project(project)

    return {"candidate_count": len(candidates), "project_id": project_id}


def _render_project_job(job_id: str, project_id: str, candidate_ids: list[str]) -> dict[str, Any]:
    project = load_project(project_id)
    candidates = load_candidates(project_id)
    transcript = load_transcript(project_id)

    if not project.get("video_path") or not Path(project["video_path"]).exists():
        raise FileNotFoundError("Source video not found. Run analyze first.")

    selected_ids = set(candidate_ids or project.get("selected_candidate_ids", []))
    selected = [c for c in candidates if c.get("id") in selected_ids]
    if not selected:
        raise ValueError("No candidates selected for render")

    project["selected_candidate_ids"] = [c.get("id") for c in selected if c.get("id")]
    project["status"] = "rendering"
    project["clips"] = []
    save_project(project)

    res = tuple(map(int, str(getattr(settings, "OUTPUT_RESOLUTION", "1080x1920")).split("x")))
    renderer = VerticalRenderer(output_dir=str(clips_dir(project_id)), resolution=res)

    selected_sorted = sorted(selected, key=lambda c: float(c.get("start_time", 0)))
    for i, highlight in enumerate(selected_sorted):
        progress = 5 + int(90 * (i / max(len(selected_sorted), 1)))
        job_manager.set_progress(job_id, progress, f"Rendering clip {i + 1}/{len(selected_sorted)}")
        clip_path = str(clips_dir(project_id) / f"clip_{i + 1:02d}.mp4")
        renderer.render_clip(project["video_path"], highlight, clip_path, transcript=transcript)
        project = add_clip(project, clip_path, highlight)

    project["status"] = "rendered"
    save_project(project)
    return {"clip_count": len(project.get("clips", [])), "project_id": project_id}


@app.get("/")
async def studio_home():
    index_path = Path("web/index.html")
    if not index_path.exists():
        return {"message": "Klippr API is running", "ui": "web/index.html not found"}
    return FileResponse(index_path)


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/api/health")
async def api_health_check():
    return {"status": "ok"}


@app.get("/api/projects")
async def api_list_projects():
    return {"projects": list_projects()}


@app.post("/api/projects")
async def api_create_project(req: ProjectCreateRequest):
    project = create_project(req.name, req.source_url)
    return project_payload(project["id"])


@app.get("/api/projects/{project_id}")
async def api_get_project(project_id: str):
    try:
        return project_payload(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Project not found")


@app.patch("/api/projects/{project_id}")
async def api_update_project(project_id: str, req: ProjectUpdateRequest):
    try:
        project = load_project(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Project not found")
    if req.name is not None:
        project["name"] = req.name.strip() or "Untitled project"
    if req.source_url is not None:
        project["source_url"] = req.source_url.strip()
    if req.notes is not None:
        project["notes"] = req.notes
    save_project(project)
    return project_payload(project_id)


@app.get("/api/projects/{project_id}/candidates")
async def api_get_candidates(project_id: str):
    try:
        load_project(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"candidates": load_candidates(project_id)}


@app.post("/api/projects/{project_id}/analyze")
async def api_analyze_project(project_id: str, req: AnalyzeRequest):
    try:
        load_project(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Project not found")

    job = job_manager.submit(
        kind="analyze",
        project_id=project_id,
        fn=lambda j: _analyze_project_job(j.id, project_id, req.candidate_count),
    )
    return job.to_dict()


@app.post("/api/projects/{project_id}/render")
async def api_render_project(project_id: str, req: RenderRequest):
    try:
        project = load_project(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Project not found")
    if req.candidate_ids:
        project["selected_candidate_ids"] = req.candidate_ids
        save_project(project)

    job = job_manager.submit(
        kind="render",
        project_id=project_id,
        fn=lambda j: _render_project_job(j.id, project_id, req.candidate_ids),
    )
    return job.to_dict()


@app.get("/api/jobs")
async def api_list_jobs():
    return {"jobs": [job.to_dict() for job in job_manager.list()]}


@app.get("/api/jobs/{job_id}")
async def api_get_job(job_id: str):
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@app.get("/api/projects/{project_id}/clips/{filename}")
async def api_get_clip(project_id: str, filename: str):
    safe_name = os.path.basename(filename)
    clip_path = clips_dir(project_id) / safe_name
    if not clip_path.exists():
        raise HTTPException(status_code=404, detail="Clip not found")
    return FileResponse(clip_path, media_type="video/mp4", filename=safe_name)


@app.post("/generate")
async def generate_clips(req: VideoRequest):
    """Compatibility endpoint.

    It now creates a project and starts analysis asynchronously instead of holding
    a long HTTP request open.
    """
    project = create_project("API generated project", str(req.url))
    job = job_manager.submit(
        kind="analyze",
        project_id=project["id"],
        fn=lambda j: _analyze_project_job(j.id, project["id"], max(req.num_clips * 4, 10)),
    )
    return {
        "project_id": project["id"],
        "job_id": job.id,
        "status_url": f"/api/jobs/{job.id}",
        "project_url": f"/api/projects/{project['id']}",
    }
