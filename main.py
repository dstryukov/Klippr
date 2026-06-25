import logging
import os
from pathlib import Path
from typing import Any, List

try:
    import torch
except Exception:  # Keep the API/UI bootable even before PyTorch is installed.
    torch = None

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
    version="0.2.2",
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


class SettingsUpdateRequest(BaseModel):
    whisper_model: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    device: str | None = None
    crop_mode: str | None = None
    output_resolution: str | None = None
    ffmpeg_preset: str | None = None
    ffmpeg_crf: int | None = None
    use_nvenc: bool | None = None
    num_clips: int | None = None
    highlight_candidate_count: int | None = None
    min_clip_duration: int | None = None
    max_clip_duration: int | None = None
    subtitle_style: str | None = None
    subtitle_font_size: int | None = None
    subtitle_color: str | None = None
    subtitle_active_color: str | None = None
    subtitle_words_per_caption: int | None = None
    subtitle_timing_offset_ms: int | None = None


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


def current_settings_payload() -> dict[str, Any]:
    return {
        "whisper_model": getattr(settings, "WHISPER_MODEL", "small"),
        "llm_provider": getattr(settings, "LLM_PROVIDER", "groq"),
        "llm_model": getattr(settings, "LLM_MODEL", "llama-3.3-70b-versatile"),
        "device": getattr(settings, "DEVICE", "cuda"),
        "crop_mode": getattr(settings, "CROP_MODE", "smart_center"),
        "output_resolution": getattr(settings, "OUTPUT_RESOLUTION", "1080x1920"),
        "ffmpeg_preset": getattr(settings, "FFMPEG_PRESET", "fast"),
        "ffmpeg_crf": int(getattr(settings, "FFMPEG_CRF", 23)),
        "use_nvenc": bool(getattr(settings, "USE_NVENC", False)),
        "num_clips": int(getattr(settings, "NUM_CLIPS", 3)),
        "highlight_candidate_count": int(getattr(settings, "HIGHLIGHT_CANDIDATE_COUNT", 12)),
        "min_clip_duration": int(getattr(settings, "MIN_CLIP_DURATION", 20)),
        "max_clip_duration": int(getattr(settings, "MAX_CLIP_DURATION", 60)),
        "subtitle_style": getattr(settings, "SUBTITLE_STYLE", "word_by_word"),
        "subtitle_font_size": int(getattr(settings, "SUBTITLE_FONT_SIZE", 70)),
        "subtitle_color": getattr(settings, "SUBTITLE_COLOR", "#FFFFFF"),
        "subtitle_active_color": getattr(settings, "SUBTITLE_ACTIVE_COLOR", "#FFFF00"),
        "subtitle_words_per_caption": int(getattr(settings, "SUBTITLE_WORDS_PER_CAPTION", 3)),
        "subtitle_timing_offset_ms": int(getattr(settings, "SUBTITLE_TIMING_OFFSET_MS", -80)),
    }


def system_payload() -> dict[str, Any]:
    if torch is None:
        configured_device = getattr(settings, "DEVICE", "cpu")
        return {
            "torch_installed": False,
            "torch_version": "not installed",
            "torch_cuda_version": None,
            "cuda_available": False,
            "gpu_count": 0,
            "gpu_name": "",
            "current_device": None,
            "configured_device": configured_device,
            "effective_device": "cpu",
            "use_nvenc": bool(getattr(settings, "USE_NVENC", False)),
            "error": "PyTorch is not installed in this Python environment.",
        }

    cuda_available = False
    gpu_name = ""
    gpu_count = 0
    current_device = None
    torch_version = getattr(torch, "__version__", "unknown")
    cuda_version = getattr(torch.version, "cuda", None)
    error = ""
    try:
        cuda_available = bool(torch.cuda.is_available())
        gpu_count = int(torch.cuda.device_count()) if cuda_available else 0
        if cuda_available and gpu_count > 0:
            current_device = int(torch.cuda.current_device())
            gpu_name = torch.cuda.get_device_name(current_device)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    configured_device = getattr(settings, "DEVICE", "cpu")
    effective_device = "cuda" if configured_device == "cuda" and cuda_available else "cpu"
    return {
        "torch_installed": True,
        "torch_version": torch_version,
        "torch_cuda_version": cuda_version,
        "cuda_available": cuda_available,
        "gpu_count": gpu_count,
        "gpu_name": gpu_name,
        "current_device": current_device,
        "configured_device": configured_device,
        "effective_device": effective_device,
        "use_nvenc": bool(getattr(settings, "USE_NVENC", False)),
        "error": error,
    }


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
    project["settings_snapshot"] = current_settings_payload()
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


@app.get("/api/system")
async def api_system():
    return system_payload()


@app.get("/api/settings")
async def api_get_settings():
    return {"settings": current_settings_payload(), "system": system_payload()}


@app.patch("/api/settings")
async def api_update_settings(req: SettingsUpdateRequest):
    incoming = req.model_dump(exclude_none=True)
    allowed = current_settings_payload()
    updates: dict[str, Any] = {}
    for key, value in incoming.items():
        if key not in allowed:
            continue
        updates[key] = value

    if updates:
        if "min_clip_duration" in updates and "max_clip_duration" in updates:
            if int(updates["min_clip_duration"]) > int(updates["max_clip_duration"]):
                raise HTTPException(status_code=400, detail="min_clip_duration cannot be greater than max_clip_duration")
        elif "min_clip_duration" in updates:
            if int(updates["min_clip_duration"]) > int(getattr(settings, "MAX_CLIP_DURATION", 60)):
                raise HTTPException(status_code=400, detail="min_clip_duration cannot be greater than max_clip_duration")
        elif "max_clip_duration" in updates:
            if int(getattr(settings, "MIN_CLIP_DURATION", 20)) > int(updates["max_clip_duration"]):
                raise HTTPException(status_code=400, detail="min_clip_duration cannot be greater than max_clip_duration")
        settings.save(updates)
    return {"settings": current_settings_payload(), "system": system_payload()}


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
