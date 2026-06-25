import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECTS_ROOT = Path("data/projects")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slugify(value: str, fallback: str = "project") -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"[^a-z0-9а-яё_-]+", "-", value, flags=re.IGNORECASE)
    value = re.sub(r"-+", "-", value).strip("-")
    return value[:60] or fallback


def ensure_projects_root() -> Path:
    PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)
    return PROJECTS_ROOT


def project_dir(project_id: str) -> Path:
    return ensure_projects_root() / project_id


def project_file(project_id: str) -> Path:
    return project_dir(project_id) / "project.json"


def transcript_file(project_id: str) -> Path:
    return project_dir(project_id) / "transcript.json"


def candidates_file(project_id: str) -> Path:
    return project_dir(project_id) / "candidates.json"


def tmp_dir(project_id: str) -> Path:
    path = project_dir(project_id) / "tmp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def clips_dir(project_id: str) -> Path:
    path = project_dir(project_id) / "clips"
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_project(name: str, source_url: str = "") -> dict[str, Any]:
    ensure_projects_root()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    project_id = f"{timestamp}-{slugify(name)}-{uuid.uuid4().hex[:6]}"
    root = project_dir(project_id)
    root.mkdir(parents=True, exist_ok=True)
    project = {
        "id": project_id,
        "name": name.strip() or "Untitled project",
        "source_url": source_url.strip(),
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "status": "created",
        "video_path": "",
        "audio_path": "",
        "transcript_path": "",
        "candidates_path": "",
        "selected_candidate_ids": [],
        "clips": [],
        "settings_snapshot": {},
        "notes": "",
    }
    save_project(project)
    return project


def load_project(project_id: str) -> dict[str, Any]:
    path = project_file(project_id)
    if not path.exists():
        raise FileNotFoundError(f"Project not found: {project_id}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_project(project: dict[str, Any]) -> dict[str, Any]:
    project["updated_at"] = utc_now_iso()
    root = project_dir(project["id"])
    root.mkdir(parents=True, exist_ok=True)
    with open(project_file(project["id"]), "w", encoding="utf-8") as f:
        json.dump(project, f, ensure_ascii=False, indent=2)
    return project


def list_projects() -> list[dict[str, Any]]:
    root = ensure_projects_root()
    projects = []
    for path in root.glob("*/project.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                projects.append(json.load(f))
        except Exception:
            continue
    return sorted(projects, key=lambda p: p.get("updated_at", ""), reverse=True)


def save_transcript(project_id: str, transcript: list[dict[str, Any]]) -> str:
    path = transcript_file(project_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)
    return str(path)


def load_transcript(project_id: str) -> list[dict[str, Any]]:
    path = transcript_file(project_id)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_candidates(project_id: str, candidates: list[dict[str, Any]]) -> str:
    normalized = []
    for idx, candidate in enumerate(candidates):
        item = dict(candidate)
        item.setdefault("id", f"candidate_{idx + 1}")
        item.setdefault("selected", False)
        normalized.append(item)
    path = candidates_file(project_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)
    return str(path)


def load_candidates(project_id: str) -> list[dict[str, Any]]:
    path = candidates_file(project_id)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def add_clip(project: dict[str, Any], clip_path: str, candidate: dict[str, Any]) -> dict[str, Any]:
    clips = project.setdefault("clips", [])
    clips.append({
        "path": clip_path,
        "candidate_id": candidate.get("id"),
        "title": candidate.get("title", "Clip"),
        "start_time": candidate.get("start_time"),
        "end_time": candidate.get("end_time"),
        "created_at": utc_now_iso(),
    })
    return save_project(project)
