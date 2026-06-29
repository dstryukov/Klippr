import json
import re
import shutil
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


def diarization_file(project_id: str) -> Path:
    return project_dir(project_id) / "diarization.json"


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
        "diarization_path": "",
        "candidates_path": "",
        "selected_candidate_ids": [],
        "clips": [],
        "settings_snapshot": {},
        "notes": "",
    }
    save_project(project)
    return project


def ensure_project_shape(project: dict[str, Any], fallback_id: str | None = None) -> dict[str, Any]:
    project = dict(project or {})
    project.setdefault("id", fallback_id or f"recovered-{uuid.uuid4().hex[:8]}")
    project.setdefault("name", "Untitled project")
    project.setdefault("source_url", "")
    project.setdefault("created_at", utc_now_iso())
    project.setdefault("updated_at", utc_now_iso())
    project.setdefault("status", "created")
    project.setdefault("video_path", "")
    project.setdefault("audio_path", "")
    project.setdefault("transcript_path", "")
    project.setdefault("diarization_path", "")
    project.setdefault("candidates_path", "")
    project.setdefault("selected_candidate_ids", [])
    project.setdefault("clips", [])
    project.setdefault("settings_snapshot", {})
    project.setdefault("notes", "")
    if not isinstance(project["selected_candidate_ids"], list):
        project["selected_candidate_ids"] = []
    if not isinstance(project["clips"], list):
        project["clips"] = []
    return project


def load_project(project_id: str) -> dict[str, Any]:
    path = project_file(project_id)
    if not path.exists():
        raise FileNotFoundError(f"Project not found: {project_id}")
    with open(path, "r", encoding="utf-8") as f:
        project = json.load(f)
    return ensure_project_shape(project, fallback_id=project_id)


def save_project(project: dict[str, Any]) -> dict[str, Any]:
    project = ensure_project_shape(project)
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
                project = json.load(f)
            project = ensure_project_shape(project, fallback_id=path.parent.name)
            projects.append(project)
        except Exception:
            # Broken project files should not break the whole UI.
            continue
    return sorted(projects, key=lambda p: p.get("updated_at", ""), reverse=True)


def save_transcript(project_id: str, transcript: list[dict[str, Any]]) -> str:
    path = transcript_file(project_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(transcript or [], f, ensure_ascii=False, indent=2)
    return str(path)


def load_transcript(project_id: str) -> list[dict[str, Any]]:
    path = transcript_file(project_id)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_diarization(project_id: str, diarization: list[dict[str, Any]]) -> str:
    path = diarization_file(project_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(diarization or [], f, ensure_ascii=False, indent=2)
    return str(path)


def load_diarization(project_id: str) -> list[dict[str, Any]]:
    path = diarization_file(project_id)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def normalize_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    used_ids = set()
    for idx, candidate in enumerate(candidates or []):
        if not isinstance(candidate, dict):
            continue
        item = dict(candidate)
        candidate_id = item.get("id") or f"candidate_{idx + 1}"
        candidate_id = str(candidate_id)
        if candidate_id in used_ids:
            candidate_id = f"candidate_{idx + 1}"
        used_ids.add(candidate_id)
        item["id"] = candidate_id
        item.setdefault("selected", False)
        item.setdefault("title", f"Candidate {idx + 1}")
        item.setdefault("start_time", 0.0)
        item.setdefault("end_time", 0.0)
        item.setdefault("score", item.get("total_score", 0))
        normalized.append(item)
    return normalized


def save_candidates(project_id: str, candidates: list[dict[str, Any]]) -> str:
    normalized = normalize_candidates(candidates)
    candidates.clear()
    candidates.extend(normalized)
    path = candidates_file(project_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)
    return str(path)


def load_candidates(project_id: str) -> list[dict[str, Any]]:
    path = candidates_file(project_id)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        candidates = normalize_candidates(data if isinstance(data, list) else [])
        # Auto-heal old candidate files that did not have ids.
        with open(path, "w", encoding="utf-8") as f:
            json.dump(candidates, f, ensure_ascii=False, indent=2)
        return candidates
    except Exception:
        return []


def add_clip(project: dict[str, Any], clip_path: str, candidate: dict[str, Any]) -> dict[str, Any]:
    project = ensure_project_shape(project)
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


def delete_project(project_id: str) -> bool:
    """Delete an entire project directory and all its contents."""
    root = project_dir(project_id)
    if not root.exists():
        raise FileNotFoundError(f"Project not found: {project_id}")
    shutil.rmtree(root, ignore_errors=True)
    return True


def delete_candidate(project_id: str, candidate_id: str) -> list[dict[str, Any]]:
    """Remove a candidate by id and persist the updated list."""
    candidates = load_candidates(project_id)
    filtered = [c for c in candidates if str(c.get("id")) != str(candidate_id)]
    if len(filtered) == len(candidates):
        raise ValueError(f"Candidate not found: {candidate_id}")
    save_candidates(project_id, filtered)

    # Also remove from selected_candidate_ids
    try:
        project = load_project(project_id)
        project["selected_candidate_ids"] = [
            cid for cid in project.get("selected_candidate_ids", [])
            if str(cid) != str(candidate_id)
        ]
        save_project(project)
    except Exception:
        pass

    return filtered


def remove_clip(project_id: str, clip_index: int) -> dict[str, Any]:
    """Remove a rendered clip by index from the project and delete the file."""
    project = load_project(project_id)
    clips = project.get("clips", [])
    if clip_index < 0 or clip_index >= len(clips):
        raise IndexError(f"Clip index out of range: {clip_index}")

    removed = clips.pop(clip_index)

    # Delete the actual file if it exists
    clip_path = removed.get("path", "")
    if clip_path:
        p = Path(clip_path)
        if p.exists():
            p.unlink()

    project["clips"] = clips
    return save_project(project)
