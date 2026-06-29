import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Job:
    id: str
    kind: str
    project_id: str
    status: str = "queued"
    progress: int = 0
    stage: str = "Queued"
    error: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    cancel_requested: bool = False
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JobManager:
    """Small in-process background job manager.

    This is intentionally simple for local/prototype use. For production this can be
    replaced with Redis/RQ/Celery without changing the API surface much.
    """

    def __init__(self, max_workers: int = 1):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()

    def submit(self, kind: str, project_id: str, fn: Callable[[Job], dict[str, Any] | None]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, project_id=project_id)
        with self.lock:
            self.jobs[job.id] = job
        self.executor.submit(self._run, job.id, fn)
        return job

    def _run(self, job_id: str, fn: Callable[[Job], dict[str, Any] | None]) -> None:
        job = self.get(job_id)
        if not job:
            return
        self.update(job_id, status="running", progress=1, stage="Starting")
        try:
            result = fn(job) or {}
            self.update(job_id, status="completed", progress=100, stage="Completed", result=result)
        except Exception as e:
            logger.error(f"Job {job_id} failed", exc_info=True)
            self.update(
                job_id,
                status="failed",
                stage=f"Error: {e.__class__.__name__}: {str(e)}",
                error=str(e),
                progress=0,
            )

    def update(self, job_id: str, **updates: Any) -> Job | None:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return None
            for key, value in updates.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            job.updated_at = utc_now_iso()
            return job

    def set_progress(self, job_id: str, progress: int, stage: str) -> None:
        self.update(job_id, progress=max(0, min(100, int(progress))), stage=stage)

    def get(self, job_id: str) -> Job | None:
        with self.lock:
            return self.jobs.get(job_id)

    def cancel(self, job_id: str) -> None:
        job = self.get(job_id)
        if job:
            job.cancel_requested = True

    def list(self) -> list[Job]:
        with self.lock:
            return sorted(self.jobs.values(), key=lambda j: j.updated_at, reverse=True)


job_manager = JobManager(max_workers=1)
