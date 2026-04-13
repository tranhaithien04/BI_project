from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..db import get_db
from ..services.artifacts import get_artifact_error, get_loaded_jobs_count
from ..services.recruitment import count_job_applications, count_job_posts
from ..services.users import count_users

router = APIRouter()


@router.get("/health")
def health_check(db: Session = Depends(get_db)):
    artifact_error = get_artifact_error()
    return {
        "status": "ok",
        "artifacts_loaded": artifact_error is None,
        "artifact_error": artifact_error,
        "total_unique_jobs_loaded": get_loaded_jobs_count(),
        "registered_users": count_users(db),
        "job_posts": count_job_posts(db),
        "job_applications": count_job_applications(db),
    }
