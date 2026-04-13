from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_current_user
from ..models import User
from ..schemas import JobResponse, UserRequest
from ..services.artifacts import recommend_jobs
from ..services.users import is_recruiter

router = APIRouter()


@router.post("/api/recommend", response_model=list[JobResponse])
def get_job_recommendations(payload: UserRequest):
    try:
        jobs = recommend_jobs(payload.user_skills, payload.top_k)
        return [
            JobResponse(
                **{k: item[k] for k in ["job_id", "job_name", "match_score"]}
            )
            for item in jobs
        ]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Loi he thong AI: {exc}") from exc


@router.get("/api/me/recommend", response_model=list[JobResponse])
def get_logged_in_recommendations(
    top_k: int = 5,
    current_user: User | None = Depends(get_current_user),
):
    if current_user is None:
        raise HTTPException(status_code=401, detail="Can dang nhap")
    if is_recruiter(current_user):
        raise HTTPException(status_code=403, detail="Role recruiter khong dung endpoint nay")

    try:
        jobs = recommend_jobs(current_user.skills, max(1, min(top_k, 20)))
        return [
            JobResponse(
                **{k: item[k] for k in ["job_id", "job_name", "match_score"]}
            )
            for item in jobs
        ]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
