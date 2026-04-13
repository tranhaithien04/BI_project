from __future__ import annotations

from pydantic import BaseModel, Field


class UserRequest(BaseModel):
    user_skills: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class JobResponse(BaseModel):
    job_id: int
    job_name: str
    match_score: float
