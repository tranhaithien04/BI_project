from __future__ import annotations

from urllib.parse import quote_plus

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from ..db import get_db
from ..dependencies import get_current_user, get_templates
from ..models import User
from ..services.artifacts import get_artifact_error, recommend_jobs, score_job_match
from ..services.recruitment import apply_to_job, get_applied_job_ids, list_open_job_posts
from ..services.users import is_recruiter, update_user_profile

router = APIRouter()


@router.get("/profile")
def profile_page(
    request: Request,
    current_user: User | None = Depends(get_current_user),
):
    if current_user is None:
        return RedirectResponse(url="/login", status_code=303)

    templates = get_templates(request)
    return templates.TemplateResponse(
        request=request,
        name="profile.html",
        context={
            "request": request,
            "title": "Cập nhật hồ sơ",
            "user": current_user,
            "saved": False,
            "is_recruiter": is_recruiter(current_user),
        },
    )


@router.post("/profile")
def profile_submit(
    request: Request,
    full_name: str = Form(""),
    email: str = Form(""),
    skills: str = Form(""),
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user),
):
    if current_user is None:
        return RedirectResponse(url="/login", status_code=303)

    skills_to_update = current_user.skills if is_recruiter(current_user) else skills.strip()

    updated_user = update_user_profile(
        db,
        user_id=int(current_user.id),
        full_name=full_name.strip(),
        email=email.strip(),
        skills=skills_to_update,
    )
    if updated_user is None:
        return RedirectResponse(url="/login", status_code=303)

    templates = get_templates(request)
    return templates.TemplateResponse(
        request=request,
        name="profile.html",
        context={
            "request": request,
            "title": "Cập nhật hồ sơ",
            "user": updated_user,
            "saved": True,
            "is_recruiter": is_recruiter(updated_user),
        },
    )


@router.get("/dashboard")
def dashboard(
    request: Request,
    top_k: int = 5,
    current_user: User | None = Depends(get_current_user),
):
    if current_user is None:
        return RedirectResponse(url="/login", status_code=303)
    if is_recruiter(current_user):
        return RedirectResponse(url="/recruiter/jobs", status_code=303)

    clamped_top_k = max(1, min(top_k, 20))
    recommendations: list[dict] = []
    notice = None

    artifact_error = get_artifact_error()
    if artifact_error is not None:
        notice = f"Model chưa sẵn sàng: {artifact_error}"
    elif not current_user.skills.strip():
        notice = "Bạn chưa có kỹ năng trong hồ sơ. Hãy cập nhật hồ sơ để nhận gợi ý."
    else:
        recommendations = recommend_jobs(current_user.skills, top_k=clamped_top_k)

    templates = get_templates(request)
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "request": request,
            "title": "Trang gợi ý việc làm",
            "user": current_user,
            "recommendations": recommendations,
            "top_k": clamped_top_k,
            "notice": notice,
        },
    )


@router.get("/jobs")
def jobs_page(
    request: Request,
    message: str | None = None,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user),
):
    if current_user is None:
        return RedirectResponse(url="/login", status_code=303)
    if is_recruiter(current_user):
        return RedirectResponse(url="/recruiter/jobs", status_code=303)

    jobs = list_open_job_posts(db)
    applied_ids = get_applied_job_ids(db, int(current_user.id))
    score_notice = None

    artifact_error = get_artifact_error()
    if artifact_error is not None:
        score_notice = "Điểm phù hợp tạm thời chưa khả dụng vì model chưa sẵn sàng"
    elif not current_user.skills.strip():
        score_notice = "Bạn chưa có kỹ năng trong hồ sơ, điểm phù hợp chưa hiển thị"

    for item in jobs:
        item["has_applied"] = int(item["id"]) in applied_ids
        item["recruiter_display_name"] = (
            str(item.get("recruiter_full_name") or "").strip()
            or str(item.get("recruiter_username") or "nhà tuyển dụng")
        )
        item["match_score"] = score_job_match(
            current_user.skills,
            str(item.get("requirements") or ""),
        )

    jobs.sort(
        key=lambda item: item.get("match_score") if item.get("match_score") is not None else -1,
        reverse=True,
    )

    templates = get_templates(request)
    return templates.TemplateResponse(
        request=request,
        name="jobs.html",
        context={
            "request": request,
            "title": "Danh sách tin tuyển dụng",
            "user": current_user,
            "jobs": jobs,
            "message": message,
            "score_notice": score_notice,
        },
    )


@router.post("/jobs/{job_post_id}/apply")
def apply_job(
    request: Request,
    job_post_id: int,
    cover_letter: str = Form(""),
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user),
):
    if current_user is None:
        return RedirectResponse(url="/login", status_code=303)
    if is_recruiter(current_user):
        return RedirectResponse(url="/recruiter/jobs", status_code=303)

    try:
        apply_to_job(
            db,
            job_post_id=job_post_id,
            candidate_id=int(current_user.id),
            cover_letter=cover_letter,
        )
        message = "Đã ứng tuyển thành công"
    except ValueError as exc:
        message = str(exc)

    return RedirectResponse(url=f"/jobs?message={quote_plus(message)}", status_code=303)
