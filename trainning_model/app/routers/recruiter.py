from __future__ import annotations

from urllib.parse import quote_plus

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from ..db import get_db
from ..dependencies import get_current_user, get_templates
from ..models import User
from ..services.recruitment import (
    create_job_post,
    list_recruiter_applications,
    list_recruiter_job_posts,
)
from ..services.users import is_recruiter

router = APIRouter()


@router.get("/recruiter/jobs")
def recruiter_jobs_page(
    request: Request,
    message: str | None = None,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user),
):
    if current_user is None:
        return RedirectResponse(url="/login", status_code=303)
    if not is_recruiter(current_user):
        return RedirectResponse(url="/dashboard", status_code=303)

    posts = list_recruiter_job_posts(db, int(current_user.id))
    templates = get_templates(request)
    return templates.TemplateResponse(
        request=request,
        name="recruiter_jobs.html",
        context={
            "request": request,
            "title": "Quản lý bài đăng tuyển dụng",
            "user": current_user,
            "posts": posts,
            "message": message,
            "error": None,
        },
    )


@router.post("/recruiter/jobs")
def recruiter_jobs_submit(
    request: Request,
    title: str = Form(...),
    description: str = Form(""),
    requirements: str = Form(""),
    location: str = Form(""),
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user),
):
    if current_user is None:
        return RedirectResponse(url="/login", status_code=303)
    if not is_recruiter(current_user):
        return RedirectResponse(url="/dashboard", status_code=303)

    try:
        create_job_post(
            db,
            recruiter_id=int(current_user.id),
            title=title,
            description=description,
            requirements=requirements,
            location=location,
        )
    except ValueError as exc:
        templates = get_templates(request)
        return templates.TemplateResponse(
            request=request,
            name="recruiter_jobs.html",
            context={
                "request": request,
                "title": "Quản lý bài đăng tuyển dụng",
                "user": current_user,
                "posts": list_recruiter_job_posts(db, int(current_user.id)),
                "message": None,
                "error": str(exc),
            },
            status_code=400,
        )

    success_message = quote_plus("Đã tạo bài đăng tuyển dụng")
    return RedirectResponse(
        url=f"/recruiter/jobs?message={success_message}",
        status_code=303,
    )


@router.get("/recruiter/applications")
def recruiter_applications_page(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user),
):
    if current_user is None:
        return RedirectResponse(url="/login", status_code=303)
    if not is_recruiter(current_user):
        return RedirectResponse(url="/dashboard", status_code=303)

    applications = list_recruiter_applications(db, int(current_user.id))
    templates = get_templates(request)
    return templates.TemplateResponse(
        request=request,
        name="recruiter_applications.html",
        context={
            "request": request,
            "title": "Danh sách ứng viên đã ứng tuyển",
            "user": current_user,
            "applications": applications,
        },
    )
