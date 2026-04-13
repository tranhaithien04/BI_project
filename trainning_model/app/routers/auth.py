from __future__ import annotations

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from ..config import DEFAULT_ROLE
from ..db import get_db
from ..dependencies import get_current_user, get_templates
from ..models import User
from ..security import verify_password
from ..services.users import (
    apply_default_role_if_empty,
    create_user,
    get_user_by_username,
    get_user_home_path,
    normalize_user_role,
)

router = APIRouter()


@router.get("/")
def index(current_user: User | None = Depends(get_current_user)):
    if current_user is None:
        return RedirectResponse(url="/login", status_code=303)
    return RedirectResponse(url=get_user_home_path(current_user), status_code=303)


@router.get("/login")
def login_page(request: Request, current_user: User | None = Depends(get_current_user)):
    if current_user is not None:
        return RedirectResponse(url=get_user_home_path(current_user), status_code=303)

    templates = get_templates(request)
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"request": request, "title": "Đăng nhập", "error": None, "user": None},
    )


@router.post("/login")
def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    user = get_user_by_username(db, username.strip())
    if user is None or not verify_password(password, user.password_hash):
        templates = get_templates(request)
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={
                "request": request,
                "title": "Đăng nhập",
                "error": "Sai tên đăng nhập hoặc mật khẩu",
                "user": None,
            },
            status_code=400,
        )

    request.session["user_id"] = int(user.id)
    return RedirectResponse(url=get_user_home_path(user), status_code=303)


@router.get("/register")
def register_page(request: Request, current_user: User | None = Depends(get_current_user)):
    if current_user is not None:
        return RedirectResponse(url=get_user_home_path(current_user), status_code=303)

    templates = get_templates(request)
    return templates.TemplateResponse(
        request=request,
        name="register.html",
        context={"request": request, "title": "Đăng ký", "error": None, "user": None},
    )


@router.post("/register")
def register_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    role: str = Form(DEFAULT_ROLE),
    full_name: str = Form(""),
    email: str = Form(""),
    skills: str = Form(""),
    db: Session = Depends(get_db),
):
    normalized_username = username.strip().lower()
    if len(normalized_username) < 3:
        templates = get_templates(request)
        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={
                "request": request,
                "title": "Đăng ký",
                "error": "Tên đăng nhập cần ít nhất 3 ký tự",
                "user": None,
            },
            status_code=400,
        )

    if len(password) < 6:
        templates = get_templates(request)
        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={
                "request": request,
                "title": "Đăng ký",
                "error": "Mật khẩu cần ít nhất 6 ký tự",
                "user": None,
            },
            status_code=400,
        )

    try:
        normalized_role = normalize_user_role(apply_default_role_if_empty(role))
    except ValueError:
        templates = get_templates(request)
        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={
                "request": request,
                "title": "Đăng ký",
                "error": "Vai trò không hợp lệ",
                "user": None,
            },
            status_code=400,
        )

    try:
        user_id = create_user(
            db,
            username=normalized_username,
            password=password,
            role=normalized_role,
            full_name=full_name.strip(),
            email=email.strip(),
            skills=skills.strip(),
        )
    except ValueError as exc:
        templates = get_templates(request)
        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={
                "request": request,
                "title": "Đăng ký",
                "error": str(exc),
                "user": None,
            },
            status_code=400,
        )

    request.session["user_id"] = user_id
    home_path = "/recruiter/jobs" if normalized_role == "recruiter" else "/dashboard"
    return RedirectResponse(url=home_path, status_code=303)


@router.post("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)
