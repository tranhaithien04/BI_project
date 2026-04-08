from __future__ import annotations

import hashlib
import hmac
import os
import pickle
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
import uvicorn
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.middleware.sessions import SessionMiddleware

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
TFIDF_PATH = ARTIFACTS_DIR / "tfidf.pkl"
KNN_PATH = ARTIFACTS_DIR / "knn_model.pkl"
JOBS_INFO_PATH = ARTIFACTS_DIR / "jobs_info.pkl"

TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

DB_PATH = BASE_DIR / "users.db"
SESSION_SECRET = os.getenv("APP_SESSION_SECRET", "change-this-secret-in-production")
PBKDF2_ITERATIONS = 120_000

# Các biến global giữ model đã load để phục vụ realtime.
vectorizer = None
knn_model = None
df_jobs = None
artifact_error = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_database()
    try:
        load_artifacts()
    except Exception as exc:
        # App vẫn khởi động để người dùng đăng nhập/chỉnh profile dù model tạm lỗi.
        set_artifact_error(str(exc))
    yield


app = FastAPI(
    title="Job Recommendation App",
    description="Web app dang nhap, cap nhat profile va goi y viec lam theo skills",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, same_site="lax")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def set_artifact_error(message: str | None) -> None:
    global artifact_error
    artifact_error = message


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database() -> None:
    with get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                full_name TEXT NOT NULL DEFAULT '',
                email TEXT NOT NULL DEFAULT '',
                skills TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    )
    return f"{salt.hex()}${password_hash.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt_hex, hash_hex = stored_hash.split("$", 1)
    except ValueError:
        return False

    salt = bytes.fromhex(salt_hex)
    expected_hash = bytes.fromhex(hash_hex)
    actual_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    )
    return hmac.compare_digest(actual_hash, expected_hash)


def get_user_by_username(username: str) -> dict[str, Any] | None:
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT id, username, password_hash, full_name, email, skills FROM users WHERE username = ?",
            (username,),
        ).fetchone()

    if row is None:
        return None
    return dict(row)


def get_user_by_id(user_id: int) -> dict[str, Any] | None:
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT id, username, full_name, email, skills FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()

    if row is None:
        return None
    return dict(row)


def create_user(
    username: str,
    password: str,
    full_name: str,
    email: str,
    skills: str,
) -> int:
    try:
        with get_db_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO users (username, password_hash, full_name, email, skills)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    username,
                    hash_password(password),
                    full_name,
                    email,
                    skills,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)
    except sqlite3.IntegrityError as exc:
        raise ValueError("Ten dang nhap da ton tai") from exc


def update_user_profile(user_id: int, full_name: str, email: str, skills: str) -> None:
    with get_db_connection() as conn:
        conn.execute(
            """
            UPDATE users
            SET full_name = ?, email = ?, skills = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (full_name, email, skills, user_id),
        )
        conn.commit()


def load_artifacts() -> None:
    global vectorizer, knn_model, df_jobs

    missing = [
        str(path) for path in [TFIDF_PATH, KNN_PATH, JOBS_INFO_PATH] if not path.exists()
    ]
    if missing:
        raise RuntimeError(
            "Thieu artifacts. Hay chay train_model.py truoc: " + ", ".join(missing)
        )

    with TFIDF_PATH.open("rb") as f:
        vectorizer = pickle.load(f)

    with KNN_PATH.open("rb") as f:
        knn_model = pickle.load(f)

    with JOBS_INFO_PATH.open("rb") as f:
        jobs_obj = pickle.load(f)

    if isinstance(jobs_obj, pd.DataFrame):
        df_jobs = jobs_obj
    else:
        df_jobs = pd.DataFrame(jobs_obj)

    required_columns = {"Job_ID", "Job_Name", "Job_Requirements"}
    if not required_columns.issubset(df_jobs.columns):
        raise RuntimeError("jobs_info.pkl thieu cot: Job_ID, Job_Name, Job_Requirements")

    if knn_model.n_samples_fit_ != len(df_jobs):
        raise RuntimeError("Artifacts khong dong bo giua KNN va jobs_info")

    set_artifact_error(None)


def get_current_user(request: Request) -> dict[str, Any] | None:
    user_id = request.session.get("user_id")
    if user_id is None:
        return None
    return get_user_by_id(int(user_id))


def recommend_jobs(user_skills: str, top_k: int = 5) -> list[dict[str, Any]]:
    if artifact_error is not None or vectorizer is None or knn_model is None or df_jobs is None:
        raise RuntimeError(artifact_error or "Model artifacts chua san sang")

    cleaned_skills = user_skills.strip().lower()
    if not cleaned_skills:
        raise ValueError("user_skills khong duoc de trong")

    user_vector = vectorizer.transform([cleaned_skills])
    n_neighbors = min(max(1, top_k), len(df_jobs))
    distances, indices = knn_model.kneighbors(user_vector, n_neighbors=n_neighbors)

    results: list[dict[str, Any]] = []
    for distance, idx in zip(distances[0], indices[0]):
        score = round(float(1 - distance), 4)
        job = df_jobs.iloc[int(idx)]
        results.append(
            {
                "job_id": int(job["Job_ID"]),
                "job_name": str(job["Job_Name"]),
                "match_score": max(0.0, score),
                "job_requirements": str(job["Job_Requirements"]),
            }
        )

    return sorted(results, key=lambda item: item["match_score"], reverse=True)


class UserRequest(BaseModel):
    user_skills: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class JobResponse(BaseModel):
    job_id: int
    job_name: str
    match_score: float


@app.get("/")
def index(request: Request):
    user = get_current_user(request)
    if user is None:
        return RedirectResponse(url="/login", status_code=303)
    return RedirectResponse(url="/dashboard", status_code=303)


@app.get("/login")
def login_page(request: Request):
    if get_current_user(request) is not None:
        return RedirectResponse(url="/dashboard", status_code=303)
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"request": request, "title": "Dang nhap", "error": None},
    )


@app.post("/login")
def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    user = get_user_by_username(username.strip())
    if user is None or not verify_password(password, str(user["password_hash"])):
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={
                "request": request,
                "title": "Dang nhap",
                "error": "Sai ten dang nhap hoac mat khau",
            },
            status_code=400,
        )

    request.session["user_id"] = int(user["id"])
    return RedirectResponse(url="/dashboard", status_code=303)


@app.get("/register")
def register_page(request: Request):
    if get_current_user(request) is not None:
        return RedirectResponse(url="/dashboard", status_code=303)
    return templates.TemplateResponse(
        request=request,
        name="register.html",
        context={"request": request, "title": "Dang ky", "error": None},
    )


@app.post("/register")
def register_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    full_name: str = Form(""),
    email: str = Form(""),
    skills: str = Form(""),
):
    normalized_username = username.strip().lower()
    if len(normalized_username) < 3:
        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={
                "request": request,
                "title": "Dang ky",
                "error": "Ten dang nhap can it nhat 3 ky tu",
            },
            status_code=400,
        )

    if len(password) < 6:
        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={
                "request": request,
                "title": "Dang ky",
                "error": "Mat khau can it nhat 6 ky tu",
            },
            status_code=400,
        )

    try:
        user_id = create_user(
            username=normalized_username,
            password=password,
            full_name=full_name.strip(),
            email=email.strip(),
            skills=skills.strip(),
        )
    except ValueError as exc:
        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={
                "request": request,
                "title": "Dang ky",
                "error": str(exc),
            },
            status_code=400,
        )

    request.session["user_id"] = user_id
    return RedirectResponse(url="/dashboard", status_code=303)


@app.post("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)


@app.get("/profile")
def profile_page(request: Request):
    user = get_current_user(request)
    if user is None:
        return RedirectResponse(url="/login", status_code=303)

    return templates.TemplateResponse(
        request=request,
        name="profile.html",
        context={"request": request, "title": "Cap nhat ho so", "user": user, "saved": False},
    )


@app.post("/profile")
def profile_submit(
    request: Request,
    full_name: str = Form(""),
    email: str = Form(""),
    skills: str = Form(""),
):
    user = get_current_user(request)
    if user is None:
        return RedirectResponse(url="/login", status_code=303)

    update_user_profile(
        user_id=int(user["id"]),
        full_name=full_name.strip(),
        email=email.strip(),
        skills=skills.strip(),
    )

    updated_user = get_user_by_id(int(user["id"]))
    return templates.TemplateResponse(
        request=request,
        name="profile.html",
        context={
            "request": request,
            "title": "Cap nhat ho so",
            "user": updated_user,
            "saved": True,
        },
    )


@app.get("/dashboard")
def dashboard(request: Request, top_k: int = 5):
    user = get_current_user(request)
    if user is None:
        return RedirectResponse(url="/login", status_code=303)

    top_k = max(1, min(top_k, 20))
    recommendations: list[dict[str, Any]] = []
    notice = None

    if artifact_error is not None:
        notice = f"Model chua san sang: {artifact_error}"
    elif not str(user.get("skills", "")).strip():
        notice = "Ban chua co ky nang trong ho so. Hay cap nhat profile de nhan goi y."
    else:
        recommendations = recommend_jobs(str(user["skills"]), top_k=top_k)

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "request": request,
            "title": "Trang goi y viec lam",
            "user": user,
            "recommendations": recommendations,
            "top_k": top_k,
            "notice": notice,
        },
    )


@app.post("/api/recommend", response_model=list[JobResponse])
def get_job_recommendations(payload: UserRequest):
    try:
        jobs = recommend_jobs(payload.user_skills, payload.top_k)
        return [JobResponse(**{k: item[k] for k in ["job_id", "job_name", "match_score"]}) for item in jobs]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Loi he thong AI: {exc}") from exc


@app.get("/api/me/recommend", response_model=list[JobResponse])
def get_logged_in_recommendations(request: Request, top_k: int = 5):
    user = get_current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Can dang nhap")

    try:
        jobs = recommend_jobs(str(user.get("skills", "")), max(1, min(top_k, 20)))
        return [JobResponse(**{k: item[k] for k in ["job_id", "job_name", "match_score"]}) for item in jobs]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health")
def health_check():
    with get_db_connection() as conn:
        users_count = conn.execute("SELECT COUNT(1) FROM users").fetchone()[0]

    return {
        "status": "ok",
        "artifacts_loaded": artifact_error is None,
        "artifact_error": artifact_error,
        "total_unique_jobs_loaded": len(df_jobs) if df_jobs is not None else 0,
        "registered_users": users_count,
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)