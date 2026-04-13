from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from .config import SESSION_SECRET, STATIC_DIR, TEMPLATES_DIR
from .db import init_database
from .routers import api, auth, candidate, health, recruiter
from .services.artifacts import load_artifacts, set_artifact_error


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_database()
    try:
        load_artifacts()
    except Exception as exc:
        # App van khoi dong de su dung auth/profile khi artifacts loi tam thoi.
        set_artifact_error(str(exc))
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Job Recommendation App",
        description="Web app dang nhap, cap nhat profile va goi y viec lam theo skills",
        version="3.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, same_site="lax")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.state.templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    app.include_router(auth.router)
    app.include_router(candidate.router)
    app.include_router(recruiter.router)
    app.include_router(api.router)
    app.include_router(health.router)

    return app


app = create_app()
