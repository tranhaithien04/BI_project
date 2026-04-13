from __future__ import annotations

from fastapi import Depends, Request
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from .db import get_db
from .models import User
from .services.users import get_user_by_id


def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
) -> User | None:
    user_id = request.session.get("user_id")
    if user_id is None:
        return None

    return get_user_by_id(db, int(user_id))


def get_templates(request: Request) -> Jinja2Templates:
    return request.app.state.templates
