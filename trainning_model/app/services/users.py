from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..config import DEFAULT_ROLE, VALID_ROLES
from ..models import User
from ..security import hash_password


def normalize_user_role(role: str) -> str:
    normalized = role.strip().lower()
    if normalized not in VALID_ROLES:
        raise ValueError("Vai trò không hợp lệ")
    return normalized


def is_recruiter(user: User) -> bool:
    return user.role == "recruiter"


def get_user_home_path(user: User) -> str:
    return "/recruiter/jobs" if is_recruiter(user) else "/dashboard"


def get_user_by_username(db: Session, username: str) -> User | None:
    return db.scalar(select(User).where(User.username == username))


def get_user_by_id(db: Session, user_id: int) -> User | None:
    return db.get(User, user_id)


def create_user(
    db: Session,
    username: str,
    password: str,
    role: str,
    full_name: str,
    email: str,
    skills: str,
) -> int:
    normalized_role = normalize_user_role(role)
    user = User(
        username=username,
        password_hash=hash_password(password),
        role=normalized_role,
        full_name=full_name,
        email=email,
        skills=skills,
    )

    try:
        db.add(user)
        db.commit()
        db.refresh(user)
    except IntegrityError as exc:
        db.rollback()
        raise ValueError("Tên đăng nhập đã tồn tại") from exc

    return int(user.id)


def update_user_profile(
    db: Session,
    user_id: int,
    full_name: str,
    email: str,
    skills: str,
) -> User | None:
    user = get_user_by_id(db, user_id)
    if user is None:
        return None

    user.full_name = full_name
    user.email = email
    user.skills = skills
    db.commit()
    db.refresh(user)
    return user


def count_users(db: Session) -> int:
    return int(db.scalar(select(func.count(User.id))) or 0)


def apply_default_role_if_empty(role: str | None) -> str:
    if role is None or not role.strip():
        return DEFAULT_ROLE
    return role
