from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..models import JobApplication, JobPost, User


def _application_count_subquery():
    return (
        select(
            JobApplication.job_post_id.label("job_post_id"),
            func.count(JobApplication.id).label("total_applied"),
        )
        .group_by(JobApplication.job_post_id)
        .subquery()
    )


def create_job_post(
    db: Session,
    recruiter_id: int,
    title: str,
    description: str,
    requirements: str,
    location: str,
) -> int:
    cleaned_title = title.strip()
    if len(cleaned_title) < 3:
        raise ValueError("Tiêu đề bài đăng cần ít nhất 3 ký tự")

    post = JobPost(
        recruiter_id=recruiter_id,
        title=cleaned_title,
        description=description.strip(),
        requirements=requirements.strip(),
        location=location.strip(),
    )
    db.add(post)
    db.commit()
    db.refresh(post)
    return int(post.id)


def list_open_job_posts(db: Session) -> list[dict]:
    counts = _application_count_subquery()
    stmt = (
        select(
            JobPost.id,
            JobPost.title,
            JobPost.description,
            JobPost.requirements,
            JobPost.location,
            JobPost.status,
            JobPost.created_at,
            User.username.label("recruiter_username"),
            User.full_name.label("recruiter_full_name"),
            func.coalesce(counts.c.total_applied, 0).label("total_applied"),
        )
        .join(User, User.id == JobPost.recruiter_id)
        .outerjoin(counts, counts.c.job_post_id == JobPost.id)
        .where(JobPost.status == "open")
        .order_by(JobPost.created_at.desc())
    )

    return [dict(row) for row in db.execute(stmt).mappings().all()]


def list_recruiter_job_posts(db: Session, recruiter_id: int) -> list[dict]:
    counts = _application_count_subquery()
    stmt = (
        select(
            JobPost.id,
            JobPost.title,
            JobPost.description,
            JobPost.requirements,
            JobPost.location,
            JobPost.status,
            JobPost.created_at,
            func.coalesce(counts.c.total_applied, 0).label("total_applied"),
        )
        .outerjoin(counts, counts.c.job_post_id == JobPost.id)
        .where(JobPost.recruiter_id == recruiter_id)
        .order_by(JobPost.created_at.desc())
    )

    return [dict(row) for row in db.execute(stmt).mappings().all()]


def get_job_post_by_id(db: Session, job_post_id: int) -> JobPost | None:
    return db.get(JobPost, job_post_id)


def get_applied_job_ids(db: Session, candidate_id: int) -> set[int]:
    stmt = select(JobApplication.job_post_id).where(JobApplication.candidate_id == candidate_id)
    return {int(job_post_id) for job_post_id in db.scalars(stmt).all()}


def apply_to_job(db: Session, job_post_id: int, candidate_id: int, cover_letter: str = "") -> None:
    post = get_job_post_by_id(db, job_post_id)
    if post is None:
        raise ValueError("Không tìm thấy bài đăng")
    if post.status != "open":
        raise ValueError("Bài đăng đã đóng")

    duplicate_stmt = select(JobApplication.id).where(
        JobApplication.job_post_id == job_post_id,
        JobApplication.candidate_id == candidate_id,
    )
    duplicate_id = db.scalar(duplicate_stmt)
    if duplicate_id is not None:
        raise ValueError("Bạn đã ứng tuyển bài đăng này")

    application = JobApplication(
        job_post_id=job_post_id,
        candidate_id=candidate_id,
        cover_letter=cover_letter.strip(),
    )
    db.add(application)
    db.commit()


def list_recruiter_applications(db: Session, recruiter_id: int) -> list[dict]:
    stmt = (
        select(
            JobApplication.id,
            JobApplication.job_post_id,
            JobPost.title.label("job_title"),
            JobApplication.candidate_id,
            User.username.label("candidate_username"),
            User.full_name.label("candidate_full_name"),
            User.email.label("candidate_email"),
            User.skills.label("candidate_skills"),
            JobApplication.cover_letter,
            JobApplication.status,
            JobApplication.created_at,
        )
        .join(JobPost, JobPost.id == JobApplication.job_post_id)
        .join(User, User.id == JobApplication.candidate_id)
        .where(JobPost.recruiter_id == recruiter_id)
        .order_by(JobApplication.created_at.desc())
    )

    return [dict(row) for row in db.execute(stmt).mappings().all()]


def count_job_posts(db: Session) -> int:
    return int(db.scalar(select(func.count(JobPost.id))) or 0)


def count_job_applications(db: Session) -> int:
    return int(db.scalar(select(func.count(JobApplication.id))) or 0)
