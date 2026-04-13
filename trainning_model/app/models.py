from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="candidate")
    full_name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    email: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    skills: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    recruiter_posts: Mapped[list[JobPost]] = relationship(
        "JobPost",
        back_populates="recruiter",
        foreign_keys="JobPost.recruiter_id",
    )
    applications: Mapped[list[JobApplication]] = relationship(
        "JobApplication",
        back_populates="candidate",
        foreign_keys="JobApplication.candidate_id",
    )


class JobPost(Base):
    __tablename__ = "job_posts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    recruiter_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    requirements: Mapped[str] = mapped_column(Text, nullable=False, default="")
    location: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="open")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    recruiter: Mapped[User] = relationship("User", back_populates="recruiter_posts")
    applications: Mapped[list[JobApplication]] = relationship("JobApplication", back_populates="job_post")


class JobApplication(Base):
    __tablename__ = "job_applications"
    __table_args__ = (
        UniqueConstraint("job_post_id", "candidate_id", name="uq_job_post_candidate"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_post_id: Mapped[int] = mapped_column(ForeignKey("job_posts.id"), nullable=False, index=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    cover_letter: Mapped[str] = mapped_column(Text, nullable=False, default="")
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="applied")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now())

    job_post: Mapped[JobPost] = relationship("JobPost", back_populates="applications")
    candidate: Mapped[User] = relationship("User", back_populates="applications")
