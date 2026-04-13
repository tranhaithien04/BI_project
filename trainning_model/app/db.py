from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from .config import DATABASE_URL


def _create_engine(database_url: str):
    url: URL = make_url(database_url)
    if not url.drivername.startswith("mysql"):
        raise RuntimeError(
            "APP_DATABASE_URL must use mysql+pymysql:// and cannot use SQLite"
        )
    return create_engine(database_url, pool_pre_ping=True, future=True)


def _quote_identifier(identifier: str) -> str:
    cleaned = identifier.replace("`", "")
    return f"`{cleaned}`"


def _ensure_mysql_database_exists(database_url: str) -> None:
    url: URL = make_url(database_url)
    if not url.drivername.startswith("mysql"):
        raise RuntimeError(
            "APP_DATABASE_URL must use mysql+pymysql:// and cannot use SQLite"
        )

    if not url.database:
        raise RuntimeError("APP_DATABASE_URL must include database name")

    admin_url = url.set(database=None)
    admin_engine = create_engine(
        admin_url,
        pool_pre_ping=True,
        future=True,
        isolation_level="AUTOCOMMIT",
    )
    database_name = _quote_identifier(url.database)

    with admin_engine.connect() as connection:
        connection.execute(
            text(
                "CREATE DATABASE IF NOT EXISTS "
                f"{database_name} "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        )

    admin_engine.dispose()


engine = _create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
Base = declarative_base()


def init_database() -> None:
    _ensure_mysql_database_exists(DATABASE_URL)

    # Import models before create_all so SQLAlchemy knows mapped tables.
    from . import models  # noqa: F401

    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
