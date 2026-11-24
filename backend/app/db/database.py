"""Database engine and session management.

The engine is driven by ``settings.database_url`` so the same code runs on
SQLite (local dev) or PostgreSQL (production) with no changes.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from backend.app.core.config import DATABASE_DIR, settings

connect_args = {}
if settings.is_sqlite:
    # SQLite lives on the local filesystem; ensure its directory exists and
    # allow use across FastAPI's threadpool.
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.database_url,
    connect_args=connect_args,
    pool_pre_ping=not settings.is_sqlite,
    future=True,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


def init_db() -> None:
    """Create tables if they do not exist.

    Convenient for local SQLite development. In production, schema changes are
    managed by Alembic migrations (``alembic upgrade head``) instead.
    """
    from backend.app.db import models  # noqa: F401  (register mappers)

    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
