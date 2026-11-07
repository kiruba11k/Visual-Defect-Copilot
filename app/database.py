"""Neon/PostgreSQL connectivity helpers.

The database connection is created lazily. Set DATABASE_URL in the environment;
never commit the connection string.
"""
from sqlalchemy import create_engine, text
from app.config import settings

_engine = None


def get_engine():
    global _engine
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is not configured")
    if _engine is None:
        url = settings.database_url
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg://", 1)
        _engine = create_engine(
            url,
            pool_pre_ping=True,
            pool_recycle=300,
            connect_args={"sslmode": "require"},
        )
    return _engine


def check_database():
    with get_engine().connect() as connection:
        return connection.execute(text("SELECT 1")).scalar_one() == 1
