from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from config import DATABASE_URL

DEFAULT_SQLITE_URL = "sqlite:///./app.db"
SQLALCHEMY_DATABASE_URL = DATABASE_URL or DEFAULT_SQLITE_URL

connect_args = {"check_same_thread": False} if SQLALCHEMY_DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


class Base(DeclarativeBase):
    pass


def ensure_database_connection():
    global engine

    try:
        with engine.connect():
            return engine
    except SQLAlchemyError:
        if str(engine.url) == DEFAULT_SQLITE_URL:
            raise

    engine = create_engine(
        DEFAULT_SQLITE_URL,
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
    )
    SessionLocal.configure(bind=engine)
    return engine


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
