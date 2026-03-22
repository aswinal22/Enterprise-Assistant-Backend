from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from config import DATABASE_URL
from logging_utils import get_logger

DEFAULT_SQLITE_URL = "sqlite:///./app.db"
SQLALCHEMY_DATABASE_URL = DATABASE_URL or DEFAULT_SQLITE_URL
logger = get_logger(__name__)

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
        logger.info("Checking database connection url=%s", engine.url)
        with engine.connect():
            logger.info("Database connection established successfully")
            return engine
    except SQLAlchemyError:
        logger.exception("Primary database connection failed url=%s", engine.url)
        if str(engine.url) == DEFAULT_SQLITE_URL:
            raise

    logger.warning("Falling back to default SQLite database url=%s", DEFAULT_SQLITE_URL)
    engine = create_engine(
        DEFAULT_SQLITE_URL,
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
    )
    SessionLocal.configure(bind=engine)
    logger.info("Database fallback configured successfully")
    return engine


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        logger.debug("Closing database session")
        db.close()
