import time
from uuid import uuid4

from fastapi import FastAPI, Request

from database import Base, ensure_database_connection
from logging_utils import configure_logging, get_logger

# Import models so tables are registered
from models import document
from models import document_chunk

from routes.document_routes import router as document_router
from routes.query_routes import router as query_router

configure_logging()
logger = get_logger(__name__)

app = FastAPI(
    title="RAG Service",
    version="1.0"
)


@app.on_event("startup")
def create_tables() -> None:
    logger.info("Application startup initiated")
    engine = ensure_database_connection()
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ensured successfully")


@app.get("/")
def healthcheck() -> dict[str, str]:
    logger.debug("Healthcheck requested")
    return {"status": "ok"}


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid4())
    start_time = time.perf_counter()

    logger.info(
        "Request started request_id=%s method=%s path=%s",
        request_id,
        request.method,
        request.url.path,
    )

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.exception(
            "Request failed request_id=%s method=%s path=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            duration_ms,
        )
        raise

    duration_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "Request completed request_id=%s method=%s path=%s status_code=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


app.include_router(document_router, prefix="/api/v1")
app.include_router(query_router, prefix="/api/v1")
