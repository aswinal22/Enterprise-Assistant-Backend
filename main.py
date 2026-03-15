from fastapi import FastAPI

from database import Base, ensure_database_connection

# Import models so tables are registered
from models import document
from models import document_chunk

from routes.document_routes import router as document_router
from routes.query_routes import router as query_router

app = FastAPI(
    title="RAG Service",
    version="1.0"
)


@app.on_event("startup")
def create_tables() -> None:
    engine = ensure_database_connection()
    Base.metadata.create_all(bind=engine)


@app.get("/")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(document_router, prefix="/api/v1")
app.include_router(query_router, prefix="/api/v1")
