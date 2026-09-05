import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import auth, chat, documents, health, reports, summarize, workflow
from app.core.config import get_settings
from app.core.exceptions import ServiceConfigurationError
from app.core.logging import RequestLoggingMiddleware, configure_logging
from app.db.init_db import init_db


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()
    init_db()
    rebuild_index_if_requested()
    yield


def rebuild_index_if_requested() -> None:
    """Rebuild the vector index at startup when the filesystem is ephemeral.

    On hosts that do not persist disk (Render's free tier, for example) the
    FAISS index is gone after every deploy, which would make retrieval return
    nothing at all while still appearing healthy. Document text is stored in
    the database, so the index can be regenerated from it.
    """
    settings = get_settings()
    if not settings.reindex_on_startup:
        return

    logger = logging.getLogger(__name__)
    try:
        from scripts.reindex import reindex

        documents, chunks = reindex(drop_existing=True)
        logger.info("Rebuilt vector index: %s document(s), %s chunk(s)", documents, chunks)
    except Exception:
        # A failed rebuild must not stop the API from serving.
        logger.exception("Startup reindex failed; retrieval may return no results")


settings = get_settings()
app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["health"])
app.include_router(auth.router, prefix=settings.api_v1_prefix, tags=["auth"])
app.include_router(chat.router, prefix=settings.api_v1_prefix, tags=["chat"])
app.include_router(reports.router, prefix=settings.api_v1_prefix, tags=["reports"])
app.include_router(summarize.router, prefix=settings.api_v1_prefix, tags=["summaries"])
app.include_router(documents.router, prefix=settings.api_v1_prefix, tags=["documents"])
app.include_router(workflow.router, prefix=settings.api_v1_prefix, tags=["workflow"])


@app.exception_handler(ServiceConfigurationError)
async def service_configuration_error_handler(
    _: Request, exc: ServiceConfigurationError
) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/", include_in_schema=False)
async def root() -> JSONResponse:
    return JSONResponse(
        {
            "message": "Internal AI Agents API is running.",
            "docs_url": "/docs",
            "streamlit_frontend": "Run `streamlit run frontend/streamlit_app.py` to start the UI.",
        }
    )
