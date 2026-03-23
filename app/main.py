from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi import Request
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
    yield


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
