import logging
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.app.api.routes import router
from backend.app.core.config import RESULTS_DIR, settings
from backend.app.core.logging_config import configure_logging
from backend.app.db.database import init_db

configure_logging()
logger = logging.getLogger("icvqi")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    if not settings.auth_enabled:
        logger.warning(
            "API authentication is DISABLED (no ICVQI_API_KEYS configured). "
            "Do not run like this in production."
        )
    logger.info(
        "Service started",
        extra={"version": settings.app_version, "environment": settings.environment},
    )
    yield


RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title=settings.app_name,
    description="AI-powered PCB defect detection API using YOLO.",
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Attach a request id and log latency for every request."""
    request_id = request.headers.get("X-Request-ID", uuid4().hex[:12])
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:  # noqa: BLE001 - convert to 500 + structured log
        elapsed = round((time.perf_counter() - start) * 1000, 2)
        logger.exception(
            "Unhandled error",
            extra={"request_id": request_id, "path": request.url.path, "ms": elapsed},
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error.", "request_id": request_id},
        )
    elapsed = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "ms": elapsed,
        },
    )
    return response


app.mount(
    "/static/results",
    StaticFiles(directory=str(RESULTS_DIR)),
    name="results",
)

app.include_router(router, prefix="/api")


@app.get("/")
def root():
    return {
        "message": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/api/health",
        "ready": "/api/ready",
    }
