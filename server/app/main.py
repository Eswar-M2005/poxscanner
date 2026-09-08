"""Pox Scanner API entrypoint."""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.routes import router as v1_router
from app.core.config import settings
from app.core.errors import APIError, api_error_handler, validation_error_handler
from app.schemas.predict import HealthResponse
from app.services.inference import get_inference_service

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_inference_service().try_load()
    yield


app = FastAPI(
    title="Pox Scanner API",
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

_origins = settings.cors_origin_list or ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials="*" not in _origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(APIError, api_error_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)

app.include_router(v1_router, prefix="/api/v1")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    service = get_inference_service()
    service.try_load()
    return HealthResponse(
        status="ok",
        service="poxscanner-api",
        model_loaded=service.model_loaded,
        demo_mode=service.using_demo,
        version=settings.APP_VERSION,
    )
