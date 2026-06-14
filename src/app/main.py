"""FastAPI app entry point."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import cv, meta, scoring
from app.core.config import settings
from app.services.embedder import preload_local


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the local embedding model at startup when it's the default, so the
    # first request isn't slowed by model loading.
    if settings.DEFAULT_EMBED == "local" and settings.ENABLE_LOCAL_EMBED:
        preload_local()
    yield


app = FastAPI(
    title="TalentMatch API",
    description=(
        "AI-powered CV parsing & scoring. "
        "Dirancang untuk dipanggil dari Spring Boot backend."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — Spring Boot dan FE bisa akses
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ganti ke domain spesifik di produksi
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cv.router,      prefix="/api/v1")
app.include_router(scoring.router, prefix="/api/v1")
app.include_router(meta.router,    prefix="/api/v1")


@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "service": "TalentMatch API", "version": "0.1.0"}


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}
