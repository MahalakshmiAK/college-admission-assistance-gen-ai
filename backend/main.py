# main.py
# FastAPI application entry point.
# Responsibilities:
#   - App factory with lifespan (startup / shutdown)
#   - CORS middleware from typed config
#   - Router registration under /api/v1
#   - RAG system initialisation (not at import time)

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from routers import chat, health
from rag_engine import CollegeAdmissionRAG


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at startup before the server accepts requests.
    Attaches the RAG system to app.state so every router can access it
    via request.app.state.rag without re-initialising per request.
    """
    print("[startup] Initialising RAG system...")

    if not settings.groq_api_key:
        print("[startup] WARNING: GROQ_API_KEY is not set. LLM responses will be disabled.")

    app.state.rag = CollegeAdmissionRAG(groq_api_key=settings.groq_api_key or None)
    print("[startup] RAG system ready.")

    yield  # server runs here

    # ── Shutdown ───────────────────────────────────────────────────────────────
    print("[shutdown] Cleaning up resources.")
    app.state.rag = None


# ── App factory ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="College Admission RAG API",
    version="1.0.0",
    description="RAG-powered assistant for Indian engineering college admissions.",
    lifespan=lifespan,
)

# ── CORS — origins driven by .env, never hardcoded ────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
# All routes live under /api/v1 for clean versioning.
# Adding v2 later is a one-line include_router call.

app.include_router(chat.router,   prefix="/api/v1")
app.include_router(health.router, prefix="/api/v1")
