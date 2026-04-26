# core/config.py
# Centralised, typed application settings loaded from environment variables.
# Uses pydantic-settings so every value is validated at startup.
# Add new env vars here — never use os.getenv() directly elsewhere.

from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # ── Groq LLM ──────────────────────────────────────────────────────────────
    groq_api_key: str = ""

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Comma-separated list of allowed frontend origins.
    # Example: "http://localhost:5173,https://my-app.vercel.app"
    allowed_origins: str = "http://localhost:5173"

    # ── Paths ─────────────────────────────────────────────────────────────────
    # Resolved relative to this file so the server can run from any cwd.
    vector_store_dir: Path = Path(__file__).resolve().parents[1] / "vectorstore"

    # ── Model ─────────────────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.3

    # ── API behaviour ─────────────────────────────────────────────────────────
    default_top_k: int = 5
    max_top_k: int = 20
    max_query_length: int = 500

    class Config:
        # Reads from backend/.env automatically
        env_file = ".env"
        env_file_encoding = "utf-8"

    def get_allowed_origins(self) -> list[str]:
        """Return CORS origins as a list, stripping whitespace."""
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


# Module-level singleton — import this everywhere instead of re-instantiating.
settings = Settings()
