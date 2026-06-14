from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # .env di root project (CVscoring/.env) — absolute biar tahan ganti CWD
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parents[3] / ".env"),
        extra="ignore",
    )

    # paths (relatif dari root project, bukan dari src/)
    BASE_DIR: Path = Path(__file__).resolve().parents[3]
    DATA_DIR: Path = BASE_DIR / "data"
    CV_DIR: Path = DATA_DIR / "cv"
    PROCESSED_DIR: Path = DATA_DIR / "processed_gemini"
    CHROMA_DIR: Path = DATA_DIR / "chroma"

    # ── API keys (dari .env) ──────────────────────────────────────────────────
    GEMINI_API_KEY: str = ""
    MIMO_API_KEY: str = ""

    # endpoints
    GEMINI_OPENAI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    GEMINI_NATIVE_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta"
    MIMO_BASE_URL: str = "https://api.xiaomimimo.com/v1"

    # ── Provider defaults (bisa dioverride per-request dari FE) ────────────────
    DEFAULT_LLM: str = "gemini"      # gemini | mimo
    DEFAULT_EMBED: str = "gemini"    # gemini | local
    DEFAULT_LANG: str = "en"         # reason output language: en | id | ms | zh

    # Maks ekstraksi CV berjalan paralel (per worker). Pelindung diri engine:
    # tiap ekstraksi = 1 call LLM + banyak call embedding → jaga rate limit Gemini.
    MAX_CONCURRENT_EXTRACT: int = 2

    # model names per LLM provider
    GEMINI_LLM_MODEL: str = "gemini-2.5-flash"
    MIMO_LLM_MODEL: str = "mimo-v2.5"

    # embedding params
    GEMINI_EMBED_MODEL: str = "gemini-embedding-001"
    GEMINI_EMBED_DIM: int = 1536
    # Small multilingual model: ~470MB, 384-dim, RAM ~600MB → fits cheap Cloud Run.
    # (e5 models REQUIRE "query:"/"passage:" prefixes — handled in embedder.)
    LOCAL_EMBED_MODEL: str = "intfloat/multilingual-e5-small"
    LOCAL_EMBED_DIM: int = 384
    ENABLE_LOCAL_EMBED: bool = True

    # scoring weights
    WEIGHT_SKILLS: float = 0.40
    WEIGHT_SEMANTIC: float = 0.35
    WEIGHT_YEARS: float = 0.15
    WEIGHT_EDUCATION: float = 0.10

    # ── Registries (dipakai service layer + endpoint /providers) ───────────────
    def llm_providers(self) -> dict:
        return {
            "gemini": {
                "label": "Gemini 2.5 Flash",
                "api_key": self.GEMINI_API_KEY,
                "base_url": self.GEMINI_OPENAI_BASE_URL,
                "model": self.GEMINI_LLM_MODEL,
            },
            "mimo": {
                "label": "MiMo v2.5",
                "api_key": self.MIMO_API_KEY,
                "base_url": self.MIMO_BASE_URL,
                "model": self.MIMO_LLM_MODEL,
            },
        }

    def embed_providers(self) -> dict:
        providers = {
            "gemini": {
                "label": "Gemini embedding (1536d)",
                "type": "gemini",
                "model": self.GEMINI_EMBED_MODEL,
                "dim": self.GEMINI_EMBED_DIM,
                "collection": "cv_chunks_gemini",
            },
        }
        # Only offer local bge-m3 where there's enough RAM + the model (e.g. local dev).
        if self.ENABLE_LOCAL_EMBED:
            providers["local"] = {
                "label": "e5-small lokal (384d)",
                "type": "local",
                "model": self.LOCAL_EMBED_MODEL,
                "dim": self.LOCAL_EMBED_DIM,
                "collection": "cv_chunks_e5small",
            }
        return providers


settings = Settings()

# pastikan folder penting ada
for d in [settings.CV_DIR, settings.PROCESSED_DIR, settings.CHROMA_DIR]:
    d.mkdir(parents=True, exist_ok=True)
