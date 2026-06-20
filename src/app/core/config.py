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
    MIMO_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    DEEPSEEK_API_KEY: str = ""

    # endpoints
    MIMO_BASE_URL: str = "https://api.xiaomimimo.com/v1"
    GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"

    # ── Provider defaults (bisa dioverride per-request dari FE) ────────────────
    DEFAULT_LLM: str = "deepseek"       # deepseek | gemini-flash | mimo
    DEFAULT_EMBED: str = "local"        # local (e5-small)
    DEFAULT_LANG: str = "en"            # reason output language: en | id | ms | zh

    # Maks ekstraksi CV berjalan paralel (per worker).
    MAX_CONCURRENT_EXTRACT: int = 10

    # model names per LLM provider
    MIMO_LLM_MODEL: str = "mimo-v2.5"
    GEMINI_FLASH_MODEL: str = "gemini-2.0-flash-lite"
    DEEPSEEK_MODEL: str = "deepseek-chat"

    # embedding params (local only)
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
            "deepseek": {
                "label": "DeepSeek Chat",
                "api_key": self.DEEPSEEK_API_KEY,
                "base_url": self.DEEPSEEK_BASE_URL,
                "model": self.DEEPSEEK_MODEL,
            },
            "gemini-flash": {
                "label": "Gemini 2.0 Flash Lite",
                "api_key": self.GEMINI_API_KEY,
                "base_url": self.GEMINI_BASE_URL,
                "model": self.GEMINI_FLASH_MODEL,
            },
            "mimo": {
                "label": "MiMo v2.5",
                "api_key": self.MIMO_API_KEY,
                "base_url": self.MIMO_BASE_URL,
                "model": self.MIMO_LLM_MODEL,
            },
        }

    def embed_providers(self) -> dict:
        # Local e5-small only — free, no external API, fits cheap Cloud Run.
        return {
            "local": {
                "label": "e5-small lokal (384d)",
                "type": "local",
                "model": self.LOCAL_EMBED_MODEL,
                "dim": self.LOCAL_EMBED_DIM,
                "collection": "cv_chunks_e5small",
            },
        }


settings = Settings()

# pastikan folder penting ada
for d in [settings.CV_DIR, settings.PROCESSED_DIR, settings.CHROMA_DIR]:
    d.mkdir(parents=True, exist_ok=True)
