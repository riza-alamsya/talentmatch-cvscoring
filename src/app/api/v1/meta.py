"""Meta endpoints — expose available AI providers so the FE can offer a selector."""
from fastapi import APIRouter

from app.core.config import settings

router = APIRouter(tags=["Meta"])


@router.get("/providers", summary="Daftar provider AI yang tersedia (LLM + embedding)")
async def providers():
    llm = settings.llm_providers()
    emb = settings.embed_providers()
    return {
        "llm": [
            {"id": k, "label": v["label"], "available": bool(v["api_key"])}
            for k, v in llm.items()
        ],
        "embed": [
            {"id": k, "label": v["label"]} for k, v in emb.items()
        ],
        "defaults": {"llm": settings.DEFAULT_LLM, "embed": settings.DEFAULT_EMBED},
    }
