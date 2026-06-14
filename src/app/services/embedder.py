"""Embedding CV chunks ke ChromaDB — multi-provider.

Provider:
  - gemini : API embedContent (taskType + outputDimensionality, 1536d)
  - local  : bge-m3 via sentence-transformers (1024d, gratis tanpa limit)

Tiap provider punya collection sendiri (dimensi/ruang vektor beda → TIDAK boleh
dicampur). Vektor di-L2-normalize untuk cosine."""
from __future__ import annotations

import json
import math
import re
import ssl
import threading
import time
import urllib.error
import urllib.request

import certifi
import chromadb

from app.core.config import settings

# Homebrew Python tidak punya CA bundle untuk SSL default urllib → pakai certifi
_SSL_CTX = ssl.create_default_context(cafile=certifi.where())

_client: chromadb.PersistentClient | None = None
_collections: dict[str, object] = {}
_local_model = None

# Endpoint `def` jalan paralel di threadpool → lazy init & tulis index harus diserialisasi.
_init_lock = threading.Lock()
_index_lock = threading.Lock()


# ── Chroma collection (cached per provider) ───────────────────────────────────
def _get_collection(provider: str):
    global _client
    with _init_lock:
        if provider not in _collections:
            if _client is None:
                _client = chromadb.PersistentClient(path=str(settings.CHROMA_DIR))
            name = settings.embed_providers()[provider]["collection"]
            _collections[provider] = _client.get_or_create_collection(
                name, metadata={"hnsw:space": "cosine"}
            )
    return _collections[provider]


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec


# ── Gemini embeddings (API) ───────────────────────────────────────────────────
def _gemini_embed(text: str, task_type: str, cfg: dict, max_retries: int = 6) -> list[float]:
    url = f"{settings.GEMINI_NATIVE_BASE_URL}/models/{cfg['model']}:embedContent"
    payload = json.dumps({
        "model": f"models/{cfg['model']}",
        "content": {"parts": [{"text": text}]},
        "taskType": task_type,
        "outputDimensionality": cfg["dim"],
    }).encode()
    headers = {"Content-Type": "application/json", "x-goog-api-key": settings.GEMINI_API_KEY}

    delay = 2.0
    for attempt in range(max_retries):
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60, context=_SSL_CTX) as r:
                data = json.loads(r.read())
            return _normalize(data["embedding"]["values"])
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "ignore")
            if e.code in (429, 500, 503) and attempt < max_retries - 1:
                m = re.search(r"retry in ([\d.]+)s", body) or re.search(r'"retryDelay":\s*"([\d.]+)s"', body)
                time.sleep(min(float(m.group(1)) + 1 if m else delay, 60))
                delay = min(delay * 2, 30)
                continue
            raise RuntimeError(f"Gemini embed gagal {e.code}: {body[:200]}")
        except (urllib.error.URLError, TimeoutError):
            if attempt < max_retries - 1:
                time.sleep(delay); delay = min(delay * 2, 30); continue
            raise
    raise RuntimeError("Gemini embed gagal setelah semua retry")


# ── Local embeddings (bge-m3 via sentence-transformers, lazy) ─────────────────
def _get_local_model():
    global _local_model
    with _init_lock:
        if _local_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RuntimeError(
                    "Embedding lokal butuh 'sentence-transformers' (belum terpasang di image ini)."
                ) from e
            _local_model = SentenceTransformer(settings.LOCAL_EMBED_MODEL)
    return _local_model


def _e5_prefix(is_query: bool) -> str:
    # e5 models REQUIRE these prefixes for good retrieval quality.
    return "query: " if is_query else "passage: "


def _local_embed(text: str, is_query: bool) -> list[float]:
    m = _get_local_model()
    v = m.encode(_e5_prefix(is_query) + text, normalize_embeddings=True)
    return v.tolist()


def _local_embed_batch(texts: list[str], is_query: bool) -> list[list[float]]:
    m = _get_local_model()
    pre = _e5_prefix(is_query)
    vecs = m.encode([pre + t for t in texts], normalize_embeddings=True, batch_size=32)
    return [v.tolist() for v in vecs]


def preload_local() -> None:
    """Warm the local model at startup so the first request isn't slow."""
    try:
        _get_local_model()
    except Exception:
        pass


# ── Unified embed API ─────────────────────────────────────────────────────────
def embed_document(text: str, provider: str) -> list[float]:
    cfg = settings.embed_providers()[provider]
    if cfg["type"] == "gemini":
        return _gemini_embed(text, "RETRIEVAL_DOCUMENT", cfg)
    return _local_embed(text, is_query=False)


def embed_query(text: str, provider: str) -> list[float]:
    cfg = settings.embed_providers()[provider]
    if cfg["type"] == "gemini":
        return _gemini_embed(text, "RETRIEVAL_QUERY", cfg)
    return _local_embed(text, is_query=True)


def embed_documents(texts: list[str], provider: str) -> list[list[float]]:
    """Batch-embed document chunks. Local runs one batched .encode (fast);
    Gemini is per-call (its API embeds one input at a time)."""
    cfg = settings.embed_providers()[provider]
    if cfg["type"] == "gemini":
        return [_gemini_embed(t, "RETRIEVAL_DOCUMENT", cfg) for t in texts]
    return _local_embed_batch(texts, is_query=False)


# ── Chunking (provider-agnostic, defensif terhadap data jelek) ────────────────
def cv_to_chunks(cv: dict, cv_id: str) -> list[dict]:
    chunks: list[dict] = []
    name = (cv.get("personal_info") or {}).get("name", "") if isinstance(cv.get("personal_info"), dict) else ""

    def add(kind: str, text: str, **meta):
        text = (text or "").strip()
        if text:
            chunks.append({
                "id":   f"{cv_id}::{kind}::{len(chunks)}",
                "text": text,
                "meta": {"cv_id": cv_id, "name": name, "type": kind, **meta},
            })

    add("summary", cv.get("summary", "") if isinstance(cv.get("summary"), str) else "")

    sk = cv.get("skills") if isinstance(cv.get("skills"), dict) else {}
    all_skills = [s for s in ((sk.get("hard_skills") or []) + (sk.get("soft_skills") or [])) if isinstance(s, str)]
    if all_skills:
        add("skills", "Skills: " + ", ".join(all_skills), skills=", ".join(all_skills))

    for e in cv.get("experience", []):
        if not isinstance(e, dict):
            continue
        ks = ", ".join(x for x in (e.get("key_skills") or []) if isinstance(x, str))
        txt = f'{e.get("role","")} at {e.get("company","")}. {e.get("summary","")} Skills: {ks}'
        add("experience", txt,
            company=e.get("company", ""), role=e.get("role", ""),
            start_date=e.get("start_date", ""), end_date=e.get("end_date", ""),
            is_current=bool(e.get("is_current", False)), key_skills=ks)

    for ed in cv.get("education", []):
        if not isinstance(ed, dict):
            continue
        add("education",
            f'{ed.get("degree","")} in {ed.get("field_of_study","")} at {ed.get("institution","")}',
            institution=ed.get("institution", ""))

    for c in cv.get("certifications", []):
        if isinstance(c, str):
            add("certification", c)
        elif isinstance(c, dict):
            add("certification", f'{c.get("name","")} - {c.get("issuer","")}', issuer=c.get("issuer", ""))

    achievements = [a for a in (cv.get("achievements") or []) if isinstance(a, str)]
    if achievements:
        add("achievements", " ".join(achievements))

    return chunks


def index_cv(cv_id: str, cv: dict, provider: str) -> int:
    """Embed semua chunk dari satu CV ke collection provider tsb. Return jumlah chunk."""
    col = _get_collection(provider)

    chunks = cv_to_chunks(cv, cv_id)
    if not chunks:
        return 0
    # Embedding (lambat, network) di luar lock; operasi Chroma (cepat) di dalam lock
    # supaya delete+add per cv_id atomik antar thread.
    embeddings = embed_documents([c["text"] for c in chunks], provider)  # batched for local
    with _index_lock:
        try:
            existing = col.get(where={"cv_id": cv_id})
            if existing["ids"]:
                col.delete(ids=existing["ids"])
        except Exception:
            pass
        col.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            embeddings=embeddings,
            metadatas=[c["meta"] for c in chunks],
        )
    return len(chunks)


def is_indexed(cv_id: str, provider: str) -> bool:
    col = _get_collection(provider)
    try:
        got = col.get(where={"cv_id": cv_id}, limit=1)
        return bool(got["ids"])
    except Exception:
        return False


def ensure_indexed(cv_id: str, cv: dict, provider: str) -> None:
    """Index CV ke collection provider kalau belum ada (lazy re-index saat ganti embedding)."""
    if not is_indexed(cv_id, provider):
        index_cv(cv_id, cv, provider)


def search_chunks(query_embedding: list[float], cv_id: str | None, provider: str, top_k: int = 3) -> list[dict]:
    col = _get_collection(provider)
    where = {"cv_id": cv_id} if cv_id else None
    res = col.query(query_embeddings=[query_embedding], n_results=top_k, where=where)
    out = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        out.append({"text": doc, "meta": meta, "similarity": round(1 - dist, 4)})
    return out


def collection_count(provider: str) -> int:
    return _get_collection(provider).count()
