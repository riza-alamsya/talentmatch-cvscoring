"""Embedding CV chunks ke ChromaDB — local e5-small (gratis, tanpa API).

Vektor di-L2-normalize (oleh sentence-transformers) untuk cosine. Collection
`cv_chunks_e5small` (384d)."""
from __future__ import annotations

import threading

import chromadb

from app.core.config import settings

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


# ── Local embeddings (e5-small via sentence-transformers, lazy) ───────────────
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


# ── Unified embed API (local e5-small only) ───────────────────────────────────
def embed_document(text: str, provider: str = "local") -> list[float]:
    return _local_embed(text, is_query=False)


def embed_query(text: str, provider: str = "local") -> list[float]:
    return _local_embed(text, is_query=True)


def embed_documents(texts: list[str], provider: str = "local") -> list[list[float]]:
    """Batch-embed document chunks in one .encode call (fast)."""
    return _local_embed_batch(texts, is_query=False)


# ── Compact chunk (1 teks per CV, hanya bagian yang relevan untuk scoring) ─────
def cv_to_compact_chunk(cv: dict, cv_id: str) -> dict | None:
    """Buat SATU teks padat per CV dari skills + experience key_skills + summary.

    Ini yang dipakai semantic_score — education/certifications sudah dihitung
    deterministik di scorer.py, tidak perlu masuk embedding.
    """
    name = (cv.get("personal_info") or {}).get("name", "") if isinstance(cv.get("personal_info"), dict) else ""

    sk = cv.get("skills") if isinstance(cv.get("skills"), dict) else {}
    all_skills = [s for s in ((sk.get("hard_skills") or []) + (sk.get("soft_skills") or [])) if isinstance(s, str)]

    exp_skills: list[str] = []
    roles: list[str] = []
    for e in cv.get("experience", []):
        if not isinstance(e, dict):
            continue
        role = e.get("role", "")
        if role:
            roles.append(role)
        exp_skills.extend(x for x in (e.get("key_skills") or []) if isinstance(x, str))

    # Deduplicate exp_skills agar teks tidak membengkak
    seen: set[str] = set()
    deduped: list[str] = []
    for s in exp_skills:
        if s.lower() not in seen:
            seen.add(s.lower())
            deduped.append(s)

    summary = (cv.get("summary") or "")[:300]  # max 300 char cukup

    parts: list[str] = []
    if summary:
        parts.append(summary)
    if roles:
        parts.append("Roles: " + ", ".join(roles))
    combined_skills = list(dict.fromkeys(all_skills + deduped))  # preserve order, dedupe
    if combined_skills:
        parts.append("Skills: " + ", ".join(combined_skills))

    text = " | ".join(parts).strip()
    if not text:
        return None

    return {
        "id":   f"{cv_id}::compact::0",
        "text": text,
        "meta": {"cv_id": cv_id, "name": name, "type": "compact"},
    }


# Legacy: masih dipakai test/debug
def cv_to_chunks(cv: dict, cv_id: str) -> list[dict]:
    chunk = cv_to_compact_chunk(cv, cv_id)
    return [chunk] if chunk else []


def index_cv(cv_id: str, cv: dict, provider: str) -> int:
    """Index 1 compact chunk dari satu CV. Return 1 jika berhasil, 0 jika kosong."""
    col = _get_collection(provider)
    chunk = cv_to_compact_chunk(cv, cv_id)
    if not chunk:
        return 0

    embedding = _local_embed(chunk["text"], is_query=False)
    with _index_lock:
        try:
            existing = col.get(where={"cv_id": cv_id})
            if existing["ids"]:
                col.delete(ids=existing["ids"])
        except Exception:
            pass
        col.add(
            ids=[chunk["id"]],
            documents=[chunk["text"]],
            embeddings=[embedding],
            metadatas=[chunk["meta"]],
        )
    return 1


def index_cvs_batch(cv_map: dict[str, dict], provider: str) -> int:
    """Batch-embed banyak CV sekaligus dalam satu model.encode() — jauh lebih cepat
    untuk bulk upload (100 CV ≈ waktu 1-2 CV jika di-encode satuan).

    cv_map: {cv_id: cv_dict, ...}
    Return: jumlah CV yang berhasil diindex.
    """
    col = _get_collection(provider)
    chunks = [(cv_id, cv_to_compact_chunk(cv, cv_id)) for cv_id, cv in cv_map.items()]
    chunks = [(cid, ch) for cid, ch in chunks if ch]
    if not chunks:
        return 0

    texts = [ch["text"] for _, ch in chunks]
    # Satu encode call untuk semua CV — bottleneck GPU/CPU inference dibayar sekali
    m = _get_local_model()
    pre = _e5_prefix(is_query=False)
    vecs = m.encode([pre + t for t in texts], normalize_embeddings=True, batch_size=64)

    with _index_lock:
        # Hapus entri lama untuk semua cv_id yang akan diindex
        for cv_id, _ in chunks:
            try:
                existing = col.get(where={"cv_id": cv_id})
                if existing["ids"]:
                    col.delete(ids=existing["ids"])
            except Exception:
                pass
        col.add(
            ids=[ch["id"] for _, ch in chunks],
            documents=texts,
            embeddings=[v.tolist() for v in vecs],
            metadatas=[ch["meta"] for _, ch in chunks],
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
