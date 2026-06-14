"""CV endpoints — upload, extract, list, delete."""
from __future__ import annotations
import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.core.config import settings
from app.models.schemas import CVData, ExtractResponse
from app.services.embedder import index_cv
from app.services.extractor import (
    extract_cv_from_pdf,
    list_processed_cvs,
    load_processed_cv,
    save_processed_cv,
)

router = APIRouter(prefix="/cv", tags=["CV"])


# NOTE: endpoint berat sengaja `def` (bukan `async def`) — kerjanya blocking
# (pymupdf, panggilan LLM/embedding yang bisa puluhan detik). FastAPI menjalankan
# `def` di threadpool anyio, jadi event loop tetap bebas melayani request lain
# dan beberapa ekstraksi bisa jalan paralel.
@router.post("/upload", response_model=ExtractResponse, summary="Upload PDF & ekstrak CV")
def upload_cv(
    file: UploadFile = File(...),
    llm: str | None = Query(None, description="LLM provider: gemini | mimo"),
    embed: str | None = Query(None, description="Embedding provider: gemini | local"),
):
    """
    Upload file PDF → ekstrak struktur CV via LLM (pilih provider) → simpan + index.

    - **llm**: provider ekstraksi (default dari config)
    - **embed**: provider embedding untuk index (default dari config)
    """
    llm = llm or settings.DEFAULT_LLM
    embed = embed or settings.DEFAULT_EMBED

    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Hanya file PDF yang diterima.")

    cv_id = Path(file.filename).stem
    pdf_dest = settings.CV_DIR / file.filename

    # simpan PDF
    with pdf_dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        data = extract_cv_from_pdf(pdf_dest, llm)
    except ValueError as e:
        # PDF scan/gambar — tandai, jangan dikirim ke LLM
        return ExtractResponse(cv_id=cv_id, status="skipped_empty", message=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ekstraksi gagal: {e}")

    save_processed_cv(cv_id, data)
    n_chunks = index_cv(cv_id, data, embed)

    data["cv_id"] = cv_id
    return ExtractResponse(
        cv_id=cv_id,
        status="ok",
        message=f"Berhasil diekstrak ({llm}). {n_chunks} chunk diindex ke Chroma ({embed}).",
        data=CVData(**data),
    )


@router.get("/", response_model=list[str], summary="Daftar semua CV yang sudah diproses")
def list_cvs():
    return list_processed_cvs()


class ExtractByPathRequest(BaseModel):
    path: str                      # absolute path file PDF di shared storage
    filename: str | None = None    # nama asli (buat nentuin cv_id)


@router.post("/extract", response_model=ExtractResponse, summary="Ekstrak CV dari file yang sudah ada (shared storage)")
def extract_cv_by_path(
    req: ExtractByPathRequest,
    llm: str | None = Query(None, description="LLM provider: gemini | mimo"),
    embed: str | None = Query(None, description="Embedding provider: gemini | local"),
):
    """Dipanggil Java setelah ia menyimpan PDF ke shared dir. Python TIDAK menyalin
    file — cukup baca dari `path` lalu ekstrak + index."""
    llm = llm or settings.DEFAULT_LLM
    embed = embed or settings.DEFAULT_EMBED

    pdf_path = Path(req.path)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"File tidak ditemukan: {req.path}")

    cv_id = Path(req.filename or pdf_path.name).stem
    try:
        data = extract_cv_from_pdf(pdf_path, llm)
    except ValueError as e:
        return ExtractResponse(cv_id=cv_id, status="skipped_empty", message=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ekstraksi gagal: {e}")

    save_processed_cv(cv_id, data)
    n_chunks = index_cv(cv_id, data, embed)

    data["cv_id"] = cv_id
    return ExtractResponse(
        cv_id=cv_id,
        status="ok",
        message=f"Berhasil diekstrak ({llm}). {n_chunks} chunk diindex ({embed}).",
        data=CVData(**data),
    )


@router.get("/{cv_id}/file", summary="Ambil file PDF asli CV (untuk ditampilkan)")
def get_cv_file(cv_id: str):
    """Kembalikan PDF asli supaya FE bisa menampilkannya (inline) buat dibandingkan."""
    pdfs = [p for p in settings.CV_DIR.glob(f"{cv_id}.*") if p.suffix.lower() == ".pdf"]
    if not pdfs:
        raise HTTPException(status_code=404, detail="PDF asli tidak ditemukan.")
    return FileResponse(
        pdfs[0],
        media_type="application/pdf",
        content_disposition_type="inline",  # tampil di iframe, bukan download
    )


@router.get("/{cv_id}", response_model=CVData, summary="Ambil detail CV")
def get_cv(cv_id: str):
    data = load_processed_cv(cv_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"CV '{cv_id}' tidak ditemukan.")
    data["cv_id"] = cv_id
    return CVData(**data)


@router.delete("/{cv_id}", summary="Hapus CV")
def delete_cv(cv_id: str):
    json_path = settings.PROCESSED_DIR / f"{cv_id}.json"
    pdf_paths = list(settings.CV_DIR.glob(f"{cv_id}.*"))

    if not json_path.exists():
        raise HTTPException(status_code=404, detail=f"CV '{cv_id}' tidak ditemukan.")

    json_path.unlink()
    for p in pdf_paths:
        p.unlink(missing_ok=True)

    return {"status": "deleted", "cv_id": cv_id}
