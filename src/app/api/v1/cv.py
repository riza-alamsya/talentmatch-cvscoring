"""CV endpoints — upload, extract, list, delete."""
from __future__ import annotations
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.core.config import settings
from app.models.schemas import CVData, ExtractResponse
from app.services.embedder import index_cv, index_cvs_batch
from app.services.extractor import (
    extract_cv_from_pdf,
    list_processed_cvs,
    load_processed_cv,
    save_processed_cv,
)

router = APIRouter(prefix="/cv", tags=["CV"])


# NOTE: heavy endpoint is intentionally `def` (not `async def`) — its work is blocking
# (pymupdf, LLM/embedding calls that can take tens of seconds). FastAPI runs
# `def` in threadpool anyio, so event loop stays free to serve other requests
# and multiple extractions can run in parallel.
@router.post("/upload", response_model=ExtractResponse, summary="Upload PDF & extract CV")
def upload_cv(
    file: UploadFile = File(...),
    llm: str | None = Query(None, description="LLM provider: mimo"),
    embed: str | None = Query(None, description="Embedding provider: local"),
):
    """
    Upload PDF file → extract CV structure via LLM (choose provider) → save + index.
    - **embed**: embedding provider for indexing (default from config)
    """
    llm = llm or settings.DEFAULT_LLM
    embed = embed or settings.DEFAULT_EMBED

    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted.")

    cv_id = Path(file.filename).stem
    pdf_dest = settings.CV_DIR / file.filename

    # save PDF
    with pdf_dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        data = extract_cv_from_pdf(pdf_dest, llm)
    except ValueError as e:
        # Scanned/image PDF — mark, don't send to LLM
        return ExtractResponse(cv_id=cv_id, status="skipped_empty", message=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")

    save_processed_cv(cv_id, data)
    n_chunks = index_cv(cv_id, data, embed)

    data["cv_id"] = cv_id
    return ExtractResponse(
        cv_id=cv_id,
        status="ok",
        message=f"Successfully extracted ({llm}). {n_chunks} chunks indexed to Chroma ({embed}).",
        data=CVData(**data),
    )


class BulkUploadResult(BaseModel):
    total: int
    succeeded: int
    failed: int
    results: list[dict[str, Any]]


@router.post("/upload-batch", response_model=BulkUploadResult,
             summary="Upload many PDFs at once (parallel extract + batch embed)")
def upload_cv_batch(
    files: list[UploadFile] = File(...),
    llm: str | None = Query(None),
    embed: str | None = Query(None),
):
    """
    Upload multiple PDFs → parallel extract via LLM → batch index embeddings (1 encode call).
    Much faster than calling /upload one-by-one for 10+ CVs.
    """
    llm_p = llm or settings.DEFAULT_LLM
    embed_p = embed or settings.DEFAULT_EMBED

    # Save all PDFs first
    saved: list[tuple[str, Path]] = []
    for f in files:
        if not f.filename or not f.filename.endswith(".pdf"):
            continue
        dest = settings.CV_DIR / f.filename
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append((Path(f.filename).stem, dest))

    # Parallel extract — each file blocking (LLM call), run in threadpool
    extracted: dict[str, dict] = {}
    results: list[dict[str, Any]] = []

    def _extract_one(cv_id: str, pdf_path: Path):
        try:
            data = extract_cv_from_pdf(pdf_path, llm_p)
            save_processed_cv(cv_id, data)
            return cv_id, data, None
        except ValueError as e:
            return cv_id, None, f"skipped: {e}"
        except Exception as e:
            return cv_id, None, f"error: {e}"

    workers = min(len(saved), settings.MAX_CONCURRENT_EXTRACT)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_extract_one, cid, path): cid for cid, path in saved}
        for fut in as_completed(futs):
            cv_id, data, err = fut.result()
            if err:
                results.append({"cv_id": cv_id, "status": "failed", "message": err})
            else:
                extracted[cv_id] = data
                results.append({"cv_id": cv_id, "status": "ok"})

    # Batch embed all successful CVs in ONE encode call
    if extracted:
        index_cvs_batch(extracted, embed_p)

    succeeded = len(extracted)
    return BulkUploadResult(
        total=len(saved),
        succeeded=succeeded,
        failed=len(saved) - succeeded,
        results=results,
    )


@router.get("/", response_model=list[str], summary="List all processed CVs")
def list_cvs():
    return list_processed_cvs()


class ExtractByPathRequest(BaseModel):
    path: str                      # absolute path to PDF file in shared storage
    filename: str | None = None    # original filename (to determine cv_id)


@router.post("/extract", response_model=ExtractResponse, summary="Extract CV from existing file (shared storage)")
def extract_cv_by_path(
    req: ExtractByPathRequest,
    llm: str | None = Query(None, description="LLM provider: mimo"),
    embed: str | None = Query(None, description="Embedding provider: local"),
):
    """Called by Java after it saves PDF to shared dir. Python does NOT copy
    the file — just reads from `path` then extracts + indexes."""
    llm = llm or settings.DEFAULT_LLM
    embed = embed or settings.DEFAULT_EMBED

    pdf_path = Path(req.path)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {req.path}")

    cv_id = Path(req.filename or pdf_path.name).stem
    try:
        data = extract_cv_from_pdf(pdf_path, llm)
    except ValueError as e:
        return ExtractResponse(cv_id=cv_id, status="skipped_empty", message=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")

    save_processed_cv(cv_id, data)
    n_chunks = index_cv(cv_id, data, embed)

    data["cv_id"] = cv_id
    return ExtractResponse(
        cv_id=cv_id,
        status="ok",
        message=f"Successfully extracted ({llm}). {n_chunks} chunks indexed ({embed}).",
        data=CVData(**data),
    )


@router.get("/{cv_id}/file", summary="Get original CV PDF file (for display)")
def get_cv_file(cv_id: str):
    """Return original PDF so FE can display it (inline) for comparison."""
    pdfs = [p for p in settings.CV_DIR.glob(f"{cv_id}.*") if p.suffix.lower() == ".pdf"]
    if not pdfs:
        raise HTTPException(status_code=404, detail="Original PDF not found.")
    return FileResponse(
        pdfs[0],
        media_type="application/pdf",
        content_disposition_type="inline",  # display in iframe, not download
    )


@router.get("/{cv_id}", response_model=CVData, summary="Get CV details")
def get_cv(cv_id: str):
    data = load_processed_cv(cv_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"CV '{cv_id}' not found.")
    data["cv_id"] = cv_id
    return CVData(**data)


@router.delete("/{cv_id}", summary="Delete CV")
def delete_cv(cv_id: str):
    json_path = settings.PROCESSED_DIR / f"{cv_id}.json"
    pdf_paths = list(settings.CV_DIR.glob(f"{cv_id}.*"))

    if not json_path.exists():
        raise HTTPException(status_code=404, detail=f"CV '{cv_id}' not found.")

    json_path.unlink()
    for p in pdf_paths:
        p.unlink(missing_ok=True)

    return {"status": "deleted", "cv_id": cv_id}
