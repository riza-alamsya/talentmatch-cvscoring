"""Scoring endpoints — rank candidates + optional LLM reasoning."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from app.core.config import settings
from app.models.schemas import JobRequest, ScoreResponse, CandidateResult, ScoreBreakdown, SkillBreakdown, SemanticBreakdown, ExperienceBreakdown, EducationBreakdown
from app.services.embedder import ensure_indexed
from app.services.extractor import list_processed_cvs, load_processed_cv
from app.services.job_parser import parse_job_description
from app.services.scorer import generate_reason, rank_candidates

router = APIRouter(prefix="/score", tags=["Scoring"])


def _autofill_job(job: JobRequest, llm: str) -> JobRequest:
    """Free-text mode: kalau FE hanya mengirim description (tanpa skills),
    AI mengekstrak title/skills/min pengalaman/pendidikan dari teks itu.
    Gagal parse → lanjut dengan scoring semantik saja (jangan tumbangkan request)."""
    if job.required_skills or not job.description.strip():
        return job
    try:
        parsed = parse_job_description(job.description, llm)
    except Exception:  # noqa: BLE001
        return job
    return JobRequest(
        title=job.title or parsed["title"],
        required_skills=parsed["required_skills"],
        min_years_experience=(job.min_years_experience
                              if job.min_years_experience is not None
                              else parsed["min_years_experience"]),
        required_education=job.required_education or parsed["required_education"],
        description=job.description,
    )


# `def` (bukan async) — scoring memanggil embedding API + LLM yang blocking;
# threadpool anyio yang menanganinya supaya event loop tidak terblok.
@router.post(
    "/",
    response_model=ScoreResponse,
    summary="Rank semua kandidat terhadap job description",
)
def score_all(
    job: JobRequest,
    with_reason: bool = Query(False, description="Generate alasan LLM per kandidat (lebih lambat)"),
    cv_ids: Optional[str] = Query(None, description="Comma-separated cv_ids. Kosong = semua CV."),
    llm: Optional[str] = Query(None, description="LLM provider: mimo"),
    embed: Optional[str] = Query(None, description="Embedding provider: local"),
    lang: Optional[str] = Query(None, description="Reason output language: en | id | ms | zh"),
):
    """
    Berikan job description → dapatkan ranking kandidat.

    - **cv_ids**: filter spesifik cv_id (comma-separated). Kalau kosong, score semua CV.
    - **llm / embed**: pilih provider (default dari config). Ganti embed → CV otomatis di-index ulang (lazy).
    - **lang**: bahasa narasi alasan (ikut pilihan FE; default dari config).
    - **with_reason=true**: tambahkan narasi alasan per kandidat (LLM, lebih lambat)
    """
    llm = llm or settings.DEFAULT_LLM
    embed = embed or settings.DEFAULT_EMBED
    lang = lang or settings.DEFAULT_LANG

    # free-text mode: ekstrak kebutuhan dari description kalau skills kosong
    job = _autofill_job(job, llm)

    # filter cv_ids kalau ada, otherwise semua
    if cv_ids:
        requested = [cid.strip() for cid in cv_ids.split(",") if cid.strip()]
    else:
        requested = list_processed_cvs()

    if not requested:
        raise HTTPException(status_code=404, detail="Belum ada CV yang diproses.")

    cvs = {}
    for cv_id in requested:
        data = load_processed_cv(cv_id)
        if data:
            cvs[cv_id] = data

    if not cvs:
        raise HTTPException(status_code=404, detail="Tidak ada CV valid ditemukan.")

    # pastikan tiap CV ter-index di collection embedding terpilih (lazy re-index)
    for cid, data in cvs.items():
        ensure_indexed(cid, data, embed)

    job_dict = job.model_dump()
    rows = rank_candidates(job_dict, cvs, embed)

    results = []
    for r in rows:
        bd = r["breakdown"]

        reason = None
        if with_reason:
            reason = generate_reason(r, cvs[r["cv_id"]], job_dict, llm, lang)

        exp_bd = None
        if bd.get("experience"):
            exp_bd = ExperienceBreakdown(**bd["experience"])

        edu_bd = None
        if bd.get("education"):
            edu_bd = EducationBreakdown(**bd["education"])

        results.append(CandidateResult(
            rank=r["rank"],
            cv_id=r["cv_id"],
            name=r["name"],
            final_score=r["final_score"],
            breakdown=ScoreBreakdown(
                skills=SkillBreakdown(**bd["skills"]),
                semantic=SemanticBreakdown(**bd["semantic"]),
                experience=exp_bd,
                education=edu_bd,
            ),
            reason=reason,
        ))

    return ScoreResponse(
        job=job,
        weights={
            "skills":    0.40,
            "semantic":  0.35,
            "years":     0.15,
            "education": 0.10,
        },
        results=results,
    )


@router.post(
    "/{cv_id}/reason",
    summary="Generate alasan LLM untuk satu kandidat",
)
def reason_one(
    cv_id: str,
    job: JobRequest,
    llm: Optional[str] = Query(None, description="LLM provider: mimo"),
    embed: Optional[str] = Query(None, description="Embedding provider: local"),
    lang: Optional[str] = Query(None, description="Reason output language: en | id | ms | zh"),
):
    """Generate narasi alasan untuk 1 kandidat saja terhadap job description."""
    cv = load_processed_cv(cv_id)
    if cv is None:
        raise HTTPException(status_code=404, detail=f"CV '{cv_id}' tidak ditemukan.")

    llm = llm or settings.DEFAULT_LLM
    embed = embed or settings.DEFAULT_EMBED
    lang = lang or settings.DEFAULT_LANG

    # FE normalnya mengirim job hasil ekstraksi dari /score, tapi tetap dukung free-text
    job = _autofill_job(job, llm)

    job_dict = job.model_dump()
    ensure_indexed(cv_id, cv, embed)
    rows = rank_candidates(job_dict, {cv_id: cv}, embed)
    reason = generate_reason(rows[0], cv, job_dict, llm, lang)
    return {"cv_id": cv_id, "name": rows[0]["name"], "reason": reason}
