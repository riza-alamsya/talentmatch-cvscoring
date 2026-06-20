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
    """Free-text mode: if FE sends only description (no skills),
    AI extracts title/skills/min experience/education from the text.
    Parse failure → continue with semantic scoring only (don't fail the request)."""
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


# `def` (not async) — scoring calls blocking embedding API + LLM;
# anyio threadpool handles it so event loop doesn't block.
@router.post(
    "/",
    response_model=ScoreResponse,
    summary="Rank all candidates against job description",
)
def score_all(
    job: JobRequest,
    with_reason: bool = Query(False, description="Generate LLM reasoning per candidate (slower)"),
    cv_ids: Optional[str] = Query(None, description="Comma-separated cv_ids. Empty = all CVs."),
    llm: Optional[str] = Query(None, description="LLM provider: mimo"),
    embed: Optional[str] = Query(None, description="Embedding provider: local"),
    lang: Optional[str] = Query(None, description="Reason output language: en | id | ms | zh"),
):
    """
    Provide job description → get candidate ranking.

    - **cv_ids**: filter specific cv_id (comma-separated). Empty = score all CVs.
    - **llm / embed**: choose provider (default from config). Change embed → CVs auto re-indexed (lazy).
    - **lang**: reasoning narrative language (from FE choice; default from config).
    - **with_reason=true**: add reasoning narrative per candidate (LLM, slower)
    """
    llm = llm or settings.DEFAULT_LLM
    embed = embed or settings.DEFAULT_EMBED
    lang = lang or settings.DEFAULT_LANG

    # free-text mode: extract requirements from description if skills empty
    job = _autofill_job(job, llm)

    # filter cv_ids if provided, otherwise all
    if cv_ids:
        requested = [cid.strip() for cid in cv_ids.split(",") if cid.strip()]
    else:
        requested = list_processed_cvs()

    if not requested:
        raise HTTPException(status_code=404, detail="No CVs have been processed yet.")

    cvs = {}
    for cv_id in requested:
        data = load_processed_cv(cv_id)
        if data:
            cvs[cv_id] = data

    if not cvs:
        raise HTTPException(status_code=404, detail="Tidak ada CV valid ditemukan.")

    # ensure each CV is indexed in chosen embedding collection (lazy re-index)
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
