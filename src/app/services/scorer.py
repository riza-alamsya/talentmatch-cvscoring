"""Scoring & reasoning engine.
Logika dari notebooks/mimo/cv_scoring_mimo_v1.ipynb (reasoning via MiMo)."""
from __future__ import annotations
import re
import time
from datetime import date
from typing import Optional

from app.core.config import settings
from app.services.embedder import embed_query, search_chunks
from app.services.llm import get_llm

TODAY_YM = date.today().year * 12 + date.today().month


# ── Helpers deterministik ─────────────────────────────────────────────────────
def candidate_skills(cv: dict) -> set[str]:
    s: set[str] = set()
    skills = cv.get("skills", {})
    for k in skills.get("hard_skills", []) + skills.get("soft_skills", []):
        s.add(k.lower())
    for e in cv.get("experience", []):
        for k in e.get("key_skills", []):
            s.add(k.lower())
    return s


def skill_match(required: list[str], cand: set[str]) -> tuple[float, list[str], list[str]]:
    matched, missing = [], []
    for req in required:
        r = req.lower()
        if any(r in c or c in r for c in cand):
            matched.append(req)
        else:
            missing.append(req)
    score = len(matched) / len(required) if required else 1.0
    return score, matched, missing


def _parse_ym(s: str) -> Optional[int]:
    if not s:
        return None
    parts = s.split("-")
    try:
        return int(parts[0]) * 12 + (int(parts[1]) if len(parts) > 1 else 1)
    except (ValueError, IndexError):
        return None


def total_years(cv: dict) -> Optional[float]:
    starts, ends = [], []
    for e in cv.get("experience", []):
        st = _parse_ym(e.get("start_date", ""))
        en = _parse_ym(e.get("end_date", ""))
        if en is None and e.get("is_current"):
            en = TODAY_YM
        if st is not None:
            starts.append(st)
        if en is not None:
            ends.append(en)
    if not starts or not ends:
        return None
    return round((max(ends) - min(starts)) / 12.0, 1)


def edu_ok(required_level: str, cv: dict) -> bool:
    if not required_level:
        return True
    rl = required_level.lower()
    for ed in cv.get("education", []):
        combined = (ed.get("degree", "") + " " + ed.get("field_of_study", "")).lower()
        if rl in combined:
            return True
    return False


def semantic_score(query_embedding: Optional[list[float]], cv_id: str,
                   embed_provider: str, top_k: int = 3) -> float:
    if query_embedding is None:
        return 0.0
    results = search_chunks(query_embedding=query_embedding, cv_id=cv_id,
                            provider=embed_provider, top_k=top_k)
    sims = [r["similarity"] for r in results]
    return round(sum(sims) / len(sims), 4) if sims else 0.0


# ── Verdict helper (dihitung di kode, bukan LLM) ─────────────────────────────
def _exp_verdict(years: Optional[float], min_years: Optional[float]) -> str:
    if not min_years:
        return "tidak ada syarat minimal"
    if years is None:
        return f"tidak diketahui (tahun tidak tercantum di CV; minimal {min_years} th)"
    if years >= min_years:
        return f"YA ({years} th >= {min_years} th)"
    return f"TIDAK ({years} th < {min_years} th)"


# ── Scoring ───────────────────────────────────────────────────────────────────
def score_candidate(job: dict, cv_id: str, cv: dict,
                    query_embedding: Optional[list[float]] = None,
                    embed_provider: str = "local") -> dict:
    cand = candidate_skills(cv)
    sk_score, matched, missing = skill_match(job.get("required_skills", []), cand)
    sem = semantic_score(query_embedding, cv_id, embed_provider)
    yrs = total_years(cv)

    weights = {
        "skills":   settings.WEIGHT_SKILLS,
        "semantic": settings.WEIGHT_SEMANTIC,
    }
    components = {"skills": sk_score, "semantic": sem}

    exp_breakdown = None
    if job.get("min_years_experience") is not None:
        min_y = job["min_years_experience"]
        exp_score = None if yrs is None else min(yrs / min_y, 1.0)
        meets = "unknown" if yrs is None else (yrs >= min_y)
        exp_breakdown = {
            "candidate_years": yrs,
            "required_years":  min_y,
            "meets_min":       meets,
            "score":           round(exp_score * 100) if exp_score is not None else None,
            "note":            "tanggal tidak tercantum di CV" if yrs is None else None,
        }
        if exp_score is not None:
            components["years"] = exp_score
            weights["years"] = settings.WEIGHT_YEARS

    edu_breakdown = None
    if job.get("required_education"):
        ok = edu_ok(job["required_education"], cv)
        edu_breakdown = {"required": job["required_education"], "met": ok}
        components["education"] = 1.0 if ok else 0.0
        weights["education"] = settings.WEIGHT_EDUCATION

    total_w = sum(weights[k] for k in components)
    final = sum(weights[k] * components[k] for k in components) / total_w

    return {
        "cv_id":   cv_id,
        "name":    cv.get("personal_info", {}).get("name", cv_id),
        "final":   final,
        "matched": matched,
        "missing": missing,
        "years":   yrs,
        "breakdown": {
            "skills":     {"score": round(sk_score * 100), "matched": matched, "missing": missing},
            "semantic":   {"score": round(sem * 100)},
            "experience": exp_breakdown,
            "education":  edu_breakdown,
        },
    }


def rank_candidates(job: dict, cvs: dict[str, dict], embed_provider: str = "local") -> list[dict]:
    # Embed job description SEKALI, reuse untuk semua CV (hemat panggilan API)
    desc = (job.get("description", "") or "").strip()
    query_embedding = embed_query(desc, embed_provider) if desc else None
    rows = [score_candidate(job, cv_id, cv, query_embedding, embed_provider)
            for cv_id, cv in cvs.items()]
    rows.sort(key=lambda r: r["final"], reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i
        r["final_score"] = round(r["final"] * 100)
    return rows


# ── LLM Reasoning ─────────────────────────────────────────────────────────────
REASON_SYSTEM = (
    "Your task is to explain the reasoning behind the candidate's match score for the job. "
    "MUST only use facts in the DATA section. ABSOLUTELY FORBIDDEN to invent skills, companies, "
    "projects, numbers, or claims not in DATA.\n"
    "STRICT RULES:\n"
    "- STRENGTHS: mention skills from MATCHED SKILLS + real experience from HISTORY.\n"
    "- WEAKNESSES: ONLY mention items listed in MISSING SKILLS, and/or if "
    "MEETS MIN EXPERIENCE = NO. DO NOT mention skills in MATCHED SKILLS as weaknesses.\n"
    "- EXPERIENCE: use the 'MEETS MIN EXPERIENCE' field as-is. DO NOT calculate yourself.\n"
    "- If MISSING SKILLS '-' and MEETS MIN EXPERIENCE = YES: state all requirements are met.\n"
    "- If experience 'unknown': write that work duration is not listed in the CV."
)

# Output-language directive — appended to system prompt so reason text
# matches language selected in FE (id | en | ms | zh).
_LANG_DIRECTIVE = {
    "id": "Answer in 2-3 concise sentences in Indonesian. Write ONLY in Indonesian.",
    "en": "Answer in 2-3 concise sentences. Write your reply in English ONLY.",
    "ms": "Answer in 2-3 concise sentences in Malay. Write ONLY in Malay.",
    "zh": "Answer in 2-3 concise sentences in Simplified Chinese. Write ONLY in Simplified Chinese.",
}


def _lang_directive(lang: str | None) -> str:
    """Resolve the output-language instruction; fall back to the configured default."""
    code = (lang or settings.DEFAULT_LANG or "en").lower()
    return _LANG_DIRECTIVE.get(code, _LANG_DIRECTIVE["en"])


def generate_reason(result: dict, cv: dict, job: dict, llm_provider: str | None = None,
                    lang: str | None = None) -> str:
    # Ambil data dari hasil scoring yang sudah dihitung
    matched = result.get("matched", [])
    missing = result.get("missing", [])
    final_score = result.get("final_score", 0)
    exp_verdict = _exp_verdict(result["years"], job.get("min_years_experience"))
    
    # 1. Kalimat Kekuatan
    if matched:
        strength = f"Kandidat memiliki skill yang sesuai: {', '.join(matched[:3])}."
    else:
        strength = "Tidak ada skill yang spesifik cocok secara langsung."
        
    # 2. Kalimat Kekurangan
    if missing:
        weakness = f"Namun, kandidat belum memiliki skill: {', '.join(missing[:3])}."
    else:
        weakness = "Kandidat memenuhi semua persyaratan skill yang dibutuhkan."
        
    # 3. Kalimat Pengalaman
    exp_note = f"Pengalaman: {exp_verdict}."
    
    # Gabungkan (sesuai permintaan 2-3 kalimat)
    return f"{strength} {weakness} {exp_note} Skor akhir: {final_score}/100."