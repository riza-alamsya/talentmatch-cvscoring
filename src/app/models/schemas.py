"""Pydantic schemas — request & response for all endpoints."""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, model_validator


# ── Extraction ────────────────────────────────────────────────────────────────
class PersonalInfo(BaseModel):
    name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    links: list[str] = []


class Skills(BaseModel):
    hard_skills: list[str] = []
    soft_skills: list[str] = []


class Experience(BaseModel):
    company: str = ""
    role: str = ""
    location: str = ""
    start_date: str = ""
    end_date: str = ""
    is_current: bool = False
    summary: str = ""
    key_skills: list[str] = []


class Education(BaseModel):
    institution: str = ""
    degree: str = ""
    field_of_study: str = ""
    start_year: str = ""
    end_year: str = ""


class Certification(BaseModel):
    name: str = ""
    issuer: str = ""
    year: str = ""

    @model_validator(mode="before")
    @classmethod
    def _coerce_str(cls, v):
        # LLM sometimes returns a bare string instead of an object.
        return {"name": v} if isinstance(v, str) else v


class Language(BaseModel):
    language: str = ""
    proficiency: str = ""

    @model_validator(mode="before")
    @classmethod
    def _coerce_str(cls, v):
        # LLM sometimes returns e.g. "English" instead of {"language": "English"}.
        return {"language": v} if isinstance(v, str) else v


class CVData(BaseModel):
    cv_id: str
    personal_info: PersonalInfo = PersonalInfo()
    summary: str = ""
    skills: Skills = Skills()
    experience: list[Experience] = []
    education: list[Education] = []
    certifications: list[Certification] = []
    achievements: list[str] = []
    languages: list[Language] = []


class ExtractResponse(BaseModel):
    cv_id: str
    status: str          # "ok" | "skipped_empty" | "error"
    message: str = ""
    data: Optional[CVData] = None


# ── Scoring ───────────────────────────────────────────────────────────────────
class JobRequest(BaseModel):
    # Semua field opsional kecuali description: FE kini cukup mengirim teks bebas,
    # title/skills/min pengalaman diekstrak otomatis oleh AI (job_parser).
    title: str = ""
    required_skills: list[str] = []
    min_years_experience: Optional[float] = None
    required_education: str = ""
    description: str = ""


class SkillBreakdown(BaseModel):
    score: int
    matched: list[str] = []
    missing: list[str] = []


class SemanticBreakdown(BaseModel):
    score: int


class ExperienceBreakdown(BaseModel):
    candidate_years: Optional[float]
    required_years: Optional[float]
    meets_min: object          # True | False | "unknown"
    score: Optional[int] = None
    note: Optional[str] = None


class EducationBreakdown(BaseModel):
    required: str
    met: bool


class ScoreBreakdown(BaseModel):
    skills: SkillBreakdown
    semantic: SemanticBreakdown
    experience: Optional[ExperienceBreakdown] = None
    education: Optional[EducationBreakdown] = None


class CandidateResult(BaseModel):
    rank: int
    cv_id: str
    name: str
    final_score: int
    breakdown: ScoreBreakdown
    reason: Optional[str] = None


class ScoreResponse(BaseModel):
    job: JobRequest
    weights: dict[str, float]
    results: list[CandidateResult]
