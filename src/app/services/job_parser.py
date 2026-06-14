"""AI job-description parser — free text masuk, kebutuhan terstruktur keluar.

FE cukup mengirim deskripsi lowongan bebas (gaya chat); LLM yang menebak
job title, required_skills, min pengalaman, dan pendidikan. Hasilnya dipakai
scorer seperti biasa, jadi tidak ada form skill/title lagi di sisi user.
"""
from __future__ import annotations

import json
import re
import time

from app.services.extractor import atomize_skills
from app.services.llm import get_llm

PARSE_SYSTEM = """You are an expert recruiter that reads a free-text job description \
(any language, any industry — never assume IT) and extracts the hiring requirements.

Output ONLY valid JSON with EXACTLY these fields:
{
  "title": string,                  // concise job title inferred from the text, in the text's language
  "required_skills": [string],      // 4-10 most important skills/competencies/tools, atomic items
  "min_years_experience": number|null, // ONLY if a minimum is explicitly stated or strongly implied (e.g. "senior" ~ 5); else null
  "required_education": string      // degree level/field ONLY if explicitly required, else ""
}

RULES:
- Never invent requirements that are not supported by the text.
- required_skills: prefer concrete, matchable terms ("Spring Boot", "negosiasi", "perawatan luka") over vague ones ("good attitude").
- Keep each skill short (1-4 words). No duplicates.
- If the text is too thin to infer a title, use a best-effort generic title (e.g. "Staf Administrasi")."""


def _coerce(data: dict) -> dict:
    """Normalisasi deterministik atas output LLM (tipe & batas wajar)."""
    title = data.get("title")
    title = title.strip() if isinstance(title, str) else ""

    skills = data.get("required_skills")
    if not isinstance(skills, list):
        skills = []
    skills = atomize_skills([s for s in skills if isinstance(s, str)])[:12]

    years = data.get("min_years_experience")
    if isinstance(years, str):
        m = re.search(r"\d+(?:\.\d+)?", years)
        years = float(m.group()) if m else None
    if isinstance(years, bool) or not isinstance(years, (int, float)):
        years = None
    elif years <= 0:
        years = None
    else:
        years = float(min(years, 40))

    edu = data.get("required_education")
    edu = edu.strip() if isinstance(edu, str) else ""

    return {
        "title": title,
        "required_skills": skills,
        "min_years_experience": years,
        "required_education": edu,
    }


def parse_job_description(description: str, llm_provider: str | None = None,
                          max_retries: int = 4) -> dict:
    """Satu panggilan LLM → kebutuhan lowongan terstruktur, dengan retry transient."""
    client, model = get_llm(llm_provider)
    delay = 5
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": PARSE_SYSTEM},
                    {"role": "user", "content": description},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            return _coerce(json.loads(resp.choices[0].message.content))
        except Exception as e:  # noqa: BLE001
            msg = str(e); last_err = e
            transient = ("429" in msg or "RESOURCE_EXHAUSTED" in msg
                         or "503" in msg or "UNAVAILABLE" in msg or "overloaded" in msg)
            if transient and attempt < max_retries - 1:
                m = re.search(r"(\d+(?:\.\d+)?)s", msg)
                wait = (float(m.group(1)) + 2) if m else delay
                time.sleep(min(wait, 60))
                delay = min(delay * 2, 60)
            else:
                raise
    raise last_err  # type: ignore[misc]
