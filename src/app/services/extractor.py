"""Ekstraksi teks PDF + structured extraction via MiMo (Xiaomi).
Semua logika dari notebooks/mimo/cv_semantic_chunking_mimo_v1.ipynb dipindah ke sini."""
from __future__ import annotations
import json
import os
import re
import threading
import time
import unicodedata
from pathlib import Path

import fitz

from app.core.config import settings
from app.services.llm import get_llm

# Batasi ekstraksi paralel: request masuk lewat threadpool (endpoint `def`),
# semaphore ini yang menjaga jumlah panggilan LLM serentak tetap kecil.
_extract_sem = threading.BoundedSemaphore(settings.MAX_CONCURRENT_EXTRACT)

# ── CV JSON Schema (sama persis dengan notebook) ──────────────────────────────
CV_SCHEMA = {
    "type": "object",
    "properties": {
        "personal_info": {
            "type": "object",
            "properties": {
                "name":     {"type": "string"},
                "email":    {"type": "string"},
                "phone":    {"type": "string"},
                "location": {"type": "string"},
                "links":    {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name", "email", "phone", "location", "links"],
        },
        "summary": {"type": "string"},
        "skills": {
            "type": "object",
            "properties": {
                "hard_skills": {"type": "array", "items": {"type": "string"}},
                "soft_skills": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["hard_skills", "soft_skills"],
        },
        "experience": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "company":    {"type": "string"},
                    "role":       {"type": "string"},
                    "location":   {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date":   {"type": "string"},
                    "is_current": {"type": "boolean"},
                    "summary":    {"type": "string"},
                    "key_skills": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["company", "role", "location", "start_date",
                             "end_date", "is_current", "summary", "key_skills"],
            },
        },
        "education": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "institution":    {"type": "string"},
                    "degree":         {"type": "string"},
                    "field_of_study": {"type": "string"},
                    "start_year":     {"type": "string"},
                    "end_year":       {"type": "string"},
                },
                "required": ["institution", "degree", "field_of_study", "start_year", "end_year"],
            },
        },
        "certifications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name":   {"type": "string"},
                    "issuer": {"type": "string"},
                    "year":   {"type": "string"},
                },
                "required": ["name", "issuer", "year"],
            },
        },
        "achievements": {"type": "array", "items": {"type": "string"}},
        "languages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "language":    {"type": "string"},
                    "proficiency": {"type": "string"},
                },
                "required": ["language", "proficiency"],
            },
        },
    },
    "required": ["personal_info", "summary", "skills", "experience",
                 "education", "certifications", "achievements", "languages"],
}

SYSTEM_PROMPT = """You are an expert CV/resume parser that works for ANY job sector \
(healthcare, finance, sales, education, engineering, hospitality, trades, etc.) - never assume an IT/technical role.

Extract the CV into the provided JSON schema. Output ONLY valid JSON, no explanation.

FIELD NAMES — use EXACTLY these, no variations:
- "personal_info" (NOT personal_information / personalInfo / contact)
- "summary" (NOT profile_summary / about / objective)
- "experience" (NOT work_experience / employment / work_history)
- "education", "skills", "certifications", "achievements", "languages"

GENERAL
- Never fabricate facts not in the CV (names, employers, dates, institutions, certifications). If absent, use "" or [].
- Preserve the original language of the content (Indonesian stays Indonesian).
- SUMMARIZE responsibilities in your own words for `summary` and each `experience[].summary`. Do NOT copy text verbatim.

SKILLS
- hard_skills = job-specific professional competencies. soft_skills = interpersonal/transferable skills.
- soft_skills: include ONLY if the CV explicitly lists them (e.g. under a "Soft Skills"/"Competencies" heading). If not, return [].
- List skills as separate items where possible (code will further normalize them afterwards).
- experience[].key_skills = skills/tools/competencies used in that specific role.

ACHIEVEMENTS
- Extract concrete, quantifiable accomplishments from anywhere in the CV, including experience bullets.

DATES (experience start_date / end_date)
- CRITICAL: only output a date EXPLICITLY written in the CV. If absent, set to "". NEVER invent or back-calculate.
- WHEN a date IS present, normalize to "YYYY-MM". Map Indonesian months (Januari=01 … Desember=12).
- Expand 2-digit years: "Maret 24" -> "2024-03".
- "Present"/"Sekarang"/"Now" => is_current=true, end_date="".
- If only a year is written, use "YYYY".

OTHER
- certifications includes professional licenses (nursing, CPA, bar, safety certs, etc.)."""

MIN_TEXT_CHARS = 100  # batas teks minimum; di bawah ini = kemungkinan PDF scan/gambar


# ── Helpers ───────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("—", " - ").replace("–", " - ").replace("•", " - ")
    text = text.replace(" ", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_clean_text(pdf_path: str | Path) -> str:
    doc = fitz.open(str(pdf_path))
    full_text = []
    for page in doc:
        data = page.get_text("dict")
        page_text = []
        for block in data["blocks"]:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                line_text = " ".join(s["text"] for s in line["spans"]).strip()
                if line_text:
                    page_text.append(line_text)
        full_text.append(" ".join(page_text))
    doc.close()
    return clean_text(" ".join(full_text))


def atomize_skills(items: list[str]) -> list[str]:
    """Normalisasi deterministik: pecah 'Kategori: a, b | c' jadi item atomic.
    TIDAK memecah di '/' supaya 'CI/CD', 'OpenAPI/Swagger' tetap utuh."""
    out = []
    for s in items:
        if not isinstance(s, str):
            continue
        if ":" in s:
            s = s.split(":", 1)[1]
        for part in re.split(r"[,|]", s):
            p = part.strip(" .;-•")
            if p:
                out.append(p)
    seen, result = set(), []
    for x in out:
        if x.lower() not in seen:
            seen.add(x.lower())
            result.append(x)
    return result


def normalize_cv(data: dict) -> dict:
    """Pulihkan field name kalau MiMo memakai nama berbeda dari schema
    (json_object mode tidak meng-enforce schema seperti Ollama `format=`)."""
    # personal_info
    if "personal_info" not in data:
        for alt in ["personal_information", "personalInfo", "contact", "personal"]:
            if alt in data:
                data["personal_info"] = data.pop(alt)
                break
    pi = data.get("personal_info", {})
    if isinstance(pi, dict) and "links" not in pi:
        for alt in ["linkedin", "github", "portfolio", "website"]:
            if alt in pi:
                pi["links"] = [pi.pop(alt)]
                break
        else:
            pi["links"] = []

    # summary
    if "summary" not in data:
        for alt in ["profile_summary", "profileSummary", "about", "objective", "profile"]:
            if alt in data:
                data["summary"] = data.pop(alt)
                break

    # experience
    if "experience" not in data:
        for alt in ["work_experience", "workExperience", "employment", "work_history"]:
            if alt in data:
                data["experience"] = data.pop(alt)
                break

    # pastikan field wajib ada
    for field in ["experience", "education", "certifications", "achievements", "languages"]:
        data.setdefault(field, [])
    data.setdefault("summary", "")

    # ── Type guards: MiMo kadang balikin tipe yang salah (list vs dict vs str) ──
    # skills harus dict {hard_skills:[], soft_skills:[]}
    if not isinstance(data.get("skills"), dict):
        data["skills"] = {"hard_skills": [], "soft_skills": []}
    for sk in ("hard_skills", "soft_skills"):
        if not isinstance(data["skills"].get(sk), list):
            data["skills"][sk] = []

    # personal_info harus dict
    if not isinstance(data.get("personal_info"), dict):
        data["personal_info"] = {"name": "", "email": "", "phone": "",
                                 "location": "", "links": []}

    # list-of-object fields: pastikan list
    for field in ("experience", "education", "certifications", "languages", "achievements"):
        if not isinstance(data.get(field), list):
            data[field] = []

    # experience/education: buang item yang bukan dict
    for field in ("experience", "education"):
        data[field] = [x for x in data[field] if isinstance(x, dict)]

    # languages/certifications: string -> object, buang yang bukan str/dict
    data["languages"] = [
        {"language": x} if isinstance(x, str) else x
        for x in data["languages"] if isinstance(x, (str, dict))
    ]
    data["certifications"] = [
        {"name": x} if isinstance(x, str) else x
        for x in data["certifications"] if isinstance(x, (str, dict))
    ]

    # achievements: hanya string
    data["achievements"] = [x for x in data["achievements"] if isinstance(x, str)]

    return data


def _year_in_source(date_str: str, text: str) -> bool:
    return (not date_str) or (date_str[:4] in text)


def validate_dates(data: dict, source_text: str) -> dict:
    """Anti-halusinasi tanggal: kosongin tanggal yang tahunnya tidak ada di teks CV."""
    for e in data.get("experience", []):
        for k in ("start_date", "end_date"):
            if e.get(k) and not _year_in_source(e[k], source_text):
                e[k] = ""
        if not e.get("start_date") and not e.get("end_date"):
            e["is_current"] = False
    return data


# ── Main service ──────────────────────────────────────────────────────────────
def extract_cv(cv_text: str, llm_provider: str | None = None) -> dict:
    """Satu panggilan LLM -> JSON terstruktur + normalisasi deterministik."""
    client, model = get_llm(llm_provider)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": cv_text},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    data = normalize_cv(json.loads(resp.choices[0].message.content))

    # normalisasi skills (deterministik)
    data["skills"]["hard_skills"] = atomize_skills(data["skills"].get("hard_skills", []))
    data["skills"]["soft_skills"] = atomize_skills(data["skills"].get("soft_skills", []))
    for e in data.get("experience", []):
        if isinstance(e, dict):
            e["key_skills"] = atomize_skills(e.get("key_skills", []))

    # anti-halusinasi tanggal (deterministik)
    return validate_dates(data, cv_text)


def extract_cv_with_retry(cv_text: str, llm_provider: str | None = None, max_retries: int = 5) -> dict:
    """Bungkus extract_cv dengan backoff untuk rate-limit (429) & overload (503)."""
    delay = 5
    for attempt in range(max_retries):
        try:
            return extract_cv(cv_text, llm_provider)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            transient = ("429" in msg or "RESOURCE_EXHAUSTED" in msg
                         or "503" in msg or "UNAVAILABLE" in msg or "overloaded" in msg)
            if transient and attempt < max_retries - 1:
                m = re.search(r"(\d+(?:\.\d+)?)s", msg)
                wait = (float(m.group(1)) + 2) if m else delay
                time.sleep(min(wait, 60))
                delay = min(delay * 2, 60)
            else:
                raise
    raise RuntimeError("Gagal setelah semua retry (rate limit / overload LLM)")


def extract_cv_from_pdf(pdf_path: str | Path, llm_provider: str | None = None) -> dict:
    """Full pipeline: PDF -> teks bersih -> LLM extraction -> normalisasi -> dict."""
    pdf_path = Path(pdf_path)
    cv_text = extract_clean_text(pdf_path)

    if len(cv_text) < MIN_TEXT_CHARS:
        raise ValueError(
            f"Teks terlalu pendek ({len(cv_text)} char). "
            "Kemungkinan PDF scan/gambar tanpa text layer — butuh OCR."
        )

    with _extract_sem:
        return extract_cv_with_retry(cv_text, llm_provider)


def save_processed_cv(cv_id: str, data: dict) -> Path:
    # Tulis atomik (tmp + rename) supaya pembaca paralel tidak pernah melihat JSON separuh.
    out = settings.PROCESSED_DIR / f"{cv_id}.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(tmp, out)
    return out


def load_processed_cv(cv_id: str) -> dict | None:
    p = settings.PROCESSED_DIR / f"{cv_id}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def list_processed_cvs() -> list[str]:
    return [f.stem for f in sorted(settings.PROCESSED_DIR.glob("*.json"))]
