"""PDF text extraction + hybrid extraction (regex personal info + LLM skills/experience).
Hybrid architecture — instant regex for personal_info, minimal LLM only for
skills & experience with text limited to 2500 characters + max_tokens=250.
Target: < 2.5 seconds per CV via DeepSeek Chat."""
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

# Limit parallel extraction: requests come through threadpool (endpoint `def`),
# semaphore keeps concurrent LLM calls small.
_extract_sem = threading.BoundedSemaphore(settings.MAX_CONCURRENT_EXTRACT)

# LITE_SCHEMA — only skills & experience. Personal info extracted via regex (instant).
# Other fields (summary, education, certifications, achievements, languages) filled
# with defaults by normalize_cv().
LITE_SCHEMA = {
    "type": "object",
    "properties": {
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
                    "start_date": {"type": "string"},
                    "end_date":   {"type": "string"},
                    "is_current": {"type": "boolean"},
                },
                "required": ["company", "role", "start_date", "end_date", "is_current"],
            },
        },
    },
    "required": ["skills", "experience"],
}

# Minimal system prompt (< 200 words) — request JSON output for skills & experience only.
SYSTEM_PROMPT = """You are a CV parser. Extract skills and work experience from the text into JSON.
- hard_skills: job-specific professional competencies. soft_skills: interpersonal skills (only if explicitly listed, else []).
- experience: list each job with company, role, start_date, end_date, is_current.
- Dates: use only dates EXPLICITLY written in the CV. Normalize to YYYY-MM. "Present"/"Now"/"Sekarang" => is_current=true, end_date="". If absent, "".
- NEVER invent facts. Output ONLY valid JSON, no explanation."""

MIN_TEXT_CHARS = 100  # minimum text threshold; below = likely scanned/image PDF


# ── Helpers ───────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2014", " - ").replace("\u2013", " - ").replace("\u2022", " - ")
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_clean_text(pdf_path: str | Path) -> str:
    """Extract PDF text using get_text('text') — much faster than 'dict' mode."""
    doc = fitz.open(str(pdf_path))
    full_text = []
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            full_text.append(text)
    doc.close()
    return clean_text(" ".join(full_text))


def atomize_skills(items: list[str]) -> list[str]:
    """Deterministic normalization: split 'Category: a, b | c' into atomic items.
    Do NOT split on '/' to preserve 'CI/CD', 'OpenAPI/Swagger' intact."""
    out = []
    for s in items:
        if not isinstance(s, str):
            continue
        if ":" in s:
            s = s.split(":", 1)[1]
        for part in re.split(r"[,|]", s):
            p = part.strip(" .;-\u2022")
            if p:
                out.append(p)
    seen, result = set(), []
    for x in out:
        if x.lower() not in seen:
            seen.add(x.lower())
            result.append(x)
    return result


def normalize_cv(data: dict) -> dict:
    """Restore field names + fill defaults for fields not requested from LLM
    (summary, education, certifications, achievements, languages set to empty)."""
    # personal_info — from regex, must always exist
    if "personal_info" not in data:
        data["personal_info"] = {"name": "", "email": "", "phone": "",
                                 "location": "", "links": []}
    pi = data.get("personal_info", {})
    if isinstance(pi, dict):
        pi.setdefault("links", [])
        for k in ("name", "email", "phone", "location"):
            pi.setdefault(k, "")

    # skills must be dict {hard_skills:[], soft_skills:[]}
    if not isinstance(data.get("skills"), dict):
        data["skills"] = {"hard_skills": [], "soft_skills": []}
    for sk in ("hard_skills", "soft_skills"):
        if not isinstance(data["skills"].get(sk), list):
            data["skills"][sk] = []

    # personal_info must be dict
    if not isinstance(data.get("personal_info"), dict):
        data["personal_info"] = {"name": "", "email": "", "phone": "",
                                 "location": "", "links": []}

    # required fields — summary/education/cert/achievements/languages filled with defaults
    data.setdefault("summary", "")
    for field in ("experience", "education", "certifications", "achievements", "languages"):
        data.setdefault(field, [])

    # list-of-object fields: ensure list type
    for field in ("experience", "education", "certifications", "languages", "achievements"):
        if not isinstance(data.get(field), list):
            data[field] = []

    # experience/education: remove non-dict items
    for field in ("experience", "education"):
        data[field] = [x for x in data[field] if isinstance(x, dict)]

    # languages/certifications: string -> object, remove non-str/dict
    data["languages"] = [
        {"language": x} if isinstance(x, str) else x
        for x in data["languages"] if isinstance(x, (str, dict))
    ]
    data["certifications"] = [
        {"name": x} if isinstance(x, str) else x
        for x in data["certifications"] if isinstance(x, (str, dict))
    ]

    # achievements: strings only
    data["achievements"] = [x for x in data["achievements"] if isinstance(x, str)]

    return data


def _year_in_source(date_str: str, text: str) -> bool:
    return (not date_str) or (date_str[:4] in text)


def validate_dates(data: dict, source_text: str) -> dict:
    """Anti-hallucination for dates: clear dates with years not in the CV text."""
    for e in data.get("experience", []):
        for k in ("start_date", "end_date"):
            if e.get(k) and not _year_in_source(e[k], source_text):
                e[k] = ""
        if not e.get("start_date") and not e.get("end_date"):
            e["is_current"] = False
    return data


# ── Regex personal info (instant, no LLM) ───────────────────────────────────
def _extract_personal_regex(text: str) -> dict:
    """Extract name, email, phone, location via regex — 0 ms latency."""
    # Email
    email = ""
    m = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    if m:
        email = m.group(0).strip()

    # Phone (Indonesia & international)
    phone = ""
    m = re.search(r"(\+?62|0|8)[\d\s\-\(\)]{8,15}", text)
    if m:
        phone = re.sub(r"\s+", " ", m.group(0).strip())

    # Name: first non-empty line (assume name at top of CV)
    name = ""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        raw = lines[0]
        raw = re.sub(r"[^\w\s\.\-\']", "", raw)
        raw = re.sub(r"\s+", " ", raw).strip()
        # Skip if contains @ (email) or only digits (phone)
        if raw and "@" not in raw and not re.match(r"^[\d\s\+\-\(\)]+$", raw):
            name = raw

    # Location: empty (difficult to regex accurately)
    location = ""

    return {"name": name, "email": email, "phone": phone, "location": location, "links": []}


# ── Main service ──────────────────────────────────────────────────────────────
def extract_cv(cv_text: str, llm_provider: str | None = None) -> dict:
    """Hybrid: regex for personal_info + minimal LLM for skills & experience.
    Total round-trip < 2.5 seconds (DeepSeek Chat)."""
    # Save original text for date validation (need full text)
    original_text = cv_text

    # Optimization: truncate for faster processing
    cv_text = cv_text[:2500]

    # 1. Regex — personal info (instant, no LLM)
    personal_info = _extract_personal_regex(cv_text)

    # 2. LLM — only skills + experience (short prompt, max_tokens=250)
    client, model = get_llm(llm_provider)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": cv_text},
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=250,
    )
    data_llm = json.loads(resp.choices[0].message.content)

    # 3. Merge: personal_info (regex) + skills & experience (LLM)
    data = {
        "personal_info": personal_info,
        "skills": data_llm.get("skills", {"hard_skills": [], "soft_skills": []}),
        "experience": data_llm.get("experience", []),
    }
    data = normalize_cv(data)

    # normalize skills (deterministic)
    data["skills"]["hard_skills"] = atomize_skills(data["skills"].get("hard_skills", []))
    data["skills"]["soft_skills"] = atomize_skills(data["skills"].get("soft_skills", []))
    for e in data.get("experience", []):
        if isinstance(e, dict):
            e["key_skills"] = atomize_skills(e.get("key_skills", []))

    # anti-hallucination for dates (deterministic, use original text)
    return validate_dates(data, original_text)


def extract_cv_with_retry(cv_text: str, llm_provider: str | None = None, max_retries: int = 5) -> dict:
    """Wrap extract_cv with backoff for rate-limit (429) & overload (503)."""
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
    raise RuntimeError("Failed after all retries (rate limit / LLM overload)")


def extract_cv_from_pdf(pdf_path: str | Path, llm_provider: str | None = None, force: bool = False) -> dict:
    """Full pipeline: PDF -> clean text -> hybrid extraction -> normalization -> dict.
    - force=False: return from cache if CV already processed.
    - force=True:  always re-extract (ignore cache)."""
    pdf_path = Path(pdf_path)
    cv_id = pdf_path.stem

    # Cache hit — return directly without LLM call
    if not force:
        cached = load_processed_cv(cv_id)
        if cached is not None:
            return cached

    cv_text = extract_clean_text(pdf_path)

    if len(cv_text) < MIN_TEXT_CHARS:
        raise ValueError(
            f"Text too short ({len(cv_text)} characters). "
            "Likely a scanned/image PDF without text layer — requires OCR."
        )

    with _extract_sem:
        return extract_cv_with_retry(cv_text, llm_provider)


# ── Caching ───────────────────────────────────────────────────────────────────
def save_processed_cv(cv_id: str, data: dict) -> Path:
    # Atomic write (tmp + rename) so parallel readers never see incomplete JSON.
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
