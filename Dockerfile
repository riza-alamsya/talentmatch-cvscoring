# TalentMatch — FastAPI scoring engine (Gemini-powered)
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install CPU-only torch first (much smaller than the default CUDA build), then the
# rest. sentence-transformers (in requirements) reuses this torch for local bge-m3.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r requirements.txt

# Bake the small multilingual embedding model into the image (~470MB) so local
# embedding works on cheap Cloud Run with no runtime download.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"

# App code (data/ and .env are provided at runtime via volume + env)
COPY src ./src

# config.py computes BASE_DIR = <repo root> = /app ; data lives at /app/data
WORKDIR /app/src
EXPOSE 8080

# Listen on $PORT (Cloud Run sets it to 8080); default 8000 for local/compose.
CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
