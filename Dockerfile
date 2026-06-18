# Base image: a slim Debian with Python 3.13 preinstalled.
FROM python:3.13-slim

# System packages our pip deps need (OCR, table extraction, OpenCV runtime).
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        ghostscript \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Don't buffer Python logs; don't write .pyc files. Cleaner container logs.
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy ONLY requirements first so code edits don't re-trigger a slow reinstall.
COPY requirements.txt .

# Two deliberate slimming steps, both saving image size:
#   1. Install CPU-only PyTorch up front. sentence-transformers depends on torch;
#      left alone it pulls the ~2GB CUDA build, useless without a GPU. Pinning the
#      CPU wheel index first means the later install sees torch already satisfied.
#   2. Drop the legacy Streamlit UI (app.py). The Django backend serves the
#      frontend itself, so streamlit + its pandas/pyarrow stack is dead weight
#      in the container. Stripped here only — requirements.txt stays intact for
#      local dev where `streamlit run app.py` may still be used.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && grep -ivE '^streamlit$' requirements.txt > /tmp/requirements.docker.txt \
    && pip install --no-cache-dir -r /tmp/requirements.docker.txt

# Now copy the rest of the project in.
COPY . .

# The port Django listens on inside the container.
EXPOSE 8000

# Default command. For a real deployment, swap runserver for gunicorn:
#   gunicorn finrag_backend.wsgi:application --bind 0.0.0.0:8000
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
