FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && adduser --disabled-password --gecos '' appuser

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e . \
    && python - <<'PY'
from pathlib import Path
from hyperquant.native_core import build_native_fwht
pkg_root = Path('/app/hyperquant')
print(build_native_fwht(force=True, build_dir=pkg_root))
PY

USER appuser
EXPOSE 8080
CMD ["hyperquant", "serve", "--bundle", "/app/bundle.npz", "--host", "0.0.0.0", "--port", "8080"]
