# EarlyDia - single image: React build served by FastAPI (same origin, no CORS setup needed).
#   docker build -t earlydia .
#   docker run -p 8000:8000 earlydia      ->  http://localhost:8000

# ---- stage 1: build the React app ------------------------------------------------------------
FROM node:20-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
# frontend/.env is excluded by .dockerignore, so the build calls the API on the same origin.
RUN npm run build

# ---- stage 2: FastAPI + locked production models ---------------------------------------------
FROM python:3.12-slim AS app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY backend/ backend/
COPY src/ src/
COPY configs/ configs/
COPY models/ models/
COPY models_v2/ models_v2/
COPY reports/ reports/
COPY --from=web /web/dist frontend/dist
EXPOSE 8000
# Hosts such as Render/Railway inject $PORT; default to 8000 locally.
CMD ["sh", "-c", "uvicorn backend.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
