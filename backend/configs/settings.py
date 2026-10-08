"""Backend settings, read from environment variables (no machine-specific paths)."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Comma-separated list of browser origins allowed to call the API (React dev server by default).
CORS_ORIGINS = [o.strip() for o in os.getenv(
    "EARLYDIA_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,"
    "http://localhost:5174,http://127.0.0.1:5174,"
    "http://localhost:4173,http://127.0.0.1:4173,http://localhost:8000,http://127.0.0.1:8000").split(",") if o.strip()]

# Regex pattern for localhost development origins (supports any local dev server port)
CORS_ORIGIN_REGEX = os.getenv("EARLYDIA_CORS_ORIGIN_REGEX", r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$")

# If the React app has been built (frontend/dist), FastAPI can also serve it, so the whole
# product can run from a single `uvicorn` process. Override with EARLYDIA_FRONTEND_DIST.
FRONTEND_DIST = Path(os.getenv("EARLYDIA_FRONTEND_DIST", PROJECT_ROOT / "frontend" / "dist"))
