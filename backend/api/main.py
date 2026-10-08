"""EarlyDia FastAPI application.

React -> FastAPI -> existing inference module (src/inference/predict_module.py) -> production models.
Run from the project root:   uvicorn backend.api.main:app --reload --port 8000
"""
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from backend.api import service  # noqa: E402
from backend.api.schemas import ScreeningRequest, ScreeningResponse  # noqa: E402
from backend.api.research import router as research_router  # noqa: E402
from backend.configs import settings  # noqa: E402
from backend.inference.engine import load_all  # noqa: E402

log = logging.getLogger("earlydia")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all()          # fail fast if a production artifact cannot be loaded
    yield


app = FastAPI(title="EarlyDia API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.CORS_ORIGINS,
                   allow_origin_regex=getattr(settings, "CORS_ORIGIN_REGEX", None),
                   allow_methods=["GET", "POST"], allow_headers=["*"])


def _err(status: int, message: str, errors=None):
    return JSONResponse(status_code=status, content={"message": message, "errors": errors or []})


@app.exception_handler(service.InputError)
async def _input_error(_: Request, exc: service.InputError):
    return _err(422, "Please check your answers.", exc.errors)


@app.exception_handler(RequestValidationError)
async def _bad_request(_: Request, exc: RequestValidationError):
    names = {"bp_systolic": "Blood pressure (systolic)", "bp_diastolic": "Blood pressure (diastolic)",
             "sex": "Sex", "age": "Age", "education": "Education", "residence": "Residence",
             "diet": "Food questions"}
    errs = []
    for e in exc.errors():
        field = str(e["loc"][-1]) if e.get("loc") else ""
        kind = "is required" if e.get("type") == "missing" else "is missing or not valid"
        errs.append(f"{names.get(field, field or 'An answer')} {kind}.")
    return _err(422, "Some answers are missing or not in the expected format.", errs[:10])


@app.exception_handler(Exception)
async def _unexpected(_: Request, exc: Exception):
    log.exception("Unhandled error")          # traceback goes to the server log only
    return _err(500, "Something went wrong while preparing your result. Please try again.")


app.include_router(research_router)


@app.get("/api/health")
def health():
    preds = load_all()
    return {"status": "ok", "models": {s: p.calibration_method for s, p in preds.items()}}


@app.get("/api/config")
def config():
    return service.get_config()


@app.post("/api/predict", response_model=ScreeningResponse)
def predict(req: ScreeningRequest):
    return service.run_screening(req)


# Optional: serve the built React app from the same process (after `npm run build`).
_dist = settings.FRONTEND_DIST
if _dist.exists() and (_dist / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path.startswith("api/"):
            return _err(404, "Not found")
        f = _dist / full_path
        return FileResponse(f if f.is_file() else _dist / "index.html")
