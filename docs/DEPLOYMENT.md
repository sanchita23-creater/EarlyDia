# EarlyDia Deployment Guide

## Local Development

### Prerequisites
- Python 3.12+
- Node.js 20+
- Git

### Backend Setup
```bash
cd Earlydia
python -m venv .venv
.venv/Scripts/activate      # Windows
pip install -r requirements.txt

# Run API
uvicorn backend.api.main:app --reload --port 8000
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev    # Starts at http://localhost:5173
```

### Run Tests
```bash
# Python tests
python -m pytest tests/ -v

# Frontend tests
cd frontend
npx vitest run
```

## Docker Deployment

### Build
```bash
docker build -t earlydia .
```

### Run
```bash
docker run -p 8000:8000 earlydia
# Access at http://localhost:8000
```

### Docker Compose
```bash
docker compose up --build
```

### Health Check
```bash
curl http://localhost:8000/api/health
# Expected: {"status": "ok", "models": {"women": "isotonic", "men": "sigmoid"}}
```

### Smoke Test
```bash
curl http://localhost:8000/api/config | python -m json.tool | head -20
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"sex":"women","age":30,"education":2,"residence":1,
       "tobacco_use":false,"drinks_alcohol":false,
       "diet":{"a":1,"b":1,"c":1,"d":1,"e":1,"f":1,"g":1,"h":1,"i":1},
       "height_cm":160,"weight_kg":60,"bp_systolic":120,"bp_diastolic":80}'
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8000 | Server port (Docker/cloud) |
| `EARLYDIA_CORS_ORIGINS` | localhost dev origins | Comma-separated allowed origins |
| `EARLYDIA_CORS_ORIGIN_REGEX` | localhost regex | CORS origin regex pattern |
| `EARLYDIA_FRONTEND_DIST` | `frontend/dist` | Path to built frontend |

## Model Artifact Requirements

The Docker image requires these artifact directories:
- `models/` — v1 production pipelines (locked, deprecated)
- `models_v2/` — v2 production pipelines (current default)
- `configs/` — Model configuration and registry
- `reports/` — Pre-computed evaluation reports

### Startup Validation
The API fails fast at startup if production artifacts cannot be loaded.
Check `docker logs <container>` for artifact loading errors.

## Model Promotion / Rollback

### Promote a New Model
1. Train and evaluate via `python -m src.v2.run_v2`
2. Run final evaluation via `python -m src.v2.final_v2 --sex <sex> --name <name> --confirm-test-access`
3. Update `configs/model_registry.json` to set `status: "approved"` for the new model
4. Restart the API server

### Rollback
1. Edit `configs/model_registry.json`: set the current approved model to `"deprecated"` and the previous model to `"approved"`
2. Ensure the rollback model's artifact file still exists
3. Restart the API server

### Programmatic Registry Management
```python
from src.models.model_registry import load_registry, promote_model, rollback_model

# Promote
promote_model("v2_xgb_women_A")

# Rollback
rollback_model("women")
```

## CI/CD Pipeline

The `.github/workflows/ci.yml` runs:
1. Python tests (`pytest`)
2. Frontend tests (`vitest`)
3. Frontend build (`npm run build`)
4. Docker build verification
