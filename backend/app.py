"""FastAPI service for the pill inspection prototype."""
import io
import json
import os
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.environ.get("MODEL_PATH", ROOT / "models" / "patchcore_pill_256.pt"))
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = FastAPI(title="Pill Anomaly Inspection")


@lru_cache(maxsize=1)
def get_inspector():
    from inference import Inspector

    if not MODEL_PATH.exists():
        raise HTTPException(503, f"Model file not found: {MODEL_PATH.name}")
    return Inspector(MODEL_PATH)


def summarize(path):
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {
        name: {key: value for key, value in run.items() if key in ("config", "overall", "by_defect", "error")}
        for name, run in data.items()
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "model_available": MODEL_PATH.exists()}


@app.get("/api/results")
def results():
    return {
        "main": summarize(ROOT / "results" / "phase3" / "phase3_results.json"),
        "exploratory": summarize(ROOT / "results" / "phase4" / "phase4_results.json"),
    }


@app.post("/api/inspect")
async def inspect(file: UploadFile = File(...), inspector=Depends(get_inspector)):
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image must be 10 MB or smaller")
    try:
        image = Image.open(io.BytesIO(content))
        image.load()
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(400, "File is not a readable image") from error
    return inspector.inspect(image)
