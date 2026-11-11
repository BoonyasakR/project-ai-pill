"""FastAPI service for the pill inspection prototype."""
import io
import json
import os
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.environ.get("MODEL_PATH", ROOT / "models" / "patchcore_pill_256.pt"))
# One memory bank and threshold per tablet type, on the same backbone. Each type has its
# own multi-pill threshold and its own way of cutting pills out of a larger image.
PILL_TYPES = {
    "mvtec": {
        "model": MODEL_PATH,
        "scenes": ROOT / "results" / "multi_pill" / "multi_pill_results.json",
        "pose": "MVTEC_POSE",
    },
    "pillqc": {
        "model": ROOT / "models" / "patchcore_pillqc_256.pt",
        "scenes": ROOT / "results" / "pillqc_model" / "pillqc_scene_results.json",
        "pose": "PILLQC_POSE",
    },
}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = FastAPI(title="Pill Anomaly Inspection")


@lru_cache(maxsize=len(PILL_TYPES))
def load_inspector(pill_type):
    import inference

    setup = PILL_TYPES[pill_type]
    if not setup["model"].exists():
        raise HTTPException(503, f"Model file not found: {setup['model'].name}")
    scenes = setup["scenes"] if setup["scenes"].exists() else None
    return inference.Inspector(setup["model"], scenes, pose=getattr(inference, setup["pose"]))


def get_inspector(pill_type: str = Form("mvtec")):
    if pill_type not in PILL_TYPES:
        raise HTTPException(400, f"Unknown pill type: {pill_type}")
    return load_inspector(pill_type)


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
    return {
        "status": "ok",
        "model_available": MODEL_PATH.exists(),
        "pill_types": {name: setup["model"].exists() for name, setup in PILL_TYPES.items()},
    }


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
