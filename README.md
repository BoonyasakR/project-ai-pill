# AI Pill Anomaly Detection

Detects and localizes defects on pills, using only normal images for training. Uses the
MVTec AD Pill category with PatchCore as the main model and PaDiM as the baseline.

This is a research prototype. It is not a certified pharmaceutical quality system.

## Results

Test set: 26 normal and 141 defective images. Thresholds come from normal validation images.

| Model | Image AUROC | Pixel AUROC | AU-PRO | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| PatchCore | 0.950 | 0.981 | 0.942 | 0.984 | 0.865 | 0.921 |
| PaDiM | 0.917 | 0.952 | 0.933 | 1.000 | 0.333 | 0.500 |

See `docs/` for the report and slides.

## Project layout

- `notebooks/`: Colab notebooks for EDA, training, evaluation, and experiments
- `src/`: dataset checks and evaluation metrics
- `splits/`: the fixed train/validation split
- `results/`: saved metrics and example heatmaps
- `backend/`: FastAPI inference service
- `frontend/`: React web app

## Run the web app

The model file is not in Git. Run `notebooks/03_exploratory_experiments.ipynb` on a Colab GPU,
then put `patchcore_pill_256.pt` in `models/`.

```bash
python3.12 -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements.txt
backend/.venv/bin/python -m uvicorn app:app --app-dir backend --port 8000
```

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173.

## Run the tests

```bash
backend/.venv/bin/python -m unittest discover -s tests
cd backend && .venv/bin/python -m unittest test_app
```

## Dataset license

MVTec AD is licensed for non-commercial use under CC BY-NC-SA 4.0. Dataset images are not stored
in this repository.
