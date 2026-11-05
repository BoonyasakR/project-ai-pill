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

## Several pills in one image

The web app also accepts several non-overlapping pills on a dark background. It separates the
pills, aligns each one like the training images, and runs the same PatchCore model on each. No model
was retrained. Scoring uses the pill area only, with its own threshold (35.25) set from validation
pills.

MVTec has no multi-pill images, so these results come from simulated scenes: each test pill was
pasted at a random rotation and size, 8 pills per scene, over 3 seeds. Each pill therefore counts
6 times, so the counts below are not independent.

| Layout | Pills | Missed by separation | Precision | Recall | F1 | AUROC |
|---|---|---|---|---|---|---|
| Spaced | 501 | 0 | 0.984 | 0.896 | 0.938 | 0.961 |
| Touching | 501 | 0 | 0.979 | 0.894 | 0.934 | 0.957 |

These numbers do not show accuracy on real photos. Real photos add shadows, uneven light, and other
backgrounds. Overlapping pills are not supported.

To repeat the evaluation, put the Pill category in `data/mvtec/pill/`, then run:

```bash
backend/.venv/bin/python src/evaluate_multi_pill.py calibrate
backend/.venv/bin/python src/evaluate_multi_pill.py test
```

## Experimental photo model

The web app can also use a second model, built from the 35 pills in one image with the same frozen
backbone. It assumes all 35 pills are normal. Each pill was scored by a bank that had not seen it:
2 of 35 were flagged, and 101 of 105 faint painted defects were caught. There are no real defect
labels, so this does not measure accuracy on real defects. Inspecting the training image again gives
results that look better than they are.

```bash
backend/.venv/bin/python src/train_from_photo.py IMAGE --bank-share 0.5 --output results/photo_model/bank50 --save models/patchcore_photo.pt
```

## Project layout

- `notebooks/`: Colab notebooks for EDA, training, evaluation, and experiments
- `src/`: dataset checks, evaluation metrics, and multi-pill simulation
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
