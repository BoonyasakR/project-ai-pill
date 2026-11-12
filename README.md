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

## Colour check

PatchCore missed 6 of the 25 `color` defects. On these pills, normal specks are dull red. The
defects are stains in new colours: bright red, pink, yellow, or blue. Many are small or on the rim.
The colour check stores every colour seen on the 227 training pills, as a Lab histogram. A pill is
abnormal if PatchCore flags it, or if it has a blob of unseen colour larger than 20 pixels on the
800-pixel canvas. On the 40 validation pills, no pixel had an unseen colour.

These runs were chosen after the test set was seen, so they are exploratory. They were trained on an
Apple M3 with `src/train_patchcore.py`.

| Setting | Image AUROC | Precision | Recall | F1 | `color` recall | False positive / negative |
|---|---|---|---|---|---|---|
| 256, PatchCore only | 0.944 | 0.983 | 0.823 | 0.896 | 0.72 | 2 / 25 |
| 256 + colour check | 0.959 | 0.984 | 0.872 | 0.925 | 0.96 | 2 / 18 |
| 512, PatchCore only | 0.976 | 0.992 | 0.894 | 0.940 | 0.84 | 1 / 15 |
| 512 + colour check (web app) | 0.979 | 0.992 | 0.915 | 0.952 | 0.96 | 1 / 12 |

The colour check added no false positives. The largest unseen-colour blob on a normal test pill
was 2 pixels. One `color` defect is still missed: a blue stain on the rim, whose colour blends into
the dark blue background. Cracks remain the weakest type, with a recall of 0.77.

The 256 PatchCore model trained here scores slightly below the Colab one (Image AUROC 0.944 against
0.950). The hardware and the random coreset start differ.

## Several pills in one image

The web app also accepts several non-overlapping pills on a dark background. It separates the
pills, aligns each one like the training images, and runs the same PatchCore model and colour check
on each. Scoring uses the pill area only, with its own thresholds set from 400 validation pills:
42.96 for PatchCore and 20 pixels for unseen colour. On those normal pills, the largest unseen-colour
blob was 10 pixels. A single pill that is turned, upside down, resized, or off centre takes the
same route. A single pill posed like the training images keeps the whole-image score and the main
threshold.

The result warns about two cases. An object between 1% and 20% of the largest pill's area is not inspected.
A pill shorter than 180 pixels may be flagged wrongly: with the 256 model, on 40 normal validation
pills, 17 were flagged at 120 pixels, 5 at 150, and 1 at 180. These pills also set the threshold, so
this shows the trend only. It was not measured again for the 512 model.

MVTec has no multi-pill images, so these results come from simulated scenes: each test pill was
pasted at a random rotation and size, 8 pills per scene, over 3 seeds. Each pill therefore counts
6 times, so the counts below are not independent.

| Model | Layout | Pills | Missed by separation | Precision | Recall | F1 | AUROC |
|---|---|---|---|---|---|---|---|
| 512 + colour check (web app) | Spaced | 501 | 0 | 0.997 | 0.875 | 0.932 | 0.987 |
| 512 + colour check (web app) | Touching | 501 | 0 | 0.997 | 0.879 | 0.935 | 0.982 |
| 256, PatchCore only (before) | Spaced | 501 | 0 | 0.984 | 0.896 | 0.938 | 0.961 |
| 256, PatchCore only (before) | Touching | 501 | 0 | 0.979 | 0.894 | 0.934 | 0.957 |

With the new model, F1 stays about the same, but the errors move. `color` recall rose from 0.78 to
0.96, and false alarms on normal pills fell from 14 to 2 of 156. Recall fell for `faulty_imprint`
(0.85 to 0.73), `scratch` (0.96 to 0.85), and `contamination` (0.95 to 0.91), because the
validation threshold is stricter.

These numbers do not show accuracy on real photos. Real photos add shadows, uneven light, and other
backgrounds. Overlapping pills are not supported.

To repeat the evaluation, put the Pill category in `data/mvtec/pill/`, then run:

```bash
backend/.venv/bin/python src/evaluate_multi_pill.py calibrate
backend/.venv/bin/python src/evaluate_multi_pill.py test
```

## Project layout

- `notebooks/`: Colab notebooks for EDA, training, evaluation, and experiments
- `src/`: dataset checks, local training, evaluation metrics, and multi-pill simulation
- `splits/`: the fixed train/validation split
- `results/`: saved metrics and example heatmaps
- `backend/`: FastAPI inference service
- `frontend/`: React web app

## Run the web app

The model file is not in Git. Put the Pill category in `data/mvtec/pill/`, then train the model.
It takes about 15 to 30 minutes on an Apple M3, and uses the Apple GPU when there is one.

```bash
python3.12 -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements.txt
backend/.venv/bin/python src/train_patchcore.py --size 512 --coreset 0.01 --out models/patchcore_pill_512.pt
backend/.venv/bin/python -m uvicorn app:app --app-dir backend --port 8000
```

A `patchcore_pill_256.pt` file exported from the Colab notebook still works. Set
`MODEL_PATH=models/patchcore_pill_256.pt` to use it. It has no colour table, so the colour check is
skipped. The multi-pill thresholds in `results/multi_pill/` were set for the 512 model, so run
`src/evaluate_multi_pill.py calibrate` again, with the same `MODEL_PATH`, after changing the model.

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
