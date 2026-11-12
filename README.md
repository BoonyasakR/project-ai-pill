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

## Colour check (MVTec Pill)

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

The colour check applies to MVTec Pill only. The pillQC model has no colour table, so pillQC pills
are judged by PatchCore alone.

## Round 2: reliability and few-shot

Run with `notebooks/04_reliability_fewshot.ipynb` on a Colab T4. Every setting was fixed before the
run and matches the main result. Results are in `results/reliability/` and `results/fewshot/`.

Five seeds on the same split, and a 95% bootstrap interval (2,000 image resamples, fixed threshold)
on the seed 42 run, which reproduces the table above exactly:

| Model | Image AUROC, 5 seeds | Recall, 5 seeds | Image AUROC, 95% interval |
|---|---|---|---|
| PatchCore | 0.948 ± 0.003 | 0.853 ± 0.018 | 0.950 [0.912, 0.980] |
| PaDiM | 0.921 ± 0.015 | 0.384 ± 0.153 | 0.917 [0.864, 0.958] |

The image AUROC gap between the models is 0.033 [-0.014, 0.083]. The interval includes zero, so
this test set does not show that PatchCore ranks images better than PaDiM.

Few-shot PatchCore builds the memory bank from a few normal images, 5 draws each. pillQC is a second
tablet type with image labels only; its normal crops are split by source photo.

| Normal images | MVTec Pill image AUROC | pillQC image AUROC |
|---|---|---|
| 1 | 0.820 ± 0.041 | 0.817 ± 0.085 |
| 5 | 0.887 ± 0.019 | 0.910 ± 0.027 |
| 10 | 0.903 ± 0.016 | 0.918 ± 0.015 |
| 25 | 0.927 ± 0.011 | 0.947 ± 0.004 |
| All (227 / 46) | 0.947 ± 0.003 | 0.957 ± 0.002 |

The validation images that set the threshold are extra to these counts. On pillQC the threshold
rule gives recall 0.288 at precision 1.000 with all images, so the scores rank well but the
threshold does not carry over to a new tablet type. The MVTec test set was already used, so the
few-shot numbers on MVTec are exploratory. pillQC photos show about 28 normal pills photographed
several times, so its images are not independent. pillQC images are not stored in this repository;
its license limits use to MathWorks products.

## pillQC: score the pill only

The first pillQC model scored the whole crop. Some normal validation crops have white dirt on the
teal floor next to the pill, which scored up to 55 and set the threshold at 55.21. Most defects
scored below that. The model now scores only the pill: the largest circle inside the pink pill
mask, grown by 5%, so floor dirt that touches the pill is left out. This was chosen from the
validation crops only, then the test set was run once.

| pillQC | Threshold | Image AUROC | Precision | Recall | F1 | Chip recall | Dirt recall | False positives |
|---|---|---|---|---|---|---|---|---|
| One pill, whole crop (before) | 55.21 | 0.959 | 1.000 | 0.249 | 0.398 | 0.44 | 0.19 | 0 / 49 |
| One pill, pill only | 47.39 | 0.990 | 1.000 | 0.746 | 0.854 | 0.88 | 0.70 | 0 / 49 |
| Multi-pill (before) | 67.64 | 0.863 | 1.000 | 0.015 | 0.029 | 0.05 | 0.00 | 0 / 294 |
| Multi-pill, clean cut-outs | 54.83 | 0.933 | 0.986 | 0.391 | 0.560 | 0.55 | 0.34 | 6 / 294 |
| One pill, bank + cut-out pool (web app) | 46.87 | 0.990 | 1.000 | 0.751 | 0.858 | 0.91 | 0.70 | 0 / 49 |
| Multi-pill, bank + cut-out pool (web app) | 48.65 | 0.980 | 0.995 | 0.597 | 0.746 | 0.55 | 0.61 | 3 / 294 |

In simulated multi-pill scenes, pills were cut out with the pink mask, so floor dirt was pasted
along with them. It is now cut off at the same circle.

Multi-pill recall stayed low because each scene pill is cut out onto a plain fill before scoring.
Its rim then touches the fill, never the teal floor of the training crops. On the 54 validation
pills, cutting out alone raised the median normal score from 33 to 45; resizing barely mattered.
The memory bank now also holds the 46 pool pills cut out the same way. On validation, the median
normal scene score fell from 46.6 to 35.1 and the threshold from 54.83 to 48.65. Single-crop
results barely changed. The test set was then run once for each mode. The multi-pill test had
already been run with the previous model, so these numbers are exploratory.

Chips are still missed in scenes (recall 0.55, against 0.91 for single crops). The simulated
scenes also turn each pill at random, which turns its lighting and side band with it; a real photo
keeps one light direction. On validation, scenes without turns had a lower 99th percentile (52.6
against 54.8 with the previous model), so the turn makes these results a little pessimistic.

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

To inspect pillQC tablets as well, build their memory bank and threshold (a few minutes on a CPU,
needs the pillQC images in `data/pillqc/`), then pick the type on the upload page. pillQC takes one
pill on its teal background, or several pills on a dark background. See "pillQC: score the pill
only" above for its results.

```bash
backend/.venv/bin/python src/build_pillqc_model.py --root data/pillqc
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

## Datasets

Download the images yourself; they are not stored in this repository.

| Dataset | Link | Put it in |
|---|---|---|
| MVTec AD (Pill category) | [mvtec.com](https://www.mvtec.com/company/research/datasets/mvtec-ad), or the copy on [Kaggle](https://www.kaggle.com/datasets/ipythonx/mvtec-ad) | `data/mvtec/pill/` |
| pillQC | [github.com/matlab-deep-learning/pillQC](https://github.com/matlab-deep-learning/pillQC) | `data/pillqc/` |

## Dataset license

MVTec AD is licensed for non-commercial use under CC BY-NC-SA 4.0. pillQC is under the MathWorks
license in its repository, which limits use to MathWorks products; this project uses it only for
coursework and does not share its images. Dataset images are not stored in this repository.
