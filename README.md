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

## Several pills in one image

The web app also accepts several non-overlapping pills on a dark background. It separates the
pills, aligns each one like the training images, and runs the same PatchCore model on each. No model
was retrained. Scoring uses the pill area only, with its own threshold (35.25) set from validation
pills. A single pill that is turned, upside down, resized, or off centre takes the same route. A
single pill posed like the training images keeps the whole-image score and the main threshold.

The result warns about two cases. An object between 1% and 20% of the largest pill's area is not inspected.
A pill shorter than 180 pixels may be flagged wrongly: on 40 normal validation pills, 17 were
flagged at 120 pixels, 5 at 150, and 1 at 180. These pills also set the threshold, so this shows the
trend only.

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
