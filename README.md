# 🛡️ Intelligent Intrusion Detection System

A machine-learning intrusion detector for network traffic, with a dashboard
that explains **why** each alert fired, in plain English.

A Random Forest classifies each network connection as **Normal**, **DoS**,
**Probe** or **Brute Force**. For every alert, the dashboard shows the
specific signals that drove the decision, for example:

> Flagged as DoS (100% confidence) mainly because only 7% of connections to the
> destination hit different services; 230 connections to the same host in the
> last 2 seconds; all of the connections to the destination host were rejected.

<!-- Add screenshots: run the app, take screenshots, save them in docs/ and uncomment:
![Alert explanation](docs/alert-explanation.png)
![Model performance](docs/model-performance.png)
-->

## Why explainability

A bare "ATTACK: 0.97" alert isn't something an analyst can act on. They have to
decide in seconds whether to block a host, and they won't trust a black box.
This project treats the explanation as the product, not an add-on.

## How it works

### The 9-stage pipeline (`ids/pipeline.py`)

| # | Stage | What happens |
|---|-------|--------------|
| 1 | Load | Download and read NSL-KDD (126k training, 22.5k test connections) |
| 2 | Clean | Drop duplicates, fail loudly on missing values |
| 3 | Label | Map ~40 raw attack names (`neptune`, `satan`, `guess_passwd`...) to 4 categories |
| 4 | Prune | Drop features that never vary (`num_outbound_cmds`) |
| 5 | Encode | One-hot encode protocol, service, TCP flag (41 → 121 features) |
| 6 | Select | Keep the features covering 99% of Random Forest importance (121 → 43) |
| 7 | Train | 200-tree Random Forest with balanced class weights |
| 8 | Evaluate | Score on the **official** test set, which contains unseen attack types |
| 9 | Save | Model + preprocessing + metrics in one artifact |

The preprocessing in stages 4–6 is wrapped in a `Preprocessor` class, so the
dashboard applies exactly the same transformations to new traffic as training
did. A service the model has never seen is handled safely instead of crashing.

### Explanations (`ids/explain.py`)

Each tree in the forest starts from the training set's class mix at its root.
Every split it makes on the way to a leaf shifts the probabilities, and that
shift is credited to the feature the split tested. Averaged over 200 trees:

```
predicted probability = baseline + Σ (contribution of each feature)
```

This adds up **exactly** (it's checked in `tests/test_ids.py`), so the
explanation is a faithful breakdown of what the model computed, not a separate
approximation. The top contributions are then turned into sentences using a
dictionary that knows what each of the 41 NSL-KDD features means
(e.g. `serror_rate = 1.0` → "all of the recent connections to this host were
half-open (SYN errors)").

## Results

On the official NSL-KDD test set (KDDTest+):

| Class | Precision | Recall |
|-------|-----------|--------|
| Normal | 0.66 | 0.97 |
| DoS | 0.96 | 0.82 |
| Probe | 0.83 | 0.61 |
| Brute Force | 0.98 | **0.05** |
| **Overall accuracy** | | **76.2%** |

### The honest part: Brute Force recall

Brute Force recall is very low, and it's the most interesting result here.
KDDTest+ was deliberately built to include **17 attack types never seen in
training**, and 718 of the 2,954 Brute Force test rows are those novel attacks.
The rest are known types (like `guess_passwd`) whose traffic in the test set
looks different from training. Brute Force is also under 1% of the training
data.

What I tried:
- **Balanced class weights** (kept): the model now weighs rare classes fairly.
- **Feature selection** (kept): dropping noisy one-hot columns raised accuracy
  from 74.4% to 76.2%, but didn't fix Brute Force.
- **Takeaway**: a supervised model can't flag an attack pattern it has never
  seen. A real deployment would pair it with anomaly detection (e.g. Isolation
  Forest) for novel attacks. That's the next thing I'd build.

When Brute Force *is* flagged, precision is 98%, so analysts aren't flooded
with false alarms.

## Run it

```bash
pip install -r requirements.txt
python train.py            # downloads data, runs the 9 stages (~30s)
streamlit run app.py       # opens the dashboard
pytest                     # 4 tests
```

The dashboard trains the model automatically on first launch if
`models/ids_model.joblib` doesn't exist, so it also deploys as-is to
[Streamlit Community Cloud](https://streamlit.io/cloud).

## Project layout

```
ids/
  data.py       dataset download, column names, attack → category mapping
  pipeline.py   the 9-stage training pipeline + Preprocessor
  explain.py    per-alert feature contributions and plain-English phrasing
app.py          Streamlit dashboard (alerts, model performance, upload your own CSV)
train.py        command-line entry point for training
tests/          pipeline, preprocessing and explanation tests
```

## Dataset

[NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html), Canadian Institute for
Cybersecurity, University of New Brunswick. It's an improved version of KDD Cup
'99 with redundant records removed.

## Tech

Python · pandas · scikit-learn · Streamlit · Altair · pytest
