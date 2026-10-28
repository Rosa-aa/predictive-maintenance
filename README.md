# Predictive Maintenance for Industrial Equipment

![Demo](docs/demo.gif)

Remaining useful life (RUL) forecasting with an LSTM, plus anomaly detection with
Isolation Forest and an Autoencoder. Results are shown as a green / yellow / red alert.

**Live demo:** [link]

## Problem
Unplanned equipment downtime threatens production, cost, and safety.
The goal is to detect degradation early enough to plan maintenance in advance.

## Data
- NASA C-MAPSS, sub-dataset FD001: 100 training and 100 test engines (open benchmark)
- The same approach can be applied to rotating equipment such as pumps, compressors,
  and gas turbines, but real industrial data would require retraining and threshold calibration
- Constant sensors were removed, leaving [N] sensors
- RUL is capped at 125 cycles (piecewise linear target). The true test RUL is capped at 125 as well

## Approach
1. MinMax scaling, fitted on the training set only
2. **RUL:** 30-cycle window → LSTM(64) → Dropout(0.2) → LSTM(32) → Dense(32, tanh) → Dense(1)
3. **Anomaly detection:** Isolation Forest (300 trees) and an Autoencoder (16–6–16), both trained on
   the healthy phase (RUL ≥ 100). Thresholds are the 99th percentile of healthy-phase scores.
   An alarm is raised when 3 consecutive cycles exceed the threshold
4. **Alert levels:** RUL < 30 red, RUL < 60 yellow

## Results

**RUL prediction (test set, 100 engines):**
| Model | RMSE | NASA score |
|---|---|---|
| Random Forest (rolling features) | 18.17 | 1070 |
| **LSTM** | **13.84** | **338** |

**Anomaly detection (20 held-out engines, split by engine):**
| Model | Median lead time (cycles before failure) | Minimum | Missed | Too early (>100 cycles) |
|---|---|---|---|---|
| Isolation Forest | 53.5 | 29 | 0/20 | 2/20 |
| Autoencoder | 43.5 | 24 | 0/20 | 2/20 |

![Anomaly score](docs/results/anomaly_unit.png)

## Limitations
- C-MAPSS is simulated turbofan data, not equipment data from SOCAR or any oil and gas operator
- Anomaly validation uses only 20 engines
- The alert thresholds (30/60 cycles) are illustrative. In practice they should reflect the maintenance lead time
- The first LSTM attempt collapsed to a constant prediction because the target was not scaled.
  Scaling the target to 0–1 fixed it

## Quick start
```bash
pip install -r requirements.txt
streamlit run app.py
```
To retrain the models, run `notebooks/01_cmapss_rul_anomaly.ipynb`.
