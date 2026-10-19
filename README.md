# Predictive Maintenance for Manufacturing

**Goal:** predict when machinery is likely to fail to avoid unplanned downtime.

The project uses the NASA **C-MAPSS turbofan engine** run-to-failure sensor dataset
(temperature, pressure, fan/core speed, etc. — 21 sensors + 3 operating settings per cycle)
and predicts each engine's **Remaining Useful Life (RUL)**.

## Techniques (as per project guidelines)
| Guideline | Implementation |
|---|---|
| Use sensor data | 21 sensors + operating settings, constant sensors removed, min-max scaled |
| Time-series forecasting / classification | RUL regression + "fails within 30 cycles" classification |
| Random Forest | `RandomForestRegressor` + `RandomForestClassifier` |
| XGBoost | `XGBRegressor` + `XGBClassifier` |
| LSTM | 2-layer LSTM on 30-cycle sliding windows (Keras) |

## Pipeline (`predictive_maintenance.py`)
1. **Data loading** – train / test / true-RUL files.
2. **Preprocessing** – RUL labels (capped at 125), zero-variance column removal, min-max scaling fitted on train only
   (per operating regime via KMeans for FD002/FD004), per-engine rolling mean/std features.
3. **EDA** – statistics printed + 6 figures saved to `outputs/`.
4. **Splitting** – engine-wise 80/20 train/validation split (no leakage between cycles of one engine) + the official NASA test set (last cycle of every test engine vs. true RUL).
5. **Models** – Random Forest, XGBoost, LSTM.
6. **Evaluation** – MAE, MSE, RMSE, R², NASA PHM08 score; Accuracy, Precision, Recall, F1, confusion matrices. Metrics saved as CSV, plots as PNG.

## Run
```bash
pip install -r requirements.txt
python predictive_maintenance.py                  # FD001 (default)
python predictive_maintenance.py --dataset FD003  # FD001 | FD002 | FD003 | FD004
python predictive_maintenance.py --epochs 60 --n_estimators 500
```
Results are written to `outputs/`.

## Structure
```
predictive_maintenance/
├── predictive_maintenance.py
├── requirements.txt
├── README.md
└── data/CMaps/        # train_FD00x.txt, test_FD00x.txt, RUL_FD00x.txt, readme, paper PDF
```
Dataset: A. Saxena et al., *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*, PHM08.
Dataset Link = "https://www.kaggle.com/datasets/behrad3d/nasa-cmaps"
