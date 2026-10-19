#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predictive Maintenance for Manufacturing
========================================
Goal      : Predict when machinery is likely to fail to avoid unplanned downtime.
Dataset   : NASA C-MAPSS turbofan engine run-to-failure sensor data (FD001-FD004).
Guidelines: * Use sensor data (vibration, temperature, pressure, etc.)
            * Apply time-series forecasting / classification models
            * Techniques: Random Forest, XGBoost, LSTM

Pipeline
--------
1. Load data                (train / test / true RUL)
2. Data preprocessing       (RUL labels, constant-sensor removal, scaling,
                             rolling-window feature engineering)
3. Exploratory Data Analysis (saved as PNG files)
4. Data splitting           (engine-wise train/validation split + official test set)
5. Models                   Random Forest, XGBoost, LSTM
                            -> regression (Remaining Useful Life)
                            -> classification (failure within N cycles)
6. Evaluation               MAE, MSE, RMSE, R^2, NASA score, Accuracy,
                            Precision, Recall, F1, Confusion matrix

Usage
-----
    python predictive_maintenance.py                  # FD001 (default)
    python predictive_maintenance.py --dataset FD003
    python predictive_maintenance.py --dataset FD002 --epochs 30
"""

import argparse
import os
import random
import sys
import warnings

import matplotlib

matplotlib.use("Agg")  # headless backend: figures are saved to disk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# Optional deep-learning / boosting libraries (the script degrades gracefully
# and tells you what was skipped if one of them is not installed).
try:
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    HAS_TF = True
except ImportError:  # pragma: no cover
    HAS_TF = False

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SEED = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "CMaps")
OUT_DIR = os.path.join(BASE_DIR, "outputs")

RUL_CAP = 125          # piece-wise linear RUL: early life is "healthy" (standard in literature)
FAIL_HORIZON = 30      # classification target: will the engine fail within 30 cycles?
SEQ_LEN = 30           # LSTM look-back window (cycles)
ROLL_WIN = 5           # rolling window for tabular models
VAL_SIZE = 0.2         # fraction of ENGINES held out for validation

COLUMNS = (
    ["unit", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"s{i}" for i in range(1, 22)]
)
SETTINGS = ["setting_1", "setting_2", "setting_3"]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if HAS_TF:
        tf.random.set_seed(seed)


# --------------------------------------------------------------------------- #
# 1. Data loading
# --------------------------------------------------------------------------- #
def read_cmapss(path: str) -> pd.DataFrame:
    """Read a whitespace-separated C-MAPSS file (26 numeric columns)."""
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, : len(COLUMNS)]
    df.columns = COLUMNS
    return df


def load_data(dataset: str):
    train = read_cmapss(os.path.join(DATA_DIR, f"train_{dataset}.txt"))
    test = read_cmapss(os.path.join(DATA_DIR, f"test_{dataset}.txt"))
    rul = pd.read_csv(
        os.path.join(DATA_DIR, f"RUL_{dataset}.txt"), sep=r"\s+", header=None
    )
    rul.columns = ["true_rul"]
    rul["unit"] = np.arange(1, len(rul) + 1)  # RUL file rows are in engine order
    return train, test, rul


# --------------------------------------------------------------------------- #
# 2. Preprocessing
# --------------------------------------------------------------------------- #
def add_rul(train: pd.DataFrame) -> pd.DataFrame:
    """RUL = (last cycle of the engine) - (current cycle); capped at RUL_CAP."""
    train = train.copy()
    max_cycle = train.groupby("unit")["cycle"].transform("max")
    train["RUL_raw"] = max_cycle - train["cycle"]
    train["RUL"] = train["RUL_raw"].clip(upper=RUL_CAP)
    return train


def find_constant_columns(train: pd.DataFrame, cols, tol: float = 1e-8):
    """Columns with (almost) zero variance carry no information."""
    return [c for c in cols if train[c].std() <= tol]


class ConditionScaler:
    """
    Min-max scaler fitted on the training set only.

    * Single operating condition (FD001 / FD003): one global scaler.
    * Six operating conditions (FD002 / FD004): sensors are scaled separately
      within each operating-condition cluster (KMeans on the 3 settings),
      which removes the strong effect of the operating regime on sensor values.
    """

    def __init__(self, feature_cols, multi_condition: bool, n_clusters: int = 6):
        self.feature_cols = list(feature_cols)
        self.multi_condition = multi_condition
        self.n_clusters = n_clusters
        self.kmeans = None
        self.scalers = {}

    def _cluster(self, df: pd.DataFrame) -> np.ndarray:
        if not self.multi_condition:
            return np.zeros(len(df), dtype=int)
        return self.kmeans.predict(df[SETTINGS].values)

    def fit(self, df: pd.DataFrame):
        if self.multi_condition:
            self.kmeans = KMeans(
                n_clusters=self.n_clusters, n_init=10, random_state=SEED
            ).fit(df[SETTINGS].values)
        labels = self._cluster(df)
        for c in np.unique(labels):
            sc = MinMaxScaler()
            sc.fit(df.loc[labels == c, self.feature_cols].values)
            self.scalers[c] = sc
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        labels = self._cluster(df)
        values = np.zeros((len(df), len(self.feature_cols)), dtype=np.float64)
        for c in np.unique(labels):
            mask = labels == c
            if c in self.scalers:
                values[mask] = self.scalers[c].transform(
                    df.loc[mask, self.feature_cols].values
                )
        df[self.feature_cols] = values
        return df


def add_rolling_features(df: pd.DataFrame, sensor_cols, win: int = ROLL_WIN):
    """
    Per-engine rolling mean and std of every sensor (time-series features for
    the tabular models). Uses only past/current cycles -> no look-ahead leakage.
    """
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    g = df.groupby("unit")[sensor_cols]
    mean = g.rolling(win, min_periods=1).mean().reset_index(level=0, drop=True)
    std = (
        g.rolling(win, min_periods=1).std().reset_index(level=0, drop=True).fillna(0.0)
    )
    mean.columns = [f"{c}_mean{win}" for c in sensor_cols]
    std.columns = [f"{c}_std{win}" for c in sensor_cols]
    return pd.concat([df, mean.sort_index(), std.sort_index()], axis=1)


def make_sequences(df: pd.DataFrame, feature_cols, seq_len: int, target_col=None):
    """
    Build LSTM windows of shape (n_samples, seq_len, n_features).
    One window ends at EVERY cycle of every engine. The start of each engine's
    history is left-padded by repeating its first row, so early cycles also get
    a full-length window and no training rows are lost.
    """
    X, y, units = [], [], []
    for unit, g in df.groupby("unit"):
        g = g.sort_values("cycle")
        feats = g[feature_cols].values.astype(np.float32)
        n = len(feats)
        padded = np.vstack([np.repeat(feats[:1], seq_len - 1, axis=0), feats])
        tgt = g[target_col].values.astype(np.float32) if target_col else None
        for i in range(n):  # window i covers original rows i-seq_len+1 .. i
            X.append(padded[i : i + seq_len])
            if tgt is not None:
                y.append(tgt[i])
            units.append(unit)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32) if target_col else None
    return X, y, np.asarray(units)


def last_window_per_unit(df: pd.DataFrame, feature_cols, seq_len: int):
    """Only the final window of each engine (official C-MAPSS test protocol)."""
    X, units = [], []
    for unit, g in df.groupby("unit"):
        g = g.sort_values("cycle")
        feats = g[feature_cols].values.astype(np.float32)
        if len(feats) < seq_len:
            pad = np.repeat(feats[:1], seq_len - len(feats), axis=0)
            feats = np.vstack([pad, feats])
        X.append(feats[-seq_len:])
        units.append(unit)
    return np.asarray(X, dtype=np.float32), np.asarray(units)


# --------------------------------------------------------------------------- #
# 3. Exploratory Data Analysis
# --------------------------------------------------------------------------- #
def run_eda(train_raw: pd.DataFrame, test_raw: pd.DataFrame, dataset: str,
            sensor_cols, const_cols):
    os.makedirs(OUT_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")

    print("\n" + "=" * 70)
    print(f"EXPLORATORY DATA ANALYSIS  ({dataset})")
    print("=" * 70)
    print(f"Train shape : {train_raw.shape}  |  engines: {train_raw['unit'].nunique()}")
    print(f"Test  shape : {test_raw.shape}  |  engines: {test_raw['unit'].nunique()}")
    print(f"Missing values (train): {int(train_raw.isna().sum().sum())}")
    print(f"Duplicate rows (train): {int(train_raw.duplicated().sum())}")
    print("\nFirst 5 rows:")
    print(train_raw.head().to_string())
    print("\nDescriptive statistics (settings + sensors):")
    print(train_raw[SETTINGS + sensor_cols].describe().T.round(3).to_string())

    life = train_raw.groupby("unit")["cycle"].max()
    print("\nEngine lifetime (cycles) summary:")
    print(life.describe().round(2).to_string())
    print(f"\nConstant / zero-variance columns removed later: {const_cols}")

    # (a) Engine lifetime distribution
    plt.figure(figsize=(8, 4.5))
    sns.histplot(life, bins=25, kde=True, color="steelblue")
    plt.xlabel("Cycles until failure")
    plt.ylabel("Number of engines")
    plt.title(f"{dataset}: distribution of engine lifetimes (train)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "eda_01_lifetime_distribution.png"), dpi=130)
    plt.close()

    # (b) Sensor variance (normalised) - shows uninformative sensors
    norm_std = (train_raw[sensor_cols].std() / (train_raw[sensor_cols].mean().abs() + 1e-9))
    plt.figure(figsize=(10, 4.5))
    norm_std.sort_values().plot(kind="bar", color="darkorange")
    plt.ylabel("Coefficient of variation (std / |mean|)")
    plt.title(f"{dataset}: relative variability of each sensor")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "eda_02_sensor_variability.png"), dpi=130)
    plt.close()

    # (c) Sensor trajectories vs. cycles for a few engines
    informative = [c for c in sensor_cols if c not in const_cols]
    show = informative[:6] if len(informative) >= 6 else informative
    units = sorted(train_raw["unit"].unique())[:5]
    fig, axes = plt.subplots(len(show), 1, figsize=(10, 2.2 * len(show)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, show):
        for u in units:
            d = train_raw[train_raw["unit"] == u]
            ax.plot(d["cycle"], d[s], lw=1, label=f"engine {u}")
        ax.set_ylabel(s)
    axes[0].legend(ncol=5, fontsize=7, loc="upper right")
    axes[-1].set_xlabel("Cycle")
    fig.suptitle(f"{dataset}: sensor degradation trends (5 engines)", y=1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "eda_03_sensor_trends.png"), dpi=130)
    plt.close()

    # (d) Correlation of sensors with RUL
    tmp = add_rul(train_raw)
    corr_cols = informative + ["RUL"]
    corr = tmp[corr_cols].corr()
    plt.figure(figsize=(11, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f",
                annot_kws={"size": 6}, square=True, cbar_kws={"shrink": 0.7})
    plt.title(f"{dataset}: correlation matrix (informative sensors + RUL)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "eda_04_correlation_heatmap.png"), dpi=130)
    plt.close()

    rul_corr = corr["RUL"].drop("RUL").sort_values(key=np.abs, ascending=False)
    print("\nTop sensors correlated with RUL:")
    print(rul_corr.head(8).round(3).to_string())

    # (e) RUL target distribution
    plt.figure(figsize=(8, 4.5))
    sns.histplot(tmp["RUL"], bins=40, color="seagreen")
    plt.xlabel(f"RUL (capped at {RUL_CAP})")
    plt.title(f"{dataset}: target (RUL) distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "eda_05_rul_distribution.png"), dpi=130)
    plt.close()

    # (f) Operating settings
    plt.figure(figsize=(7, 5))
    plt.scatter(train_raw["setting_1"], train_raw["setting_2"], s=3, alpha=0.3)
    plt.xlabel("setting_1")
    plt.ylabel("setting_2")
    plt.title(f"{dataset}: operating conditions")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "eda_06_operating_conditions.png"), dpi=130)
    plt.close()
    print(f"\nEDA figures saved to: {OUT_DIR}")


# --------------------------------------------------------------------------- #
# 6. Evaluation helpers
# --------------------------------------------------------------------------- #
def nasa_score(y_true, y_pred) -> float:
    """Asymmetric PHM08 score: late predictions are penalised more than early."""
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def regression_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
        "NASA_score": nasa_score(y_true, y_pred),
    }


def classification_metrics(y_true, y_pred) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }


def print_block(title: str, metrics: dict) -> None:
    print(f"  {title}")
    for k, v in metrics.items():
        print(f"    {k:<11}: {v:,.4f}")


def plot_pred_vs_true(y_true, y_pred, title: str, fname: str) -> None:
    lim = max(float(np.max(y_true)), float(np.max(y_pred))) * 1.05
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(y_true, y_pred, s=18, alpha=0.7, edgecolor="k", linewidth=0.3)
    plt.plot([0, lim], [0, lim], "r--", label="perfect prediction")
    plt.xlim(0, lim)
    plt.ylim(0, lim)
    plt.xlabel("Actual RUL")
    plt.ylabel("Predicted RUL")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=130)
    plt.close()


def plot_confusion(cm, title: str, fname: str) -> None:
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Healthy", "Fail soon"],
                yticklabels=["Healthy", "Fail soon"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=130)
    plt.close()


# --------------------------------------------------------------------------- #
# 5. Models
# --------------------------------------------------------------------------- #
def build_lstm(seq_len: int, n_features: int):
    model = keras.Sequential(
        [
            layers.Input(shape=(seq_len, n_features)),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse",
                  metrics=["mae"])
    return model


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Predictive maintenance (C-MAPSS)")
    parser.add_argument("--dataset", default="FD001",
                        choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--epochs", type=int, default=40, help="LSTM max epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="LSTM batch size")
    parser.add_argument("--n_estimators", type=int, default=300,
                        help="trees for Random Forest / XGBoost")
    args = parser.parse_args()

    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    dataset = args.dataset
    multi_condition = dataset in ("FD002", "FD004")

    print("=" * 70)
    print(f"PREDICTIVE MAINTENANCE FOR MANUFACTURING  |  dataset = {dataset}")
    print("=" * 70)
    if not HAS_XGB:
        print("[!] xgboost not installed -> XGBoost models skipped (pip install xgboost)")
    if not HAS_TF:
        print("[!] tensorflow not installed -> LSTM skipped (pip install tensorflow)")

    # ---------------- 1. load ---------------- #
    train_raw, test_raw, rul_true = load_data(dataset)
    sensor_cols = [f"s{i}" for i in range(1, 22)]

    # ---------------- 3. EDA (on raw data) ---------------- #
    const_cols = find_constant_columns(train_raw, SETTINGS + sensor_cols) \
        if not multi_condition else find_constant_columns(train_raw, sensor_cols)
    run_eda(train_raw, test_raw, dataset, sensor_cols, const_cols)

    # ---------------- 2. preprocessing ---------------- #
    print("\n" + "=" * 70)
    print("DATA PREPROCESSING")
    print("=" * 70)
    train = add_rul(train_raw)

    # Operating settings are inputs for single-condition sets only if they vary;
    # for multi-condition sets they are used to normalise sensors per regime.
    candidate_cols = sensor_cols + (SETTINGS if not multi_condition else [])
    feature_cols = [c for c in candidate_cols if c not in const_cols]
    print(f"Dropped constant columns : {const_cols}")
    print(f"Features used ({len(feature_cols)}): {feature_cols}")

    scaler = ConditionScaler(feature_cols, multi_condition).fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test_raw)

    # time-series features for tabular models
    sensors_for_roll = [c for c in feature_cols if c.startswith("s")]
    train_fe = add_rolling_features(train, sensors_for_roll)
    test_fe = add_rolling_features(test, sensors_for_roll)
    tab_cols = (
        feature_cols
        + [f"{c}_mean{ROLL_WIN}" for c in sensors_for_roll]
        + [f"{c}_std{ROLL_WIN}" for c in sensors_for_roll]
    )
    print(f"Tabular feature count (raw + rolling mean/std): {len(tab_cols)}")

    # classification target: failure within FAIL_HORIZON cycles (true RUL, not capped)
    train_fe["FAIL_SOON"] = (train_fe["RUL_raw"] <= FAIL_HORIZON).astype(int)
    print(f"Classification target: RUL <= {FAIL_HORIZON} cycles -> 'Fail soon'")
    print("Class balance (train rows):")
    print(train_fe["FAIL_SOON"].value_counts(normalize=True).round(3).to_string())

    # ---------------- 4. splitting ---------------- #
    print("\n" + "=" * 70)
    print("DATA SPLITTING")
    print("=" * 70)
    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=SEED)
    tr_idx, va_idx = next(gss.split(train_fe, groups=train_fe["unit"]))
    tr_df = train_fe.iloc[tr_idx].reset_index(drop=True)
    va_df = train_fe.iloc[va_idx].reset_index(drop=True)
    print(f"Train      : {tr_df['unit'].nunique()} engines, {len(tr_df)} rows")
    print(f"Validation : {va_df['unit'].nunique()} engines, {len(va_df)} rows "
          "(whole engines held out -> no leakage)")

    # official test set: last observed cycle of each engine vs. true RUL
    test_last = test_fe.sort_values(["unit", "cycle"]).groupby("unit").tail(1)
    test_last = test_last.merge(rul_true, on="unit", how="left")
    y_test_reg = test_last["true_rul"].clip(upper=RUL_CAP).values
    y_test_cls = (test_last["true_rul"].values <= FAIL_HORIZON).astype(int)
    print(f"Test       : {len(test_last)} engines (official NASA test set, last cycle)")

    X_tr, X_va, X_te = tr_df[tab_cols].values, va_df[tab_cols].values, test_last[tab_cols].values
    y_tr, y_va = tr_df["RUL"].values, va_df["RUL"].values
    c_tr, c_va = tr_df["FAIL_SOON"].values, va_df["FAIL_SOON"].values

    # ---------------- 5. models ---------------- #
    print("\n" + "=" * 70)
    print("MODEL TRAINING")
    print("=" * 70)
    reg_results = []   # one row per (model, split)
    cls_results = []
    preds_test = {}

    def record_reg(name, split, y_true, y_pred):
        m = regression_metrics(y_true, y_pred)
        reg_results.append({"Model": name, "Split": split, **m})
        print_block(f"[{name}] {split} - regression", m)
        # failure-within-horizon accuracy derived from the predicted RUL
        return m

    def record_cls(name, split, y_true, y_pred):
        m = classification_metrics(y_true, y_pred)
        cls_results.append({"Model": name, "Split": split, **m})
        print_block(f"[{name}] {split} - classification", m)
        return m

    # ---- Random Forest (regression + classification) ---- #
    print("\n>>> Random Forest")
    rf_reg = RandomForestRegressor(
        n_estimators=args.n_estimators, max_depth=18, min_samples_leaf=3,
        max_features="sqrt", n_jobs=-1, random_state=SEED,
    ).fit(X_tr, y_tr)
    p_va, p_te = rf_reg.predict(X_va), rf_reg.predict(X_te)
    record_reg("Random Forest", "Validation (all cycles)", y_va, p_va)
    record_reg("Random Forest", "Test (last cycle / engine)", y_test_reg, p_te)
    preds_test["Random Forest"] = p_te
    record_cls("Random Forest (derived from RUL)", "Test",
               y_test_cls, (p_te <= FAIL_HORIZON).astype(int))

    rf_cls = RandomForestClassifier(
        n_estimators=args.n_estimators, max_depth=18, min_samples_leaf=3,
        max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=SEED,
    ).fit(X_tr, c_tr)
    record_cls("Random Forest Classifier", "Validation (all cycles)", c_va, rf_cls.predict(X_va))
    rf_cls_te = rf_cls.predict(X_te)
    record_cls("Random Forest Classifier", "Test (last cycle / engine)", y_test_cls, rf_cls_te)
    cm_rf = confusion_matrix(y_test_cls, rf_cls_te, labels=[0, 1])
    plot_confusion(cm_rf, "Random Forest - test", "cm_random_forest.png")

    # feature importance (RF)
    imp = pd.Series(rf_reg.feature_importances_, index=tab_cols).sort_values(ascending=False)
    plt.figure(figsize=(8, 6))
    imp.head(15)[::-1].plot(kind="barh", color="teal")
    plt.title("Random Forest - top 15 feature importances")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rf_feature_importance.png"), dpi=130)
    plt.close()

    # ---- XGBoost (regression + classification) ---- #
    if HAS_XGB:
        print("\n>>> XGBoost")
        xgb_reg = XGBRegressor(
            n_estimators=args.n_estimators, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            objective="reg:squarederror", random_state=SEED, n_jobs=-1,
        )
        xgb_reg.fit(X_tr, y_tr)
        p_va, p_te = xgb_reg.predict(X_va), xgb_reg.predict(X_te)
        record_reg("XGBoost", "Validation (all cycles)", y_va, p_va)
        record_reg("XGBoost", "Test (last cycle / engine)", y_test_reg, p_te)
        preds_test["XGBoost"] = p_te
        record_cls("XGBoost (derived from RUL)", "Test",
                   y_test_cls, (p_te <= FAIL_HORIZON).astype(int))

        pos_weight = float((c_tr == 0).sum()) / max(float((c_tr == 1).sum()), 1.0)
        xgb_cls = XGBClassifier(
            n_estimators=args.n_estimators, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pos_weight,
            eval_metric="logloss", random_state=SEED, n_jobs=-1,
        )
        xgb_cls.fit(X_tr, c_tr)
        record_cls("XGBoost Classifier", "Validation (all cycles)", c_va, xgb_cls.predict(X_va))
        xgb_cls_te = xgb_cls.predict(X_te)
        record_cls("XGBoost Classifier", "Test (last cycle / engine)", y_test_cls, xgb_cls_te)
        cm_x = confusion_matrix(y_test_cls, xgb_cls_te, labels=[0, 1])
        plot_confusion(cm_x, "XGBoost - test", "cm_xgboost.png")

    # ---- LSTM (sequence regression) ---- #
    if HAS_TF:
        print("\n>>> LSTM")
        keras.backend.clear_session()
        set_seed(SEED)
        Xs_tr, ys_tr, _ = make_sequences(tr_df, feature_cols, SEQ_LEN, "RUL")
        Xs_va, ys_va, _ = make_sequences(va_df, feature_cols, SEQ_LEN, "RUL")
        test_scaled_sorted = test.sort_values(["unit", "cycle"])
        Xs_te, units_te = last_window_per_unit(test_scaled_sorted, feature_cols, SEQ_LEN)
        # align with the order of `test_last` (sorted by unit)
        order = pd.Series(np.arange(len(units_te)), index=units_te).loc[test_last["unit"].values].values
        Xs_te = Xs_te[order]
        print(f"LSTM input shapes  train={Xs_tr.shape}  val={Xs_va.shape}  test={Xs_te.shape}")

        lstm = build_lstm(SEQ_LEN, len(feature_cols))
        lstm.summary()
        cbs = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                          restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                                              min_lr=1e-5),
        ]
        hist = lstm.fit(Xs_tr, ys_tr, validation_data=(Xs_va, ys_va),
                        epochs=args.epochs, batch_size=args.batch_size,
                        callbacks=cbs, verbose=2)

        plt.figure(figsize=(7, 4.5))
        plt.plot(hist.history["loss"], label="train loss (MSE)")
        plt.plot(hist.history["val_loss"], label="validation loss (MSE)")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title("LSTM training curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "lstm_training_curve.png"), dpi=130)
        plt.close()

        p_va = lstm.predict(Xs_va, batch_size=1024, verbose=0).ravel()
        p_te = lstm.predict(Xs_te, batch_size=1024, verbose=0).ravel()
        p_va, p_te = np.clip(p_va, 0, RUL_CAP), np.clip(p_te, 0, RUL_CAP)
        record_reg("LSTM", "Validation (all cycles)", ys_va, p_va)
        record_reg("LSTM", "Test (last cycle / engine)", y_test_reg, p_te)
        preds_test["LSTM"] = p_te
        lstm_cls_te = (p_te <= FAIL_HORIZON).astype(int)
        record_cls("LSTM (derived from RUL)", "Test", y_test_cls, lstm_cls_te)
        cm_l = confusion_matrix(y_test_cls, lstm_cls_te, labels=[0, 1])
        plot_confusion(cm_l, "LSTM - test", "cm_lstm.png")

    # ---------------- 6. evaluation summary ---------------- #
    print("\n" + "=" * 70)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 70)
    reg_df = pd.DataFrame(reg_results)
    cls_df = pd.DataFrame(cls_results)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\nRegression metrics (Remaining Useful Life, in cycles):")
    print(reg_df.round(4).to_string(index=False))
    print(f"\nClassification metrics (failure within {FAIL_HORIZON} cycles):")
    print(cls_df.round(4).to_string(index=False))

    reg_df.to_csv(os.path.join(OUT_DIR, f"regression_metrics_{dataset}.csv"), index=False)
    cls_df.to_csv(os.path.join(OUT_DIR, f"classification_metrics_{dataset}.csv"), index=False)

    for name, p in preds_test.items():
        plot_pred_vs_true(y_test_reg, p, f"{name} - test set ({dataset})",
                          f"pred_vs_actual_{name.lower().replace(' ', '_')}.png")

    test_df = reg_df[reg_df["Split"].str.startswith("Test")].set_index("Model")
    if len(test_df):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
            test_df[metric].plot(kind="bar", ax=ax, color="slateblue")
            ax.set_title(f"Test {metric}")
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=20)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "model_comparison.png"), dpi=130)
        plt.close()
        best = test_df["RMSE"].idxmin()
        print(f"\nBest model on the official test set (lowest RMSE): {best}")

    print(f"\nAll metrics (CSV) and figures (PNG) saved in: {OUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    sys.exit(main())
