"""
model_trainer.py — train the FinGuard ensemble on PaySim1.

Pipeline:
  1. Load PaySim1 csv (≈6.36M rows).
  2. Stratified down-sample the negatives so training fits in memory while
     keeping every fraud row.
  3. Encode (one-hot for `type`), scale numeric features, project with PCA.
  4. Train an XGBoost classifier + a Keras Autoencoder anomaly detector.
  5. Ensemble = 0.7 * XGB probability + 0.3 * normalised AE reconstruction error.
  6. Persist artefacts to MODEL_DIR and write a metrics snapshot to Supabase.

Run:  python model_trainer.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import tensorflow as tf
layers = tf.keras.layers
models = tf.keras.models

from database import insert_metrics_snapshot

load_dotenv()

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PAYSIM_CSV = BASE_DIR / "data" / "PS1.csv"
MODEL_DIR = Path(os.getenv("MODEL_DIR", "./artifacts"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_VERSION = "v1.0-xgb+ae"
NUMERIC_COLS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]
TYPE_CATEGORIES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

# Data

def load_paysim(path: str) -> pd.DataFrame:
    print(f"[data] reading {path}")
    df = pd.read_csv(path)
    print(f"[data] rows={len(df):,}  fraud={int(df.isFraud.sum()):,}")
    return df


def stratified_sample(df: pd.DataFrame, neg_ratio: int = 20) -> pd.DataFrame:
    """Keep every fraud row, randomly down-sample non-fraud to neg_ratio×."""
    pos = df[df.isFraud == 1]
    neg = df[df.isFraud == 0].sample(
        n=min(len(pos) * neg_ratio, len(df) - len(pos)),
        random_state=42,
    )
    out = pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"[data] balanced sample: {len(out):,} rows  (fraud rate {out.isFraud.mean():.4f})")
    return out


def build_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    # one-hot encode `type` against a fixed vocabulary
    type_dummies = pd.get_dummies(df["type"]).reindex(columns=TYPE_CATEGORIES, fill_value=0)
    feat = pd.concat([df[NUMERIC_COLS].astype(float), type_dummies.astype(float)], axis=1)
    feature_names = feat.columns.tolist()
    return feat.values, df["isFraud"].astype(int).values, feature_names


# Models

def build_autoencoder(input_dim: int) -> tf.keras.Model:
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(16, activation="relu")(inputs)
    x = layers.Dense(8, activation="relu")(x)
    x = layers.Dense(4, activation="relu")(x)
    x = layers.Dense(8, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(input_dim, activation="linear")(x)
    ae = models.Model(inputs, outputs)
    ae.compile(optimizer="adam", loss="mse")
    return ae


# Train + evaluate

def main() -> None:
    raw = load_paysim(PAYSIM_CSV)
    sample = stratified_sample(raw, neg_ratio=20)

    X, y, feature_names = build_features(sample)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    pca = PCA(n_components=min(12, X_train_s.shape[1])).fit(X_train_s)
    X_train_p = pca.transform(X_train_s)
    X_test_p = pca.transform(X_test_s)

    print("[xgb] training XGBoost…")
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=float((y_train == 0).sum() / max((y_train == 1).sum(), 1)),
        eval_metric="auc",
        tree_method="hist",
        n_jobs=-1,
    )
    xgb.fit(X_train_p, y_train)

    print("[ae] training Autoencoder on benign-only data…")
    benign = X_train_p[y_train == 0]
    ae = build_autoencoder(X_train_p.shape[1])
    ae.fit(benign, benign, epochs=10, batch_size=512, verbose=2, validation_split=0.1)

    # ----- ensemble scoring -----
    def ensemble_proba(Xp: np.ndarray) -> np.ndarray:
        xgb_p = xgb.predict_proba(Xp)[:, 1]
        recon = ae.predict(Xp, verbose=0)
        err = np.mean((Xp - recon) ** 2, axis=1)
        err_norm = (err - err.min()) / (err.max() - err.min() + 1e-9)
        return 0.7 * xgb_p + 0.3 * err_norm

    t0 = time.perf_counter()
    proba = ensemble_proba(X_test_p)
    avg_latency_ms = (time.perf_counter() - t0) / len(X_test_p) * 1000

    preds = (proba >= 0.5).astype(int)
    metrics = {
        "model_version": MODEL_VERSION,
        "auc_roc": float(roc_auc_score(y_test, proba)),
        "precision_score": float(precision_score(y_test, preds)),
        "recall_score": float(recall_score(y_test, preds)),
        "f1_score": float(f1_score(y_test, preds)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "avg_latency_ms": float(avg_latency_ms),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "fraud_rate": float(sample.isFraud.mean()),
        "notes": "XGBoost + Autoencoder ensemble (0.7 / 0.3) on PCA-12 features",
    }
    print("[eval]", json.dumps(metrics, indent=2))

    # ----- curves for the dashboard -----
    fpr, tpr, _ = roc_curve(y_test, proba)
    pr_p, pr_r, _ = precision_recall_curve(y_test, proba)
    cm = confusion_matrix(y_test, preds).tolist()

    importances = sorted(
        zip(feature_names, xgb.feature_importances_.tolist()
            if hasattr(xgb, "feature_importances_") else [0] * len(feature_names)),
        key=lambda kv: kv[1], reverse=True,
    )

    curves = {
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr_curve": {"precision": pr_p.tolist(), "recall": pr_r.tolist()},
        "confusion_matrix": cm,
        "feature_importance": [{"feature": f, "importance": i} for f, i in importances],
        **{k: v for k, v in metrics.items() if k != "notes"},
    }

    # ----- persist -----
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    joblib.dump(pca,    MODEL_DIR / "pca.joblib")
    joblib.dump(xgb,    MODEL_DIR / "xgb.joblib")
    ae.save(MODEL_DIR / "autoencoder.keras")
    (MODEL_DIR / "feature_names.json").write_text(json.dumps(feature_names))
    (MODEL_DIR / "curves.json").write_text(json.dumps(curves))

    insert_metrics_snapshot(metrics)
    print(f"[ok] artefacts written to {MODEL_DIR}/  and metrics row pushed to Supabase.")


if __name__ == "__main__":
    main()
