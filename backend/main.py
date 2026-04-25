"""
main.py — FastAPI service for FinGuard real-time fraud scoring.

Endpoints:
  GET  /health          → service + model status
  GET  /metrics         → cached training metrics + curves for the dashboard
  POST /score           → score a single transaction
  POST /score/batch     → score many transactions (used by CSV upload)
  GET  /predictions/recent  → last 25 stored predictions

Run:  uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Deque

import joblib
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import tensorflow as tf
load_model = tf.keras.models.load_model

from database import (
    insert_predictions,
    insert_transactions,
    latest_metrics,
    new_batch_id,
    recent_predictions,
)
from model_trainer import NUMERIC_COLS, TYPE_CATEGORIES

load_dotenv()
MODEL_DIR = Path(os.getenv("MODEL_DIR", "./artifacts"))

app = FastAPI(title="FinGuard", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class Transaction(BaseModel):
    step: int = Field(ge=0)
    type: str
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float


class BatchRequest(BaseModel):
    transactions: list[Transaction]


# ---------------------------------------------------------------------------
# Model loading (lazy + tolerant of missing artefacts)
# ---------------------------------------------------------------------------

_state: dict = {"loaded": False, "scaler": None, "pca": None, "xgb": None, "ae": None}
_latencies: Deque[float] = deque(maxlen=200)


def _try_load_models() -> None:
    if _state["loaded"]:
        return
    try:
        _state["scaler"] = joblib.load(MODEL_DIR / "scaler.joblib")
        _state["pca"]    = joblib.load(MODEL_DIR / "pca.joblib")
        _state["xgb"]    = joblib.load(MODEL_DIR / "xgb.joblib")
        _state["ae"]     = load_model(MODEL_DIR / "autoencoder.keras", compile=False)
        _state["loaded"] = True
        print("[model] artefacts loaded from", MODEL_DIR)
    except Exception as e:  # noqa: BLE001
        print(f"[model] not loaded yet ({e}). Train with: python model_trainer.py")


@app.on_event("startup")
def _startup() -> None:
    _try_load_models()


# ---------------------------------------------------------------------------
# Feature engineering — must match model_trainer.build_features
# ---------------------------------------------------------------------------

def _vectorise(txs: list[Transaction]) -> np.ndarray:
    rows = []
    for t in txs:
        numeric = [
            t.step,
            t.amount,
            t.oldbalanceOrg,
            t.newbalanceOrig,
            t.oldbalanceDest,
            t.newbalanceDest,
        ]
        one_hot = [1.0 if t.type == c else 0.0 for c in TYPE_CATEGORIES]
        rows.append(numeric + one_hot)
    assert len(NUMERIC_COLS) == 6
    return np.asarray(rows, dtype=float)


def _ensemble(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (fraud_probability, anomaly_score)."""
    Xs = _state["scaler"].transform(X)
    Xp = _state["pca"].transform(Xs)
    xgb_p = _state["xgb"].predict_proba(Xp)[:, 1]
    recon = _state["ae"].predict(Xp, verbose=0)
    err = np.mean((Xp - recon) ** 2, axis=1)
    err_norm = (err - err.min()) / (err.max() - err.min() + 1e-9) if len(err) > 1 else np.zeros_like(err)
    proba = 0.7 * xgb_p + 0.3 * err_norm
    return proba, err


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    _try_load_models()
    avg = float(np.mean(_latencies)) if _latencies else 0.0
    m = latest_metrics() or {}
    return {
        "status": "ok",
        "model_loaded": _state["loaded"],
        "model_version": m.get("model_version", "untrained"),
        "avg_latency_ms": round(avg, 2),
        "f1_score": m.get("f1_score", 0.0),
    }


@app.get("/metrics")
def metrics() -> dict:
    curves_path = MODEL_DIR / "curves.json"
    if not curves_path.exists():
        raise HTTPException(404, "Metrics not available — train the model first.")
    return json.loads(curves_path.read_text())


@app.post("/score")
def score_one(tx: Transaction) -> dict:
    _try_load_models()
    if not _state["loaded"]:
        raise HTTPException(503, "Model not loaded — run model_trainer.py first.")

    t0 = time.perf_counter()
    X = _vectorise([tx])
    proba, anomaly = _ensemble(X)
    latency_ms = (time.perf_counter() - t0) * 1000
    _latencies.append(latency_ms)

    batch_id = new_batch_id()
    tx_ids = insert_transactions([tx.model_dump()], batch_id)
    is_fraud = bool(proba[0] >= 0.5)
    insert_predictions([{
        "transaction_id": tx_ids[0],
        "batch_id": batch_id,
        "fraud_probability": float(proba[0]),
        "anomaly_score": float(anomaly[0]),
        "is_fraud": is_fraud,
        "latency_ms": float(latency_ms),
    }])
    return {
        "transaction_id": tx_ids[0],
        "fraud_probability": float(proba[0]),
        "anomaly_score": float(anomaly[0]),
        "is_fraud": is_fraud,
        "latency_ms": round(latency_ms, 2),
    }


@app.post("/score/batch")
def score_batch(req: BatchRequest) -> dict:
    _try_load_models()
    if not _state["loaded"]:
        raise HTTPException(503, "Model not loaded — run model_trainer.py first.")
    if not req.transactions:
        return {"results": []}

    t0 = time.perf_counter()
    X = _vectorise(req.transactions)
    proba, anomaly = _ensemble(X)
    total_ms = (time.perf_counter() - t0) * 1000
    per_row_ms = total_ms / len(req.transactions)
    _latencies.extend([per_row_ms] * len(req.transactions))

    batch_id = new_batch_id()
    payload = [t.model_dump() for t in req.transactions]
    tx_ids = insert_transactions(payload, batch_id)

    pred_rows = []
    results = []
    for tx_id, p, a in zip(tx_ids, proba, anomaly):
        is_fraud = bool(p >= 0.5)
        pred_rows.append({
            "transaction_id": tx_id,
            "batch_id": batch_id,
            "fraud_probability": float(p),
            "anomaly_score": float(a),
            "is_fraud": is_fraud,
            "latency_ms": float(per_row_ms),
        })
        results.append({
            "transaction_id": tx_id,
            "fraud_probability": float(p),
            "anomaly_score": float(a),
            "is_fraud": is_fraud,
            "latency_ms": round(per_row_ms, 2),
        })
    insert_predictions(pred_rows)

    return {
        "batch_id": batch_id,
        "count": len(results),
        "fraud_count": sum(1 for r in results if r["is_fraud"]),
        "avg_latency_ms": round(per_row_ms, 2),
        "results": results,
    }


@app.get("/predictions/recent")
def predictions_recent(limit: int = 25) -> dict:
    return {"items": recent_predictions(limit=limit)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=False,
    )
