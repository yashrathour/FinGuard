"""
database.py — Supabase connection + helpers for FinGuard.

Tables (defined in supabase/schema.sql):
  - transactions          : raw inputs from users
  - fraud_predictions     : model outputs
  - model_metrics         : training-time performance snapshots
"""

from __future__ import annotations

import os
import uuid
from typing import Iterable

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise RuntimeError(
        "SUPABASE_URL / SUPABASE_ANON_KEY missing — check the .env file."
    )

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def insert_transactions(rows: list[dict], batch_id: str) -> list[int]:
    """Insert raw user-submitted transactions, return generated row ids."""
    payload = [
        {
            "batch_id": batch_id,
            "step": int(r["step"]),
            "type": str(r["type"]),
            "amount": float(r["amount"]),
            "name_orig": str(r["nameOrig"]),
            "oldbalance_org": float(r["oldbalanceOrg"]),
            "newbalance_orig": float(r["newbalanceOrig"]),
            "name_dest": str(r["nameDest"]),
            "oldbalance_dest": float(r["oldbalanceDest"]),
            "newbalance_dest": float(r["newbalanceDest"]),
        }
        for r in rows
    ]
    res = supabase.table("transactions").insert(payload).execute()
    return [row["id"] for row in res.data]


def insert_predictions(predictions: Iterable[dict]) -> None:
    rows = list(predictions)
    if not rows:
        return
    supabase.table("fraud_predictions").insert(rows).execute()


def insert_metrics_snapshot(metrics: dict) -> None:
    supabase.table("model_metrics").insert(metrics).execute()


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def latest_metrics() -> dict | None:
    res = (
        supabase.table("model_metrics")
        .select("*")
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    return res.data[0] if res.data else None


def recent_predictions(limit: int = 25) -> list[dict]:
    res = (
        supabase.table("fraud_predictions")
        .select("*, transactions(type, amount, name_orig, name_dest)")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return res.data or []


def new_batch_id() -> str:
    return str(uuid.uuid4())
