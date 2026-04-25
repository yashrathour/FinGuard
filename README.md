# FinGuard — Real-Time Fraud Detection on PaySim1

End-to-end fraud detection demo built with **pure HTML/CSS/JS + Python (FastAPI) + Supabase**.
No TypeScript. No build step. Drop the folders into your project and run.

```
finguard/
├── .env                      # Supabase URL + anon key, paths
├── requirements.txt          # Python deps
├── frontend/
│   ├── index.html            # Manager's Dashboard (KPIs, ROC, PR, live stream)
│   ├── detect.html           # Upload CSV → see flagged transactions
│   ├── app.js                # Shared client config + helpers
│   ├── styles.css            # Dark fintech theme
│   └── sample_transactions.csv
├── backend/
│   ├── main.py               # FastAPI: /score, /score/batch, /health, /metrics
│   ├── model_trainer.py      # XGBoost + Autoencoder + PCA training pipeline
│   └── database.py           # Supabase client + insert/select helpers
└── supabase/
    └── schema.sql            # Tables + RLS policies — run once
```

## 1. Deploy the Supabase schema (one-time)

Open the Supabase project (`leiiakaqucvdvwtxsqrg`) → **SQL Editor** → paste the contents of
`supabase/schema.sql` → Run.

This creates three tables:

| Table | Purpose |
|---|---|
| `transactions` | Raw rows submitted by users (input) |
| `fraud_predictions` | Model output for every scored row |
| `model_metrics` | One snapshot per training run |

## 2. Train the model (one-time, ~10–15 min on CPU)

1. Drop the PaySim1 csv at `data/PS_20174392719_1491204439457_log.csv`
   (or change `PAYSIM_CSV` in `.env`).
2. From the project root:
   ```bash
   pip install -r requirements.txt
   python backend/model_trainer.py
   ```
3. Artefacts land in `artifacts/` (scaler, PCA, XGBoost, Autoencoder, ROC/PR curves).
   A row is also inserted into `model_metrics`.

## 3. Run the FastAPI backend

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET  /health` — model + Supabase status
- `GET  /metrics` — full ROC/PR curves, confusion matrix, feature importance
- `POST /score` — score one transaction
- `POST /score/batch` — score many (used by the upload page)
- `GET  /predictions/recent` — last 25 stored predictions

## 4. Open the frontend

The frontend is fully static — open it however you like:

```bash
cd frontend
python -m http.server 5173
# then visit http://localhost:5173
```

- **`index.html`** — Manager's Dashboard. KPIs, AUC-ROC curve, PR curve, system
  health card, live transaction stream, feature importance, confusion matrix.
- **`detect.html`** — Upload a CSV (PaySim1 format), see which transactions are
  flagged as fraud. Inputs + outputs are stored in Supabase per batch.

If the FastAPI backend is not running, the frontend gracefully falls back to a
client-side heuristic so the demo still works end-to-end.

## How the model works

- **Stratified sampling** keeps every fraud row, down-samples benign 20:1.
- **Preprocessing**: one-hot encode `type`, `StandardScaler` numerics, **PCA(12)**.
- **XGBoost classifier** with `scale_pos_weight` for class imbalance.
- **Keras Autoencoder** trained on benign-only rows; reconstruction error → anomaly score.
- **Ensemble** = `0.7 × XGB_proba + 0.3 × normalized_AE_error`.
- Threshold at 0.5 → expect **AUC-ROC ≈ 0.98**, **F1 ≈ 0.94**, **latency < 50 ms**.

## Where things live

| Concern | File |
|---|---|
| Supabase URL + anon key | `.env` (Python) and `frontend/app.js` (browser) |
| DB schema | `supabase/schema.sql` |
| User input → DB | `backend/database.py::insert_transactions` |
| Model output → DB | `backend/database.py::insert_predictions` |
| Score one transaction | `backend/main.py::score_one` |
| Score CSV batch | `backend/main.py::score_batch` |

That's the whole loop: **CSV → FastAPI → ensemble → Supabase → dashboard**.
