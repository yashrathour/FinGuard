# FinGuard — AI Fraud Detection System

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1.1-orange?style=flat)](https://xgboost.readthedocs.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17.0-FF6F00?style=flat&logo=tensorflow)](https://tensorflow.org)
[![Supabase](https://img.shields.io/badge/Supabase-Database-3ECF8E?style=flat&logo=supabase)](https://supabase.com)

Real-time financial fraud detection powered by an **XGBoost + Autoencoder ensemble** trained on the PaySim1 synthetic banking dataset. Flags suspicious transactions with **0.98+ AUC-ROC** at sub-50ms inference latency. Every prediction is logged to Supabase for full auditability.

---

## Live Demo

| Service | URL |
|--------|-----|
| Frontend | finguard-1.vercel.app |
| Backend API | Deployed on Railway |

---

## Features

- Upload a CSV of transactions and get back fraud predictions instantly
- XGBoost classifier + deep Autoencoder anomaly detector ensemble (0.7 / 0.3 blend)
- AUC-ROC 0.98+, F1 Score 0.94, sub-50ms inference per transaction
- Full audit trail — every input and prediction logged to Supabase
- Interactive dashboard with ROC curve, Precision-Recall curve, confusion matrix, and feature importance charts
- Live transaction stream simulator on the dashboard
- Graceful offline fallback — browser-side heuristic scorer when backend is unavailable
- Export flagged transactions as CSV

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML, CSS, JavaScript (vanilla) |
| Charts | Chart.js |
| CSV Parsing | PapaParse |
| Backend | FastAPI (Python) |
| ML Models | XGBoost + Keras Autoencoder |
| Feature Engineering | scikit-learn (StandardScaler, PCA) |
| Database | Supabase (PostgreSQL) |
| Backend Deploy | Railway |
| Frontend Deploy | Vercel |

---

## Project Structure

```
finguard/
├── main.py              # FastAPI service — scoring endpoints
├── database.py          # Supabase connection and helpers
├── model_trainer.py     # Training pipeline for XGBoost + Autoencoder
├── requirements.txt     # Python dependencies
├── Procfile             # Railway start command
├── artifacts/           # Trained model files (generated after training)
│   ├── scaler.joblib
│   ├── pca.joblib
│   ├── xgb.joblib
│   ├── autoencoder.keras
│   └── curves.json
├── frontend/
│   ├── index.html       # Dashboard page
│   ├── detect.html      # CSV upload and scoring page
│   ├── styles.css       # Shared dark fintech theme
│   └── app.js           # Shared config, API helpers, chart options
│   └── sample_transactions.csv
└── schema.sql           # Supabase table definitions
```

---

## ML Pipeline

**Training data:** PaySim1 — 6.36M synthetic mobile money transactions

**Feature engineering:**
- Numeric features: `step`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`
- One-hot encoded transaction type: `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER`
- StandardScaler normalization → PCA (12 components)

**Models:**
- XGBoost classifier trained on PCA-projected features with class imbalance correction
- Keras Autoencoder (4-layer bottleneck) trained on benign-only transactions for anomaly detection
- Ensemble: `0.7 × XGBoost probability + 0.3 × normalized reconstruction error`

**Results on hold-out test set:**

| Metric | Score |
|--------|-------|
| AUC-ROC | 0.9847 |
| Precision | 96.2% |
| Recall | 91.8% |
| F1 Score | 0.9395 |
| Accuracy | 99.91% |
| Avg Latency | ~38ms |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service and model status |
| GET | `/metrics` | Training metrics and curve data |
| POST | `/score` | Score a single transaction |
| POST | `/score/batch` | Score many transactions (used by CSV upload) |
| GET | `/predictions/recent` | Last 25 stored predictions |

---

## CSV Format

Upload a CSV with these required columns:

```
step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest
```

Download the template from the Detect page or use `frontend/sample_transactions.csv`.

---

## Local Setup

### 1. Clone the repo

```bash
git clone https://github.com/yashrathour/FinGuard.git
cd FinGuard
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Create a `.env` file in the root:

```
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key
MODEL_DIR=./artifacts
API_HOST=0.0.0.0
API_PORT=8000
```

### 4. Set up Supabase

Run `schema.sql` in your Supabase SQL Editor to create the required tables.

### 5. Train the model

Download the [PaySim1 dataset](https://www.kaggle.com/datasets/ealaxi/paysim1) and place the CSV at `./data/PS_20174392719_1491204439457_log.csv`, then run:

```bash
python model_trainer.py
```

This generates all artifacts in the `./artifacts/` folder and pushes a metrics snapshot to Supabase. Training takes 10–20 minutes depending on hardware.

### 6. Start the backend

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 7. Open the frontend

Open `frontend/index.html` in your browser or serve it with any static file server:

```bash
npx serve frontend
```

---

## Deployment

**Backend — Railway:**
- Connect GitHub repo on Railway
- Set root directory to repo root
- Add environment variables: `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `MODEL_DIR=./artifacts`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

**Frontend — Vercel:**
- Connect GitHub repo on Vercel
- Set root directory to `frontend`
- Framework preset: Other
- No build command needed

---

## Database Schema

Three tables in Supabase:

- `transactions` — raw inputs submitted by users
- `fraud_predictions` — model output for every scored transaction
- `model_metrics` — performance snapshots written after each training run

Full schema in `schema.sql`.

---

## Team

Built by Yash Rathour

---

## Dataset

[PaySim1 — Synthetic Financial Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1) by Edgar Lopez-Rojas. 6.36M transactions simulating mobile money transfers with labeled fraud cases.
