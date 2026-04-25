-- ============================================================
-- FinGuard — Supabase schema
-- Run this once in Supabase SQL Editor (project: leiiakaqucvdvwtxsqrg).
-- ============================================================

-- 1. Raw transactions submitted by users (input from frontend)
create table if not exists public.transactions (
    id              bigserial primary key,
    batch_id        uuid not null,
    step            int  not null,
    type            text not null,
    amount          numeric(18,2) not null,
    name_orig       text not null,
    oldbalance_org  numeric(18,2) not null,
    newbalance_orig numeric(18,2) not null,
    name_dest       text not null,
    oldbalance_dest numeric(18,2) not null,
    newbalance_dest numeric(18,2) not null,
    created_at      timestamptz default now()
);
create index if not exists idx_tx_batch on public.transactions (batch_id);

-- 2. Model output for every scored transaction
create table if not exists public.fraud_predictions (
    id                  bigserial primary key,
    transaction_id      bigint references public.transactions(id) on delete cascade,
    batch_id            uuid not null,
    fraud_probability   double precision not null,
    anomaly_score       double precision not null,
    is_fraud            boolean not null,
    model_version       text default 'v1.0-xgb+ae',
    latency_ms          double precision,
    created_at          timestamptz default now()
);
create index if not exists idx_pred_batch on public.fraud_predictions (batch_id);
create index if not exists idx_pred_isfraud on public.fraud_predictions (is_fraud);

-- 3. Model performance snapshots written after each training run
create table if not exists public.model_metrics (
    id              bigserial primary key,
    model_version   text not null,
    auc_roc         double precision,
    precision_score double precision,
    recall_score    double precision,
    f1_score        double precision,
    accuracy        double precision,
    avg_latency_ms  double precision,
    train_rows      bigint,
    test_rows       bigint,
    fraud_rate      double precision,
    notes           text,
    created_at      timestamptz default now()
);

-- 4. Allow the anon key (used by the demo app) to read & write.
--    Lock this down before going to production.
alter table public.transactions       enable row level security;
alter table public.fraud_predictions  enable row level security;
alter table public.model_metrics      enable row level security;

drop policy if exists "anon all transactions"      on public.transactions;
drop policy if exists "anon all predictions"       on public.fraud_predictions;
drop policy if exists "anon all metrics"           on public.model_metrics;

create policy "anon all transactions"
  on public.transactions for all to anon using (true) with check (true);
create policy "anon all predictions"
  on public.fraud_predictions for all to anon using (true) with check (true);
create policy "anon all metrics"
  on public.model_metrics for all to anon using (true) with check (true);
