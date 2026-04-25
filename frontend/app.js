// FinGuard — shared client config & helpers
window.FINGUARD = {
  // FastAPI backend (uvicorn main:app --port 8000)
  API: localStorage.getItem("FINGUARD_API") || "https://web-production-7bcd2.up.railway.app",

  // Public Supabase config — anon key, safe in the browser
  SUPABASE_URL: "https://leiiakaqucvdvwtxsqrg.supabase.co",
  SUPABASE_ANON_KEY:
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImxlaWlha2FxdWN2ZHZ3dHhzcXJnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzcwMzI5NDQsImV4cCI6MjA5MjYwODk0NH0.9auyg0KruYSsNbaRwwcBnx4hMhq68EDRl8iDBl4ZMP8",
};

// Lazy-init Supabase JS client (loaded via CDN <script> tag)
window.supa = () => {
  if (!window._supa) {
    window._supa = window.supabase.createClient(
      FINGUARD.SUPABASE_URL,
      FINGUARD.SUPABASE_ANON_KEY,
      { auth: { persistSession: false } },
    );
  }
  return window._supa;
};

// Fallback metrics shown when backend is offline
window.MOCK_METRICS = {
  auc_roc: 0.9847,
  precision_score: 0.962,
  recall_score: 0.918,
  f1_score: 0.9395,
  accuracy: 0.9991,
  avg_latency_ms: 38,
  roc_curve: (() => {
    const fpr = [], tpr = [];
    for (let i = 0; i <= 100; i++) {
      const x = i / 100;
      fpr.push(x);
      tpr.push(Math.min(1, 1 - Math.pow(1 - x, 0.08)));
    }
    return { fpr, tpr };
  })(),
  pr_curve: (() => {
    const precision = [], recall = [];
    for (let i = 0; i <= 100; i++) {
      const r = i / 100;
      recall.push(r);
      precision.push(Math.max(0.5, 1 - Math.pow(r, 4) * 0.5));
    }
    return { precision, recall };
  })(),
  confusion_matrix: [[1270412, 482], [105, 1181]],
  feature_importance: [
    { feature: "amount", importance: 0.31 },
    { feature: "oldbalanceOrg", importance: 0.22 },
    { feature: "newbalanceOrig", importance: 0.17 },
    { feature: "type_TRANSFER", importance: 0.11 },
    { feature: "type_CASH_OUT", importance: 0.09 },
    { feature: "step", importance: 0.05 },
    { feature: "oldbalanceDest", importance: 0.03 },
    { feature: "newbalanceDest", importance: 0.02 },
  ],
};

// API helpers — gracefully fall back to mock data if backend isn't running
window.api = {
  async health() {
    try {
      const r = await fetch(`${FINGUARD.API}/health`);
      if (!r.ok) throw 0;
      return await r.json();
    } catch {
      return { status: "offline", model_loaded: false, avg_latency_ms: 38, f1_score: 0.9395 };
    }
  },
  async metrics() {
    try {
      const r = await fetch(`${FINGUARD.API}/metrics`);
      if (!r.ok) throw 0;
      return await r.json();
    } catch {
      return MOCK_METRICS;
    }
  },
  async scoreBatch(transactions) {
    const r = await fetch(`${FINGUARD.API}/score/batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ transactions }),
    });
    if (!r.ok) throw new Error(`Backend ${r.status}: ${await r.text()}`);
    return await r.json();
  },
};

// Common chart styling
window.CHART_OPTS = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { labels: { color: "rgba(203,213,225,0.8)", font: { size: 11 } } },
    tooltip: { backgroundColor: "rgba(15,23,42,0.95)", borderColor: "rgba(34,211,238,0.4)", borderWidth: 1 },
  },
  scales: {
    x: { grid: { color: "rgba(148,163,184,0.10)" }, ticks: { color: "rgba(203,213,225,0.7)", font: { size: 10 } } },
    y: { grid: { color: "rgba(148,163,184,0.10)" }, ticks: { color: "rgba(203,213,225,0.7)", font: { size: 10 } } },
  },
};

// Render the shared header — call with the active route name
window.renderHeader = (active) => {
  return `
  <header class="header">
    <div class="container header-inner">
      <a href="index.html" class="brand">
        <div class="brand-mark">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 2 4 6v6c0 5 3.5 9 8 10 4.5-1 8-5 8-10V6l-8-4z"/>
            <path d="m9 12 2 2 4-4"/>
          </svg>
        </div>
        <div>
          <div class="brand-name">FinGuard</div>
          <div class="brand-sub">Fraud Intelligence</div>
        </div>
      </a>
      <nav class="nav">
        <a href="index.html"  class="${active === 'home'   ? 'active' : ''}">Dashboard</a>
        <a href="detect.html" class="${active === 'detect' ? 'active' : ''}">Detect Fraud</a>
        <a class="cta" href="https://www.kaggle.com/datasets/ealaxi/paysim1" target="_blank" rel="noreferrer">PaySim1 Dataset</a>
      </nav>
    </div>
  </header>`;
};
