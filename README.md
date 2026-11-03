# ChargePilot — ML Model Training & Serving

Real ML models (scikit-learn, joblib artifacts) trained on public EV charging
datasets, wired into the backend serving path **and** served independently by
this folder's own FastAPI service (`ml/service`). Both the API and the
standalone service keep their deterministic baselines as a fallback; the
trained models only ever **correct or score on top of** the baseline — they
never bypass the safety layers.

## Directory layout

```
ml/
├── load_artifacts.py        # canonical loader shim (re-exports backend loader)
├── docs/
│   ├── functionality.md     # how the app works: forecast / faults / insights, data flow
│   └── models_explained.md  # beginner-friendly guide: how the models were built
├── notebooks/
│   ├── 01_data_ingestion_eda.ipynb      # datasets → sessions_training_store.csv (+ EDA)
│   ├── 02_train_forecast_model.ipynb    # RandomForest demand forecast (+ evaluation)
│   └── 03_train_fault_model.ipynb       # GBDT fault detector + IF fallback (+ evaluation)
├── artifacts/               # staged .pkl bundles (git-ignored)
│   ├── forecast_model_v1.pkl            # rf-forecast-v0.3 (~154 MB)
│   └── fault_model_v1.pkl               # gbdt-fault-v0.2 (~1.1 MB)
├── data/                    # training store + raw sources (git-ignored)
│   ├── sessions_training_store.csv
│   └── raw/
│       ├── korea/                       # Kim et al. 2024 figshare CSV
│       ├── barnet/                      # CityEV quarterly + 2020 transactions CSVs
│       └── acn/                         # per-(site, year) JSONL API caches
├── service/                 # standalone FastAPI serving app (see service/README.md)
├── reports/                 # EDA figures + data-quality notes (git-ignored, local)
└── .env                     # shared ML-side config: ACN token, DB URL, JWT (git-ignored)
```

## Datasets

1. **Kim et al. (2024), *Scientific Data* — 72,856 charging sessions, Korea**
   Figshare DOI `10.6084/m9.figshare.22495141.v1`
   (direct: `https://ndownloader.figshare.com/files/39952252`)
2. **Barnet (UK) open data** — CityEV quarterly charger CSVs
   (`cityev_YYYY_qN.csv`, cumulative snapshots → deduplicated) and 2020
   charge transactions from `open.barnet.gov.uk`.
3. **ACN-Data (Caltech Adaptive Charging Network)** — real workplace charging
   sessions from three US sites (Caltech campus, NASA JPL, Office 1),
   2018-04 → 2021-12, fetched via the REST API (`ev.caltech.edu/api/v1/`)
   with the user's free token (`ACN_DATA_API_KEY` in `ml/.env`, read by
   notebook 01 only — the serving path needs no token). Raw responses are
   cached per `(site, year)` under `ml/data/raw/acn/` so re-running notebook
   01 is a pure cache hit; UTC timestamps are converted to site-local
   `America/Los_Angeles` before hour-of-day features are built.

After cleaning: **146,278 sessions across 2,391 chargers** (subset: 43,718
ACN-workplace sessions after quality rules) land in
`ml/data/sessions_training_store.csv` (the single training input for
notebooks 02/03). See `ml/reports/data_quality.md` for filters and caveats.
Dundee was skipped (gated download).

## Training workflow

Run notebooks in order with outputs preserved — they double as the training
report (each keeps EDA, evaluation, and honest caveat sections in its
narrative):

```powershell
cd ml\notebooks
python -m nbconvert --to notebook --execute --inplace 01_data_ingestion_eda.ipynb
python -m nbconvert --to notebook --execute --inplace 02_train_forecast_model.ipynb
python -m nbconvert --to notebook --execute --inplace 03_train_fault_model.ipynb
```

Each training notebook:

1. loads `ml/data/sessions_training_store.csv`,
2. builds features (time-of-day, dow/weekend, connector type, history lags),
3. evaluates against the deterministic baseline the API serves,
4. only stages the artifact if it **beats the baseline**
   (`beats_baseline=True` is embedded in the bundle — the serving loader
   refuses artifacts without it and silently falls back to the baseline).

Model / vintage summary:

| Artifact | Models | vs baseline (own test split) | Honest caveat |
|---|---|---|---|
| `rf-forecast-v0.3` (ACN-era, 146k sessions) | RandomForestRegressor ×2 (daily sessions, hourly demand) | daily MAE 0.348 vs 0.418; hourly MAE 10.41 vs 10.42 | hourly gain stays marginal — `hist_demand` dominates (importance 0.76); true slot-level occupancy isn't in these session-level sources |
| `rf-forecast-v0.2` (pre-ACN, 79.5k sessions) | RandomForestRegressor ×2 | daily MAE 0.357 vs 0.394; hourly MAE 10.148 vs 10.151 | superseded by v0.3 |
| `gbdt-fault-v0.2` (ACN-era, 146k sessions) | CalibratedClassifierCV(GradientBoosting) + IsolationForest fallback | per-class F1 1.000 | labels derive from the same features (weak-label circular); IF recall 0.564 (was 0.23 on the pre-ACN store) |
| `gbdt-fault-v0.1` | CalibratedClassifierCV(GradientBoosting) + IsolationForest fallback | per-class F1 1.000 | superseded by v0.2 |

## Serving integration

```
app/core/ml_artifacts.py      # cached loader; resolves settings.ML_ARTIFACTS_DIR
                              # → repo-root ml/artifacts; never raises
ml/load_artifacts.py          # shim for the original plan path
```

**Forecast** (`demand_forecast_service.py`): baseline profile forecast is
computed first; then each connector is re-scored per predictor step with
`model_hourly_demand`. The correction is applied as a **bounded ratio**
(`pred / baseline` clamped to [0.5, 1.5]) so the model never invents load in
slots where the baseline has no history. Persisted `demand_forecasts` records
stamp `model_version=rf-forecast-v0.2`, artifact `mae`/`rmse`, and the
`ml_adjustment` / `baseline_demand_kw` metadata in `forecast_data`.

**Faults** (`fault_detection_service.py`): deterministic guardrails run first
and remain the **independent safety layer** (e.g. overtemperature always
wins); the staged GBDT scores the telemetry window in parallel and its result
(#1 by probability, threshold from the bundle's `decision_threshold=0.60`) is
used when stronger. Artifact labels map to app fault types:
`no_energy_failure → connector_failure`, `power_anomaly / overlong_session →
power_anomaly`. Detector evidence records `kind: "ml"` vs `"guardrail"`.

## Standalone service (`ml/service`, runs on :8001)

A **fully standalone** FastAPI app inside `ml/` — own config/env, **own
dedicated database** (`chargepilot_ml`; never the backend's `chargepilot`
DB), own bootstrap (schema + deterministic demo seed), own users and JWT
secret, own artifact loader. Nothing imports from `backend/`, and nothing in
`backend/` imports from `ml/service`. Endpoints: `POST /api/v1/auth/login`,
`GET /api/v1/auth/me`, `GET+POST /api/v1/forecasts` (`/generate`), `GET+PATCH
/api/v1/insights`, plus `GET /health` reporting staged model versions.

Pick up the whole `ml/` folder, run three commands on any machine:

```bash
python scripts_setup_db.py                 # from ml/service: creates DB + schema + demo data
python -m uvicorn app.main:app --port 8001
python scripts_verify.py                   # HTTP-only end-to-end checks
```

See `ml/service/README.md` for the bootstrap details, the SQLite fallback,
and how the fault detector (telemetry → GBDT/guardrails → fault events)
is fed.

## Verification (done against live Postgres)

- `staged_versions()` (backend loader *and* ml/service loader) →
  `{'forecast': 'rf-forecast-v0.3', 'fault': 'gbdt-fault-v0.2'}`
- Live forecasts (1h/4h/24h) persisted with ML stamps, e.g. `mean_ratio≈0.99`,
  `mae=10.4067`, `rmse=14.4417` — from the backend (`:8000`)
- Simulated `connector_failure` produced an open fault event with
  `kind: ml`, `p=0.9927` (`no_energy_failure` → `connector_failure`)
- `backend/scripts_smoke_test.py`: 48/48 pass (SQLite target)
- `ml/service` end-to-end (`scripts_verify.py` on its own `chargepilot_ml`
  database): **35/35** — health + own-DB ping, login on its own users,
  insights list/acknowledge, forecast generate/list/detail with ML stamps,
  and the fault pipeline: ingested anomaly burst → trained GBDT detects
  `connector_failure` at p=0.9927 → open fault with detector evidence →
  linked insight generated live → operator triage; the backend DB and its
  state are never touched

## Windowing note (fresh failures score correctly)

The fault model scores the telemetry **since the most recent anomalous reading**
(the point a failure would start), not the whole 50-reading window. Without
this, a freshly failed connector drowns in pre-fault healthy readings the
window aggregates are dominated by (~49 normal + 1 zero-power reading still
"looks normal"). Guardrails always score the full window as the independent
safety layer.

## Known limitations / next steps

- Fault labels are weakly circular (derived from the same features used for
  scoring) — replace with real maintenance-record labels when available.
- Hourly-demand model remains only marginally better than baseline: session-
  level event records (KR/UK/ACN) can't fully express per-slot occupancy.
  Candidates: per-slot aggregation of ACN's `doneChargingTime`/pilot-derived
  charging-current series (API `/ts` endpoint) or a dedicated slot-occupancy
  target from the demo DB telemetry. ACN improved daily-count accuracy
  substantially (MAE 0.4184 → 0.3478 vs baseline).

