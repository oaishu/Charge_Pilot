# ChargePilot ML — Application Functionality Document

How the standalone ML codebase (`ml/`) actually works: the three main
functionalities — **demand forecast**, **fault detection**, **insights** —
the data each one requires, and the end-to-end data flow of the application.

Companion docs: `ml/README.md` (layout, training, how to run) and
`ml/service/README.md` (service quickstart, endpoints, bootstrap). A
beginner-friendly version of the model story lives in
`docs/models_explained.md`.

---

## 1. What this codebase is

`ml/` is a **fully self-contained application**: data ingestion → model
training → serving API → database. It runs *without* the main backend:

```
ml/
├── notebooks/               OFFLINE LAYER — train + evaluate the models
│   ├── 01_data_ingestion_eda.ipynb      public datasets → training store
│   ├── 02_train_forecast_model.ipynb    RandomForest demand forecaster
│   └── 03_train_fault_model.ipynb       GBDT fault detector + IF fallback
├── artifacts/               MODEL STORE — the .pkl bundles (git-ignored)
│   ├── forecast_model_v1.pkl   → rf-forecast-v0.3
│   └── fault_model_v1.pkl      → gbdt-fault-v0.2
├── service/                 ONLINE LAYER — FastAPI app (port 8001)
│   ├── scripts_setup_db.py     bootstrap: create DB + schema + demo data
│   ├── scripts_verify.py       end-to-end HTTP checks
│   └── app/                    routers / services / models / schemas
└── .env                     DB URL, JWT secret, ACN dataset token (git-ignored)
```

**Division of labour (the single most important idea in this codebase):**

| Layer | When it runs | What it does | Cost |
|---|---|---|---|
| Training | offline, when you rerun the notebooks | fits the RandomForests and the GBDT, evaluates them against the deterministic baseline, and stages an artifact **only if it beats the baseline** | hours |
| Serving | per HTTP request | aggregates DB history into a baseline, applies the *already-trained* models as bounded corrections/scorers, persists results | milliseconds–seconds |

Models are **never** retrained at request time. The 90-day "training window"
in the serving path is a *SQL aggregation window*, not a fitting window.

---

## 2. The data the application works on (entities)

All entities live in the service's own dedicated database (default
`chargepilot_ml`; never the backend's `chargepilot`). The whole schema is
created from `app/models.py` (`create_all` — no migration tooling) and the
demo data is seeded deterministically (`random.Random(42)`) by
`scripts_setup_db.py`:

| Table | Role in the application | Required for |
|---|---|---|
| `users` | login accounts (bcrypt), roles `operator` / `driver` | auth on every data endpoint (operator-gated) |
| `stations` | charge sites: name, area, coordinates, **rated capacity kW** | forecast (pressure classification), insights |
| `connectors` | 4 per station: type (`CCS/Type2/J1772/CHAdeMO/NACS`), `max_power_kw` | forecast (`is_fast = max_power_kw ≥ 50` feature + which connector absorbs load), fault detection |
| `charging_sessions` | history: start/end, energy, peak kW, status; ~16k rows over ~90 days seeded, live rows accumulate | the **input** to forecasting (the single source of the baseline) |
| `telemetry_readings` | per-connector streams: `recorded_at, power_kw, voltage_v, current_a, temperature_c, is_anomalous` | the **input** to fault detection |
| `fault_events` | detector output: type, severity, status, evidence JSON | the *output* of fault detection; input to insights |
| `demand_forecasts` | forecast records incl. `forecast_data` JSON (time series + baseline comparison + ml adjustment) | the *output* of forecasting; input to insights |
| `operational_insights` | operator work queue: title/summary/action, priority, status, `source_type`/`source_id` link | the *human-facing* product of both models |

Data disciplines worth knowing:

- **Sessions are the ground truth for demand.** The forecasting service
  selects only `started_at` within a 90-day window before the current
  reference time; active sessions (no `ended_at`) are treated as ending
  after one hour.
- **Telemetry is append-only evidence.** Rows are never edited; the
  detector reads a recent slice, faults reference it, operators see it.
- **Every seeded row is reproducible.** `scripts_setup_db.py --reset`
  wipes all eight tables in FK order and reseeds identically (seed 42),
  which makes demos and verification deterministic.

---

## 3. Functionality A — Demand forecast

**Purpose:** predict station-level charging load over a horizon so an
operator sees pressure before it happens.

**Endpoint:** `POST /api/v1/forecasts/generate`
`{"station_id": <uuid>, "horizon": "1h" | "4h" | "24h"}` → 201.

**Data required** (all read live from the DB at call time):
1. The station (`stations` row → **rated capacity kW**).
2. Up to **90 days** of that station's `charging_sessions`
   (`started_at`, `ended_at`, `peak_power_kw`).
3. The same lines **joined to `connectors`** (for `max_power_kw` →
   `is_fast`), only when the ML correction is active.
4. The staged forecast artifact (`ml/artifacts/forecast_model_v1.pkl`),
   loaded once per process, **used only if `beats_baseline=True`**.

### Step by step

```
① BASELINE AGGREGATION (pure SQL/Python statistics, recomputed every call)
   For the station's last-90-day sessions, bucket each session into the
   slot (hour-of-day, is_weekend):
      station profile[slot]  = Σ session peak kW in slot / 90 days
      session profile[slot]  = round(#sessions in slot / 90 days)
   This is the deterministic model: "this station, on this kind of day,
   at this hour, historically draws X kW with about N sessions."
   Connectors without history are omitted downstream — no history, no
   invented load.

② DETERMINISTIC FORECAST
   Walk the horizon in steps (1h→12×5-min, 4h→16×15-min, 24h→24×1h points);
   each point = profile[(hour, weekend)] of its timestamp.
   Totals: predicted_demand_kw = Σ point × step-fraction;
           predicted_sessions  = rounded sum of slot session counts.

③ ML CORRECTION (only when the artifact is staged AND promoted)
   For every point × connector profile: the trained RandomForest
   (feature order from the artifact: hour, dow, is_weekend, month,
   is_fast, is_kr, hist_demand, lag_any, roll3, lag_recent) predicts a
   value; the serving change is the RATIO pred/baseline CLAMPED to
   [0.5, 1.5].
        point.predicted_kw = Σ over connectors: hist * ratio
   - ratio < 1  → model says "quieter than usual"
   - ratio > 1  → model says "busier than usual"
   - no history for the slot → baseline kept (the clamp's lower bound of 0
     plus the no-key skip guarantee the model never invents load)
   The correction recomputes totals and stamps the record:
      model_version=rf-forecast-v0.3, mae=10.4067, rmse=14.4417,
      baseline_demand_kw (pre-correction total),
      ml_adjustment={mean_ratio, connectors_scored}.

④ PRESSURE CLASSIFICATION
   avg_kw = total / horizon-hours; ratio to the station's rated capacity:
      < 0.60 → low     0.60–0.85 → medium     ≥ 0.85 (or no capacity) → high

⑤ PERSISTENCE + INSIGHT
   One demand_forecasts row with forecast_data JSON:
      time_series[{timestamp, predicted_kw, actual_kw: null}],
      baseline_comparison{historical_average_kw, deviation_pct},
      ml_adjustment, baseline_demand_kw.
   If capacity_pressure ∈ {medium, high} → OperationalInsightService
   raises a capacity_pressure insight (deduped per source forecast).
```

**What is *not* true:** the model is not trained per call, and the baseline
is not an average of averages — it is Σ power / 90 days per slot, i.e. an
average *concurrent-load* per slot across all days.

**Failure behaviour:** if the artifact is missing/corrupt/demoted, steps ③–④
degrade silently to the pure baseline; the endpoint still works.

---

## 4. Functionality B — Fault detection

**Purpose:** turn raw charge-point telemetry into actionable, evidence-backed
fault events without a human watching a dashboard.

**Endpoint:** `POST /api/v1/telemetry`
`{"readings": [...≤500], "run_detection": true}`.

**Data required per reading:**
`connector_id` (FK), `recorded_at` (defaults to now), `power_kw`,
`voltage_v`, `current_a`, `temperature_c`, `is_anomalous` flag, optional
`raw_payload`. The endpoint validates the connectors exist, inserts the
rows, and **immediately scores each touched connector** in the same request.

### Step by step (per touched connector)

```
① WINDOW  newest ≤ 50 readings from the last 2h (HISTORY_WINDOW_MINUTES
          30 × 4), newest-first. Reading ≠ training: this is a query slice.

② GUARDRAILS FIRST  (deterministic physics rules; the INDEPENDENT safety
   layer — they run even with no artifact and always as the fallback)
     G1  max(temperature_c) ≥ 85 °C
         → overtemperature, p=0.95, high severity at ≥95 °C   ← always wins
     G2  ≥50% of window readings flagged is_anomalous
         → power_anomaly, p=0.75
     G3  all readings within the last 20 min at 0 kW AND ≥1 anomalous
         → connector_failure, p=0.90 (10-min streak rule × 2)

③ ML SCORING  (staged fault artifact = calibrated GradientBoosting, v0.2)
   The model does not score the whole window: it scores the slice AFTER —
   i.e. SINCE — the most recent anomalous reading, because that is where a
   failure begins; a fresh zero-power failure is not drowned out by the
   healthy pre-fault data. Features (artifact order): duration_hours,
   avg_kw, hour_start, dow, is_weekend, is_fast, is_kr. The bundle's own
   decision_threshold (0.60) is used so training and serving agree.
   Returns a DetectionResult with class probabilities recorded in evidence.

④ FUSION   winner = ML result if p(ml) ≥ p(guardrail), else guardrail.
   So: ML can beat a modifier guardrail but NEVER overtemperature.

⑤ PERSIST IF p ≥ 0.60 — with connector-level dedupe
   Before persisting, an existing open/investigating fault for the SAME
   connector suppresses a new one (a continuous feed, retries, or re-runs
   keep pointing at the standing event instead of minting a fault every
   push). When persisted:
   fault_event(open) with evidence JSON:
     telemetry_readings   sample raw readings (always present)
     connector_status_history
     detector{model_version, kind: "ml"|"guardrail", label,
              class_probabilities, window_features, probability}
   → OperationalInsightService.generate_from_fault links a fault_detected
     insight in the same commit. Its dedupe is source-based: an already
     ACTIVE insight for the same fault id is never duplicated.

⑥ LIFECYCLE  operators triage via GET/PATCH /faults:
   open → investigating → resolved | false_positive (+notes into evidence).
```

**Worked example** (the one `scripts_verify.py` performs on every run):
seed the service, then ingest 4 zero-kW anomalous readings for a fast
connector over 15 minutes. Guardrail G3 fires at 0.90, but the trained model
scores the post-anomaly slice stronger: **connector_failure at p=0.9927,
kind="ml"** → an `open` event with detector evidence, a linked insight, then
the operator PATCH. Exactly the intended behaviour: guardrails are the net,
the trained model is the better classifier.

**Cron / daemon:** none. Detection runs synchronously inside the POST.
If you connect a real charger network, the *producer* (gateway, cron,
MQTT bridge) decides the cadence and calls `POST /telemetry`; the API stays
stateless. Un-fed data is never scored retroactively.

---

## 5. Functionality C — Insights

**Purpose:** the operator-facing work queue that turns model output into
"do something, something specific, by a deadline".

**Endpoints:** `GET /api/v1/insights` (filters: station, status, priority,
type, paging) and `PATCH /api/v1/insights/{id}` (status /
acknowledged_at / acknowledged_by stamps).

Insights are **generated, not authored** — the endpoints only read and
triage. Production happens in exactly three places:

| # | Producer | When | Mechanism |
|---|---|---|---|
| 1 | **Live faults** (primary) | every fault persisted via `POST /telemetry` | `generate_from_fault`: insight type `fault_detected`, title "`<Type> detected at <Station> (<connector>)`", summary = the fault's description, suggested action = the detector's action, evidence links `fault_id`/`severity`, priority mapped from severity |
| 2 | **Live forecasts** | `POST /forecasts/generate` when `capacity_pressure` is medium/high | `generate_from_forecast`: type `capacity_pressure`, summary carries predicted kW vs rated capacity ("exceeding N % of rated capacity"), priority medium/high |
| 3 | **Bootstrap seed** | `scripts_setup_db.py` | two historical fault→insight pairs so a fresh database opens with a non-empty board |

Text is deterministic templating (no LLM). Dedupe is two-layered: at the
**fault level**, `detect()` never stacks a new fault on a connector that
already has an open/investigating fault — repeated anomaly bursts (a
continuous feed or retries) reference the standing event, so exactly one
fault stays in flight per connector; at the **insight level**, an already
*active* insight for the same `source_type + source_id` is never duplicated,
so every fault row is represented by exactly one insight and repeated
forecast calls don't spam the board.

**Lifecycle:** an insight is not immortal "seeded data shown forever" —
it sits `active` until an operator PATCHes it to `acknowledged` (kit-stamped
by whom/when) or `resolved`, or `scripts_setup_db.py --reset` wipes it.
Faults, not insights, are the permanent audit trail; insights are the
to-do list on top of it. Nothing expires on its own and nothing runs
outside HTTP requests.

---

## 6. Overall data flow

```
OFFLINE
   public datasets (Korea / Barnet / ACN)
        │  notebook 01: clean, filter, merge → training store CSV
        │  notebook 02/03: fit, evaluate vs baseline, gate
        ▼
   ml/artifacts/*.pkl   (beats_baseline gate embedded)

BOOTSTRAP (once)
   scripts_setup_db.py → CREATE DATABASE (Postgres) → create_all schema
                        → seed: users, stations, connectors, sessions,
                                telemetry, faults, insights  [seed 42]

ONLINE (per HTTP request)
   ┌──────────────────────┐        ┌───────────────────────────┐
   │ POST /forecasts/generate │    │ POST /telemetry           │
   └──────────┬───────────┘        └────────────┬──────────────┘
              │                                 │
   sessions 90d ─┐                     telemetry_readings 2h window
   station/conn ─┤                                  │
              ▼                                     ▼
   baseline profile ── RF correction ──    guardrails ── GBDT slice-since-anomaly
        (clamped 0.5–1.5)                   (p, fusion at p≥ threshold 0.60)
              │                                     │
              ▼                                     ▼
   demand_forecasts row  ──(medium/high pressure)──► operational_insights
   fault_events row ────────────────────────────────► operational_insights
              │                                     │
              ▼                                     ▼
   GET/PATCH /forecasts                GET/PATCH /faults, /insights
   (operator triage: acknowledge / resolve / false-positive stamps)
```

Endpoint map (all data endpoints operator-gated):

| Method & path | Writes / reads |
|---|---|
| `POST /auth/login`, `GET /auth/me` | own users table, own JWT |
| `POST /forecasts/generate` | computes + writes `demand_forecasts` (+ insight) |
| `GET /forecasts`, `GET /forecasts/{id}` | reads |
| `POST /telemetry` | writes `telemetry_readings`; computes + writes `fault_events` (+ insight) |
| `GET /faults`, `GET /faults/{id}`, `PATCH /faults/{id}` | reads / triage writes |
| `GET /insights`, `PATCH /insights/{id}` | reads / triage writes |
| `GET /health` | liveness + DB ping + staged model versions |

---

## 7. Design invariants (the promises the pieces make to each other)

1. **Safety layer independence** — deterministic guardrails always run; the
   ML is an enhancement, never a dependency. Overtemperature can never be
   talked out of by the model.
2. **Baseline never bypassed** — the forecast can deviate only within the
   [0.5, 1.5] ratio clamp, per connector, and only where history exists.
3. **Promotion gate** — artifacts stage only when they beat the baseline;
   the loader refuses anything without `beats_baseline=True`.
4. **Request-scoped work** — no cron, no background daemon; each HTTP
   request owns its computation from input to persisted side effects.
5. **Deterministic demo state** — seed 42 makes `--reset` reproduce the
   exact database, which is what makes `scripts_verify.py` meaningful.
6. **Full isolation** — nothing imports from `backend/`; the service points
   at its own database; backend tokens are meaningless here.

---

## 8. Honest limitations

- Fault labels are weakly circular (derived from the same features used for
  scoring); real maintenance records would end that.
- Hourly demand stays only marginally better than the baseline —
  session-level records can't express per-slot occupancy.
- No refresh-token rotation, register/lockout, or rate limiting (full-app
  concerns).
- Detection latency = ingestion latency; nothing re-scores readings that
  arrive with old timestamps outside the 2h window.

*Document generated from the implementation as of 2026-11-01 —
`ml/service` code is the source of truth where wording and code disagree.*
