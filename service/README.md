# ChargePilot ML Service (standalone)

A **fully standalone FastAPI service** that serves the models trained in this
`ml/` folder: own config, **own dedicated database**, own schema + demo-data
bootstrap, own users and JWT secret, and own artifact loader. Nothing in this
folder imports from `backend/`, and the service never touches the main
backend's `chargepilot` database. Copy the whole `ml/` folder to another
machine and it runs as-is.

## Quickstart (fresh machine)

```bash
# 1. configure (once): copy ml/.env.example to ml/.env and set DATABASE_URL
#    (any Postgres user with CREATEDB rights works; SQLite is supported too)
# 2. bootstrap — creates the database, schema, and demo data:
python scripts_setup_db.py            # from ml/service; add --reset to wipe + reseed
# 3. run:
python -m uvicorn app.main:app --port 8001
# 4. verify end-to-end:
python scripts_verify.py
```

Defaults (no env file) point at `postgresql+asyncpg://chargepilot:chargepilot_dev@localhost:5432/chargepilot_ml`.
To run fully offline without Postgres:

```bash
# in ml/.env
DATABASE_URL=sqlite+aiosqlite:///ml_demo.db
```

Notebooks must have run first (01 → 02 → 03) so `ml/artifacts/*.pkl` exist.

## Bootstrap (`scripts_setup_db.py`)

| Step | What happens |
|---|---|
| create database | Postgres: connects to the server admin DB (`postgres`) and issues `CREATE DATABASE` when `chargepilot_ml` is missing (no-op on SQLite) |
| create schema | `Base.metadata.create_all` from `app/models.py` — no migration tooling needed |
| seed demo data | only when the DB is empty (or `--reset` wipes first) |

Seeded demo data (deterministic, `random.Random(42)` — `--reset` reproduces it):

- users `alex.operator` / `Operator!123` (operator) and `sam.driver` / `Driver!123` (driver)
- 4 stations with distinct demand profiles (fast hub / workplace / retail /
  residential) × 4 connectors each (CCS, Type2, J1772, CHAdeMO, NACS mix)
- ~16k charging sessions over ~90 days following each station's
  hour-of-day curve (a few sessions still `active`)
- ~250 healthy telemetry readings (last ~3h, 5-min cadence) for the fast
  connectors, so live fault detection has a realistic window
- 4 resolved guardrail-style historical faults + 2 operational insights

## What it serves

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness, DB reachability, staged model versions |
| `POST /api/v1/auth/login` | JSON login against the service's own users table (access tokens only) |
| `GET /api/v1/auth/me` | current user from the bearer token |
| `GET /api/v1/forecasts` | stored forecasts, filterable by station/horizon/pressure/from |
| `GET /api/v1/forecasts/{id}` | one forecast incl. its `time_series` and `ml_adjustment` |
| `POST /api/v1/forecasts/generate` | run the forecast service for one station, persist, return |
| `POST /api/v1/telemetry` | ingest charge-point readings; scores each touched connector and persists faults above the detection threshold |
| `GET /api/v1/faults` | fault events, filterable by station/status/severity/type/from + paging |
| `GET /api/v1/faults/{id}` | one fault incl. detector evidence (`kind: ml`/`guardrail`, sample telemetry) |
| `PATCH /api/v1/faults/{id}` | operator triage: status / false-positive flag / notes |
| `GET /api/v1/insights` | operational insights, filterable (${station_id}, ${status}, ${priority}, ${type}, paging) |
| `PATCH /api/v1/insights/{id}` | acknowledge / resolve an insight |

All data endpoints are operator-gated (`require_role("operator")`).

## Fault detection — how the detector works

Faults are **not** hand-entered: they are produced by
`POST /api/v1/telemetry` (charge-point readings: `power_kw`,
`voltage_v`, `current_a`, `temperature_c`, `is_anomalous`, timestamps) and
scored by `app/services/fault_detection_service.py`:

1. **Window**: the newest ≤ 50 readings for that connector from the last
   2 hours (`HISTORY_WINDOW_MINUTES × 4`), newest-first.
2. **Guardrails first** (independent safety layer, thresholds are constants):
   — `temperature_c ≥ 85` → `overtemperature` p=0.95 (high at ≥95 °C); the
   always-wins safety class; ≥ 50 % of window readings flagged anomalous →
   `power_anomaly` p=0.75; zero-power readings within the last 20 min with
   any anomaly → `connector_failure` p=0.90.
3. **Trained model** (`fault_model_v1.pkl`, `gbdt-fault-v0.2`): scores the
   slice *since the most recent anomalous reading* (a fresh failure isn't
   drowned out by pre-fault data), features `duration_hours, avg_kw,
   hour_start, dow, is_weekend, is_fast, is_kr`, using the bundle's
   `decision_threshold`. Its result wins when ≥ the guardrail probability.
4. When the winner's probability ≥ 0.60, an `open` `fault_event` is persisted
   with detector evidence (`kind: ml`/`guardrail`, class probabilities,
   sample telemetry readings), and `OperationalInsightService` creates the
   matching operator insight.

Above every persisted fault there is an operator triage path:
list → detail → `PATCH` (status / false-positive / notes).

## Forecasts — how the trained model is served

The deterministic baseline (per-station, per-(hour, weekend) historical
average computed live from session history) is **always** computed first.
When `ml/artifacts/forecast_model_v1.pkl` is staged *and* its
`beats_baseline` flag is true, the RandomForest re-scores each hourly point
per connector as a **bounded ratio** on the baseline (`MIN_RATIO 0.5`,
`MAX_RATIO 1.5`): the ML can dial an hour up or down but never invent load
where history is silent. The persisted record carries `model_version`,
artifact metrics (`mae`/`rmse`), `baseline_demand_kw`, and the
`ml_adjustment` metadata.

## Insights

Insights are **generated by the ML workflows, not authored** — `GET
/PATCH /insights` only read and triage them. They come from three places:

1. **Live fault detection** — every fault the detector persists (via
   `POST /telemetry`) gets a companion `fault_detected` insight wrapping the
   fault's description + suggested action (same workflow step as the
   backend's event workflow).
2. **Live forecasts** — a `POST /forecasts/generate` whose `capacity_pressure`
   is medium/high raises a `capacity_pressure` insight with the predicted kW,
   rated capacity, and usage % (deduped per source forecast).
3. **Bootstrap seed** — two historical fault→insight pairs so a fresh DB
   opens with a non-empty operator board.

They persist in this service's own DB until an operator acknowledges/resolves
them (PATCH) or `scripts_setup_db.py --reset` wipes them — there is no
self-expiry and no cron; nothing runs outside HTTP requests. Acknowledge/
resolve timestamps are the operator audit trail.

## Layout

```
ml/service/
├── scripts_setup_db.py               # bootstrap: create DB + schema + seed
├── scripts_verify.py                 # HTTP end-to-end checker
├── app/
│   ├── main.py                       # app wiring, /health, routers
│   ├── core/
│   │   ├── config.py                 # pydantic-settings; env order: .env → ml/.env
│   │   ├── database.py               # engine + get_db + create DB/create tables
│   │   ├── security.py               # bcrypt + JWT (this service's own secret)
│   │   ├── errors.py                 # {"error": {code, message, details}} envelope
│   │   └── ml_artifacts.py           # artifact loader + beats_baseline gate
│   ├── models.py                     # this service's own schema
│   ├── schemas/                      # auth / forecast / fault / insight / telemetry / common
│   ├── services/
│   │   ├── demand_forecast_service.py   # baseline + RF bounded-ratio correction
│   │   ├── fault_detection_service.py   # guardrails + GBDT window scoring
│   │   ├── insight_service.py           # forecast→insight, fault→insight
│   │   └── forecast_api_helpers.py      # row→schema serializer
│   └── routers/                      # auth / forecasts / faults / insights / telemetry
```

## Considered differences (documented on purpose)

- **No refresh tokens, no register/lockout/rate-limiting** — the ML service
  issues access tokens only; account lifecycle belongs to a full backend.
- **No migrations** — schema is `create_all` from the ORM metadata; alter the
  models and rerun bootstrap.
- **No simulation endpoints** — demo state comes from the seeded data in this
  service's own DB.
- `refresh_token` in the login response is `null`.

## Verified

`python scripts_verify.py` passes **35/35** against the standalone DB,
including: health (own DB reachable, `rf-forecast-v0.3` + `gbdt-fault-v0.2`
staged), login on the service's own users, insights acknowledge, forecast
generation with the trained-model stamps (`model_version`, `mae=10.4067`,
`rmse=14.4417`), and the live fault pipeline — ingesting an anomalous
zero-power burst raises `connector_failure` at **p=0.9927 from the trained
GBDT**, the open event carries detector evidence + sample telemetry, the
linked operator insight is generated in the same request, and the verify run
then marks the fault `false_positive` (so repeated runs leave at most one
triaged test event behind).
