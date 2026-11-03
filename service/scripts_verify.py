"""End-to-end verification for the standalone ML service (HTTP only).

Run with the service up (``python -m uvicorn app.main:app --port 8001``):

    python scripts_verify.py            # checks http://localhost:8001
    BASE_URL=http://other-host:9000 python scripts_verify.py

Covers: health (DB ping + staged models), bcrypt login on the service's own
users, role-protected /me, insights list + acknowledge, forecast generation
with the trained-model stamps, and the fault pipeline (seeded fault list +
detail, telemetry ingestion triggering live detection, open-fault evidence,
operator patch). Exits non-zero when any check fails.
"""
import os
import sys
from datetime import datetime, timedelta, timezone

import httpx

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8001").rstrip("/")
API = f"{BASE_URL}/api/v1"

EXPECTED_MODEL_VERSION = "rf-forecast-v0.3"  # staged artifact promoted via beats_baseline
EXPECTED_MAE = 10.4067                       # artifact evaluation metric (hourly demand)

failures: list[str] = []
passed = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed
    if condition:
        passed += 1
        print(f"PASS  {name}" + (f" — {detail}" if detail else ""))
    else:
        failures.append(f"{name} {detail}")
        print(f"FAIL  {name} {detail}")


def main() -> int:
    token = None
    with httpx.Client(timeout=120) as client:
        # 1. health
        r = client.get(f"{BASE_URL}/health")
        check("GET /health", r.status_code == 200, f"status={r.status_code}")
        health = r.json()
        check("health DB reachable", health.get("database") == "ok", str(health.get("database")))
        models = health.get("models") or {}
        check("health exposes staged models", bool(models), str(models))

        # 2. login (own users, own JWT secret)
        r = client.post(f"{API}/auth/login",
                        json={"username": "alex.operator", "password": "Operator!123"})
        check("POST /auth/login", r.status_code == 200, f"status={r.status_code}")
        if r.status_code == 200:
            token = r.json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}
        else:
            print("Cannot continue without a token — stopping after auth checks.")
            return finish()

        # 3. me
        r = client.get(f"{API}/auth/me", headers=headers)
        check("GET /auth/me", r.status_code == 200, f"status={r.status_code}")
        me = r.json() if r.status_code == 200 else {}
        check("/me returns alex.operator", me.get("username") == "alex.operator",
              str(me.get("username")))
        check("/me role operator", me.get("role") == "operator", str(me.get("role")))

        # 4. insights list + acknowledge
        r = client.get(f"{API}/insights", headers=headers, params={"page_size": 5})
        check("GET /insights", r.status_code == 200, f"status={r.status_code}")
        items = r.json().get("data", []) or []
        check("insights non-empty", len(items) > 0, f"total={r.json().get('pagination', {}).get('total')}")
        station_id = None
        if items:
            first = items[0]
            station_id = first.get("station_id")
            r = client.patch(f"{API}/insights/{first['id']}", headers=headers,
                             json={"status": "acknowledged"})
            check("PATCH /insights/{id} acknowledge", r.status_code == 200, f"status={r.status_code}")
            ack = r.json() if r.status_code == 200 else {}
            check("acknowledge stamped", bool(ack.get("acknowledged_by")), str(ack.get("acknowledged_by")))

        # 5. forecast generation (needs a real station with seeded history)
        check("insight carries station_id", bool(station_id), str(station_id))
        if station_id:
            r = client.post(f"{API}/forecasts/generate", headers=headers,
                            json={"station_id": station_id, "horizon": "4h"})
            check("POST /forecasts/generate", r.status_code == 201, f"status={r.status_code}")
            if r.status_code == 201:
                check("forecast message", r.json().get("message") == "Forecast generated successfully",
                      str(r.json().get("message")))

            # 6. list — newest record carries the trained-model stamps
            r = client.get(f"{API}/forecasts", headers=headers, params={"page_size": 5})
            check("GET /forecasts", r.status_code == 200, f"status={r.status_code}")
            flist = r.json().get("data", []) or []
            check("forecasts non-empty", len(flist) > 0, f"count={len(flist)}")
            newest = flist[0] if flist else {}
            check("forecast stamped model_version",
                  newest.get("model_version") == EXPECTED_MODEL_VERSION,
                  str(newest.get("model_version")))
            mae = newest.get("mae")
            try:
                mae_ok = mae is not None and abs(float(mae) - EXPECTED_MAE) < 0.01
            except (TypeError, ValueError):
                mae_ok = False
            check("forecast stamped mae", mae_ok, str(mae))

            # 7. detail
            if newest.get("id"):
                r = client.get(f"{API}/forecasts/{newest['id']}", headers=headers)
                check("GET /forecasts/{id}", r.status_code == 200, f"status={r.status_code}")
                detail = r.json() if r.status_code == 200 else {}
                check("detail has forecast_data", (detail.get("forecast_data") or {}) != {},
                      "empty" if not detail.get("forecast_data") else "present")

        # ---- 8. faults: seeded history ----
        r = client.get(f"{API}/faults", headers=headers, params={"page_size": 5})
        check("GET /faults", r.status_code == 200, f"status={r.status_code}")
        faults_items = r.json().get("data", []) or []
        check("faults non-empty", len(faults_items) >= 4,
              f"total={r.json().get('pagination', {}).get('total')}")
        seeded_fault = None
        if faults_items:
            seeded_fault = faults_items[0]
            r = client.get(f"{API}/faults/{seeded_fault['id']}", headers=headers)
            check("GET /faults/{id}", r.status_code == 200, f"status={r.status_code}")
            fdetail = r.json() if r.status_code == 200 else {}
            check("fault detail carries connector", bool(fdetail.get("connector_id")),
                  str(fdetail.get("connector_id")))

        # ---- 9. telemetry ingestion + live fault detection ----
        if seeded_fault and seeded_fault.get("connector_id"):
            cid = seeded_fault["connector_id"]
            now = datetime.now(timezone.utc)
            r = client.post(f"{API}/telemetry", headers=headers, json={
                "readings": [
                    {"connector_id": cid,
                     "recorded_at": (now - timedelta(minutes=m)).isoformat(),
                     "power_kw": 0.0, "temperature_c": 40.0,
                     "is_anomalous": True, "raw_payload": {"source": "verify-script"}}
                    for m in (15, 10, 5, 0)
                ]
            })
            check("POST /telemetry ingest", r.status_code == 200, f"status={r.status_code}")
            ingest = r.json() if r.status_code == 200 else {}
            check("ingest inserted readings", ingest.get("inserted") == 4, str(ingest.get("inserted")))
            check("detection raised a fault", (ingest.get("faults_detected") or []),
                  str([(f.get("fault_type"), f.get("probability")) for f in ingest.get("faults_detected") or []]))
            check("fault insight generated", (ingest.get("insights_created") or 0) >= 1,
                  f"insights_created={ingest.get('insights_created')}")

            # open-fault list shows the fresh event with detector evidence
            r = client.get(f"{API}/faults", headers=headers,
                           params={"status": "open", "page_size": 5})
            check("GET /faults?status=open", r.status_code == 200, f"status={r.status_code}")
            new_open = [
                f for f in (r.json().get("data", []) or [])
                if f.get("connector_id") == cid and f.get("status") == "open"
            ]
            check("open fault from live detection", bool(new_open), f"count={len(new_open)}")
            if new_open:
                r = client.get(f"{API}/faults/{new_open[0]['id']}", headers=headers)
                fd = r.json() if r.status_code == 200 else {}
                check("fault detail has telemetry evidence",
                      bool((fd.get("evidence") or {}).get("telemetry_readings")),
                      "readings" if (fd.get("evidence") or {}).get("telemetry_readings") else "none")
                # the linked operator insight exists and points back at this fault
                r = client.get(f"{API}/insights", headers=headers, params={"page_size": 20})
                linked = [
                    i for i in (r.json().get("data", []) or [])
                    if i.get("source_type") == "fault_detection"
                    and str(i.get("source_id")) == str(new_open[0]["id"])
                ]
                check("insight linked to live fault", bool(linked), f"count={len(linked)}")
                # connector-level dedupe: a repeat burst while the fault is
                # still open must not mint a second fault (+ insight)
                r = client.post(f"{API}/telemetry", headers=headers, json={
                    "readings": [
                        {"connector_id": cid,
                         "recorded_at": (now - timedelta(minutes=m)).isoformat(),
                         "power_kw": 0.0, "temperature_c": 40.0,
                         "is_anomalous": True, "raw_payload": {"source": "verify-script-repeat"}}
                        for m in (14, 9, 4, 0)
                    ]
                })
                repeat = r.json() if r.status_code == 200 else {}
                check("repeat burst deduped",
                      (repeat.get("faults_detected") or []) == []
                      and (repeat.get("insights_created") or 0) == 0,
                      f"faults={repeat.get('faults_detected')} insights={repeat.get('insights_created')}")
            if new_open:
                r = client.get(f"{API}/faults/{new_open[0]['id']}", headers=headers)
                fd = r.json() if r.status_code == 200 else {}
                check("fault detail has telemetry evidence",
                      bool((fd.get("evidence") or {}).get("telemetry_readings")),
                      "readings" if (fd.get("evidence") or {}).get("telemetry_readings") else "none")
                # keep the demo list clean: mark it a verified-noise false positive
                r = client.patch(f"{API}/faults/{new_open[0]['id']}", headers=headers,
                                 json={"status": "false_positive", "is_false_positive": True,
                                       "notes": "verify-script anomaly burst"})
                check("PATCH /faults/{id} false_positive", r.status_code == 200, f"status={r.status_code}")

    return finish()


def finish() -> int:
    total = passed + len(failures)
    print(f"\n{passed}/{total} checks passed")
    if failures:
        print("FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
