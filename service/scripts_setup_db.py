"""Bootstrap the ML service's dedicated database.

One command takes a fresh machine from "nothing" to a working service DB:

    python scripts_setup_db.py            # create DB + schema + demo data (idempotent)
    python scripts_setup_db.py --reset    # wipe tables and reseed from scratch

What it does:
1. Creates the target database when it doesn't exist (Postgres: connects to the
   server's admin ``postgres`` DB and issues ``CREATE DATABASE``; SQLite: no-op,
   the file is created on first connect — see ``ml/.env.example`` for the
   ``sqlite+aiosqlite`` fallback).
2. Applies the ORM schema (``app.models.Base.metadata.create_all``).
3. Seeds deterministic demo data — only when the DB is empty (or ``--reset``):
   * two users (operator ``alex.operator`` / ``Operator!123``,
     driver ``sam.driver`` / ``Driver!123``),
   * four stations with distinct demand profiles (fast hub / workplace /
     retail / residential) and four connectors each,
   * ~90 days of charging sessions (~17k) following each station's
     hour-of-day curve, with a sprinkle of active sessions in the last hours,
   * four resolved, guardrail-style historical faults,
   * operational insights generated from those faults.
"""
import argparse
import asyncio
import random
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import List

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import (
    AsyncSessionLocal,
    create_database_if_missing,
    create_tables,
)
from app.core.security import hash_password
from app.models import (
    ChargingSession,
    Connector,
    DemandForecast,
    FaultEvent,
    OperationalInsight,
    Station,
    TelemetryReading,
    User,
    new_uuid,
)
from app.services.insight_service import OperationalInsightService

SEED = 42  # deterministic demo data — --reset reproduces the same DB
DAYS_OF_HISTORY = 90

# --------------------------------------------------------------------------- #
# Station profiles: demand curve shape + connector mix
# --------------------------------------------------------------------------- #
STATION_SEEDS: List[dict] = [
    {
        "name": "Riverside Fast Hub",
        "location_area": "Riverside",
        "latitude": 45.464203, "longitude": 9.189982,
        "capacity_kw": 300.0,
        "daily_sessions": 55,            # busy all day, calmer weekends
        "weekend_factor": 0.75,
        "hour_weights": [2, 1, 1, 1, 1, 2, 3, 5, 7, 9, 11, 13, 15, 14, 12, 10, 9, 11, 13, 12, 10, 7, 5, 3],
        "connectors": [("CCS", 150.0), ("CCS", 150.0), ("CHAdeMO", 50.0), ("NACS", 250.0)],
    },
    {
        "name": "Tech Park Workplace",
        "location_area": "Innovation Park",
        "latitude": 45.485911, "longitude": 9.231678,
        "capacity_kw": 88.0,
        "daily_sessions": 40,            # weekday-driven, quiet weekends
        "weekend_factor": 0.25,
        "hour_weights": [1, 1, 1, 1, 1, 2, 6, 14, 20, 18, 14, 12, 10, 12, 15, 13, 8, 4, 2, 1, 1, 1, 1, 1],
        "connectors": [("Type2", 22.0), ("Type2", 22.0), ("J1772", 7.4), ("J1772", 7.4)],
    },
    {
        "name": "Market Square Retail",
        "location_area": "Market Square",
        "latitude": 45.440847, "longitude": 9.173472,
        "capacity_kw": 150.0,
        "daily_sessions": 60,            # shopping hours, busier weekends
        "weekend_factor": 1.15,
        "hour_weights": [2, 1, 1, 1, 1, 1, 2, 4, 7, 10, 13, 15, 16, 14, 13, 13, 12, 10, 8, 6, 5, 4, 3, 2],
        "connectors": [("CCS", 50.0), ("Type2", 22.0), ("J1772", 7.4), ("NACS", 150.0)],
    },
    {
        "name": "Harborview Residential",
        "location_area": "Harborview",
        "latitude": 45.472155, "longitude": 9.045120,
        "capacity_kw": 60.0,
        "daily_sessions": 35,            # evening/overnight peaks
        "weekend_factor": 1.10,
        "hour_weights": [3, 2, 2, 1, 1, 1, 2, 4, 5, 4, 4, 4, 4, 4, 5, 6, 9, 14, 18, 17, 15, 13, 10, 6],
        "connectors": [("Type2", 22.0), ("Type2", 22.0), ("J1772", 7.4), ("CHAdeMO", 50.0)],
    },
]


def _weighted_hour(rng: random.Random, weights: List[int]) -> int:
    return rng.choices(range(24), weights=weights, k=1)[0]


def _build_sessions(
    rng: random.Random, now: datetime, profiles: List[dict]
) -> List[dict]:
    """Deterministic session history honoring each station's demand curve.

    ``profiles`` entries carry ``{"connectors": [(conn_row, max_kw), ...],
    "daily_sessions", "weekend_factor", "hour_weights"}``.
    """
    rows: List[dict] = []
    for profile in profiles:
        connectors = profile["connectors"]  # [(Connector, max_kw)]
        weights = [max_kw for _conn, max_kw in connectors]  # heavier connectors draw more load
        for d in range(DAYS_OF_HISTORY, 0, -1):
            day_start = (now - timedelta(days=d)).replace(hour=0, minute=0, second=0, microsecond=0)
            weekend = day_start.weekday() >= 5
            count = max(
                0,
                int(round(
                    profile["daily_sessions"]
                    * (profile["weekend_factor"] if weekend else 1.0)
                    * rng.uniform(0.85, 1.15)
                )),
            )
            for _ in range(count):
                conn, max_kw = rng.choices(connectors, weights=weights, k=1)[0]
                is_fast = max_kw >= 50.0
                start = (
                    day_start
                    + timedelta(hours=_weighted_hour(rng, profile["hour_weights"]))
                    + timedelta(minutes=rng.randrange(60))
                )
                # fast sessions are short top-ups; slow sessions park for hours
                minutes = rng.uniform(20, 75) if is_fast else rng.uniform(45, 330)
                hours = minutes / 60.0
                low, high = (0.45, 0.95) if is_fast else (0.60, 1.00)
                draw = min(max_kw, max_kw * rng.uniform(low, high))
                energy = draw * hours * rng.uniform(0.75, 1.0)  # charge tapers near full
                peak = min(max_kw, draw * rng.uniform(0.85, 1.0))
                # a couple of very recent sessions still on the cable
                active = (now - start) < timedelta(hours=24) and rng.random() < 0.04
                rows.append({
                    "connector": conn,
                    "started_at": start,
                    "minutes": minutes,
                    "energy_kwh": round(energy, 3),
                    "peak_kw": round(peak, 2),
                    "active": active,
                })
    return rows


def _build_telemetry(rng: random.Random, now: datetime, profiles: List[dict]) -> List[dict]:
    """Healthy trailing telemetry for the fast connectors (last 3h, 5-min
    cadence) so the live detector has a realistic window to score against."""
    rows: List[dict] = []
    for profile in profiles:
        for conn, max_kw in profile["connectors"]:
            if max_kw < 50.0:
                continue
            base = max_kw * rng.uniform(0.55, 0.85)  # steady session draw
            # +20 min gap before "now" — the detector's recent-window guardrail
            # only sees readings ingested live, keeping standalone runs testable
            for n in range(36, 0, -1):
                recorded = now - timedelta(minutes=5 * n + 20)
                power = min(max_kw, base * rng.uniform(0.92, 1.04))
                rows.append({
                    "connector": conn,
                    "recorded_at": recorded,
                    "power_kw": round(power, 2),
                    "voltage_v": round(rng.uniform(394, 402), 1),
                    "current_a": round(power * 1000 / 400, 1),
                    "temperature_c": round(30 + (power / max_kw) * 22 + rng.uniform(-1.5, 1.5), 1),
                    "is_anomalous": False,
                })
    return rows


def _build_faults(rng: random.Random, now: datetime, profiles: List[dict]) -> List[dict]:
    specs = [
        {
            "days_ago": 68, "type": "connector_failure", "severity": "high",
            "description": "Connector reported zero power output with anomalous readings during an active session",
            "suggested_action": "Inspect connector hardware and check communication logs",
            "evidence": {
                "detector": {"kind": "guardrail", "guardrail": "zero_power_streak"},
                "streak_minutes": 12,
            },
        },
        {
            "days_ago": 54, "type": "power_anomaly", "severity": "medium",
            "description": "Recurring anomalous telemetry readings detected in the recent window",
            "suggested_action": "Inspect connector power electronics and review telemetry history",
            "evidence": {
                "detector": {"kind": "guardrail", "guardrail": "recurring_anomalous_readings"},
                "anomalous_fraction": 0.38,
            },
        },
        {
            "days_ago": 33, "type": "overtemperature", "severity": "high",
            "description": "Connector temperature exceeded the 85°C safety threshold during charging",
            "suggested_action": "Pause connector and inspect cooling before resuming service",
            "evidence": {
                "detector": {"kind": "guardrail", "guardrail": "temperature_threshold"},
                "peak_temperature_c": 87.5,
            },
        },
        {
            "days_ago": 14, "type": "communication_loss", "severity": "medium",
            "description": "Telemetry stream was unreachable for an extended period",
            "suggested_action": "Check network connectivity and reconnect the charge point",
            "evidence": {
                "detector": {"kind": "guardrail", "guardrail": "telemetry_gap"},
                "gap_hours": 3.5,
            },
        },
    ]
    faults = []
    for spec in specs:
        chosen = rng.choice(profiles)
        # fast connectors are the safety-critical ones for these fault classes
        candidates = [(c, kw) for c, kw in chosen["connectors"] if kw >= 50.0] or chosen["connectors"]
        conn, _kw = rng.choice(candidates)
        detected = now - timedelta(days=spec["days_ago"], hours=rng.uniform(0, 6))
        faults.append({
            "station_id": chosen["station_id"],
            "connector_id": conn.id,
            "fault_type": spec["type"],
            "severity": spec["severity"],
            "status": "resolved",
            "detected_at": detected,
            "resolved_at": detected + timedelta(hours=rng.uniform(1.5, 8)),
            "evidence": spec["evidence"],
            "description": spec["description"],
            "suggested_action": spec["suggested_action"],
        })
    return faults


async def _wipe(session: AsyncSession) -> None:
    """Delete all demo rows, respecting FK order."""
    for model in (DemandForecast, OperationalInsight, FaultEvent, TelemetryReading, ChargingSession, Connector, Station, User):
        await session.execute(delete(model))


async def _seed(session: AsyncSession) -> dict:
    rng = random.Random(SEED)
    now = datetime.now(timezone.utc)

    # --- users ---
    session.add_all([
        User(username="alex.operator", email="alex@chargepilot.dev",
             password_hash=hash_password("Operator!123"), role="operator",
             full_name="Alex Morgan"),
        User(username="sam.driver", email="sam@chargepilot.dev",
             password_hash=hash_password("Driver!123"), role="driver",
             full_name="Sam Ortiz"),
    ])

    # --- stations + connectors ---
    profiles: List[dict] = []
    for spec in STATION_SEEDS:
        station = Station(
            id=new_uuid(),  # assigned now: connectors below reference it
            name=spec["name"],
            location_area=spec["location_area"],
            latitude=Decimal(str(spec["latitude"])),
            longitude=Decimal(str(spec["longitude"])),
            total_capacity_kw=Decimal(str(spec["capacity_kw"])),
            operator_notes=f"Demo station — {spec['location_area']} profile",
            is_simulated=True,
            status="operational",
        )
        session.add(station)
        await session.flush()  # stations first: no ORM relationships, order is manual
        connectors = [
            Connector(station_id=station.id, connector_type=ctype,
                      max_power_kw=Decimal(str(max_kw)), status="available",
                      is_simulated=True)
            for ctype, max_kw in spec["connectors"]
        ]
        session.add_all(connectors)
        profiles.append({
            "station_id": station.id,
            "connectors": [(conn, float(conn.max_power_kw)) for conn in connectors],
            "daily_sessions": spec["daily_sessions"],
            "weekend_factor": spec["weekend_factor"],
            "hour_weights": spec["hour_weights"],
            "obj": station,
        })

    await session.flush()  # IDs assigned before sessions/faults reference them

    # --- charging sessions (~17k over ~90 days) ---
    generated = _build_sessions(rng, now, profiles)
    active: dict = {}
    session_rows: List[ChargingSession] = []
    for row in generated:
        conn = row["connector"]
        started = row["started_at"]
        if row["active"]:
            busy = active.setdefault(conn.id, conn)
            busy.status = "charging"
            busy.last_status_changed_at = started
            session_rows.append(ChargingSession(
                connector_id=conn.id, station_id=conn.station_id,
                started_at=started, ended_at=None,
                energy_kwh=None, peak_power_kw=None,
                session_status="active", is_simulated=True,
            ))
        else:
            session_rows.append(ChargingSession(
                connector_id=conn.id, station_id=conn.station_id,
                started_at=started,
                ended_at=started + timedelta(minutes=row["minutes"]),
                energy_kwh=Decimal(str(row["energy_kwh"])),
                peak_power_kw=Decimal(str(row["peak_kw"])),
                session_status="completed", is_simulated=True,
            ))
    for i in range(0, len(session_rows), 2500):
        session.add_all(session_rows[i:i + 2500])
        await session.flush()

    # --- telemetry (healthy trailing window for fast connectors) ---
    telemetry_specs = _build_telemetry(rng, now, profiles)
    session.add_all([
        TelemetryReading(
            connector_id=row["connector"].id,
            station_id=row["connector"].station_id,
            recorded_at=row["recorded_at"],
            power_kw=Decimal(str(row["power_kw"])),
            voltage_v=Decimal(str(row["voltage_v"])),
            current_a=Decimal(str(row["current_a"])),
            temperature_c=Decimal(str(row["temperature_c"])),
            is_anomalous=row["is_anomalous"],
            is_simulated=True,
        )
        for row in telemetry_specs
    ])

    # --- historical faults (guardrail-style, all resolved) ---
    fault_specs = _build_faults(rng, now, profiles)
    faults = [FaultEvent(is_simulated=True, is_false_positive=False, **spec) for spec in fault_specs]
    session.add_all(faults)
    await session.flush()  # fault IDs needed for insight generation below

    # --- operational insights (from two of the faults) ---
    insight_service = OperationalInsightService(session)
    made: List[OperationalInsight] = []
    for fault in faults[:2]:
        conn = await session.get(Connector, fault.connector_id)
        insight = await insight_service.generate_from_fault(fault, connector=conn)
        if insight is not None:
            made.append(insight)

    await session.flush()
    return {
        "users": 2,
        "stations": len(profiles),
        "connectors": sum(len(p["connectors"]) for p in profiles),
        "sessions": len(session_rows),
        "active_sessions": len(active),
        "telemetry_readings": len(telemetry_specs),
        "faults": len(faults),
        "insights": len(made),
    }


DB_EMPTY_MARKER = User  # presence of any user ⇒ seeded


async def main(reset: bool) -> None:
    from app.core.config import settings  # noqa: PLC0415 - late import keeps --help fast

    print(f"Target database: {settings.DATABASE_URL.rpartition('/')[2] or settings.DATABASE_URL}")

    created = await create_database_if_missing(settings.DATABASE_URL)
    print("Database created." if created else "Database already exists (or file-based).")

    await create_tables()
    print("Schema ensured (create_all).")

    async with AsyncSessionLocal() as session:
        existing = int(
            (await session.execute(select(func.count()).select_from(DB_EMPTY_MARKER))).scalar() or 0
        )
        if existing and not reset:
            print(f"Database already seeded ({existing} users). Use --reset to wipe and reseed.")
            return
        if reset:
            await _wipe(session)
            print("Existing demo data wiped.")
        counts = await _seed(session)
        await session.commit()

    print("Seeded:", ", ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--reset", action="store_true",
                        help="wipe all demo tables and reseed from scratch")
    args = parser.parse_args()
    asyncio.run(main(args.reset))
