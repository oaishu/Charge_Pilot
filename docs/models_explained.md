# How ChargePilot's ML models work — a plain-language guide

*This document assumes you know nothing about the application. No math
background required. (Engineers: the precise version lives in
`ml/docs/functionality.md`.)*

---

## 1. What is this application, in one paragraph?

Imagine you run a company with **electric car charging stations** spread
around a city. Three everyday problems annoy you:

1. **Will my station be overloaded tomorrow at 6 pm?** If yes, drivers will
   queue and complain.
2. **Is a charger secretly broken?** A charging cable that plugs in but
   delivers *no* electricity looks "fine" on a screen until someone checks.
3. **Who needs to do what?** Your small ops team can't stare at dashboards
   all day — they need a short list: "Station A tomorrow lunch: busy.
   Station B: charger #2 probably dead — go look."

This application answers all three with ML models + fair rules, and talks to
the world through an **API** (a web address other programs call to get
answers). It comes with its own small database, demo data, and a login
system, so it runs by itself — copy the folder, one command, it works.

---

## 2. The main characters (before any ML)

**Humans use cars... chargers are the "gas pumps."**
Every time someone charges, the system writes two kinds of records in the
database:

- **A session** — "a car plugged in at 9:14, unplugged at 10:05, took 7.2 kWh
  while drawing up to 11 kW." Thousands of these make up the history.
- **Telemetry readings** — the charger reporting its pulse every few minutes
  while a car is on the cable: how much power, what temperature, plus a flag
  if it noticed something weird.

Keep these two apart — they are the inputs to the two different models:

| Model | Eats | Warns about |
|---|---|---|
| **The forecaster** (RandomForest) | session history (minutes/hours/days grids of "how busy") | "heavy load coming" |
| **The fault detector** (GradientBoosting) | telemetry readings (the charger's pulse) | "this charger is misbehaving" |

---

## 3. The golden rule: the "obvious answer" always exists

Both models live on top of something dumber but more reliable:

- For demand: the **baseline**. Look at this station's own history and
  average it: "on Fridays at 6 pm, this station drew about 40 kW on
  average." Simple, explainable, invisible-proof.
- For faults: the **guardrails**. Fixed physical rules like "if a charging
  cable reports 85 °C, that's a fire risk — raise it now" and "if a session
  is *active* but reads 0 kW for 10+ minutes, the charger is dead." No
  statistics, no black box.

**The trained models are only allowed to be the opinion on top of the
obvious answer — never the replacement for it.** If the ML file is missing
or broken, the app still works and still catches fires. The ML makes the
obvious answer a little smarter; it never makes it dumber.

---

## 4. How the forecaster was built (the simple story)

**Step 1 — collect the diaries.** Public records of real charging sessions
from three places (a Korean study, the London borough of Barnet, and the US
Caltech workplace network) were merged and cleaned into one big history:
~146,000 sessions.

**Step 2 — ask the right question.** Not "car A plugs in at 9:14?" (too
fine, unknowable) but "how busy in general?" Two questions, two lookup
tables the model learns:

- per station, per **day**: how many sessions? ("Friday + downtown = 20")
- per station, per **hour**: how much power while in use? ("9 am = 60 kW")

**Step 3 — the exam before the training.** Because predicting the future is
the whole point, the records were split **by date**: older data to learn
from (~80%), newest ~20% kept hidden as the final exam. Never mixed, never
shuffled — the model may not peek at its exam.

**Step 4 — the only homework the model may copy.** Two honest ingredients
that also exist live in the app:

- *profile*: "what this station usually does on Saturday mornings" (an
  average from older data only)
- *recent history*: "what the station did yesterday / last week"

**Step 5 — train two RandomForests.** Two were trained (one per question). A random
forest is like **asking 200 mildly-different advisors and averaging their
answers** — each sees the data slightly differently, and the crowd is
steadier than any single expert.

**Step 6 — the report card (kept honest on purpose):**

| Question | Obvious answer was wrong by | Model was wrong by | Verdict |
|---|---|---|---|
| sessions/day | 0.42 | **0.35** | genuinely better (~1 in 6 fewer errors) |
| hourly load | 10.42 | **10.41** | barely better — the obvious answer is already very good here |

That last row is *published on purpose*. The model's whole power comes from
"what is this station's habit" — which is also what the obvious answer uses.
So on hourly load, the model can only make small corrections. The app
respects that: no matter what the model says, "**the corrected hour must stay
within a half-to-one-and-a-half-times band around the obvious answer**" — the
model can say "a bit quieter" or "a bit busier", it cannot invent a stampede.

**Step 7 — the tripwire.** The trained file carries a flag: `I beat the
obvious answer on BOTH questions`. The app refuses files without it. A bad
retrain cannot silently degrade the service — it's simply ignored, and the
obvious answer keeps serving.

---

## 5. How the fault detector was built (the simple story)

**Step 1 — make the answer key.** No dataset screams "this charger broke!"
So the notebook wrote rules an engineer would approve and stamped each
historical session:

- power ≈ 0 the whole "charge" → **dead charger** (`no_energy_failure`)
- still "charging" after 24 h → **stuck relay** (`overlong_session`)
- absurd power readings → **metering anomaly** (`power_anomaly`)
- everything else, ~99% of rows → **normal**

These are called **weak labels**: convenient, but a bit circular — the model
may just be relearning the rules. The report card takes this seriously (next
step).

**Step 2 — tune the honesty dial.** The classifier is wrapped in
**calibration**, so when it says "`0.9` chance fault", 9 out of 10 such
windows really are faults. That matters because the API uses these numbers
as hard thresholds.

**Step 3 — the report card.**

- The classifier reproduces the rule-made labels with (nearly) perfect
  scores — *consistent*, but unconvincing for the circularity above.
- The number worth trusting most comes from a **second model that never saw
  any labels**: an IsolationForest shown ONLY normal traffic, asked to flag
  weird things on its own. It catches **56% of true misbehaviors now (was
  23% before the workplace data improved the dataset)**. That breakout
  proves the weirdness is real and learnable, independent of our answer key.

**Step 4 — the deployment deal.** The fault model file also carries its own
threshold (`0.60`) frozen at training time, so the notebook's exam and the
live service use the same number.

---

## 6. What actually happens when the app is used

### A. "Will it be busy?" — `POST /api/v1/forecasts/generate`

```
operator asks: station X, next 4 hours
   1. look up the station's last 90 days of sessions → the "habit table"
   2. walk forward hour by hour reading the habit (the obvious answer)
   3. the RandomForest may nudge each hour up/down (the band-bound),
      per charger type
   4. compare predicted load vs the station's rated capacity:
      <60% → "low"     60–85% → "medium"     85%+ → "high" pressure
   5. save the forecast (with model name + scorecard inside)
   6. if pressure = medium/high → write an "insight" so an operator sees it
```

### B. "Is a charger broken?" — `POST /api/v1/telemetry`

The charger (or the demo/verify script) sends its readings:

```
push readings (power, temperature, anomaly flags)
   1. store them
   2. grab the LAST ~2 hours of readings for that charger
   3. guardrails first:   temperature ≥85 °C → fire risk, IMMEDIATE fault
                          readings flagged weird ≥50% → suspicious
                          0 kW streak in the last 20 min → dead? fault
   4. the model reads the readings SINCE THE LAST WEIRD ONE
      (a fresh failure isn't drowned by hours of the charger being fine)
   5. whoever is more confident wins — but the temperature rule wins ALWAYS
   6. if the winner's probability ≥ 0.60 → a FAULT EVENT is saved
      - automatically accompanied by evidence: the sample readings, the
        model's per-possibility odds, and a "generated by model" vs
        "generated by rule" label
   7. one linked INSIGHT is written for the fault — always
   8. guard against spam: a charger that already has an OPEN fault — that is
      being worked on — will not produce another (the same one stays "open")
```

So: a healthy charger pushes readings and the app quietly nods. A broken
one triggers a fault event *and* a human's to-do in the same breath.

### C. "What should my team do?" — `GET /api/v1/insights`

The insights list is the **operator's to-do**, and it's *produced*, never
hand-written. Three sources: a detected fault → one insight; a forecast
under pressure → one insight ("elevated demand predicted at Station A —
consider load balancing"); two seeded starter ones on a fresh install. Once
an operator handles one (`PATCH` → acknowledged → resolved), the audit-trail
stamps who/when. It never shows duplicates: the same charger doesn't
accumulate open faults; the same forecast doesn't re-summon its insight.

### D. The rest of the menu

| Endpoint | plain words |
|---|---|
| `POST /api/v1/auth/login` | "prove who you are (username + password)" → a token for the hour |
| `GET /api/v1/auth/me` | "who am I logged in as?" |
| `GET /api/v1/forecasts` | "show me past forecasts (filter by station/horizon/etc.)" |
| `GET /api/v1/forecasts/{id}` | "open one forecast and show all its details" |
| `GET /api/v1/faults` | "list all faults (filter unresolved / by station)" |
| `GET /api/v1/faults/{id}` | "open one fault: evidence + model's reasoning" |
| `PATCH /api/v1/faults/{id}` | "I looked: it's real → investigating/resolved, or false alarm" |
| `PATCH /api/v1/insights/{id}` | "I handled this: acknowledge it" or "close it" |
| `GET /health` | "are you alive?" (service + a standalone database + which model files are in use) |

Everything except login requires the operator's token (the applications'
owners). There is no background job: everything happens when a request
arrives.

---

## 7. Words you'll hear (tiny glossary)

| Term | Meaning here |
|---|---|
| **baseline** | the "historical average" obvious answer for demand |
| **guardrails** | fixed physical rules that catch faults without ML |
| **telemetry** | a charger's periodic readings (power, temperature, etc.) |
| **p / probability** | the model's odds that a window is a fault (0.9927 = 99.27%) |
| **MAE** | "on average, how far off the predictions were" (lower is better) |
| **horizon** | how far ahead a forecast looks: 1h / 4h / 24h |
| **capacity pressure** | predicted load vs rated capacity (%: low / medium / high) |
| **weak labels** | answer keys made by rules instead of humans |
| **IsolationForest** | the second fault model that finds weirdness *without* labels |
| **bounded ratio** | the demand model's leash: ±50% around the obvious answer |
| **insight** | an operator's to-do; produced automatically, closed by a human |
| **artifact** | the trained model file (`.pkl`) with its self-check flags |

---

## 8. The 10-second summary

> *"A folder containing: historic real-world charging data, two honest
> models trained on it (a load forecaster with a measured, moderate edge,
> and a fault classifier wrapped in physical safety rules), a self-contained
> API that serves them against its own database, and a to-do list that
> writes itself whenever the data says something*" — you copy the folder,
> run three commands, and a fresh machine has a working network-monitoring
> brain.*

*Derived from the notebooks' actual outputs as of Nov 2026; source of truth
is the code (`ml/notebooks`, `ml/service`).*
