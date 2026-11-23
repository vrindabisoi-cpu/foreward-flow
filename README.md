# ForeWard Flow — Halvian Medical Center (fictional, 320 beds)

A staff-facing Streamlit dashboard for **emergency department (ED) patient flow and boarding** at Halvian Medical Center, a fictional 320-bed hospital, built for a hospital case competition.

> **All data is synthetic, calibrated to the Halvian case brief.** No real patients, staff, or hospital records are used. Every dataset is generated in code from a fixed random seed (`42`), so the numbers are the same on every run. The only non-synthetic numbers are the case brief's equity baseline, shown in the Equity tab.

## What it shows

### AI Control Center tab

- **Live bed counts** per unit: 🟢 open, 🟡 needs cleaning, 🔴 occupied.
- **ED room status:** 32 treatment rooms, 8 fast-track rooms, 4 resuscitation/trauma bays, 4 behavioral health rooms, and a 40-seat waiting room. Shows how many treatment rooms boarders are holding (e.g., "6 of 32 ED rooms held by boarders") and how many patients are waiting.
- **ED boarding:** admitted patients boarding, total boarding hours, and today's boarding cost at $350 per boarding hour.
- **8-hour bed forecast (9 AM–5 PM)** for a selected unit, with a red warning banner when the unit is projected to run out of beds. The banner lists the reasons and shows how many hours of head start staff have.
- **Recommended capacity tier** with backup actions. The house supervisor approves or declines each action.
- **Public health & news alerts** (simulated).
- **Decision log** of every decision, with time, role, and reason. It can be downloaded as a CSV.

### ED Admissions tab

- One card per ED patient: age, chief complaint, triage acuity, vitals, admit probability (progress bar), the top 3 reasons, likely bed type and the bed rule that fired, and hours boarding (red over 2 hours).
- **Physician confirms level of care · Bed manager assigns room.** Only the **Physician** or **Bed manager** role can **Accept** the AI bed type or **Override** it.
- An override needs a corrected bed type (Medicine, Telemetry, Surgery, ICU, Women's & Peds, Not admitting) and a reason (different level of care, isolation needed, likely discharge, room constraint, other). "Other" also needs a note.
- Every decision goes to the decision log with time, role, and reason.

### Barrier Flags tab

- Inpatients ranked by **likely to go home today**, with probability and top reasons.
- **Scan notes with AI** sends each note to the Claude API (official `anthropic` Python SDK, `POST /v1/messages`, structured JSON output). Claude returns each barrier tagged **Medical** or **Social** with a suggested owner team (Social work, Case management, Pharmacy, Hospitalist). It is told not to make clinical decisions, and only the note text is sent, never insurance or demographics.
- **API key:** entered in the sidebar password box or set as `ANTHROPIC_API_KEY`. It is never hard-coded, saved, or logged. The model name is editable in the sidebar; the default is `claude-opus-5-5`. Server-side refusal fallback is on for models that support it.
- **Offline fallback:** with no key, or for any note whose call fails, keyword rules flag the barriers instead. The page always says which source was used ("Claude API" or "Offline fallback") and why.
- Totals at the top, orange **Social** and blue **Medical** flags per patient, the note in an expander, and an **Assign to [owner team]** button. Only the **Case manager** or **Social worker** role can assign; each assignment is logged with the barrier text.
- A bar chart shows the % of scanned patients with a social barrier by insurance group. Help is routed by documented need, never by insurance type.

### What-If tab

Choose how many hours less each admitted patient boards in the ED. Earlier discharges and faster room cleaning are what make beds ready sooner. Two presets set the slider: **Month-18 goal: 1 hour less boarding (3.5 → 2.5 h)** and **Conservative: 45 minutes**. Every number shows its formula underneath.

**Per year (case brief figures, to match the slides):** 85,000 ED visits × 16% admitted = **13,600 admissions**. Hours saved per admission is the slider value, capped at the 3.5-hour median boarding time.

| | Month-18 goal (1 h) | Conservative (45 min) | Formula |
|---|---|---|---|
| **Value line 1 — Boarding cost avoided** | $4,760,000 | $3,570,000 | 13,600 × hours × $350 |
| ED room-hours freed | 13,600 | 10,200 | 13,600 × hours |
| Patients roomed sooner | 4,124 | 3,093 | room-hours × 0.303 patients/room-hour (233 arrivals/day ÷ 32 rooms ÷ 24 h) |
| Recovered visits (estimate) | 165 | 124 | patients roomed sooner × 4% LWBS rate |
| **Value line 2 — Conservative estimate (freed ED rooms only)** | ≈ $250,000 | ≈ $190,000 | recovered visits × $1,500, rounded to the nearest $10K |

**Upper estimate:** If the Month-18 goal is met (LWBS 4.0% → 2.8%): 85,000 × 1.2% = 1,020 visits × $1,500 ≈ $1.53M/yr.

**Today** uses the 6 current boarders: each leaves up to the slider value sooner, never more than they actually boarded. Value line 2 is labelled in the app as an estimate, not a forecast.

### Sidebar

- **ED staffing for the current shift**, as context only: Day shift 7a–3p, 4 physicians, 3 APPs, 14 nurses, 8 techs/registration.

### Equity tab

- **Halvian baseline** from the case brief: door-to-provider time and left-without-being-seen (LWBS) rate by payer, with gaps against Commercial.
- **Boarding time by payer** from the synthetic 30-day boarding history.

## Models

Both models are **logistic regressions** (scikit-learn), chosen because they're explainable: every prediction can be broken down into how much each input moved it. Each is trained on 75% of its synthetic history and tested on the other 25%.

| Model | Trained on | Inputs | Test AUC |
|---|---|---|---|
| Admission at triage | 6,000 past ED visits | age, triage acuity, heart rate, O₂ sat, temperature, blood pressure, chief complaint, cardiac history, prior ED visits | 0.84 |
| Likely home today | 3,000 past inpatient days | days past expected stay, pending test, IV meds, mobility, physician noted likely discharge today, documented social barrier | 0.89 |

**Reasons.** Each reason is an input's coefficient times its standardized value: how far it moves this patient's score compared with the average patient in the training data.

**Likely bed type** uses fixed rules, not the model, checked in this order:
1. **Women's & Peds** if age is under 18.
2. **ICU** if triage acuity is 1, O₂ sat is below 89%, or systolic BP is below 90.
3. **Telemetry** for chest pain, or cardiac history with heart rate above 105.
4. **Surgery** for abdominal pain or fall/injury.
5. Otherwise **Medicine**.

## Bed forecast

Projected open beds at each hour from 9 AM to 5 PM:

> **open now + rooms cleaned + expected discharges − expected admissions**

- **Rooms cleaned:** beds that need cleaning come back at one per hour.
- **Expected discharges** come from the likely-home-today model. The unit's average probability is applied to every occupied bed, and discharges are weighted toward the afternoon, mostly after 2 PM.
- **Expected admissions** are:
  - boarders already admitted to the unit, placed one per hour as beds open;
  - ED patients in workup whose likely bed is that unit, weighted by admit probability;
  - new ED arrivals by hour × the 16% admit rate × the unit's share of ED admissions (Medicine 60%), adjusted by:
    - **day of week:** Monday +15% (Sunday −8%), from the arrival pattern;
    - **local flu/RSV trend:** Normal, Elevated +20%, or High +35%. This stands in for the brief's +20% winter respiratory season, so the season is not counted twice;
    - **news alerts** a house supervisor has applied.

A unit **runs out** when projected open beds drop below zero.

The sidebar day-of-week and flu/RSV trend are set automatically from the synthetic "today" (Monday) and the simulated surveillance feed. Staff can override both.

With the default inputs (Monday, elevated flu/RSV), **Medicine is projected to run out of beds by 12 PM and be about 2 beds short by 2 PM**. That gives staff a **3-hour head start** from the 9 AM forecast.

### Capacity tiers

The tier depends on how soon the unit runs out and by how many beds. With the default inputs, Medicine runs out in 3 hours and is short about 2 beds, so the recommendation is **Level 1 — Tight**: there's still time to free beds before escalating. Monday with a High flu/RSV trend plus the applied ambulance-diversion alert reaches **Level 3 — Crisis**.

| Tier | When | Backup actions |
|---|---|---|
| Normal | Never runs out; about 3 or more open beds all day | Routine huddles |
| Level 1 — Tight | Down to 0–2 open beds, or runs out 3+ hours away and short 2 beds or fewer | Work the likely-home-today list first; open discharge lounge; prioritize cleaning; extra bed huddle |
| Level 2 — Full | Runs out within 2 hours, or short more than 2 beds | Flex beds with extra nurse shifts; hallway-bed protocol for stable patients (RN approval); inpatient teams round on ED boarders; fast-track |
| Level 3 — Crisis | Short 3+ beds within 2 hours, or short 4+ beds at any time | Delay non-urgent planned surgeries (CMO approval); transfer stable patients; ambulance diversion as a last resort |

### Human in the loop

- **Decision owner: House supervisor.** Only the house supervisor role can approve or decline actions, or apply or dismiss news alerts. Other roles see the page as view-only.
- A reason is required to decline an action or dismiss an alert.
- **Public health data** (simulated CDC/state surveillance) adjusts the forecast automatically.
- **News alerts** (simulated) never change the forecast unless a house supervisor applies them.
- Every decision is logged with time, role, and reason to `data/decision_log.csv`, so it survives a page refresh. The file is git-ignored. Only the house supervisor can clear it, and clearing leaves one entry recording who cleared it.

## Synthetic datasets

All generators live in [`flow_engine.py`](flow_engine.py).

**Calibration to the Halvian case brief**

| Target | Brief | Synthetic data |
|---|---|---|
| ED admit rate | about 16% | 15.9% of 6,000 past visits |
| Payer mix | Medicaid/CHIP 35%, Medicare 22%, Commercial 33%, Uninsured 8%, Other 2% | Same shares wherever insurance is generated |
| ED arrivals | about 233/day (85,000/year) | 85,247 in the past year (233.6/day) |
| Arrival pattern | Weekday morning and evening peaks; Monday busiest; winter respiratory season +20% | Weekday peaks at 10–11 AM and 6–7 PM; Monday about 268/day; December–February +20% |
| Beds | 320 total incl. 24 ICU, about 88% occupied, about 40 open | 281 occupied (87.8%); 39 unoccupied (29 open + 10 needs cleaning); Medicine tightest at 95% |
| Boarding history | median about 3.5 hours | median 211 minutes |

| Function | Contents |
|---|---|
| `generate_bed_status()` | Medicine (120 beds), Surgery (80), Telemetry (56), ICU (24), Women's & Peds (40). Medicine is deliberately tight: 3 open, 3 needing cleaning. |
| `generate_ed_arrivals()` | One year of hourly ED arrivals with day-of-week, time-of-day, and respiratory-season patterns. |
| `generate_ed_visits()` | 6,000 past ED visits: age, triage acuity (1–5), heart rate, O₂ saturation, temperature, blood pressure, chief complaint, cardiac history, prior visits, insurance group, and an `admitted` outcome. |
| `generate_current_ed_patients()` | 14 patients in the ED now: 6 admitted and boarding (3 of them waiting for Medicine, the tightest unit), 8 still in workup. |
| `generate_inpatients()` | 40 inpatients with unit, room, age, insurance, length of stay, and a short nurse note. Notes may mention medical barriers (pending MRI, IV antibiotics, PT eval…) or social barriers (no ride home, can't afford meds, nursing-home placement, shelter, insurance authorization). |
| `generate_past_inpatient_stays()` | 3,000 past inpatient days with days past expected stay, pending test, IV meds, mobility, physician discharge note, social barrier, and a `discharged_that_day` outcome. |
| `generate_boarding_history()` | 30 days of ED boarding times by insurance group. |

**Built-in disparities.** The data includes patterns for the dashboard to surface:
- Social discharge barriers are more common for Medicaid/CHIP and Uninsured patients.
- Medicaid/CHIP patients board about 37 minutes longer than the overall average.

## Fairness safeguards

- ED visits include `race_ethnicity`, `age_group`, and `primary_language`. These are for **fairness checks only** and are listed in `FAIRNESS_ONLY_COLUMNS`.
- The admission outcome is generated from clinical factors only.
- **Insurance, race/ethnicity, age group, and language are never model inputs.** Age as a clinical number is allowed. The code enforces this: building a model with any of these inputs (`FORBIDDEN_MODEL_INPUTS`) raises an error.

## Running it locally

Requires Python 3.12.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

## Project layout

```
app.py            Streamlit UI
flow_engine.py    Synthetic data, models, bed-type rules, and bed forecast
requirements.txt  streamlit, pandas, numpy, scikit-learn, anthropic
```
