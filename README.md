# ForeWard Flow

A staff-facing Streamlit dashboard for **emergency department (ED) patient flow and boarding** at a fictional 320-bed hospital, built for a hospital case competition.

> **All data is synthetic.** No real patients, staff, or hospital records are used. Every dataset is generated in code from a fixed random seed (`42`), so the numbers are the same on every run.

## What it shows

- **Bed status** for five inpatient units: open, needs cleaning, and occupied beds, plus occupancy %.

More views (ED boarding, discharge barriers, equity checks) are planned and use the datasets below.

## Synthetic datasets

All generators live in [`flow_engine.py`](flow_engine.py).

| Function | Contents |
|---|---|
| `generate_bed_status()` | Medicine (120 beds), Surgery (80), Telemetry (56), ICU (24), Women's & Peds (40). Medicine is deliberately tight: 3 open, 3 needing cleaning. |
| `generate_ed_visits()` | 6,000 past ED visits: age, triage acuity (1–5), heart rate, O₂ saturation, temperature, blood pressure, chief complaint, cardiac history, prior visits, insurance group, and an `admitted` outcome. |
| `generate_current_ed_patients()` | 14 patients in the ED now: 6 admitted and boarding (with hours boarding), 8 still in workup. |
| `generate_inpatients()` | 40 inpatients with unit, room, age, insurance, length of stay, and a short nurse note. Notes may mention medical barriers (pending MRI, IV antibiotics, PT eval…) or social barriers (no ride home, can't afford meds, nursing-home placement, shelter, insurance authorization). |
| `generate_past_inpatient_stays()` | 3,000 past inpatient days with a `discharged_that_day` outcome. |
| `generate_boarding_history()` | 30 days of ED boarding times by insurance group. |

**Insurance mix:** Commercial 33%, Medicare 22%, Medicaid 35%, Uninsured 10%.

**Built-in disparities.** The data includes patterns for the dashboard to surface:
- Social discharge barriers are more common for Medicaid and Uninsured patients.
- Medicaid patients board about 36 minutes longer than the overall average.

## Fairness safeguards

- ED visits include `race_ethnicity`, `age_group`, and `primary_language`. These are for **fairness checks only** and are listed in `FAIRNESS_ONLY_COLUMNS`.
- The admission outcome is generated from clinical factors only.
- Model inputs are limited to `ED_MODEL_FEATURES`, which leaves out the fairness fields and insurance.

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
flow_engine.py    Synthetic data generation and logic
requirements.txt  streamlit, pandas, numpy, scikit-learn
```
