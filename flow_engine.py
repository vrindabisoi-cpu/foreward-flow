"""ForeWard Flow — synthetic data and logic engine.

ALL DATA IN THIS MODULE IS SYNTHETIC. No real patients, staff, or hospital
records are used. Every generator uses a fixed seed so results are
reproducible across runs.

Each generator gets its own seeded random stream (SEED + offset), so adding
or changing one dataset never shifts the values of another.
"""

from __future__ import annotations

import csv
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
REFERENCE_DATE = pd.Timestamp("2026-11-23")  # fixed "today" for reproducibility

HOSPITAL_NAME = "ForeWard Flow — Halvian Medical Center (fictional, 320 beds)"

UNIT_CAPACITY = {
    "Medicine": 120,
    "Surgery": 80,
    "Telemetry": 56,
    "ICU": 24,
    "Women's & Peds": 40,
}

# Payer mix from the Halvian case brief, used everywhere insurance is generated.
MEDICAID = "Medicaid/CHIP"
INSURANCE_GROUPS = ["Commercial", "Medicare", MEDICAID, "Uninsured", "Other"]
INSURANCE_SHARES = [0.33, 0.22, 0.35, 0.08, 0.02]
SOCIAL_RISK_PAYERS = (MEDICAID, "Uninsured")

CHIEF_COMPLAINTS = {
    # complaint: (sampling weight, admission log-odds effect)
    "Chest pain": (0.12, 0.6),
    "Shortness of breath": (0.10, 0.9),
    "Abdominal pain": (0.14, 0.2),
    "Fever": (0.07, 0.3),
    "Fall / injury": (0.11, 0.1),
    "Altered mental status": (0.05, 1.2),
    "Headache": (0.06, -0.6),
    "Back pain": (0.07, -0.8),
    "Syncope": (0.05, 0.5),
    "Laceration": (0.06, -1.5),
    "Nausea / vomiting": (0.08, -0.1),
    "Psychiatric concern": (0.05, 0.0),
    "Weakness": (0.04, 0.7),
}

# Columns that exist ONLY for fairness auditing. They must never be used as
# model inputs.
FAIRNESS_ONLY_COLUMNS = ["race_ethnicity", "age_group", "primary_language"]

# Clinical inputs permitted for the ED admission model. Insurance is kept out
# of the model too, so it can be audited rather than learned from.
ED_MODEL_FEATURES = [
    "age",
    "triage_acuity",
    "heart_rate",
    "o2_sat",
    "temperature_c",
    "systolic_bp",
    "diastolic_bp",
    "chief_complaint",
    "cardiac_history",
    "prior_visits_12mo",
]

AGE_GROUP_BINS = [-1, 17, 39, 64, 79, 200]
AGE_GROUP_LABELS = ["0-17", "18-39", "40-64", "65-79", "80+"]


def _rng(offset: int) -> np.random.Generator:
    return np.random.default_rng(SEED + offset)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Bed status
# ---------------------------------------------------------------------------

# Calibrated to the case brief: 320 beds, ~88% occupied (281), 39 unoccupied
# (29 open + 10 needs cleaning). Medicine is the tightest unit.
BED_BOARD = {
    # unit: (open, needs_cleaning)
    "Medicine": (3, 3),
    "Surgery": (10, 2),
    "Telemetry": (5, 2),
    "ICU": (2, 1),
    "Women's & Peds": (9, 2),
}


def generate_bed_status() -> pd.DataFrame:
    """Current bed board: open / needs cleaning / occupied per unit."""
    rows = []
    for unit, total in UNIT_CAPACITY.items():
        open_beds, cleaning = BED_BOARD[unit]
        occupied = total - open_beds - cleaning
        rows.append(
            {
                "unit": unit,
                "total_beds": total,
                "open": open_beds,
                "needs_cleaning": cleaning,
                "occupied": occupied,
                "occupancy_pct": round(100 * occupied / total, 1),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ED visits (historical + current)
# ---------------------------------------------------------------------------

def _ages_for_insurance(insurance: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Ages conditional on payer, so insurance marginals stay exact."""
    n = len(insurance)
    ages = np.empty(n)
    for i, ins in enumerate(insurance):
        if ins == "Medicare":
            # mostly 65+, a few younger (disability)
            ages[i] = rng.normal(75, 8) if rng.random() < 0.9 else rng.normal(50, 10)
        elif ins in ("Commercial", MEDICAID) and rng.random() < 0.13:
            ages[i] = rng.uniform(0, 17)  # pediatric visits
        else:
            ages[i] = rng.normal(44, 15)
    return np.clip(np.round(ages), 0, 98).astype(int)


def _generate_visit_features(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Demographics, vitals, and history for n synthetic ED visits."""
    insurance = rng.choice(INSURANCE_GROUPS, size=n, p=INSURANCE_SHARES)
    age = _ages_for_insurance(insurance, rng)

    acuity = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.04, 0.18, 0.45, 0.25, 0.08])
    sick = (acuity <= 2).astype(float)

    heart_rate = np.clip(rng.normal(84 + 10 * sick + 0.08 * np.maximum(age - 60, 0), 14), 40, 180)
    o2_sat = np.clip(np.round(rng.normal(97.5 - 3.0 * sick, 1.8)), 75, 100)
    fever = rng.random(n) < (0.08 + 0.10 * sick)
    temperature = np.where(fever, rng.normal(38.8, 0.5, n), rng.normal(36.9, 0.35, n))
    systolic = np.clip(rng.normal(132 + 0.3 * (age - 45) - 12 * sick * (rng.random(n) < 0.3), 20), 70, 220)
    diastolic = np.clip(systolic * rng.normal(0.6, 0.05, n), 40, 130)

    complaints = list(CHIEF_COMPLAINTS)
    weights = np.array([CHIEF_COMPLAINTS[c][0] for c in complaints])
    chief_complaint = rng.choice(complaints, size=n, p=weights / weights.sum())

    cardiac_history = rng.random(n) < _sigmoid(-3.2 + 0.05 * (age - 40))

    visit_rate = np.select(
        [insurance == MEDICAID, insurance == "Uninsured", insurance == "Medicare"],
        [1.3, 1.1, 1.2],
        default=0.7,
    )
    prior_visits = rng.poisson(visit_rate)

    return pd.DataFrame(
        {
            "age": age,
            "triage_acuity": acuity,
            "heart_rate": heart_rate.round().astype(int),
            "o2_sat": o2_sat.astype(int),
            "temperature_c": temperature.round(1),
            "systolic_bp": systolic.round().astype(int),
            "diastolic_bp": diastolic.round().astype(int),
            "chief_complaint": chief_complaint,
            "cardiac_history": cardiac_history.astype(int),
            "prior_visits_12mo": prior_visits,
            "insurance": insurance,
        }
    )


ADMISSION_INTERCEPT = -2.93  # tuned so ~16% of historical ED visits are admitted


def _admission_logit(df: pd.DataFrame) -> np.ndarray:
    """Clinical-only ground truth. Fairness columns play no role."""
    acuity_effect = df["triage_acuity"].map({1: 3.0, 2: 1.7, 3: 0.4, 4: -1.1, 5: -2.2})
    complaint_effect = df["chief_complaint"].map({c: v[1] for c, v in CHIEF_COMPLAINTS.items()})
    return (
        ADMISSION_INTERCEPT
        + acuity_effect
        + complaint_effect
        + 0.022 * (df["age"] - 50)
        + 0.025 * np.maximum(df["heart_rate"] - 100, 0)
        + 0.30 * np.maximum(94 - df["o2_sat"], 0)
        + 0.9 * (df["temperature_c"] >= 38.3)
        + 1.1 * (df["systolic_bp"] < 90)
        + 0.5 * df["cardiac_history"]
        + 0.08 * df["prior_visits_12mo"]
    ).to_numpy()


def _fairness_fields(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(df)
    race = rng.choice(
        ["White", "Black", "Hispanic/Latino", "Asian", "Other/Multiracial"],
        size=n,
        p=[0.52, 0.22, 0.15, 0.06, 0.05],
    )
    language = np.where(
        race == "Hispanic/Latino",
        rng.choice(["English", "Spanish"], size=n, p=[0.55, 0.45]),
        rng.choice(["English", "Arabic", "Other"], size=n, p=[0.93, 0.04, 0.03]),
    )
    age_group = pd.cut(df["age"], bins=AGE_GROUP_BINS, labels=AGE_GROUP_LABELS).astype(str)
    return pd.DataFrame(
        {"race_ethnicity": race, "age_group": age_group, "primary_language": language},
        index=df.index,
    )


def generate_ed_visits(n: int = 6000) -> pd.DataFrame:
    """Historical ED visits with an `admitted` (1/0) outcome.

    Includes race_ethnicity, age_group, primary_language for FAIRNESS CHECKS
    ONLY (see FAIRNESS_ONLY_COLUMNS); they never feed the model.
    """
    rng = _rng(2)
    df = _generate_visit_features(n, rng)
    df["admitted"] = (rng.random(n) < _sigmoid(_admission_logit(df))).astype(int)
    df = pd.concat([df, _fairness_fields(df, _rng(3))], axis=1)
    df.insert(0, "visit_id", [f"V{100000 + i}" for i in range(n)])
    return df


MEDICINE_BOARDERS = 3


def generate_current_ed_patients() -> pd.DataFrame:
    """14 patients in the ED now: 6 admitted & boarding, 8 still in workup."""
    rng = _rng(4)
    # Boarders come from synthetic patients whose sampled outcome was
    # "admitted", so they look like real admits. Workup patients are unselected.
    pool = _generate_visit_features(300, rng)
    true_p = _sigmoid(_admission_logit(pool))
    was_admitted = rng.random(len(pool)) < true_p
    bed = pool.apply(lambda r: predict_bed_type(r)[0], axis=1)
    admits = pool[was_admitted]
    to_medicine = bed[was_admitted] == "Medicine"
    # Boarding piles up on the tightest unit, so half the boarders need Medicine.
    # Demo: lead with the clearest-cut adult Medicine admit (highest true probability).
    candidates = admits[to_medicine & (admits["age"] >= 18)]
    lead = candidates.loc[[pd.Series(true_p, index=pool.index)[candidates.index].idxmax()]]
    medicine = admits[to_medicine].drop(lead.index).head(MEDICINE_BOARDERS - len(lead))
    other = admits[~to_medicine].head(6 - len(lead) - len(medicine))
    boarders = pd.concat([lead, medicine, other])
    workup = pool.drop(boarders.index).head(8)
    df = pd.concat([boarders, workup], ignore_index=True)

    status = ["Boarding"] * 6 + ["In workup"] * 8
    df.insert(0, "patient_id", [f"ED-{201 + i}" for i in range(14)])
    df.insert(1, "bed", [f"ED {i + 1}" for i in range(14)])
    df["status"] = status

    df["admit_unit"] = None
    df.loc[:5, "admit_unit"] = [predict_bed_type(r)[0] for _, r in df.loc[:5].iterrows()]
    df["hours_boarding"] = 0.0
    df.loc[:5, "hours_boarding"] = np.round(rng.uniform(1.5, 14, 6), 1)
    df["hours_in_ed"] = np.round(rng.uniform(0.5, 6, 14), 1)
    df.loc[:5, "hours_in_ed"] = df.loc[:5, "hours_boarding"] + np.round(rng.uniform(2, 5, 6), 1)
    return df


# ---------------------------------------------------------------------------
# ED arrivals (calibrated to the case brief)
# ---------------------------------------------------------------------------

ED_ARRIVALS_PER_YEAR = 85_000  # ~233/day
ED_ADMIT_RATE = 0.16
# Monday is the busiest day; multipliers average to 1.0 across the week.
DOW_ARRIVAL_MULTIPLIER = {
    "Monday": 1.15, "Tuesday": 1.03, "Wednesday": 0.99, "Thursday": 0.98,
    "Friday": 0.99, "Saturday": 0.94, "Sunday": 0.92,
}
RESPIRATORY_SEASON_MONTHS = (12, 1, 2)  # winter respiratory season
RESPIRATORY_SEASON_BOOST = 0.20

# Relative arrivals by hour (0 = midnight). Weekdays peak mid-morning and
# early evening; weekends are flatter.
_WEEKDAY_HOURLY = np.array([2.2, 1.8, 1.5, 1.3, 1.3, 1.6, 2.4, 3.4, 4.6, 5.6, 6.2, 6.1,
                            5.6, 5.3, 5.2, 5.3, 5.6, 6.0, 6.3, 6.2, 5.5, 4.6, 3.7, 2.9])
_WEEKEND_HOURLY = np.array([2.8, 2.4, 2.0, 1.7, 1.5, 1.6, 2.0, 2.8, 3.8, 4.7, 5.3, 5.6,
                            5.6, 5.5, 5.4, 5.3, 5.3, 5.4, 5.5, 5.4, 5.0, 4.4, 3.7, 3.1])
WEEKDAY_HOURLY_SHARE = _WEEKDAY_HOURLY / _WEEKDAY_HOURLY.sum()
WEEKEND_HOURLY_SHARE = _WEEKEND_HOURLY / _WEEKEND_HOURLY.sum()


def hourly_arrival_share(day_of_week: str) -> np.ndarray:
    return WEEKEND_HOURLY_SHARE if day_of_week in ("Saturday", "Sunday") else WEEKDAY_HOURLY_SHARE


def _season_days(year_days: pd.DatetimeIndex) -> np.ndarray:
    return np.isin(year_days.month, RESPIRATORY_SEASON_MONTHS)


# Base daily arrivals with the seasonal boost removed, so the yearly total
# still averages ED_ARRIVALS_PER_YEAR.
_YEAR = pd.date_range("2025-01-01", "2025-12-31")
BASE_DAILY_ARRIVALS = (ED_ARRIVALS_PER_YEAR / 365) / (1 + RESPIRATORY_SEASON_BOOST * _season_days(_YEAR).mean())


def generate_ed_arrivals(days: int = 365) -> pd.DataFrame:
    """Hourly ED arrivals for the past year (one row per date and hour)."""
    rng = _rng(8)
    dates = pd.date_range(end=REFERENCE_DATE - pd.Timedelta(days=1), periods=days)
    in_season = _season_days(dates)
    rows = []
    for date, season in zip(dates, in_season):
        dow = date.day_name()
        daily = BASE_DAILY_ARRIVALS * DOW_ARRIVAL_MULTIPLIER[dow] * (1 + RESPIRATORY_SEASON_BOOST * season)
        counts = rng.poisson(daily * hourly_arrival_share(dow))
        for hour, n in enumerate(counts):
            rows.append(
                {"date": date, "day_of_week": dow, "hour": hour,
                 "respiratory_season": bool(season), "arrivals": int(n)}
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Inpatients with nurse notes
# ---------------------------------------------------------------------------

MEDICAL_BARRIERS = {
    "Pending MRI": "MRI still pending, not yet scheduled.",
    "IV antibiotics": "Remains on IV antibiotics, awaiting ID input on PO switch.",
    "PT eval": "PT eval not yet completed; unsafe to ambulate independently.",
    "Pending echo": "Echo ordered yesterday, still pending read.",
    "Pending labs": "Repeat labs and blood cultures still pending.",
}
PENDING_TEST_BARRIERS = {"Pending MRI", "Pending echo", "Pending labs"}

SOCIAL_BARRIERS = {
    "No ride home": "Pt states no ride home; family unavailable until weekend.",
    "Can't afford meds": "Pt reports they can't afford discharge meds.",
    "Nursing-home placement": "Needs nursing-home placement; SW sending referrals.",
    "Housing / shelter": "Pt currently staying at a shelter; needs safe discharge plan.",
    "Insurance authorization": "Insurance authorization pending for post-acute care.",
}

MOBILITY_LEVELS = ["Independent", "Needs assistance", "Bedbound"]
MOBILITY_NOTES = {
    "Needs assistance": "Ambulates with walker and 1 assist.",
    "Bedbound": "Bedbound; 2-person assist for repositioning.",
}

NOTE_OPENERS = [
    "Pt resting comfortably, VSS.",
    "A&O x4, pain controlled on current regimen.",
    "Slept well overnight, tolerating diet.",
    "Afebrile overnight, no acute events.",
]
INDEPENDENT_OPENER = "Ambulating in hallway with steady gait."

NOTE_CLOSERS = [
    "Will continue to monitor.",
    "Plan per primary team.",
    "Family updated at bedside.",
    "Discussed plan of care with pt.",
]

PHYSICIAN_DC_NOTE = "Per MD, likely discharge today."

UNIT_ROOM_PREFIX = {
    "Medicine": "M",
    "Surgery": "S",
    "Telemetry": "T",
    "ICU": "I",
    "Women's & Peds": "W",
}


def generate_inpatients(n: int = 40) -> pd.DataFrame:
    """Current inpatients with a short synthetic nurse note.

    The structured columns (pending_test, iv_meds, mobility,
    physician_noted_dc_today, social_barrier) match what the note says.
    `medical_barriers` / `social_barriers` hold the specific barrier tags.
    Social barriers are more common for Medicaid/CHIP and Uninsured patients.
    """
    rng = _rng(5)
    units = rng.choice(list(UNIT_CAPACITY), size=n, p=[0.40, 0.22, 0.18, 0.08, 0.12])
    insurance = rng.choice(INSURANCE_GROUPS, size=n, p=INSURANCE_SHARES)
    ages = _ages_for_insurance(insurance, rng)

    rows = []
    used_rooms = set()
    for i in range(n):
        unit, ins = units[i], insurance[i]
        while True:
            room = f"{UNIT_ROOM_PREFIX[unit]}-{rng.integers(1, 5)}{rng.integers(1, 30):02d}"
            if room not in used_rooms:
                used_rooms.add(room)
                break

        expected_los = int(rng.integers(2, 7))
        los = max(1, expected_los + int(rng.integers(-3, 5)))

        med = []
        if rng.random() < 0.40:
            k = 1 if rng.random() < 0.8 else 2
            med = list(rng.choice(list(MEDICAL_BARRIERS), size=k, replace=False))

        if "PT eval" in med:
            mobility = "Needs assistance"
        else:
            mobility = str(rng.choice(MOBILITY_LEVELS, p=[0.75, 0.18, 0.07]))

        social_p = 0.55 if ins in SOCIAL_RISK_PAYERS else 0.15
        soc = []
        if rng.random() < social_p:
            options = list(SOCIAL_BARRIERS)
            if ages[i] < 65:
                options.remove("Nursing-home placement")
            k = 1 if rng.random() < 0.75 else 2
            soc = list(rng.choice(options, size=k, replace=False))

        physician_noted = rng.random() < (0.65 if not med and mobility != "Bedbound" else 0.08)

        openers = NOTE_OPENERS + ([INDEPENDENT_OPENER] if mobility == "Independent" else [])
        parts = [rng.choice(openers)]
        parts += [MEDICAL_BARRIERS[b] for b in med]
        if mobility in MOBILITY_NOTES and "PT eval" not in med:
            parts.append(MOBILITY_NOTES[mobility])
        parts += [SOCIAL_BARRIERS[b] for b in soc]
        if physician_noted:
            parts.append(PHYSICIAN_DC_NOTE)
        parts.append(rng.choice(NOTE_CLOSERS))

        rows.append(
            {
                "patient_id": f"IP-{3001 + i}",
                "unit": unit,
                "room": room,
                "age": int(ages[i]),
                "insurance": ins,
                "los_days": los,
                "expected_los_days": expected_los,
                "days_past_expected": los - expected_los,
                "pending_test": int(any(b in PENDING_TEST_BARRIERS for b in med)),
                "iv_meds": int("IV antibiotics" in med),
                "mobility": mobility,
                "physician_noted_dc_today": int(physician_noted),
                "social_barrier": int(bool(soc)),
                "nurse_note": " ".join(parts),
                "medical_barriers": "; ".join(med),
                "social_barriers": "; ".join(soc),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Historical inpatient stays (discharge-today model training data)
# ---------------------------------------------------------------------------

def generate_past_inpatient_stays(n: int = 3000) -> pd.DataFrame:
    """Past inpatient days with a `discharged_that_day` (1/0) outcome."""
    rng = _rng(6)
    units = rng.choice(list(UNIT_CAPACITY), size=n, p=[0.40, 0.22, 0.18, 0.08, 0.12])
    insurance = rng.choice(INSURANCE_GROUPS, size=n, p=INSURANCE_SHARES)
    age = _ages_for_insurance(insurance, rng)

    expected_los = rng.integers(2, 7, size=n)
    los = np.maximum(1, expected_los + rng.integers(-3, 6, size=n))
    days_past = los - expected_los

    pending_test = (rng.random(n) < 0.25).astype(int)
    iv_meds = (rng.random(n) < 0.22).astype(int)
    older = age >= 75
    mobility = np.where(
        older,
        rng.choice(MOBILITY_LEVELS, size=n, p=[0.50, 0.32, 0.18]),
        rng.choice(MOBILITY_LEVELS, size=n, p=[0.75, 0.18, 0.07]),
    )
    social_p = np.where(np.isin(insurance, SOCIAL_RISK_PAYERS), 0.45, 0.15)
    social = (rng.random(n) < social_p).astype(int)

    limited = (mobility != "Independent").astype(int)
    physician_noted = (
        rng.random(n)
        < _sigmoid(-0.4 - 1.8 * pending_test - 1.8 * iv_meds - 0.9 * limited + 0.25 * np.clip(days_past, -3, 3))
    ).astype(int)

    logit = (
        -1.5
        + 0.30 * np.clip(days_past, -3, 4)
        - 1.2 * pending_test
        - 1.3 * iv_meds
        - 0.7 * (mobility == "Needs assistance")
        - 1.8 * (mobility == "Bedbound")
        + 2.2 * physician_noted
        - 1.4 * social
    )
    discharged = (rng.random(n) < _sigmoid(logit)).astype(int)

    return pd.DataFrame(
        {
            "stay_id": [f"S{500000 + i}" for i in range(n)],
            "unit": units,
            "age": age,
            "insurance": insurance,  # fairness checks only
            "los_days": los,
            "expected_los_days": expected_los,
            "days_past_expected": days_past,
            "pending_test": pending_test,
            "iv_meds": iv_meds,
            "mobility": mobility,
            "physician_noted_dc_today": physician_noted,
            "social_barrier": social,
            "discharged_that_day": discharged,
        }
    )


# ---------------------------------------------------------------------------
# Boarding times by insurance (30 days)
# ---------------------------------------------------------------------------

BOARDING_GAMMA_SCALE = 33.5  # tuned so the overall median is ~210 min (3.5 h)


def generate_boarding_history(days: int = 30) -> pd.DataFrame:
    """One row per boarded ED patient over the last `days` days.

    Calibrated to a ~3.5-hour overall median. Medicaid/CHIP boarders wait
    ~30+ minutes longer than the overall average.
    """
    rng = _rng(7)
    insurance_offset = {"Commercial": -5, "Medicare": 0, MEDICAID: 60, "Uninsured": 15, "Other": 0}
    rows = []
    for d in range(days, 0, -1):
        date = REFERENCE_DATE - pd.Timedelta(days=d)
        n_boarders = int(rng.integers(8, 16))
        ins = rng.choice(INSURANCE_GROUPS, size=n_boarders, p=INSURANCE_SHARES)
        for g in ins:
            minutes = rng.gamma(shape=6.0, scale=BOARDING_GAMMA_SCALE) + insurance_offset[g]
            rows.append(
                {
                    "date": date.normalize(),
                    "insurance": g,
                    "boarding_minutes": int(max(30, round(minutes))),
                }
            )
    return pd.DataFrame(rows)


def boarding_gap_by_insurance(boarding: pd.DataFrame) -> pd.DataFrame:
    """Mean boarding minutes per insurance group vs. the overall mean."""
    overall = boarding["boarding_minutes"].mean()
    out = (
        boarding.groupby("insurance")["boarding_minutes"]
        .agg(patients="count", mean_minutes="mean", median_minutes="median")
        .reindex(INSURANCE_GROUPS)
        .reset_index()
    )
    out["mean_minutes"] = out["mean_minutes"].round(1)
    out["vs_overall_minutes"] = (out["mean_minutes"] - overall).round(1)
    return out


# ---------------------------------------------------------------------------
# Explainable models (logistic regression)
# ---------------------------------------------------------------------------

# Never allowed as model inputs. Age as a clinical number is allowed; the
# binned `age_group` is not.
FORBIDDEN_MODEL_INPUTS = frozenset({"insurance", *FAIRNESS_ONLY_COLUMNS})


@dataclass(frozen=True)
class Feature:
    name: str
    kind: str  # "numeric" | "binary" | "categorical"
    categories: tuple = ()


class ExplainableLogit:
    """Logistic regression on standardized inputs, with per-patient reasons.

    A feature's contribution is coefficient x standardized value: how far it
    moves this patient's log-odds compared with the average training patient.
    One-hot columns are summed back into their original feature.
    """

    def __init__(self, features: list[Feature], target: str, labeler):
        names = {f.name for f in features}
        forbidden = names & FORBIDDEN_MODEL_INPUTS
        if forbidden:
            raise ValueError(f"Not allowed as model inputs: {sorted(forbidden)}")
        self.features = features
        self.target = target
        self.labeler = labeler

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = {}
        for f in self.features:
            if f.kind == "categorical":
                for c in f.categories:
                    cols[f"{f.name}={c}"] = (df[f.name] == c).astype(float)
            else:
                cols[f.name] = df[f.name].astype(float)
        return pd.DataFrame(cols, index=df.index)

    def fit(self, df: pd.DataFrame) -> "ExplainableLogit":
        X, y = self._encode(df), df[self.target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=SEED, stratify=y
        )
        self.pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        self.pipeline.fit(X_train, y_train)
        self.auc = float(roc_auc_score(y_test, self.pipeline.predict_proba(X_test)[:, 1]))
        self.n_train, self.n_test = len(X_train), len(X_test)
        self.test_index = X_test.index  # held-out rows, kept for fairness checks
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(self._encode(df))[:, 1]

    def contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Log-odds contribution of each original feature, per row."""
        X = self._encode(df)
        scaler = self.pipeline.named_steps["standardscaler"]
        coef = self.pipeline.named_steps["logisticregression"].coef_[0]
        contrib = pd.DataFrame(scaler.transform(X) * coef, index=df.index, columns=X.columns)
        groups = [c.split("=")[0] for c in X.columns]
        return contrib.T.groupby(groups, sort=False).sum().T

    def reasons(self, df: pd.DataFrame, n: int = 3, direction: int = 1, min_effect: float = 0.05):
        """Top-n readable reasons per row. direction=1 pushes the score up, -1 down."""
        contrib = self.contributions(df) * direction
        out = []
        for idx, row in contrib.iterrows():
            top = row[row > min_effect].sort_values(ascending=False).head(n)
            out.append([self.labeler(f, df.at[idx, f], direction) for f in top.index])
        return out

    def coefficients(self) -> pd.DataFrame:
        coef = self.pipeline.named_steps["logisticregression"].coef_[0]
        cols = self._encode(pd.DataFrame({f.name: [0] for f in self.features})).columns
        return (
            pd.DataFrame({"input": cols, "coefficient_per_sd": coef.round(3)})
            .sort_values("coefficient_per_sd", key=abs, ascending=False)
            .reset_index(drop=True)
        )


# ----- Admission at triage ------------------------------------------------

ADMISSION_FEATURES = [
    Feature("age", "numeric"),
    Feature("triage_acuity", "categorical", (1, 2, 3, 4, 5)),
    Feature("heart_rate", "numeric"),
    Feature("o2_sat", "numeric"),
    Feature("temperature_c", "numeric"),
    Feature("systolic_bp", "numeric"),
    Feature("diastolic_bp", "numeric"),
    Feature("chief_complaint", "categorical", tuple(CHIEF_COMPLAINTS)),
    Feature("cardiac_history", "binary"),
    Feature("prior_visits_12mo", "numeric"),
]
assert [f.name for f in ADMISSION_FEATURES] == ED_MODEL_FEATURES


def _admission_label(feature: str, value, direction: int = 1) -> str:
    if feature == "triage_acuity":
        # A reason that raises admission risk means sicker than the average patient.
        urgency = "more" if direction > 0 else "less"
        return f"Triage acuity {value} ({urgency} urgent than typical)"
    return {
        "age": lambda v: f"Age {v}",
        "heart_rate": lambda v: f"Heart rate {v}",
        "o2_sat": lambda v: f"O₂ sat {v}%",
        "temperature_c": lambda v: f"Temp {v}°C",
        "systolic_bp": lambda v: f"Systolic BP {v}",
        "diastolic_bp": lambda v: f"Diastolic BP {v}",
        "chief_complaint": lambda v: f"Complaint: {v.lower()}",
        "cardiac_history": lambda v: "Cardiac history" if v else "No cardiac history",
        "prior_visits_12mo": lambda v: f"{v} ED visit(s) in past year",
    }[feature](value)


def train_admission_model(ed_visits: pd.DataFrame) -> ExplainableLogit:
    return ExplainableLogit(ADMISSION_FEATURES, "admitted", _admission_label).fit(ed_visits)


def predict_bed_type(row) -> tuple[str, str]:
    """Rule-based likely bed type, returned with the rule that fired."""
    if row["age"] < 18:
        return "Women's & Peds", f"age {row['age']} (<18)"
    icu = []
    if row["triage_acuity"] == 1:
        icu.append("acuity 1")
    if row["o2_sat"] < 89:
        icu.append(f"O₂ {row['o2_sat']}% (<89)")
    if row["systolic_bp"] < 90:
        icu.append(f"SBP {row['systolic_bp']} (<90)")
    if icu:
        return "ICU", ", ".join(icu)
    if row["chief_complaint"] == "Chest pain":
        return "Telemetry", "chest pain"
    if row["cardiac_history"] and row["heart_rate"] > 105:
        return "Telemetry", f"cardiac history + HR {row['heart_rate']} (>105)"
    if row["chief_complaint"] in ("Abdominal pain", "Fall / injury"):
        return "Surgery", row["chief_complaint"].lower()
    return "Medicine", "no ICU, telemetry, or surgery criteria"


def score_current_ed(model: ExplainableLogit, ed_current: pd.DataFrame) -> pd.DataFrame:
    out = ed_current.copy()
    out["admit_probability"] = model.predict_proba(out)
    out["top_reasons"] = ["; ".join(r) for r in model.reasons(out, n=3, direction=1)]
    beds = out.apply(predict_bed_type, axis=1, result_type="expand")
    out["likely_bed"], out["bed_rule"] = beds[0], beds[1]
    return out


# ----- Likely to go home today -------------------------------------------

DISCHARGE_FEATURES = [
    Feature("days_past_expected", "numeric"),
    Feature("pending_test", "binary"),
    Feature("iv_meds", "binary"),
    Feature("mobility", "categorical", tuple(MOBILITY_LEVELS)),
    Feature("physician_noted_dc_today", "binary"),
    Feature("social_barrier", "binary"),
]


def _discharge_label(feature: str, value, direction: int = 1) -> str:
    if feature == "days_past_expected":
        v = int(value)
        if v > 0:
            return f"{v} day(s) past expected stay"
        if v < 0:
            return f"{-v} day(s) before expected discharge"
        return "At expected length of stay"
    return {
        "pending_test": lambda v: "Test still pending" if v else "No pending tests",
        "iv_meds": lambda v: "On IV medications" if v else "No IV medications",
        "mobility": lambda v: f"Mobility: {v.lower()}",
        "physician_noted_dc_today": lambda v: "MD noted likely discharge today" if v else "No MD discharge note",
        "social_barrier": lambda v: "Documented social barrier" if v else "No social barrier documented",
    }[feature](value)


def train_discharge_model(past_stays: pd.DataFrame) -> ExplainableLogit:
    return ExplainableLogit(DISCHARGE_FEATURES, "discharged_that_day", _discharge_label).fit(past_stays)


def rank_inpatients(model: ExplainableLogit, inpatients: pd.DataFrame) -> pd.DataFrame:
    """Current inpatients sorted by probability of going home today."""
    out = inpatients.copy()
    out["discharge_probability"] = model.predict_proba(out)
    out["reasons_for"] = ["; ".join(r) for r in model.reasons(out, n=3, direction=1)]
    out["reasons_against"] = ["; ".join(r) for r in model.reasons(out, n=3, direction=-1)]
    return out.sort_values("discharge_probability", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# AI Control Center: boarding cost, 8-hour bed forecast, capacity tiers
# ---------------------------------------------------------------------------

BOARDING_COST_PER_HOUR = 350  # dollars per boarding hour

FORECAST_START_HOUR = 9  # 9 AM
FORECAST_HOURS = 8  # through 5 PM

DAYS_OF_WEEK = list(DOW_ARRIVAL_MULTIPLIER)
# Day-of-week effect on admissions, from the arrival pattern (Monday +15%).
DAY_ADJUSTMENT = {d: round(m - 1, 2) for d, m in DOW_ARRIVAL_MULTIPLIER.items()}

TREND_LEVELS = {"Normal": 0.0, "Elevated flu/RSV": 0.20, "High": 0.35}

# Share of ED admissions that go to each unit.
UNIT_ADMIT_SHARE = {
    "Medicine": 0.60,
    "Surgery": 0.12,
    "Telemetry": 0.12,
    "ICU": 0.06,
    "Women's & Peds": 0.10,
}
# Share of the DAY's expected discharges that happen in each slot.
# Most discharges land in the afternoon; the rest go after 5 PM.
DISCHARGE_PROFILE = [0.01, 0.02, 0.03, 0.03, 0.04, 0.12, 0.17, 0.16]
# Share of the day's discharges in the two slots after 5 PM (used when a
# discharge lounge pulls discharges earlier).
DISCHARGE_PROFILE_EVENING = [0.10, 0.08]
CLEANING_PER_HOUR = 1.0  # needs-cleaning rooms turned over per hour per unit
AFTERNOON_START_HOUR = 14  # "after 2 PM"


def _discharge_share(slot: int, shift_hours: int = 0) -> float:
    """Share of the day's discharges in forecast slot `slot`, optionally pulled
    `shift_hours` earlier (e.g., a discharge lounge frees rooms ~2 hours sooner)."""
    profile = DISCHARGE_PROFILE + DISCHARGE_PROFILE_EVENING
    if shift_hours == 0:
        return profile[slot]
    if slot == 0:  # the first slot also absorbs the hours pulled in from before
        return sum(profile[: shift_hours + 1])
    j = slot + shift_hours
    return profile[j] if j < len(profile) else 0.0


def hour_label(hour: int) -> str:
    suffix = "AM" if hour < 12 else "PM"
    return f"{(hour - 1) % 12 + 1} {suffix}"


def default_day_of_week() -> str:
    return REFERENCE_DATE.day_name()


def public_health_feed() -> dict:
    """SIMULATED public health surveillance (styled after CDC/state reports).

    Not real data. The local level adjusts the forecast automatically.
    """
    return {
        "source": "Simulated CDC/state surveillance data (synthetic, not real)",
        "as_of": (REFERENCE_DATE - pd.Timedelta(days=2)).strftime("%b %d, %Y"),
        "local": {
            "level": "Elevated flu/RSV",
            "trend": "Rising 3 weeks in a row",
            "flu_ed_visits_pct": 4.8,
            "rsv_ed_visits_pct": 2.1,
        },
        "national": {
            "level": "Moderate",
            "trend": "Rising",
            "flu_ed_visits_pct": 3.2,
            "rsv_ed_visits_pct": 1.4,
        },
    }


def news_alerts() -> list[dict]:
    """SIMULATED news alerts. They never change the forecast on their own;
    a house supervisor must choose "Apply to forecast".
    """
    return [
        {
            "id": "measles-county",
            "headline": "Measles outbreak reported in county",
            "source": "Simulated local news feed",
            "detail": "County health department reports 6 confirmed cases, mostly children.",
            "effect": {"Women's & Peds": 0.10, "Medicine": 0.03},
        },
        {
            "id": "nearby-diversion",
            "headline": "Nearby hospital diverting ambulances",
            "source": "Simulated regional EMS feed",
            "detail": "A nearby hospital is on ambulance diversion until about 3 PM.",
            "effect": {unit: 0.15 for unit in UNIT_CAPACITY},
        },
        {
            "id": "ice-storm",
            "headline": "Ice storm expected tonight",
            "source": "Simulated weather service feed",
            "detail": "Freezing rain expected after 6 PM; more falls and crashes likely overnight.",
            "effect": {"Surgery": 0.08},
        },
    ]


def boarding_metrics(ed_scored: pd.DataFrame) -> dict:
    boarders = ed_scored[ed_scored["status"] == "Boarding"]
    hours = float(boarders["hours_boarding"].sum())
    return {
        "patients": len(boarders),
        "total_hours": hours,
        "cost": hours * BOARDING_COST_PER_HOUR,
    }


def expected_discharges_today(unit: str, beds: pd.DataFrame, inpatients_ranked: pd.DataFrame) -> float:
    """The charted inpatients are a sample, so the unit's average model
    probability is applied to every occupied bed on the unit."""
    occupied = int(beds.set_index("unit").at[unit, "occupied"])
    probs = inpatients_ranked.loc[inpatients_ranked["unit"] == unit, "discharge_probability"]
    mean_p = float(probs.mean()) if len(probs) else 0.0
    return mean_p * occupied


@dataclass
class Forecast:
    unit: str
    table: pd.DataFrame  # one row per hour, 9 AM to 5 PM
    admission_adjustment: float  # e.g. 0.38 for +38%
    day_of_week: str
    trend_level: str
    applied_alerts: list
    ed_boarders: int
    ed_workup: int
    discharges_today: float

    @property
    def min_open(self) -> float:
        return float(self.table["projected_open"].min())

    @property
    def runout_hour(self):
        """First hour the unit has no open bed left, or None."""
        hit = self.table[self.table["projected_open"] < 0]
        return int(hit["hour"].iloc[0]) if len(hit) else None

    @property
    def worst_hour(self) -> int:
        return int(self.table.loc[self.table["projected_open"].idxmin(), "hour"])

    @property
    def head_start_hours(self):
        return None if self.runout_hour is None else self.runout_hour - FORECAST_START_HOUR

    @property
    def afternoon_discharge_share(self) -> float:
        slots = range(FORECAST_HOURS)
        after = sum(DISCHARGE_PROFILE[i] for i in slots if FORECAST_START_HOUR + i >= AFTERNOON_START_HOUR)
        return after / sum(DISCHARGE_PROFILE)

    def warning(self):
        if self.runout_hour is None:
            return None
        text = f"{self.unit} projected to run out of beds by {hour_label(self.runout_hour)}"
        short = round(-self.min_open)
        if short >= 1:
            text += f" — short about {short} bed{'s' if short != 1 else ''} by {hour_label(self.worst_hour)}"
        return text

    def reasons(self) -> list[str]:
        out = []
        day = DAY_ADJUSTMENT.get(self.day_of_week, 0)
        if day:
            out.append(f"{self.day_of_week}: {day:+.0%} expected admissions")
        trend = TREND_LEVELS[self.trend_level]
        if trend:
            out.append(f"Flu/RSV trend {self.trend_level.split(' ')[0]}: +{trend:.0%} expected admissions (simulated surveillance data)")
        for a in self.applied_alerts:
            out.append(f"Applied news alert: {a['headline']} (+{a['effect'].get(self.unit, 0):.0%})")
        n_ed = self.ed_boarders + self.ed_workup
        if n_ed:
            out.append(
                f"{n_ed} ED patient{'s' if n_ed != 1 else ''} likely needing {self.unit} "
                f"({self.ed_boarders} already admitted and boarding, {self.ed_workup} in workup)"
            )
        out.append(
            f"Discharges usually come after 2 PM ({self.afternoon_discharge_share:.0%} of those "
            f"expected by 5 PM happen after 2 PM)"
        )
        return out


def forecast_unit(
    unit: str,
    beds: pd.DataFrame,
    ed_scored: pd.DataFrame,
    inpatients_ranked: pd.DataFrame,
    day_of_week: str,
    trend_level: str,
    applied_alerts: list | None = None,
    flex_beds: int = 0,
    discharge_shift_hours: int = 0,
) -> Forecast:
    """Projected open beds by hour:
    open now + rooms cleaned + expected discharges - expected admissions.

    Expected admissions = ED patients (boarders, plus workup patients weighted
    by admit probability) + new ED arrivals by hour x 16% admit rate x the
    unit's share of admissions, scaled by day of week, public health trend
    (which stands in for the winter respiratory season), and any news alerts
    a human applied.
    """
    applied_alerts = applied_alerts or []
    b = beds.set_index("unit").loc[unit]

    adjustment = (
        DAY_ADJUSTMENT.get(day_of_week, 0)
        + TREND_LEVELS[trend_level]
        + sum(a["effect"].get(unit, 0) for a in applied_alerts)
    )
    multiplier = 1 + adjustment

    boarders = ed_scored[(ed_scored["status"] == "Boarding") & (ed_scored["admit_unit"] == unit)]
    workup = ed_scored[(ed_scored["status"] == "In workup") & (ed_scored["likely_bed"] == unit)]
    workup_expected = float(workup["admit_probability"].sum())
    dc_today = expected_discharges_today(unit, beds, inpatients_ranked)
    # Typical arrivals for this weekday by hour; season is applied via trend_level.
    arrivals_by_hour = BASE_DAILY_ARRIVALS * hourly_arrival_share(day_of_week)

    rows = [{"hour": FORECAST_START_HOUR, "cleaned": 0.0, "discharges": 0.0,
             "ed_admits": 0.0, "new_arrival_admits": 0.0}]
    for i in range(FORECAST_HOURS):
        prev = rows[-1]
        ed_now = 1.0 if i < len(boarders) else 0.0  # boarders placed one per hour as beds open
        if i in (1, 2):
            ed_now += workup_expected / 2  # workup decisions over the next 2-3 hours
        rows.append(
            {
                "hour": FORECAST_START_HOUR + i + 1,
                "cleaned": min(float(b["needs_cleaning"]), (i + 1) * CLEANING_PER_HOUR),
                "discharges": prev["discharges"] + dc_today * _discharge_share(i, discharge_shift_hours),
                "ed_admits": prev["ed_admits"] + ed_now,
                "new_arrival_admits": prev["new_arrival_admits"]
                + arrivals_by_hour[FORECAST_START_HOUR + i] * ED_ADMIT_RATE
                * UNIT_ADMIT_SHARE[unit] * multiplier,
            }
        )
    table = pd.DataFrame(rows)
    table.insert(0, "time", table["hour"].map(hour_label))
    table.insert(2, "open_now", int(b["open"]))
    table["flex_beds"] = [0] + [flex_beds] * FORECAST_HOURS  # flex beds open from 10 AM
    table["projected_open"] = (
        table["open_now"] + table["cleaned"] + table["discharges"] + table["flex_beds"]
        - table["ed_admits"] - table["new_arrival_admits"]
    )
    return Forecast(
        unit=unit,
        table=table,
        admission_adjustment=adjustment,
        day_of_week=day_of_week,
        trend_level=trend_level,
        applied_alerts=applied_alerts,
        ed_boarders=len(boarders),
        ed_workup=len(workup),
        discharges_today=dc_today,
    )


# ----- What-If: Medicine operational actions ------------------------------

LOUNGE_SHIFT_HOURS = 2


def social_barrier_patients(inpatients: pd.DataFrame, scan: dict | None) -> set:
    """Patients with a social barrier: from the Barrier Flags scan when available,
    otherwise from the documented social_barrier field."""
    if scan:
        return {pid for pid, r in scan["results"].items()
                if any(b["type"] == "Social" for b in r["barriers"])}
    return set(inpatients.loc[inpatients["social_barrier"] == 1, "patient_id"])


def clear_social_barriers(model: "ExplainableLogit", inpatients_ranked: pd.DataFrame, pids: set) -> pd.DataFrame:
    """Re-score discharge probability as if these patients' social barriers were cleared."""
    out = inpatients_ranked.copy()
    mask = out["patient_id"].isin(pids)
    out.loc[mask, "social_barrier"] = 0
    out["discharge_probability"] = model.predict_proba(out)
    return out


def shortage_patient_hours(forecast: Forecast) -> float:
    """Patient-hours without a bed: each hour the unit is short, the patients
    short are boarding in the ED."""
    return float(np.maximum(-forecast.table["projected_open"].iloc[1:], 0).sum())


def medicine_scenario(beds, ed_scored, inpatients_ranked, discharge_model, day_of_week, trend_level,
                      applied_alerts, lounge: bool, clear_social: bool, flex_beds: int, scan=None) -> dict:
    unit = "Medicine"
    base = forecast_unit(unit, beds, ed_scored, inpatients_ranked, day_of_week, trend_level, applied_alerts)
    pids = social_barrier_patients(inpatients_ranked, scan) if clear_social else set()
    medicine_pids = set(inpatients_ranked.loc[inpatients_ranked["unit"] == unit, "patient_id"]) & pids
    scen_inpatients = clear_social_barriers(discharge_model, inpatients_ranked, pids) if pids else inpatients_ranked
    scen = forecast_unit(unit, beds, ed_scored, scen_inpatients, day_of_week, trend_level, applied_alerts,
                         flex_beds=flex_beds, discharge_shift_hours=LOUNGE_SHIFT_HOURS if lounge else 0)
    saved = max(shortage_patient_hours(base) - shortage_patient_hours(scen), 0.0)
    return {
        "base": base, "scenario": scen,
        "boarding_hours_saved": saved,
        "dollars_saved": saved * BOARDING_COST_PER_HOUR,
        "medicine_social_cleared": len(medicine_pids),
        "extra_discharges_today": scen.discharges_today - base.discharges_today,
    }


CAPACITY_TIERS = {
    "Normal": {
        "when": "Never runs out; about 3 or more open beds all day",
        "actions": ["Routine bed huddles"],
    },
    "Level 1 — Tight": {
        "when": "Down to 0–2 open beds, or runs out 3+ hours away and short 2 beds or fewer",
        "actions": [
            "Work the \"likely home today\" list first",
            "Open the discharge lounge",
            "Prioritize cleaning of discharged rooms",
            "Add an extra bed huddle",
        ],
    },
    "Level 2 — Full": {
        "when": "Runs out within 2 hours, or short more than 2 beds",
        "actions": [
            "Open flex beds with extra nurse shifts",
            "Hallway-bed protocol for stable patients (RN approval)",
            "Inpatient teams round on ED boarders",
            "Use fast-track for low-acuity ED patients",
        ],
    },
    "Level 3 — Crisis": {
        "when": "Short 3+ beds within 2 hours, or short 4+ beds at any time",
        "actions": [
            "Delay non-urgent planned surgeries (CMO approval)",
            "Transfer stable patients to partner facilities",
            "Coordinate ambulance diversion (last resort)",
        ],
    },
}


def recommend_tier(forecast: Forecast) -> str:
    """Tier from how soon the unit runs out (below 0 beds) and by how much.

    - Never runs out: Normal if ~3+ beds all day, otherwise Tight.
    - Runs out 3+ hours away and short 2 beds or fewer: Tight (time to act).
    - Runs out within 2 hours, or short more than 2 beds: Full.
    - Short 3+ beds within 2 hours, or short 4+ beds at any time: Crisis.
    """
    if forecast.runout_hour is None:
        return "Normal" if forecast.min_open >= 2.5 else "Level 1 — Tight"
    short = round(-forecast.min_open)
    hours_away = forecast.head_start_hours
    if short >= 4 or (short >= 3 and hours_away <= 2):
        return "Level 3 — Crisis"
    if hours_away <= 2 or short > 2:
        return "Level 2 — Full"
    return "Level 1 — Tight"


def tier_rationale(forecast: Forecast) -> str:
    if forecast.runout_hour is None:
        return f"Does not run out; lowest point is about {max(forecast.min_open, 0):.1f} open beds."
    short = round(-forecast.min_open)
    hours = forecast.head_start_hours
    return (
        f"Runs out {hours} hour{'s' if hours != 1 else ''} from now "
        f"({hour_label(forecast.runout_hour)}), short about {short} bed{'s' if short != 1 else ''} at most."
    )


# ---------------------------------------------------------------------------
# ED rooms and staffing (case brief)
# ---------------------------------------------------------------------------

ED_ROOM_CAPACITY = {
    "Treatment rooms": 32,
    "Fast-track rooms": 8,
    "Resuscitation/trauma bays": 4,
    "Behavioral health rooms": 4,
}
WAITING_ROOM_SEATS = 40

# Synthetic 9 AM snapshot: rooms in use by patients who are NOT boarders.
# The 14 detailed ED patients are part of this census.
_ED_ROOMS_IN_USE = {
    "Treatment rooms": 23,
    "Fast-track rooms": 6,
    "Resuscitation/trauma bays": 2,
    "Behavioral health rooms": 4,
}
WAITING_ROOM_PATIENTS = 17

CURRENT_SHIFT_STAFFING = {
    "shift": "Day shift 7a–3p",
    "staff": {"Physicians": 4, "APPs": 3, "Nurses": 14, "Techs/registration": 8},
}


def ed_room_status(ed_scored: pd.DataFrame) -> pd.DataFrame:
    """ED rooms by area. Boarders (admitted, waiting for a bed) hold treatment rooms."""
    boarders = int((ed_scored["status"] == "Boarding").sum())
    rows = []
    for area, total in ED_ROOM_CAPACITY.items():
        held = boarders if area == "Treatment rooms" else 0
        in_use = _ED_ROOMS_IN_USE[area]
        rows.append(
            {"area": area, "total": total, "held_by_boarders": held,
             "other_patients": in_use, "open": total - held - in_use}
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# What-if: beds ready earlier
# ---------------------------------------------------------------------------

LWBS_RATE = 0.04  # case brief: 4% of ED patients leave without being seen
LWBS_GOAL_RATE = 0.028  # case brief Month-18 goal: 4.0% -> 2.8%
NET_REVENUE_PER_VISIT = 1_500  # case brief: average net revenue per ED visit
# Patients one ED treatment room serves per hour on an average day.
PATIENTS_PER_ROOM_HOUR = (ED_ARRIVALS_PER_YEAR / 365) / (ED_ROOM_CAPACITY["Treatment rooms"] * 24)


def _value_lines(boarding_hours_saved: float) -> dict:
    room_hours = boarding_hours_saved  # each boarder holds one ED treatment room
    patients_roomed = room_hours * PATIENTS_PER_ROOM_HOUR
    recovered = patients_roomed * LWBS_RATE
    return {
        "boarding_hours_saved": boarding_hours_saved,
        "boarding_cost_avoided": boarding_hours_saved * BOARDING_COST_PER_HOUR,
        "room_hours_freed": room_hours,
        "patients_roomed_sooner": patients_roomed,
        "recovered_visits": recovered,
        "recovered_revenue": recovered * NET_REVENUE_PER_VISIT,
    }


# Annual figures from the case brief, so results match the slides.
ANNUAL_ADMISSIONS = round(ED_ARRIVALS_PER_YEAR * ED_ADMIT_RATE)  # 85,000 x 16% = 13,600
MEDIAN_BOARDING_HOURS = 3.5  # case brief; cap on hours saved per admission

WHAT_IF_PRESETS = {
    "Month-18 goal: 1 hour less boarding (3.5 → 2.5 h)": 1.0,
    "Conservative: 45 minutes": 0.75,
}


def what_if_beds_earlier(hours_earlier: float, ed_scored: pd.DataFrame) -> dict:
    """Value of inpatient beds being ready `hours_earlier` hours sooner.

    Per year (case brief figures): 13,600 admissions x hours saved per
    admission (the slider, capped at the 3.5-hour median boarding time).
    Today: each current boarder leaves up to `hours_earlier` hours sooner,
    never more than they have actually boarded.

    Two value lines:
      1. Boarding cost avoided: hours saved x $350.
      2. Fewer patients leave without being seen (ESTIMATE): freed ED room-hours
         x patients per room-hour x 4% LWBS rate = recovered visits, x $1,500.
    """
    per_admission = min(hours_earlier, MEDIAN_BOARDING_HOURS)
    today = ed_scored.loc[ed_scored["status"] == "Boarding", "hours_boarding"]
    today_lines = _value_lines(float(np.minimum(today, hours_earlier).sum()))
    today_lines["boarders"] = len(today)
    year_lines = _value_lines(ANNUAL_ADMISSIONS * per_admission)
    year_lines["hours_per_admission"] = per_admission
    return {"today": today_lines, "per_year": year_lines}


def upper_lwbs_estimate() -> dict:
    """If the Month-18 LWBS goal is met: 85,000 x (4.0% - 2.8%) visits x $1,500."""
    visits = ED_ARRIVALS_PER_YEAR * (LWBS_RATE - LWBS_GOAL_RATE)
    return {"visits": visits, "revenue": visits * NET_REVENUE_PER_VISIT}


# ---------------------------------------------------------------------------
# Barrier flags: read notes with Claude, with an offline keyword fallback
# ---------------------------------------------------------------------------

BARRIER_TYPES = ["Medical", "Social"]
BARRIER_OWNERS = ["Social work", "Case management", "Pharmacy", "Hospitalist"]
DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
# Models that accept server-side refusal fallbacks (fallbacks="default").
_FALLBACK_MODELS = {"claude-fable-5-2", "claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
SOURCE_CLAUDE = "Claude API"
SOURCE_OFFLINE = "Offline fallback"

BARRIER_SYSTEM_PROMPT = """You read short synthetic inpatient notes for a hospital capacity dashboard and list the discharge barriers each note documents.

For each barrier the note explicitly documents:
- Write a short phrase describing it, using the note's own facts.
- Tag it "Medical" for clinical items still pending (tests, IV medications, therapy evaluations, consults, limited mobility) or "Social" for non-clinical items (transportation, medication cost, placement, housing, insurance authorization, family or caregiver availability).
- Suggest one owner team: Social work, Case management, Pharmacy, or Hospitalist.

You are not making clinical decisions. Do not recommend treatments, diagnoses, discharge timing, or levels of care, and do not judge whether the patient is ready to leave. A team member reviews every flag. Do not infer barriers the note does not state; a note that documents none gets an empty list."""

BARRIER_SCHEMA = {
    "type": "object",
    "properties": {
        "barriers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "barrier": {"type": "string"},
                    "type": {"type": "string", "enum": BARRIER_TYPES},
                    "owner": {"type": "string", "enum": BARRIER_OWNERS},
                },
                "required": ["barrier", "type", "owner"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["barriers"],
    "additionalProperties": False,
}

# Offline fallback: (pattern, barrier, type, owner)
_KEYWORD_RULES = [
    (r"MRI\b.*pending", "MRI pending", "Medical", "Hospitalist"),
    (r"\b[Ee]cho\b.*pending", "Echo read pending", "Medical", "Hospitalist"),
    (r"(labs|cultures).*pending", "Labs/cultures pending", "Medical", "Hospitalist"),
    (r"IV antibiotics", "On IV antibiotics, awaiting oral switch", "Medical", "Pharmacy"),
    (r"PT eval not yet completed", "PT evaluation not completed", "Medical", "Hospitalist"),
    (r"walker and 1 assist", "Needs assistance to walk", "Medical", "Case management"),
    (r"[Bb]edbound", "Bedbound, 2-person assist", "Medical", "Case management"),
    (r"no ride home", "No ride home", "Social", "Social work"),
    (r"can't afford", "Can't afford discharge medications", "Social", "Pharmacy"),
    (r"nursing-home placement", "Needs nursing-home placement", "Social", "Case management"),
    (r"shelter", "Staying at a shelter; needs a safe discharge plan", "Social", "Social work"),
    (r"[Ii]nsurance authorization pending", "Insurance authorization pending", "Social", "Case management"),
]


def offline_barriers(note: str) -> list[dict]:
    return [
        {"barrier": text, "type": kind, "owner": owner}
        for pattern, text, kind, owner in _KEYWORD_RULES
        if re.search(pattern, note)
    ]


def _validated(barriers) -> list[dict]:
    out = []
    for b in barriers if isinstance(barriers, list) else []:
        if (
            isinstance(b, dict)
            and b.get("type") in BARRIER_TYPES
            and b.get("owner") in BARRIER_OWNERS
            and str(b.get("barrier", "")).strip()
        ):
            out.append({"barrier": str(b["barrier"]).strip(), "type": b["type"], "owner": b["owner"]})
    return out


def claude_barriers(client, model: str, note: str) -> list[dict]:
    """One note -> barriers, via the Claude Messages API (structured JSON output)."""
    params = dict(
        model=model,
        max_tokens=2000,
        system=BARRIER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"<note>\n{note}\n</note>"}],
        output_config={"format": {"type": "json_schema", "schema": BARRIER_SCHEMA}},
    )
    if model in _FALLBACK_MODELS:
        # Server-side refusal fallback: if this model declines, another model answers.
        response = client.beta.messages.create(
            betas=["server-side-fallback-2026-07-01"], fallbacks="default", **params
        )
    else:
        response = client.messages.create(**params)
    if response.stop_reason == "refusal":
        raise RuntimeError("Model declined to read this note")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("Response was cut off")
    text = next((b.text for b in response.content if b.type == "text"), "")
    return _validated(json.loads(text).get("barriers"))


def _describe_error(exc: Exception) -> str:
    """Short, key-free error label for the UI."""
    import anthropic

    if isinstance(exc, anthropic.AuthenticationError):
        return "API key was rejected (401)"
    if isinstance(exc, anthropic.PermissionDeniedError):
        return "API key lacks permission (403)"
    if isinstance(exc, anthropic.NotFoundError):
        return "Model not found (404); check the model name"
    if isinstance(exc, anthropic.RateLimitError):
        return "Rate limited (429)"
    if isinstance(exc, anthropic.BadRequestError):
        return "Request rejected (400)"
    if isinstance(exc, anthropic.APIStatusError):
        return f"API error ({exc.status_code})"
    if isinstance(exc, anthropic.APIConnectionError):
        return "Could not reach the Claude API"
    if isinstance(exc, json.JSONDecodeError):
        return "Response was not valid JSON"
    return str(exc) or type(exc).__name__


def scan_notes(inpatients: pd.DataFrame, api_key: str | None, model: str, client=None) -> dict:
    """Scan every note. Uses Claude when a key is given; any note whose call
    fails (or every note, with no key) uses the offline keyword fallback.

    The API key is only passed to the SDK client. It is never stored,
    returned, or written to the decision log.
    """
    notes = dict(zip(inpatients["patient_id"], inpatients["nurse_note"]))
    results, errors = {}, []

    if api_key or client is not None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key, max_retries=2, timeout=60)

        def one(item):
            pid, note = item
            try:
                return pid, claude_barriers(client, model, note), None
            except Exception as exc:  # fall back per note; keep the reason
                return pid, None, _describe_error(exc)

        with ThreadPoolExecutor(max_workers=8) as pool:
            for pid, barriers, err in pool.map(one, notes.items()):
                if err is None:
                    results[pid] = {"source": SOURCE_CLAUDE, "barriers": barriers}
                else:
                    errors.append(err)
                    results[pid] = {"source": SOURCE_OFFLINE, "barriers": offline_barriers(notes[pid]), "error": err}
    else:
        for pid, note in notes.items():
            results[pid] = {"source": SOURCE_OFFLINE, "barriers": offline_barriers(note), "error": "No API key"}

    n_claude = sum(r["source"] == SOURCE_CLAUDE for r in results.values())
    return {
        "results": results,
        "model": model,
        "n_claude": n_claude,
        "n_offline": len(results) - n_claude,
        "errors": sorted(set(errors)),
    }


def barrier_summary(inpatients: pd.DataFrame, scan: dict) -> pd.DataFrame:
    """One row per scanned patient with social/medical flags (for totals and the chart)."""
    rows = []
    for pid, ins in zip(inpatients["patient_id"], inpatients["insurance"]):
        r = scan["results"].get(pid)
        if r is None:
            continue
        kinds = {b["type"] for b in r["barriers"]}
        rows.append({"patient_id": pid, "insurance": ins,
                     "social": "Social" in kinds, "medical": "Medical" in kinds})
    return pd.DataFrame(rows)


def social_barrier_by_insurance(summary: pd.DataFrame) -> pd.DataFrame:
    """% of scanned patients with a social barrier, by insurance group (display only)."""
    g = summary.groupby("insurance").agg(patients=("social", "size"), with_social=("social", "sum"))
    g = g.reindex([i for i in INSURANCE_GROUPS if i in g.index]).reset_index()
    g["pct_social"] = (100 * g["with_social"] / g["patients"]).round(0)
    return g


# ---------------------------------------------------------------------------
# Decision log (saved to disk so it survives a page refresh)
# ---------------------------------------------------------------------------

DECISION_LOG_COLUMNS = ["time", "role", "decision", "item", "context", "reason"]
DECISION_LOG_PATH = Path(
    os.environ.get("FOREWARD_DECISION_LOG", Path(__file__).resolve().parent / "data" / "decision_log.csv")
)


def read_decision_log(path: Path | None = None) -> pd.DataFrame:
    path = Path(path or DECISION_LOG_PATH)
    if not path.exists():
        return pd.DataFrame(columns=DECISION_LOG_COLUMNS)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def append_decision(entry: dict, path: Path | None = None) -> None:
    path = Path(path or DECISION_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DECISION_LOG_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow({c: entry.get(c, "") for c in DECISION_LOG_COLUMNS})


def clear_decision_log(path: Path | None = None) -> None:
    Path(path or DECISION_LOG_PATH).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Physician overrides -> forecast
# ---------------------------------------------------------------------------

NOT_ADMITTING = "Not admitting"


def apply_physician_overrides(ed_scored: pd.DataFrame, log: pd.DataFrame):
    """Copy of the ED patients with each patient's latest physician override applied,
    for the forecast only. A later Accept restores the AI bed type.

    Returns (adjusted copy, list of applied overrides as dicts).
    """
    out = ed_scored.copy()
    applied = []
    ed_log = log[(log["context"] == "ED Admissions") & log["decision"].isin(["Accepted", "Overridden"])]
    for pid in out["patient_id"]:
        rows = ed_log[ed_log["item"].str.startswith(f"{pid} bed type:")]
        if not len(rows):
            continue
        last = rows.iloc[-1]
        if last["decision"] != "Overridden" or last["role"] != "Physician" or " → " not in last["item"]:
            continue
        old, new = last["item"].split(": ", 1)[1].split(" → ")
        idx = out.index[out["patient_id"] == pid][0]
        if out.at[idx, "status"] == "Boarding":
            out.at[idx, "admit_unit"] = None if new == NOT_ADMITTING else new
        out.at[idx, "likely_bed"] = new
        if new == NOT_ADMITTING:
            out.at[idx, "admit_probability"] = 0.0
        applied.append({"patient_id": pid, "from": old, "to": new, "time": last["time"]})
    return out, applied


# ---------------------------------------------------------------------------
# Equity & oversight
# ---------------------------------------------------------------------------

FAIRNESS_AUC_GAP = 0.05  # flag any group this far below overall AUC
FAIRNESS_MIN_GROUP = 30  # smaller groups are shown but marked "too few to judge"
FAIRNESS_MIN_ADMITTED = 10  # AUC with fewer admitted patients is too noisy to judge
OVERRIDE_BAND = (0.05, 0.40)
OVERRIDE_MIN_DECISIONS = 5
SYNTHETIC_OVERRIDE_BASELINE = 0.18  # synthetic 30-day baseline, shown when the log is thin


def fairness_auc(model: "ExplainableLogit", ed_visits: pd.DataFrame) -> pd.DataFrame:
    """Admission model AUC on the held-out test set, by insurance, race/ethnicity,
    age group, and language. These fields are NOT model inputs."""
    test = ed_visits.loc[model.test_index]
    p = model.predict_proba(test)
    y = test["admitted"].to_numpy()
    overall = roc_auc_score(y, p)
    rows = []
    for attr, label in [("insurance", "Insurance"), ("race_ethnicity", "Race/ethnicity"),
                        ("age_group", "Age group"), ("primary_language", "Language")]:
        for group, idx in test.groupby(attr).indices.items():
            yg, pg = y[idx], p[idx]
            enough = (len(idx) >= FAIRNESS_MIN_GROUP and yg.sum() >= FAIRNESS_MIN_ADMITTED
                      and yg.sum() < len(yg))
            auc = roc_auc_score(yg, pg) if 0 < yg.sum() < len(yg) else np.nan
            gap = auc - overall if not np.isnan(auc) else np.nan
            if not enough:
                status = "Too few to judge"
            elif gap < -FAIRNESS_AUC_GAP:
                status = "Do not go live: retrain and re-test"
            else:
                status = "OK"
            rows.append({"attribute": label, "group": group, "patients": len(idx),
                         "admitted": int(yg.sum()), "auc": auc, "gap": gap, "status": status})
    out = pd.DataFrame(rows)
    out.attrs["overall_auc"] = overall
    return out


def override_rate(log: pd.DataFrame) -> dict:
    """Share of ED bed-type decisions that overrode the AI, from the decision log.
    With fewer than 5 decisions, falls back to a labeled synthetic baseline."""
    d = log[(log["context"] == "ED Admissions") & log["decision"].isin(["Accepted", "Overridden"])]
    n = len(d)
    if n >= OVERRIDE_MIN_DECISIONS:
        rate, source = (d["decision"] == "Overridden").mean(), f"Decision log ({n} decisions)"
    else:
        rate, source = SYNTHETIC_OVERRIDE_BASELINE, (
            f"Synthetic 30-day baseline (only {n} logged decision{'s' if n != 1 else ''} so far)")
    low, high = OVERRIDE_BAND
    if rate > high:
        status = "Too high: staff may not trust the tool"
    elif rate < low:
        status = "Too low: possible rubber-stamping"
    else:
        status = "Healthy"
    return {"rate": float(rate), "source": source, "status": status, "decisions": n}


def generate_discharge_trust(days: int = 7) -> pd.DataFrame:
    """Synthetic predicted vs. actual hospital-wide discharges for the last `days` days."""
    rng = _rng(9)
    dates = pd.date_range(end=REFERENCE_DATE - pd.Timedelta(days=1), periods=days)
    predicted = np.round(rng.normal(64, 5, days) * np.where(dates.dayofweek >= 5, 0.8, 1.0), 1)
    actual = np.round(predicted + rng.normal(0, 3.5, days)).astype(int)
    return pd.DataFrame({"date": dates, "predicted": predicted, "actual": actual,
                         "miss": np.abs(actual - predicted).round(1)})


PAUSE_RETIRE_RULES = {
    "Pause the tool if": [
        f"any group's AUC falls more than {FAIRNESS_AUC_GAP} below overall",
        "calibration drifts (predicted vs. actual discharges stop matching)",
        "a safety event is linked to the tool",
    ],
    "Review the tool if": ["the override rate stays outside 5–40% for 2 weeks"],
    "Retire the tool after": ["2 failed retraining cycles"],
}
OVERSIGHT_OWNERS = {
    "CMIO": "Accountable",
    "Data analyst": "Monthly monitoring",
    "AI oversight committee": "Decides to pause, retrain, or retire",
}


# ---------------------------------------------------------------------------
# Silent mode, manual mode, footer
# ---------------------------------------------------------------------------

FOOTER_TEXT = ("AI recommendations only — a qualified human approves every action. "
               "Insurance, race, and language are never model inputs. "
               "All data is synthetic, calibrated to the Halvian case brief.")

SILENT_LOG_PATH = Path(
    os.environ.get("FOREWARD_SILENT_LOG", Path(__file__).resolve().parent / "data" / "silent_predictions.csv")
)
SILENT_LOG_COLUMNS = ["logged_at", "kind", "subject", "prediction", "detail"]


def log_silent_predictions(ed_scored: pd.DataFrame, medicine_forecast: "Forecast", path: Path | None = None) -> int:
    """Silent mode: record predictions for later validation against what happened."""
    from datetime import datetime

    path = Path(path or SILENT_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = [
        {"logged_at": now, "kind": "ED admission", "subject": r.patient_id,
         "prediction": f"{r.admit_probability:.3f}", "detail": f"likely bed: {r.likely_bed}"}
        for r in ed_scored.itertuples()
    ]
    rows.append({"logged_at": now, "kind": "Medicine forecast", "subject": "Medicine",
                 "prediction": f"{medicine_forecast.min_open:.2f}",
                 "detail": f"run-out: {hour_label(medicine_forecast.runout_hour) if medicine_forecast.runout_hour else 'none'}"})
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SILENT_LOG_COLUMNS)
        if new_file:
            w.writeheader()
        w.writerows(rows)
    return len(rows)


MANUAL_PROTOCOL = [
    "Declare downtime: the house supervisor announces it and notes the time.",
    "Use this whiteboard plus the last auto-printed bed board.",
    "Bed huddles every 2 hours (house supervisor, charge nurses, bed manager, ED charge).",
    "Request beds by phone to the bed manager.",
    "Record every placement and decision in the paper log.",
    "Reconcile the paper log into the system after recovery.",
    "Drill this protocol monthly.",
]


# ---------------------------------------------------------------------------
# Equity baseline (from the Halvian case brief; not synthetic)
# ---------------------------------------------------------------------------

HALVIAN_BASELINE = pd.DataFrame(
    {
        "group": ["Commercial", MEDICAID, "Uninsured", "Overall"],
        "door_to_provider_min": [48, 58, 64, 55],
        "lwbs_pct": [3.0, 4.6, 5.8, 4.0],
    }
)

DATA_SOURCES_NOTE = "Synthetic data calibrated to the Halvian case brief."


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def load_all() -> dict[str, pd.DataFrame]:
    return {
        "beds": generate_bed_status(),
        "ed_visits": generate_ed_visits(),
        "ed_current": generate_current_ed_patients(),
        "inpatients": generate_inpatients(),
        "past_stays": generate_past_inpatient_stays(),
        "boarding": generate_boarding_history(),
        "ed_arrivals": generate_ed_arrivals(),
    }
