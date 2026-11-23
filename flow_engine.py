"""ForeWard Flow — synthetic data and logic engine.

ALL DATA IN THIS MODULE IS SYNTHETIC. No real patients, staff, or hospital
records are used. Every generator uses a fixed seed so results are
reproducible across runs.

Each generator gets its own seeded random stream (SEED + offset), so adding
or changing one dataset never shifts the values of another.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42
REFERENCE_DATE = pd.Timestamp("2026-11-23")  # fixed "today" for reproducibility

HOSPITAL_NAME = "ForeWard Flow Demo Hospital (fictional, 320 beds)"

UNIT_CAPACITY = {
    "Medicine": 120,
    "Surgery": 80,
    "Telemetry": 56,
    "ICU": 24,
    "Women's & Peds": 40,
}

INSURANCE_GROUPS = ["Commercial", "Medicare", "Medicaid", "Uninsured"]
INSURANCE_SHARES = [0.33, 0.22, 0.35, 0.10]

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

def generate_bed_status() -> pd.DataFrame:
    """Current bed board: open / needs cleaning / occupied per unit."""
    rng = _rng(1)
    rows = []
    for unit, total in UNIT_CAPACITY.items():
        if unit == "Medicine":
            open_beds, cleaning = 3, 3  # Medicine is deliberately tight
        else:
            open_beds = int(rng.integers(max(2, total // 20), max(4, total // 8) + 1))
            cleaning = int(rng.integers(1, max(2, total // 20) + 1))
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
        elif ins in ("Commercial", "Medicaid") and rng.random() < 0.13:
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
        [insurance == "Medicaid", insurance == "Uninsured", insurance == "Medicare"],
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


def _admission_logit(df: pd.DataFrame) -> np.ndarray:
    """Clinical-only ground truth. Fairness columns play no role."""
    acuity_effect = df["triage_acuity"].map({1: 3.0, 2: 1.7, 3: 0.4, 4: -1.1, 5: -2.2})
    complaint_effect = df["chief_complaint"].map({c: v[1] for c, v in CHIEF_COMPLAINTS.items()})
    return (
        -1.9
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


def generate_current_ed_patients() -> pd.DataFrame:
    """14 patients in the ED now: 6 admitted & boarding, 8 still in workup."""
    rng = _rng(4)
    df = _generate_visit_features(14, rng)
    status = ["Boarding"] * 6 + ["In workup"] * 8
    df.insert(0, "patient_id", [f"ED-{201 + i}" for i in range(14)])
    df.insert(1, "bed", [f"ED {i + 1}" for i in range(14)])
    df["status"] = status

    # Boarders: skew toward Medicine since it's the tightest unit.
    df["admit_unit"] = None
    boarding_units = rng.choice(
        list(UNIT_CAPACITY), size=6, p=[0.5, 0.15, 0.2, 0.1, 0.05]
    )
    df.loc[:5, "admit_unit"] = boarding_units
    df["hours_boarding"] = 0.0
    df.loc[:5, "hours_boarding"] = np.round(rng.uniform(1.5, 14, 6), 1)
    df["hours_in_ed"] = np.round(rng.uniform(0.5, 6, 14), 1)
    df.loc[:5, "hours_in_ed"] = df.loc[:5, "hours_boarding"] + np.round(rng.uniform(2, 5, 6), 1)
    return df


# ---------------------------------------------------------------------------
# Inpatients with nurse notes
# ---------------------------------------------------------------------------

MEDICAL_BARRIERS = {
    "Pending MRI": "MRI still pending, not yet scheduled.",
    "IV antibiotics": "Remains on IV antibiotics, awaiting ID input on PO switch.",
    "PT eval": "PT eval not yet completed; unsafe to ambulate independently.",
    "Pending echo": "Echo ordered yesterday, still pending read.",
    "Awaiting consult": "Awaiting cardiology consult note.",
}

SOCIAL_BARRIERS = {
    "No ride home": "Pt states no ride home; family unavailable until weekend.",
    "Can't afford meds": "Pt reports they can't afford discharge meds.",
    "Nursing-home placement": "Needs nursing-home placement; SW sending referrals.",
    "Housing / shelter": "Pt currently staying at a shelter; needs safe discharge plan.",
    "Insurance authorization": "Insurance authorization pending for post-acute care.",
}

NOTE_OPENERS = [
    "Pt resting comfortably, VSS.",
    "A&O x4, pain controlled on current regimen.",
    "Slept well overnight, tolerating diet.",
    "Ambulating in hallway with steady gait.",
    "Afebrile overnight, no acute events.",
]

NOTE_CLOSERS = [
    "Will continue to monitor.",
    "Plan per primary team.",
    "Family updated at bedside.",
    "Discussed plan of care with pt.",
]

UNIT_ROOM_PREFIX = {
    "Medicine": "M",
    "Surgery": "S",
    "Telemetry": "T",
    "ICU": "I",
    "Women's & Peds": "W",
}


def generate_inpatients(n: int = 40) -> pd.DataFrame:
    """Current inpatients with a short synthetic nurse note.

    `medical_barriers` / `social_barriers` are the ground-truth tags used to
    build the note text (handy for checking any note-parsing logic).
    Social barriers are more common for Medicaid and Uninsured patients.
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

        med = []
        if rng.random() < 0.40:
            k = 1 if rng.random() < 0.8 else 2
            med = list(rng.choice(list(MEDICAL_BARRIERS), size=k, replace=False))

        social_p = 0.55 if ins in ("Medicaid", "Uninsured") else 0.15
        soc = []
        if rng.random() < social_p:
            options = list(SOCIAL_BARRIERS)
            if ages[i] < 65:
                options.remove("Nursing-home placement")
            k = 1 if rng.random() < 0.75 else 2
            soc = list(rng.choice(options, size=k, replace=False))

        parts = [rng.choice(NOTE_OPENERS)]
        parts += [MEDICAL_BARRIERS[b] for b in med]
        parts += [SOCIAL_BARRIERS[b] for b in soc]
        if not med and not soc:
            parts.append("Medically ready; anticipate discharge today.")
        parts.append(rng.choice(NOTE_CLOSERS))

        rows.append(
            {
                "patient_id": f"IP-{3001 + i}",
                "unit": unit,
                "room": room,
                "age": int(ages[i]),
                "insurance": ins,
                "los_days": int(rng.integers(1, 12)),
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
    """Patient-days from past stays with a `discharged_that_day` (1/0) outcome."""
    rng = _rng(6)
    units = rng.choice(list(UNIT_CAPACITY), size=n, p=[0.40, 0.22, 0.18, 0.08, 0.12])
    insurance = rng.choice(INSURANCE_GROUPS, size=n, p=INSURANCE_SHARES)
    age = _ages_for_insurance(insurance, rng)
    los = rng.integers(1, 15, size=n)
    day_of_week = rng.integers(0, 7, size=n)  # 0 = Monday

    medical = (rng.random(n) < 0.40).astype(int)
    social_p = np.where(np.isin(insurance, ["Medicaid", "Uninsured"]), 0.50, 0.15)
    social = (rng.random(n) < social_p).astype(int)

    unit_effect = pd.Series(units).map(
        {"Medicine": 0.0, "Surgery": 0.2, "Telemetry": 0.1, "ICU": -1.5, "Women's & Peds": 0.6}
    ).to_numpy()
    logit = (
        -0.9
        + 0.18 * np.minimum(los, 7)
        - 1.7 * medical
        - 1.2 * social
        - 0.6 * (day_of_week >= 5)
        - 0.012 * np.maximum(age - 65, 0)
        + unit_effect
    )
    discharged = (rng.random(n) < _sigmoid(logit)).astype(int)

    return pd.DataFrame(
        {
            "stay_id": [f"S{500000 + i}" for i in range(n)],
            "unit": units,
            "age": age,
            "insurance": insurance,
            "los_days": los,
            "day_of_week": day_of_week,
            "is_weekend": (day_of_week >= 5).astype(int),
            "has_medical_barrier": medical,
            "has_social_barrier": social,
            "discharged_that_day": discharged,
        }
    )


# ---------------------------------------------------------------------------
# Boarding times by insurance (30 days)
# ---------------------------------------------------------------------------

def generate_boarding_history(days: int = 30) -> pd.DataFrame:
    """One row per boarded ED patient over the last `days` days.

    Medicaid boarders wait ~30+ minutes longer than the overall average.
    """
    rng = _rng(7)
    insurance_offset = {"Commercial": -5, "Medicare": 0, "Medicaid": 60, "Uninsured": 15}
    rows = []
    for d in range(days, 0, -1):
        date = REFERENCE_DATE - pd.Timedelta(days=d)
        n_boarders = int(rng.integers(8, 16))
        ins = rng.choice(INSURANCE_GROUPS, size=n_boarders, p=INSURANCE_SHARES)
        for g in ins:
            minutes = rng.gamma(shape=6.0, scale=38.0) + insurance_offset[g]
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
        .agg(patients="count", mean_minutes="mean")
        .reindex(INSURANCE_GROUPS)
        .reset_index()
    )
    out["mean_minutes"] = out["mean_minutes"].round(1)
    out["vs_overall_minutes"] = (out["mean_minutes"] - overall).round(1)
    return out


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
    }
