"""ForeWard Flow — synthetic data and logic engine.

ALL DATA IN THIS MODULE IS SYNTHETIC. No real patients, staff, or hospital
records are used. Every generator uses a fixed seed so results are
reproducible across runs.

Each generator gets its own seeded random stream (SEED + offset), so adding
or changing one dataset never shifts the values of another.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
    # Boarders come from synthetic patients whose sampled outcome was
    # "admitted", so they look like real admits. Workup patients are unselected.
    pool = _generate_visit_features(300, rng)
    was_admitted = rng.random(len(pool)) < _sigmoid(_admission_logit(pool))
    boarders = pool[was_admitted].head(6)
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

        social_p = 0.55 if ins in ("Medicaid", "Uninsured") else 0.15
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
    social_p = np.where(np.isin(insurance, ["Medicaid", "Uninsured"]), 0.45, 0.15)
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
            out.append([self.labeler(f, df.at[idx, f]) for f in top.index])
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


def _admission_label(feature: str, value) -> str:
    return {
        "age": lambda v: f"Age {v}",
        "triage_acuity": lambda v: f"Triage acuity {v}",
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


def _discharge_label(feature: str, value) -> str:
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
