"""ForeWard Flow — staff-facing ED patient flow & boarding dashboard.

Uses SYNTHETIC data only (see flow_engine.py).
"""

import streamlit as st

import flow_engine as fe

st.set_page_config(page_title="ForeWard Flow", layout="wide")


@st.cache_data
def load_data():
    return fe.load_all()


@st.cache_resource
def load_models():
    data = load_data()
    return (
        fe.train_admission_model(data["ed_visits"]),
        fe.train_discharge_model(data["past_stays"]),
    )


data = load_data()
admission_model, discharge_model = load_models()

# --- Sidebar: model performance ---------------------------------------------
with st.sidebar:
    st.header("Model performance")
    st.metric("Admission at triage — AUC", f"{admission_model.auc:.2f}")
    st.metric("Likely home today — AUC", f"{discharge_model.auc:.2f}")
    st.caption("Tested on held-out synthetic data.")
    st.divider()
    st.caption(
        "Both models are logistic regressions, chosen because they're explainable. "
        "Insurance, race/ethnicity, age group, and language are never model inputs; "
        "they're kept only for fairness checks."
    )

st.title("ForeWard Flow")
st.caption(f"{fe.HOSPITAL_NAME} · Synthetic data only — not real patients")

# --- Bed status ---------------------------------------------------------------
beds = data["beds"]

st.subheader("Bed status")
total = beds[["total_beds", "open", "needs_cleaning", "occupied"]].sum()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total beds", int(total["total_beds"]))
c2.metric("Open", int(total["open"]))
c3.metric("Needs cleaning", int(total["needs_cleaning"]))
c4.metric("Occupied", int(total["occupied"]))

for col, row in zip(st.columns(len(beds)), beds.itertuples()):
    with col:
        st.markdown(f"**{row.unit}**")
        st.metric("Open", row.open)
        st.metric("Needs cleaning", row.needs_cleaning)
        st.caption(f"{row.occupied}/{row.total_beds} occupied ({row.occupancy_pct}%)")

with st.expander("Bed status table"):
    st.dataframe(
        beds.rename(
            columns={
                "unit": "Unit",
                "total_beds": "Total",
                "open": "Open",
                "needs_cleaning": "Needs cleaning",
                "occupied": "Occupied",
                "occupancy_pct": "Occupancy %",
            }
        ),
        hide_index=True,
        width="stretch",
    )

# --- ED: admission prediction --------------------------------------------------
st.subheader("ED patients — admission prediction")
ed = fe.score_current_ed(admission_model, data["ed_current"])
boarding = ed[ed["status"] == "Boarding"]
c1, c2, c3 = st.columns(3)
c1.metric("Patients in ED", len(ed))
c2.metric("Boarding (admitted, no bed)", len(boarding))
c3.metric("Longest boarding", f"{boarding['hours_boarding'].max():.1f} h")

st.dataframe(
    ed[
        [
            "patient_id", "bed", "status", "age", "triage_acuity", "chief_complaint",
            "admit_probability", "top_reasons", "likely_bed", "bed_rule",
            "admit_unit", "hours_boarding",
        ]
    ],
    hide_index=True,
    width="stretch",
    column_config={
        "patient_id": "Patient",
        "bed": "ED bed",
        "status": "Status",
        "age": "Age",
        "triage_acuity": "Acuity",
        "chief_complaint": "Chief complaint",
        "admit_probability": st.column_config.ProgressColumn(
            "Admit probability", min_value=0.0, max_value=1.0, format="percent"
        ),
        "top_reasons": st.column_config.TextColumn("Top reasons pushing score up", width="large"),
        "likely_bed": "Likely bed type",
        "bed_rule": "Bed rule",
        "admit_unit": "Admitted to",
        "hours_boarding": st.column_config.NumberColumn("Hours boarding", format="%.1f"),
    },
)
st.caption(
    "Reasons are compared with the average ED patient in the training data. "
    "Likely bed type uses fixed rules: ICU if acuity 1, O₂ < 89%, or systolic BP < 90; "
    "Telemetry for chest pain, or cardiac history with HR > 105; "
    "Surgery for abdominal pain or fall/injury; otherwise Medicine."
)

# --- Inpatients: likely to go home today ----------------------------------------
st.subheader("Inpatients — likely to go home today")
ranked = fe.rank_inpatients(discharge_model, data["inpatients"])
ranked.insert(0, "rank", range(1, len(ranked) + 1))

c1, c2 = st.columns(2)
c1.metric("Inpatients listed", len(ranked))
c2.metric("Expected discharges today (sum of probabilities)", f"{ranked['discharge_probability'].sum():.1f}")

st.dataframe(
    ranked[
        [
            "rank", "patient_id", "unit", "room", "los_days", "discharge_probability",
            "reasons_for", "reasons_against", "nurse_note",
        ]
    ],
    hide_index=True,
    width="stretch",
    column_config={
        "rank": "#",
        "patient_id": "Patient",
        "unit": "Unit",
        "room": "Room",
        "los_days": "Days in hospital",
        "discharge_probability": st.column_config.ProgressColumn(
            "Home-today probability", min_value=0.0, max_value=1.0, format="percent"
        ),
        "reasons_for": st.column_config.TextColumn("Top reasons for", width="medium"),
        "reasons_against": st.column_config.TextColumn("Top barriers", width="medium"),
        "nurse_note": st.column_config.TextColumn("Nurse note", width="large"),
    },
)
st.caption(
    "Model inputs: days past expected stay, pending test, IV meds, mobility, "
    "physician-noted likely discharge, and documented social barrier."
)
