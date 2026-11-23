"""ForeWard Flow — staff-facing ED patient flow & boarding dashboard.

Uses SYNTHETIC data only (see flow_engine.py).
"""

import streamlit as st

import flow_engine as fe

st.set_page_config(page_title="ForeWard Flow", layout="wide")


@st.cache_data
def load_beds():
    return fe.generate_bed_status()


st.title("ForeWard Flow")
st.caption(f"{fe.HOSPITAL_NAME} · Synthetic data only — not real patients")

beds = load_beds()

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
