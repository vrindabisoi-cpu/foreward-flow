"""ForeWard Flow — staff-facing ED patient flow & boarding dashboard.

Uses SYNTHETIC data only (see flow_engine.py).
"""

from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

import flow_engine as fe

st.set_page_config(page_title="ForeWard Flow — Halvian Medical Center", layout="wide")

DECISION_OWNER = "House supervisor"
ROLES = [DECISION_OWNER, "Charge nurse", "Bed manager", "ED physician"]
SERIES_BLUE = "#2a78d6"
CRITICAL_RED = "#d03b3b"


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
ed = fe.score_current_ed(admission_model, data["ed_current"])
ranked = fe.rank_inpatients(discharge_model, data["inpatients"])
beds = data["beds"]
feed = fe.public_health_feed()
alerts = fe.news_alerts()

# --- Decision log (saved to data/decision_log.csv) ----------------------------------


def log_decision(role, decision, item, context, reason_key, default_reason, require_reason):
    reason = st.session_state.get(reason_key, "").strip()
    if require_reason and not reason:
        verb = {"Declined": "decline", "Dismissed": "dismiss"}.get(decision, decision.lower())
        st.session_state["flash"] = f"Add a reason before you {verb}."
        return False
    fe.append_decision(
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "role": role,
            "decision": decision,
            "item": item,
            "context": context,
            "reason": reason or default_reason,
        }
    )
    st.session_state[reason_key] = ""
    return True


def clear_log(role):
    fe.clear_decision_log()
    # Keep a record that the log was cleared, and by whom.
    fe.append_decision(
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "role": role,
            "decision": "Cleared log",
            "item": "Decision log",
            "context": "Decision log",
            "reason": "Log cleared by house supervisor",
        }
    )


def alert_status(log, alert):
    rows = log[log["item"] == f"News alert: {alert['headline']}"]
    return rows["decision"].iloc[-1] if len(rows) else "Pending review"


decision_log = fe.read_decision_log()


def decide_action(role, decision, action, context):
    log_decision(
        role, decision, action, context, "action_reason",
        "Approved as recommended", require_reason=(decision == "Declined"),
    )


def review_alert(role, alert, decision):
    key = f"alert_reason_{alert['id']}"
    log_decision(
        role, decision, f"News alert: {alert['headline']}", "Public health & news alerts", key,
        "Applied after review", require_reason=(decision == "Dismissed"),
    )


# --- Sidebar ---------------------------------------------------------------------
with st.sidebar:
    st.header("Your role")
    role = st.selectbox("Acting as", ROLES, index=0)
    is_owner = role == DECISION_OWNER
    if not is_owner:
        st.caption(f"View only. Decisions belong to the {DECISION_OWNER.lower()}.")

    st.header("Forecast inputs")
    day_of_week = st.selectbox(
        "Day of week",
        fe.DAYS_OF_WEEK,
        index=fe.DAYS_OF_WEEK.index(fe.default_day_of_week()),
        help="Monday adds 15% to expected admissions.",
    )
    trend_options = list(fe.TREND_LEVELS)
    feed_level = feed["local"]["level"]
    trend_level = st.selectbox(
        "Local flu/RSV trend",
        trend_options,
        index=trend_options.index(feed_level),
        format_func=lambda t: f"{t} (+{fe.TREND_LEVELS[t]:.0%})" if fe.TREND_LEVELS[t] else t,
        help="Set automatically from the public health feed. Change it only to override.",
    )
    st.caption(f"Source: {feed['source']}, as of {feed['as_of']}.")

    st.header("ED staffing — current shift")
    staffing = fe.CURRENT_SHIFT_STAFFING
    st.caption(f"{staffing['shift']} · context only, not used in any calculation")
    st.markdown("\n".join(f"- {n} {role}" for role, n in staffing["staff"].items()))

    st.header("Data sources")
    st.caption(fe.DATA_SOURCES_NOTE)

    st.header("Model performance")
    st.metric("Admission at triage — AUC", f"{admission_model.auc:.2f}")
    st.metric("Likely home today — AUC", f"{discharge_model.auc:.2f}")
    st.caption("Tested on held-out synthetic data.")
    st.caption(
        "Both models are logistic regressions, chosen because they're explainable. "
        "Insurance, race/ethnicity, age group, and language are never model inputs; "
        "they're kept only for fairness checks."
    )

st.title(fe.HOSPITAL_NAME)
st.caption("Synthetic data only — not real patients")

if "flash" in st.session_state:
    st.error(st.session_state.pop("flash"))

tab_control, tab_flow, tab_whatif, tab_equity = st.tabs(["AI Control Center", "Patient flow", "What-If", "Equity"])

# =====================================================================================
# AI Control Center
# =====================================================================================
with tab_control:
    # --- Live bed counts ---------------------------------------------------------
    st.subheader("Live bed counts")
    for col, row in zip(st.columns(len(beds)), beds.itertuples()):
        with col, st.container(border=True):
            st.markdown(f"**{row.unit}**")
            st.markdown(
                f"🟢 **{row.open}** open  \n"
                f"🟡 **{row.needs_cleaning}** needs cleaning  \n"
                f"🔴 **{row.occupied}** occupied"
            )
            st.caption(f"{row.total_beds} beds · {row.occupancy_pct}% occupied")

    # --- Boarding metrics ----------------------------------------------------------
    st.subheader("ED room status")
    rooms = fe.ed_room_status(ed)
    treat = rooms.set_index("area").loc["Treatment rooms"]
    st.markdown(f"#### {treat['held_by_boarders']} of {treat['total']} ED rooms held by boarders")
    room_cols = st.columns(len(rooms) + 1)
    for col, row in zip(room_cols[:-1], rooms.itertuples()):
        with col, st.container(border=True):
            st.markdown(f"**{row.area}**")
            lines = []
            if row.held_by_boarders:
                lines.append(f"🛏️ **{row.held_by_boarders}** held by boarders")
            lines += [f"🔴 **{row.other_patients}** other patients", f"🟢 **{row.open}** open"]
            st.markdown("  \n".join(lines))
            st.caption(f"{row.total} total")
    with room_cols[-1], st.container(border=True):
        st.markdown("**Waiting room**")
        st.markdown(f"🪑 **{fe.WAITING_ROOM_PATIENTS}** patients waiting")
        st.caption(f"{fe.WAITING_ROOM_SEATS} seats")

    bm = fe.boarding_metrics(ed)
    c1, c2, c3 = st.columns(3)
    c1.metric("Admitted patients boarding", bm["patients"])
    c2.metric("Total boarding hours", f"{bm['total_hours']:.1f}")
    c3.metric("Today's boarding cost", f"${bm['cost']:,.0f}")
    st.caption(f"Cost estimate: ${fe.BOARDING_COST_PER_HOUR} per boarding hour.")

    # --- 8-hour forecast ------------------------------------------------------------
    st.subheader("8-hour bed forecast (9 AM–5 PM)")
    units = list(fe.UNIT_CAPACITY)
    unit = st.selectbox("Unit", units, index=units.index("Medicine"))
    applied = [a for a in alerts if alert_status(decision_log, a) == "Applied to forecast"]
    forecast = fe.forecast_unit(unit, beds, ed, ranked, day_of_week, trend_level, applied)
    tier = fe.recommend_tier(forecast)

    warning = forecast.warning()
    left, right = st.columns([3, 1])
    with left:
        if warning:
            st.error(
                f"**{warning}**\n\n**Why:**\n"
                + "\n".join(f"- {r}" for r in forecast.reasons()),
                icon="⚠️",
            )
        else:
            st.success(
                f"**{unit} projected to keep open beds through 5 PM** "
                f"(lowest point: about {max(forecast.min_open, 0):.0f})",
                icon="✅",
            )
    with right:
        if forecast.head_start_hours is not None:
            hs = forecast.head_start_hours
            st.markdown(f"### ⏱️ {hs}-hour head start" if hs else "### ⏱️ No head start")
            st.caption(f"From the 9 AM forecast to the projected {fe.hour_label(forecast.runout_hour)} run-out.")
        st.metric("Expected admissions adjustment", f"+{forecast.admission_adjustment:.0%}")

    chart_df = forecast.table.assign(projected_open=lambda d: d["projected_open"].round(1))
    x = alt.X("hour:O", title=None, sort=None, axis=alt.Axis(labelExpr=
        "datum.value < 12 ? datum.value + ' AM' : (datum.value == 12 ? '12 PM' : (datum.value - 12) + ' PM')",
        labelAngle=0))
    hover = alt.selection_point(fields=["hour"], nearest=True, on="pointerover", empty=False)
    line = alt.Chart(chart_df).mark_line(color=SERIES_BLUE, strokeWidth=2).encode(
        x=x, y=alt.Y("projected_open:Q", title="Projected open beds")
    )
    points = alt.Chart(chart_df).mark_point(filled=True, size=70, color=SERIES_BLUE).encode(
        x=x,
        y="projected_open:Q",
        opacity=alt.condition(hover, alt.value(1), alt.value(0.35)),
        tooltip=[
            alt.Tooltip("time:N", title="Time"),
            alt.Tooltip("projected_open:Q", title="Projected open beds", format=".1f"),
            alt.Tooltip("cleaned:Q", title="Rooms cleaned", format=".0f"),
            alt.Tooltip("discharges:Q", title="Expected discharges", format=".1f"),
            alt.Tooltip("ed_admits:Q", title="ED patients admitted", format=".1f"),
            alt.Tooltip("new_arrival_admits:Q", title="New-arrival admissions", format=".1f"),
        ],
    ).add_params(hover)
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=CRITICAL_RED, strokeDash=[4, 3]).encode(y="y:Q")
    zero_label = alt.Chart(pd.DataFrame({"y": [0], "label": ["No open beds"]})).mark_text(
        align="left", dx=4, dy=-6, color=CRITICAL_RED, fontSize=11
    ).encode(y="y:Q", text="label:N", x=alt.value(0))
    st.altair_chart((zero + zero_label + line + points).properties(height=280), width="stretch")
    st.caption(
        "Projected open beds = open now + rooms cleaned + expected discharges "
        "(likely-home-today model) − expected admissions (ED patients from the admission "
        "model + new arrivals by hour)."
    )
    with st.expander("Forecast table"):
        st.dataframe(
            forecast.table.drop(columns="hour").round(1),
            hide_index=True,
            width="stretch",
            column_config={
                "time": "Time",
                "open_now": "Open now",
                "cleaned": "Rooms cleaned",
                "discharges": "Expected discharges",
                "ed_admits": "ED patients admitted",
                "new_arrival_admits": "New-arrival admissions",
                "projected_open": "Projected open beds",
            },
        )

    with st.expander("All units at a glance"):
        overview = []
        for u in units:
            f = fe.forecast_unit(u, beds, ed, ranked, day_of_week, trend_level, applied)
            overview.append(
                {
                    "Unit": u,
                    "Lowest projected open beds": round(f.min_open, 1),
                    "At": fe.hour_label(f.worst_hour),
                    "Runs out by": fe.hour_label(f.runout_hour) if f.runout_hour else "—",
                    "Recommended tier": fe.recommend_tier(f),
                }
            )
        st.dataframe(pd.DataFrame(overview), hide_index=True, width="stretch")

    # --- Capacity tiers & decisions ---------------------------------------------------
    st.subheader("Recommended capacity tier")
    tier_info = fe.CAPACITY_TIERS[tier]
    icon = {"Normal": "✅", "Level 1 — Tight": "🟡", "Level 2 — Full": "🟠", "Level 3 — Crisis": "🔴"}[tier]
    st.markdown(f"### {icon} {unit}: {tier}")
    st.caption(f"Why this tier: {fe.tier_rationale(forecast)}")
    st.markdown(f"**Decision owner: {DECISION_OWNER}**")

    context = f"{unit} · {tier} · {day_of_week} · {trend_level}"
    latest = {}
    for entry in decision_log[decision_log["context"] == context].to_dict("records"):
        latest[entry["item"]] = entry

    st.text_input(
        "Reason for your decision (required to decline)",
        key="action_reason",
        disabled=not is_owner,
        placeholder="e.g. Discharge lounge already open since 8 AM",
    )
    for i, action in enumerate(tier_info["actions"]):
        a_col, s_col, ok_col, no_col = st.columns([5, 2, 1, 1])
        a_col.markdown(f"**{action}**")
        if action in latest:
            e = latest[action]
            mark = "✅ Approved" if e["decision"] == "Approved" else "❌ Declined"
            s_col.caption(f"{mark} · {e['time'][11:16]}")
        else:
            s_col.caption("Awaiting decision")
        ok_col.button(
            "Approve", key=f"approve_{i}", disabled=not is_owner, type="primary",
            on_click=decide_action, args=(role, "Approved", action, context),
        )
        no_col.button(
            "Decline", key=f"decline_{i}", disabled=not is_owner,
            on_click=decide_action, args=(role, "Declined", action, context),
        )

    with st.expander("All capacity tiers"):
        for name, info in fe.CAPACITY_TIERS.items():
            marker = " ← recommended" if name == tier else ""
            st.markdown(f"**{name}**{marker} — {info['when']}")
            st.markdown("\n".join(f"- {a}" for a in info["actions"]))

    # --- Public health & news alerts ------------------------------------------------------
    st.subheader("Public health & news alerts")
    ph_col, news_col = st.columns(2)
    with ph_col, st.container(border=True):
        st.markdown("**Official public health data**")
        st.caption(f"{feed['source']}, as of {feed['as_of']}.")
        loc, nat = feed["local"], feed["national"]
        c1, c2 = st.columns(2)
        c1.metric("Local flu/RSV level", loc["level"].replace(" flu/RSV", ""), loc["trend"], delta_color="inverse")
        c1.caption(f"Flu {loc['flu_ed_visits_pct']}% · RSV {loc['rsv_ed_visits_pct']}% of ED visits")
        c2.metric("National level", nat["level"], nat["trend"], delta_color="inverse")
        c2.caption(f"Flu {nat['flu_ed_visits_pct']}% · RSV {nat['rsv_ed_visits_pct']}% of ED visits")
        if trend_level == feed_level:
            st.info(
                f"Applied automatically: local level **{feed_level}** adds "
                f"**+{fe.TREND_LEVELS[feed_level]:.0%}** to expected admissions."
            )
        else:
            st.warning(
                f"Manual override: forecast uses **{trend_level}** instead of the "
                f"surveillance level ({feed_level})."
            )

    with news_col, st.container(border=True):
        st.markdown("**News alerts — for house supervisor review**")
        st.caption("Simulated news. Alerts never change the forecast unless a person applies them.")
        for alert in alerts:
            status = alert_status(decision_log, alert)
            effect = ", ".join(f"{u} +{p:.0%}" for u, p in alert["effect"].items())
            if len(alert["effect"]) == len(fe.UNIT_CAPACITY):
                effect = f"all units +{next(iter(alert['effect'].values())):.0%}"
            body = (
                f"**{alert['headline']}**  \n{alert['detail']}  \n"
                f"*{alert['source']}* · If applied: {effect} expected admissions  \n"
                f"Status: **{status}**"
            )
            if status == "Pending review":
                st.warning(body, icon="📰")
                st.text_input(
                    "Reason (required to dismiss)", key=f"alert_reason_{alert['id']}",
                    disabled=not is_owner, label_visibility="collapsed",
                    placeholder="Reason (required to dismiss)",
                )
                b1, b2 = st.columns(2)
                b1.button(
                    "Apply to forecast", key=f"apply_{alert['id']}", disabled=not is_owner,
                    on_click=review_alert, args=(role, alert, "Applied to forecast"), width="stretch",
                )
                b2.button(
                    "Dismiss", key=f"dismiss_{alert['id']}", disabled=not is_owner,
                    on_click=review_alert, args=(role, alert, "Dismissed"), width="stretch",
                )
            else:
                st.info(body, icon="✅" if status == "Applied to forecast" else "🗑️")

    # --- Decision log ------------------------------------------------------------------------
    st.subheader("Decision log")
    log = decision_log
    if log.empty:
        st.caption("No decisions yet.")
    else:
        st.dataframe(
            log.iloc[::-1],
            hide_index=True,
            width="stretch",
            column_config={c: c.capitalize() for c in log.columns},
        )
        d_col, c_col, _ = st.columns([1, 1, 3])
        d_col.download_button(
            "Download log (CSV)", log.to_csv(index=False), "foreward_flow_decision_log.csv", "text/csv"
        )
        with c_col.popover("Clear log", disabled=not is_owner):
            st.markdown(f"Delete all {len(log)} entries? A single entry recording the clear is kept.")
            st.button("Yes, clear the log", key="confirm_clear_log", type="primary",
                      on_click=clear_log, args=(role,))
    st.caption(
        "Saved to data/decision_log.csv, so it survives a page refresh. "
        f"Only the {DECISION_OWNER.lower()} can clear it."
    )

# =====================================================================================
# Patient flow: ED predictions and discharge list
# =====================================================================================
with tab_flow:
    st.subheader("ED patients — admission prediction")
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
        "Likely bed type uses fixed rules, checked in order: Women's & Peds if age < 18; "
        "ICU if acuity 1, O₂ < 89%, or systolic BP < 90; "
        "Telemetry for chest pain, or cardiac history with HR > 105; "
        "Surgery for abdominal pain or fall/injury; otherwise Medicine."
    )

    st.subheader("Inpatients — likely to go home today")
    ranked_view = ranked.copy()
    ranked_view.insert(0, "rank", range(1, len(ranked_view) + 1))
    c1, c2 = st.columns(2)
    c1.metric("Inpatients listed", len(ranked_view))
    c2.metric(
        "Expected discharges today (sum of probabilities)",
        f"{ranked_view['discharge_probability'].sum():.1f}",
    )
    st.dataframe(
        ranked_view[
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

# =====================================================================================
# What-If: beds ready earlier
# =====================================================================================
with tab_whatif:
    st.subheader("What if inpatient beds were ready earlier?")
    st.caption(
        "Earlier discharges and faster room cleaning mean boarders leave the ED sooner. "
        "Each boarder moves up to this many hours earlier (never more than they actually boarded)."
    )
    hours_earlier = st.slider("Hours earlier", 0.5, 4.0, 2.0, 0.5, format="%.1f h")
    wi = fe.what_if_beds_earlier(hours_earlier, ed, data["boarding"])
    today, year = wi["today"], wi["per_year"]

    with st.container(border=True):
        st.markdown("**Value line 1 — Boarding cost avoided**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Boarding hours avoided today", f"{today['boarding_hours_saved']:.1f}")
        c2.metric("Cost avoided today", f"${today['boarding_cost_avoided']:,.0f}")
        c3.metric("Cost avoided per year", f"${year['boarding_cost_avoided']:,.0f}")
        st.caption(f"Boarding hours avoided × ${fe.BOARDING_COST_PER_HOUR} per boarding hour.")

    with st.container(border=True):
        st.markdown("**Value line 2 — Fewer patients leave without being seen (estimate)**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ED room-hours freed per year", f"{year['room_hours_freed']:,.0f}")
        c2.metric("Patients roomed sooner per year", f"{year['patients_roomed_sooner']:,.0f}")
        c3.metric("Recovered visits per year (est.)", f"{year['recovered_visits']:,.0f}")
        c4.metric("Recovered net revenue per year (est.)", f"${year['recovered_revenue']:,.0f}")
        st.caption(
            f"Today: {today['room_hours_freed']:.1f} room-hours freed → "
            f"{today['recovered_visits']:.2f} recovered visits (${today['recovered_revenue']:,.0f})."
        )
        st.warning(
            "**This is an estimate, not a forecast.** Each ED room-hour freed from boarding serves "
            f"about {fe.PATIENTS_PER_ROOM_HOUR:.2f} more patients (233 arrivals/day ÷ 32 treatment rooms "
            f"÷ 24 h). Of those patients, the case's {fe.LWBS_RATE:.0%} left-without-being-seen rate is "
            f"assumed to be recovered, at ${fe.NET_REVENUE_PER_VISIT:,} average net revenue per ED visit.",
            icon="📐",
        )

    st.caption(
        "\"Today\" uses the 6 patients boarding now. \"Per year\" uses an average day from the "
        "synthetic 30-day boarding history × 365."
    )


# =====================================================================================
# Equity
# =====================================================================================
with tab_equity:
    st.subheader("Halvian baseline")
    st.caption("From the Halvian case brief (not synthetic): ED performance by payer before ForeWard Flow.")
    base = fe.HALVIAN_BASELINE.copy()
    ref = base.set_index("group").loc["Commercial"]
    base["dtp_gap"] = base["door_to_provider_min"] - ref["door_to_provider_min"]
    base["lwbs_gap"] = (base["lwbs_pct"] - ref["lwbs_pct"]).round(1)
    base.loc[base["group"].isin(["Commercial", "Overall"]), ["dtp_gap", "lwbs_gap"]] = None

    overall = base.set_index("group").loc["Overall"]
    c1, c2 = st.columns(2)
    c1.metric("Door-to-provider, overall", f"{overall['door_to_provider_min']:.0f} min")
    c2.metric("Left without being seen, overall", f"{overall['lwbs_pct']:.1f}%")
    st.dataframe(
        base,
        hide_index=True,
        width="stretch",
        column_config={
            "group": "Payer",
            "door_to_provider_min": st.column_config.NumberColumn("Door-to-provider (min)", format="%d"),
            "dtp_gap": st.column_config.NumberColumn("vs Commercial (min)", format="%+d"),
            "lwbs_pct": st.column_config.NumberColumn("Left without being seen (%)", format="%.1f"),
            "lwbs_gap": st.column_config.NumberColumn("vs Commercial (pts)", format="%+.1f"),
        },
    )
    st.caption(
        "Uninsured patients wait 16 minutes longer than Commercial patients to see a provider "
        "and leave without being seen almost twice as often (5.8% vs 3.0%)."
    )

    st.subheader("Boarding time by payer (synthetic, last 30 days)")
    gap = fe.boarding_gap_by_insurance(data["boarding"])
    st.dataframe(
        gap,
        hide_index=True,
        width="stretch",
        column_config={
            "insurance": "Payer",
            "patients": "Boarded patients",
            "mean_minutes": st.column_config.NumberColumn("Mean (min)", format="%.0f"),
            "median_minutes": st.column_config.NumberColumn("Median (min)", format="%.0f"),
            "vs_overall_minutes": st.column_config.NumberColumn("vs overall mean (min)", format="%+.0f"),
        },
    )
    medicaid_gap = gap.set_index("insurance").at[fe.MEDICAID, "vs_overall_minutes"]
    st.caption(
        f"Overall median boarding time: {data['boarding']['boarding_minutes'].median() / 60:.1f} hours. "
        f"Medicaid/CHIP patients board about {medicaid_gap:.0f} minutes longer than average."
    )
