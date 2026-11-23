"""ForeWard Flow — staff-facing ED patient flow & boarding dashboard.

Uses SYNTHETIC data only (see flow_engine.py).
"""

from datetime import datetime

import os

import altair as alt
import pandas as pd
import streamlit as st

import flow_engine as fe

st.set_page_config(page_title="ForeWard Flow — Halvian Medical Center", layout="wide")

DECISION_OWNER = "House supervisor"
ROLES = [DECISION_OWNER, "Charge nurse", "Bed manager", "Physician", "Case manager", "Social worker"]
PLACEMENT_ROLES = ["Physician", "Bed manager"]  # accept/override AI bed type
ASSIGN_ROLES = ["Case manager", "Social worker"]  # assign barrier flags
OVERRIDE_BED_TYPES = ["Medicine", "Telemetry", "Surgery", "ICU", "Women's & Peds", "Not admitting"]
OVERRIDE_REASONS = [
    "Physician determined different level of care",
    "Isolation needed",
    "Likely discharge",
    "Room constraint",
    "Other",
]
SOCIAL_ORANGE = "#e07b39"
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


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def accept_bed(role, pid, bed, rule):
    fe.append_decision({"time": _now(), "role": role, "decision": "Accepted",
                        "item": f"{pid} bed type: {bed}", "context": "ED Admissions",
                        "reason": f"AI bed type confirmed (rule: {rule})"})


def override_bed(role, pid, ai_bed):
    new_bed = st.session_state.get(f"ov_bed_{pid}")
    reason = st.session_state.get(f"ov_reason_{pid}")
    note = st.session_state.get(f"ov_note_{pid}", "").strip()
    if not new_bed or not reason:
        st.session_state["flash"] = f"{pid}: choose a corrected bed type and a reason to override."
        return
    if reason == "Other" and not note:
        st.session_state["flash"] = f"{pid}: add a note when the reason is Other."
        return
    fe.append_decision({"time": _now(), "role": role, "decision": "Overridden",
                        "item": f"{pid} bed type: {ai_bed} → {new_bed}", "context": "ED Admissions",
                        "reason": reason + (f": {note}" if note else "")})
    for k in (f"ov_bed_{pid}", f"ov_reason_{pid}"):
        st.session_state[k] = None
    st.session_state[f"ov_note_{pid}"] = ""


def assign_barrier(role, pid, barrier, source):
    fe.append_decision({"time": _now(), "role": role, "decision": "Assigned",
                        "item": f"{pid}: {barrier['barrier']}", "context": "Barrier Flags",
                        "reason": f"Routed to {barrier['owner']} ({barrier['type']} barrier; flagged by {source})"})


def run_scan():
    # The key is read here and passed straight to the SDK; it is never stored or logged.
    key = st.session_state.get("api_key_input") or os.environ.get("ANTHROPIC_API_KEY")
    model = st.session_state.get("claude_model") or fe.DEFAULT_CLAUDE_MODEL
    with st.spinner("Scanning notes..."):
        st.session_state["barrier_scan"] = fe.scan_notes(ranked, key, model)


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
    can_place = role in PLACEMENT_ROLES
    can_assign = role in ASSIGN_ROLES
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

    st.header("Claude API (Barrier Flags)")
    st.text_input("Anthropic API key", type="password", key="api_key_input",
                  help="Or set the ANTHROPIC_API_KEY environment variable. Never saved or logged.")
    st.text_input("Model", value=fe.DEFAULT_CLAUDE_MODEL, key="claude_model")
    if st.session_state.get("api_key_input"):
        key_source = "sidebar"
    elif os.environ.get("ANTHROPIC_API_KEY"):
        key_source = "environment"
    else:
        key_source = None
    api_key = bool(key_source)
    model_name = st.session_state.get("claude_model") or fe.DEFAULT_CLAUDE_MODEL
    st.caption(f"Key: {key_source or 'none — notes use the offline fallback'}")

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

tab_control, tab_ed, tab_barriers, tab_whatif, tab_equity = st.tabs(
    ["AI Control Center", "ED Admissions", "Barrier Flags", "What-If", "Equity"]
)

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
# ED Admissions: AI admit prediction + bed type, confirmed by people
# =====================================================================================
with tab_ed:
    st.subheader("ED Admissions")
    boarding = ed[ed["status"] == "Boarding"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Patients in ED", len(ed))
    c2.metric("Boarding (admitted, no bed)", len(boarding))
    c3.metric("Longest boarding", f"{boarding['hours_boarding'].max():.1f} h")
    st.info("**Physician confirms level of care · Bed manager assigns room.** "
            "The AI suggests; people decide.", icon="🩺")
    if not can_place:
        st.caption(f"Acting as {role}: view only. Only {' or '.join(PLACEMENT_ROLES)} can accept or override.")

    ed_log = decision_log[decision_log["context"] == "ED Admissions"]

    def place_status(pid):
        rows = ed_log[ed_log["item"].str.startswith(f"{pid} ")]
        return rows.iloc[-1] if len(rows) else None

    for row in ed.itertuples():
        pid = row.patient_id
        with st.container(border=True):
            info, model_col, bed_col = st.columns([3, 3, 3])
            with info:
                st.markdown(f"**{pid}** · {row.bed} · {row.status}")
                st.markdown(f"Age **{row.age}** · {row.chief_complaint} · Triage acuity **{row.triage_acuity}**")
                st.caption(
                    f"HR {row.heart_rate} · O₂ {row.o2_sat}% · Temp {row.temperature_c}°C · "
                    f"BP {row.systolic_bp}/{row.diastolic_bp}"
                )
                if row.status == "Boarding":
                    hours = f"{row.hours_boarding:.1f} h boarding"
                    st.markdown(f":red[**{hours}**]" if row.hours_boarding > 2 else f"**{hours}**")
                else:
                    st.caption(f"In workup · {row.hours_in_ed:.1f} h in ED")
            with model_col:
                st.progress(float(row.admit_probability), text=f"Admit probability: **{row.admit_probability:.0%}**")
                reasons = [r for r in row.top_reasons.split("; ") if r]
                st.markdown("Top reasons:\n" + "\n".join(f"- {r}" for r in reasons) if reasons else "No strong reasons")
            with bed_col:
                st.markdown(f"Likely bed type: **{row.likely_bed}**")
                st.caption(f"Bed rule: {row.bed_rule}")
                last = place_status(pid)
                if last is not None:
                    mark = "✅" if last["decision"] == "Accepted" else "✏️"
                    st.markdown(f"{mark} **{last['decision']}** by {last['role']} · {last['time'][11:16]}")
                    st.caption(f"{last['item'].split(': ', 1)[-1]} · {last['reason']}")
                b1, b2 = st.columns(2)
                b1.button("Accept", key=f"accept_{pid}", type="primary", disabled=not can_place,
                          on_click=accept_bed, args=(role, pid, row.likely_bed, row.bed_rule), width="stretch")
                with b2.popover("Override", disabled=not can_place, width="stretch"):
                    st.selectbox("Corrected bed type", OVERRIDE_BED_TYPES, key=f"ov_bed_{pid}",
                                 index=None, placeholder="Choose bed type")
                    st.selectbox("Reason", OVERRIDE_REASONS, key=f"ov_reason_{pid}",
                                 index=None, placeholder="Choose reason")
                    st.text_input("Note (required for Other)", key=f"ov_note_{pid}")
                    st.button("Submit override", key=f"ov_submit_{pid}", type="primary",
                              on_click=override_bed, args=(role, pid, row.likely_bed))


# =====================================================================================
# Barrier Flags: likely-home-today list + note scanning
# =====================================================================================
with tab_barriers:
    st.subheader("Barrier Flags")
    st.caption("Inpatients ranked by how likely they are to go home today. "
               "Scan the notes to flag what is holding each discharge up.")
    if not can_assign:
        st.caption(f"Acting as {role}: view only. Only {' or '.join(ASSIGN_ROLES)} can assign barriers.")

    scan_col, info_col = st.columns([1, 3])
    scan_col.button("Scan notes with AI", key="scan_notes", type="primary", on_click=run_scan, width="stretch")
    scan = st.session_state.get("barrier_scan")
    with info_col:
        if scan is None:
            st.caption(
                f"Will use: **{fe.SOURCE_CLAUDE if api_key else fe.SOURCE_OFFLINE}**"
                + (f" ({model_name}, key from {key_source})" if api_key else " (no API key)")
            )
        elif scan["n_offline"] == 0:
            st.success(f"Source: **{fe.SOURCE_CLAUDE}** · {scan['model']} · all {scan['n_claude']} notes", icon="🤖")
        elif scan["n_claude"] == 0:
            why = "no API key" if not scan["errors"] else "; ".join(scan["errors"])
            st.warning(f"Source: **{fe.SOURCE_OFFLINE}** (keyword rules) for all {scan['n_offline']} notes · {why}",
                       icon="🔌")
        else:
            st.warning(
                f"Source: **{fe.SOURCE_CLAUDE}** for {scan['n_claude']} notes, **{fe.SOURCE_OFFLINE}** for "
                f"{scan['n_offline']} ({'; '.join(scan['errors'])})", icon="⚠️",
            )

    if scan is not None:
        summary = fe.barrier_summary(ranked, scan)
        t1, t2, t3 = st.columns(3)
        t1.metric("Patients with social barriers", int(summary["social"].sum()))
        t2.metric("Patients with medical barriers", int(summary["medical"].sum()))
        t3.metric("Patients scanned", len(summary))
        st.markdown(
            f"**{int(summary['social'].sum())} patients with social barriers, "
            f"{int(summary['medical'].sum())} with medical.**"
        )

        chart_df = fe.social_barrier_by_insurance(summary)
        chart_df["label"] = chart_df["insurance"] + " (n=" + chart_df["patients"].astype(str) + ")"
        bars = alt.Chart(chart_df).mark_bar(color=SOCIAL_ORANGE, cornerRadiusEnd=4, size=22).encode(
            y=alt.Y("label:N", title=None, sort=None),
            x=alt.X("pct_social:Q", title="% of scanned patients with a social barrier",
                    scale=alt.Scale(domain=[0, 100])),
            tooltip=[alt.Tooltip("insurance:N", title="Insurance"),
                     alt.Tooltip("patients:Q", title="Patients scanned"),
                     alt.Tooltip("with_social:Q", title="With a social barrier"),
                     alt.Tooltip("pct_social:Q", title="%", format=".0f")],
        )
        text = bars.mark_text(align="left", dx=4, color="gray").encode(text=alt.Text("pct_social:Q", format=".0f"))
        st.altair_chart((bars + text).properties(height=34 * len(chart_df) + 30), width="stretch")
        st.caption("Help is routed by documented need, never by insurance type. Clearing these barriers early "
                   "frees beds sooner and helps the groups that board longest.")

    bar_log = decision_log[decision_log["context"] == "Barrier Flags"]
    for rank, row in enumerate(ranked.itertuples(), start=1):
        pid = row.patient_id
        with st.container(border=True):
            head, prob = st.columns([2, 3])
            head.markdown(f"**#{rank} · {pid}** · {row.unit} {row.room} · age {row.age}")
            prob.progress(float(row.discharge_probability),
                          text=f"Likely home today: **{row.discharge_probability:.0%}**")
            st.caption(f"For: {row.reasons_for or '—'}  ·  Against: {row.reasons_against or '—'}")
            if scan is not None and pid in scan["results"]:
                r = scan["results"][pid]
                if not r["barriers"]:
                    st.caption(f"No barriers flagged · {r['source']}")
                for j, b in enumerate(r["barriers"]):
                    badge = ":orange-badge[Social]" if b["type"] == "Social" else ":blue-badge[Medical]"
                    item = f"{pid}: {b['barrier']}"
                    done = bar_log[bar_log["item"] == item]
                    flag_col, btn_col = st.columns([4, 2])
                    flag_col.markdown(f"{badge} {b['barrier']} · suggested owner: **{b['owner']}**  \n"
                                      f":gray[{r['source']}]")
                    if len(done):
                        e = done.iloc[-1]
                        btn_col.caption(f"✅ Assigned to {b['owner']} by {e['role']} · {e['time'][11:16]}")
                    else:
                        btn_col.button(f"Assign to {b['owner']}", key=f"assign_{pid}_{j}", disabled=not can_assign,
                                       on_click=assign_barrier, args=(role, pid, b, r["source"]), width="stretch")
            with st.expander("Note"):
                st.write(row.nurse_note)


# =====================================================================================
# What-If: beds ready earlier
# =====================================================================================
with tab_whatif:
    st.subheader("What if inpatient beds were ready earlier?")
    st.caption(
        "Earlier discharges and faster room cleaning mean admitted patients board in the ED for less time."
    )

    def set_hours(h):
        st.session_state["hours_earlier"] = h

    st.session_state.setdefault("hours_earlier", 1.0)
    preset_cols = st.columns(len(fe.WHAT_IF_PRESETS))
    for i, (col, (label, h)) in enumerate(zip(preset_cols, fe.WHAT_IF_PRESETS.items())):
        col.button(label, key=f"preset_{i}", on_click=set_hours, args=(h,), width="stretch",
                   type="primary" if st.session_state["hours_earlier"] == h else "secondary")
    hours_earlier = st.slider(
        "Hours less boarding per admitted patient", 0.25, fe.MEDIAN_BOARDING_HOURS, step=0.25,
        key="hours_earlier", format="%.2f h",
        help=f"Capped at the {fe.MEDIAN_BOARDING_HOURS}-hour median boarding time.",
    )
    wi = fe.what_if_beds_earlier(hours_earlier, ed)
    today, year = wi["today"], wi["per_year"]
    h = year["hours_per_admission"]
    adm = fe.ANNUAL_ADMISSIONS
    rate = fe.BOARDING_COST_PER_HOUR
    pprh = fe.PATIENTS_PER_ROOM_HOUR

    def stat(col, label, value, formula):
        col.metric(label, value)
        col.caption(formula)

    with st.container(border=True):
        st.markdown("**Value line 1 — Boarding cost avoided**")
        c1, c2, c3 = st.columns(3)
        stat(c1, "Boarding hours avoided per year", f"{year['boarding_hours_saved']:,.0f}",
             f"{adm:,} admissions × {h:.2f} h")
        stat(c2, "Cost avoided per year", f"${year['boarding_cost_avoided']:,.0f}",
             f"{adm:,} admissions × {h:.2f} h × ${rate}/h")
        stat(c3, "Cost avoided today", f"${today['boarding_cost_avoided']:,.0f}",
             f"{today['boarders']} current boarders, up to {hours_earlier:.2f} h each "
             f"= {today['boarding_hours_saved']:.1f} h × ${rate}/h")
        st.caption(
            f"Per year: 85,000 ED visits × 16% admitted = {adm:,} admissions; hours saved per admission "
            f"is capped at the {fe.MEDIAN_BOARDING_HOURS}-hour median boarding time."
        )

    with st.container(border=True):
        st.markdown("**Value line 2 — Fewer patients leave without being seen (estimate)**")
        c1, c2, c3, c4 = st.columns(4)
        stat(c1, "ED room-hours freed per year", f"{year['room_hours_freed']:,.0f}",
             f"{adm:,} admissions × {h:.2f} h (each boarder holds one room)")
        stat(c2, "Patients roomed sooner per year", f"{year['patients_roomed_sooner']:,.0f}",
             f"{year['room_hours_freed']:,.0f} room-hours × {pprh:.3f} patients/room-hour")
        stat(c3, "Recovered visits per year (est.)", f"{year['recovered_visits']:,.0f}",
             f"{year['patients_roomed_sooner']:,.0f} patients × {fe.LWBS_RATE:.0%} LWBS rate")
        stat(c4, "Conservative estimate (freed ED rooms only)",
             f"≈ ${round(year['recovered_revenue'], -4):,.0f}/yr",
             f"{year['recovered_visits']:,.1f} visits × ${fe.NET_REVENUE_PER_VISIT:,}/visit "
             f"= ${year['recovered_revenue']:,.0f}, rounded to the nearest $10K")
        st.caption(
            f"Today: {today['room_hours_freed']:.1f} room-hours × {pprh:.3f} × {fe.LWBS_RATE:.0%} = "
            f"{today['recovered_visits']:.2f} recovered visits × ${fe.NET_REVENUE_PER_VISIT:,} = "
            f"${today['recovered_revenue']:,.0f}."
        )
        st.warning(
            "**This is an estimate, not a forecast.** Each ED room-hour freed from boarding serves about "
            f"{pprh:.2f} more patients (233 arrivals/day ÷ 32 treatment rooms ÷ 24 h). Of those patients, the "
            f"case's {fe.LWBS_RATE:.0%} left-without-being-seen rate is assumed to be recovered, at "
            f"${fe.NET_REVENUE_PER_VISIT:,} average net revenue per ED visit.",
            icon="📐",
        )
        up = fe.upper_lwbs_estimate()
        st.markdown(
            f"**Upper estimate:** If the Month-18 goal is met (LWBS {fe.LWBS_RATE:.1%} → {fe.LWBS_GOAL_RATE:.1%}): "
            f"85,000 × {fe.LWBS_RATE - fe.LWBS_GOAL_RATE:.1%} = {up['visits']:,.0f} visits × "
            f"${fe.NET_REVENUE_PER_VISIT:,} ≈ ${up['revenue'] / 1e6:.2f}M/yr"
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
