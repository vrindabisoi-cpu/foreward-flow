"""ForeWard Flow — staff-facing ED patient flow & boarding dashboard.

Uses SYNTHETIC data only (see flow_engine.py).
"""

import html
import os
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

import flow_engine as fe

st.set_page_config(page_title="ForeWard Flow — Halvian Medical Center", layout="wide")

DECISION_OWNER = "House supervisor"
ROLES = [DECISION_OWNER, "Charge nurse", "Bed manager", "Physician", "Case manager", "Social worker"]
PLACEMENT_ROLES = ["Physician", "Bed manager"]  # accept/override suggested bed type
ASSIGN_ROLES = ["Case manager", "Social worker"]  # assign barrier flags
ROLE_CAN = {
    "House supervisor": "Approves capacity actions and news alerts. Can clear the decision log.",
    "Charge nurse": "View only.",
    "Bed manager": "Accepts or overrides suggested bed types.",
    "Physician": "Accepts or overrides suggested bed types.",
    "Case manager": "Assigns discharge barriers to the right team.",
    "Social worker": "Assigns discharge barriers to the right team.",
}
OVERRIDE_BED_TYPES = ["Medicine", "Telemetry", "Surgery", "ICU", "Women's & Peds", "Not admitting"]
OVERRIDE_REASONS = [
    "Physician determined different level of care",
    "Isolation needed",
    "Likely discharge",
    "Room constraint",
    "Other",
]

# Palette: one hospital blue, neutral slate for "occupied", reserved status colors.
HOSPITAL_BLUE = "#0b4f8a"
SERIES_BLUE = "#2a6fb0"
SERIES_ORANGE = "#d4652a"
SOCIAL_ORANGE = "#c8692c"
CRITICAL_RED = "#b42318"
FONT = '"Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, "Helvetica Neue", Arial, sans-serif'

APP_CSS = """
<style>
:root {
  --ff-navy: #0a3d6b; --ff-blue: #0b4f8a; --ff-ink: #1b2733; --ff-muted: #5b6876; --ff-line: #d9e0e8;
  --ff-bg: #f3f5f8; --ff-card: #ffffff;
  --ff-ok: #1f7a4d; --ff-ok-bg: #e7f3ec; --ff-warn: #8a5a00; --ff-warn-bg: #fdf3dc; --ff-warn-bar: #d99a1e;
  --ff-crit: #b42318; --ff-crit-bg: #fdecea; --ff-info-bg: #e9f1f9; --ff-occ: #9aa6b2; --ff-open: #2e9461;
}
html, body, .stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp button, .stApp textarea {
  font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, "Helvetica Neue", Arial, sans-serif;
}
.stApp {background: var(--ff-bg);}
[data-testid="stHeader"] {background: var(--ff-bg);}
[data-testid="stDecoration"] {display: none;}
.block-container {padding-top: 2.6rem; padding-bottom: 2rem; max-width: 1440px;}
[data-testid="stSidebar"] {background: #ffffff; border-right: 1px solid var(--ff-line);}
[data-testid="stSidebar"] .block-container, [data-testid="stSidebarUserContent"] {padding-top: 1rem;}

/* Top bar */
.ff-topbar {display: flex; justify-content: space-between; align-items: center; gap: 1.5rem; flex-wrap: wrap;
  background: var(--ff-navy); color: #fff; border-radius: 6px; padding: 0.85rem 1.2rem;}
.ff-brand {display: flex; align-items: center; gap: 0.8rem;}
.ff-logo {width: 38px; height: 38px; border-radius: 4px; background: #fff; color: var(--ff-navy);
  display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 0.8rem; letter-spacing: 0.02em;}
.ff-title {font-size: 1.15rem; font-weight: 700; line-height: 1.2;}
.ff-sub {font-size: 0.82rem; opacity: 0.85;}
.ff-meta {display: flex; gap: 1.6rem; flex-wrap: wrap;}
.ff-meta div {display: flex; flex-direction: column; line-height: 1.25;}
.ff-meta span {font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.75;}
.ff-meta b {font-size: 0.92rem; font-weight: 600;}
.ff-synth {font-size: 0.78rem; color: var(--ff-muted); margin: 0.35rem 0 0.9rem 0.1rem;}

/* KPI strip */
.ff-kpis {display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 0.6rem; margin-bottom: 0.6rem;}
.ff-kpi {background: var(--ff-card); border: 1px solid var(--ff-line); border-radius: 6px; padding: 0.6rem 0.85rem;}
.ff-kpi .l {font-size: 0.74rem; color: var(--ff-muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em;}
.ff-kpi .v {font-size: 1.45rem; font-weight: 700; color: var(--ff-ink); font-variant-numeric: tabular-nums; line-height: 1.3;}
.ff-kpi .s {font-size: 0.78rem; color: var(--ff-muted);}
.ff-kpi.crit {border-left: 4px solid var(--ff-crit);} .ff-kpi.warn {border-left: 4px solid var(--ff-warn-bar);}

/* Section headings */
.ff-sec {display: flex; align-items: baseline; gap: 0.75rem; margin: 1.4rem 0 0.55rem 0;
  border-bottom: 1px solid var(--ff-line); padding-bottom: 0.3rem;}
.ff-sec .t {font-size: 1.08rem; font-weight: 700; color: var(--ff-ink);}
.ff-sec .n {font-size: 0.82rem; color: var(--ff-muted);}

/* Cards */
.ff-grid {display: grid; gap: 0.6rem;}
.ff-card {background: var(--ff-card); border: 1px solid var(--ff-line); border-radius: 6px; padding: 0.7rem 0.9rem;}
.ff-card .hd {display: flex; flex-direction: column; margin-bottom: 0.25rem;}
.ff-card .name {font-weight: 700; color: var(--ff-ink); font-size: 0.95rem;}
.ff-card .meta {font-size: 0.78rem; color: var(--ff-muted);}
.ff-big {font-size: 1.9rem; font-weight: 700; color: var(--ff-ink); line-height: 1.15; font-variant-numeric: tabular-nums;}
.ff-big small {font-size: 0.85rem; font-weight: 500; color: var(--ff-muted); margin-left: 0.25rem;}
.ff-bar {display: flex; height: 8px; border-radius: 2px; overflow: hidden; background: #eef1f4; margin: 0.45rem 0 0.4rem 0; gap: 2px;}
.ff-bar i {display: block; height: 100%;}
.ff-bar .o {background: var(--ff-open);} .ff-bar .c {background: var(--ff-warn-bar);} .ff-bar .x {background: var(--ff-occ);}
.ff-bar .b {background: var(--ff-crit);}
.ff-legend {font-size: 0.8rem; color: var(--ff-muted); display: flex; flex-wrap: wrap; gap: 0.2rem 0.7rem;}
.ff-legend b {color: var(--ff-ink); font-variant-numeric: tabular-nums;}
.ff-dot {display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 4px; vertical-align: middle;}
.ff-dot.o {background: var(--ff-open);} .ff-dot.c {background: var(--ff-warn-bar);} .ff-dot.x {background: var(--ff-occ);}
.ff-dot.b {background: var(--ff-crit);}
.ff-outlook {margin-top: 0.45rem; padding-top: 0.4rem; border-top: 1px solid #eef1f4; font-size: 0.8rem; font-weight: 600;}
.ff-outlook.crit {color: var(--ff-crit);} .ff-outlook.ok {color: var(--ff-ok);} .ff-outlook.warn {color: var(--ff-warn);}

/* Pills */
.ff-pill {display: inline-block; font-size: 0.74rem; font-weight: 600; padding: 0.08rem 0.5rem; border-radius: 3px;
  border: 1px solid transparent; white-space: nowrap; vertical-align: middle;}
.ff-pill.ok {background: var(--ff-ok-bg); color: var(--ff-ok); border-color: #b9dcc7;}
.ff-pill.warn {background: var(--ff-warn-bg); color: var(--ff-warn); border-color: #efd28e;}
.ff-pill.crit {background: var(--ff-crit-bg); color: var(--ff-crit); border-color: #f2b8b1;}
.ff-pill.info {background: var(--ff-info-bg); color: var(--ff-blue); border-color: #bcd3ea;}
.ff-pill.neutral {background: #eef1f4; color: #44515e; border-color: #d9e0e8;}
.ff-pill.social {background: #fbeee4; color: #9a4a18; border-color: #efc9ad;}

/* Banners */
.ff-banner {border: 1px solid var(--ff-line); border-left: 4px solid var(--ff-blue); background: var(--ff-card);
  border-radius: 4px; padding: 0.65rem 0.9rem; margin: 0.3rem 0 0.6rem 0; color: var(--ff-ink); font-size: 0.92rem;}
.ff-banner .bt {font-weight: 700;}
.ff-banner ul {margin: 0.35rem 0 0 1.1rem; padding: 0;} .ff-banner li {margin: 0.1rem 0; font-size: 0.88rem;}
.ff-banner.crit {border-left-color: var(--ff-crit); background: #fff8f7;}
.ff-banner.warn {border-left-color: var(--ff-warn-bar); background: #fffcf3;}
.ff-banner.ok {border-left-color: var(--ff-ok); background: #f6fbf8;}
.ff-banner.info {border-left-color: var(--ff-blue); background: #f5f9fd;}
.ff-banner.neutral {border-left-color: var(--ff-occ); background: #fafbfc; color: #44515e;}
.ff-mode {background: #3d4a57; color: #fff; border-radius: 4px; padding: 0.5rem 0.9rem; font-size: 0.9rem; margin-bottom: 0.7rem;}

/* Forecast status panel */
.ff-status .lbl {font-size: 0.74rem; color: var(--ff-muted); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;}
.ff-status .when {font-size: 2.1rem; font-weight: 700; line-height: 1.15;}
.ff-status .when.crit {color: var(--ff-crit);} .ff-status .when.ok {color: var(--ff-ok);}
.ff-status .row {display: flex; justify-content: space-between; font-size: 0.86rem; padding: 0.32rem 0;
  border-top: 1px solid #eef1f4;}
.ff-status .row span {color: var(--ff-muted);} .ff-status .row b {font-variant-numeric: tabular-nums;}

/* Patient rows */
.ff-pt-top {display: flex; align-items: center; gap: 0.45rem; flex-wrap: wrap;}
.ff-bed {font-weight: 700; font-size: 1rem; color: var(--ff-ink);}
.ff-pid {font-size: 0.82rem; color: var(--ff-muted);}
.ff-pt-main {font-size: 0.92rem; margin-top: 0.2rem;}
.ff-small {font-size: 0.8rem; color: var(--ff-muted); margin-top: 0.15rem;}
.ff-lbl {font-size: 0.72rem; color: var(--ff-muted); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;}
.ff-meter {display: flex; align-items: center; gap: 0.55rem;}
.ff-meter .num {font-size: 1.25rem; font-weight: 700; font-variant-numeric: tabular-nums; min-width: 3rem;}
.ff-meter .track {flex: 1; height: 6px; background: #eef1f4; border-radius: 2px; overflow: hidden;}
.ff-meter .fill {height: 100%; background: var(--ff-blue);}
.ff-factors {color: #3b4855; margin: 0.3rem 0 0 0; padding-left: 1rem;}
.ff-factors li {font-size: 0.82rem !important; margin: 0 !important; padding: 0 !important; line-height: 1.45;}
.ff-flag {font-size: 0.88rem; padding: 0.15rem 0;}
.ff-flag .src {font-size: 0.75rem; color: var(--ff-muted);}

/* Streamlit widgets */
div[class*="st-key-ffcard_"] {background: var(--ff-card); border-radius: 6px; border-color: var(--ff-line) !important;}
.stTabs [data-baseweb="tab-list"] {gap: 0.2rem; border-bottom: 1px solid var(--ff-line);}
.stTabs [data-baseweb="tab"] {padding: 0.5rem 0.95rem;}
.stTabs [data-baseweb="tab"] p {font-size: 0.97rem !important; font-weight: 600 !important;}
.stButton button, .stDownloadButton button, [data-testid="stPopover"] button {border-radius: 4px; font-weight: 600;}
[data-testid="stMetric"] {background: var(--ff-card); border: 1px solid var(--ff-line); border-radius: 6px; padding: 0.55rem 0.8rem;}
[data-testid="stMetricLabel"] p {font-weight: 600; color: var(--ff-muted); font-size: 0.82rem; white-space: normal;}
[data-testid="stMetricLabel"] > div {overflow: visible;}
[data-testid="stMetricValue"] {font-size: 1.45rem; font-variant-numeric: tabular-nums;}
[data-testid="stExpander"] details {background: var(--ff-card); border-radius: 6px;}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] .ff-side {font-size: 0.74rem; text-transform: uppercase;
  letter-spacing: 0.07em; color: var(--ff-muted); font-weight: 700; margin: 1rem 0 0.3rem 0;}

.ff-footer {margin-top: 2.2rem; padding: 0.8rem 1rem; border-top: 1px solid var(--ff-line);
  color: var(--ff-muted); font-size: 0.8rem;}

/* Manual-mode whiteboard */
.wb-unit {border: 1px solid #c9d6e3; border-radius: 6px; padding: 0.8rem 1rem; margin-bottom: 0.9rem; background: #fff;
          break-inside: avoid; page-break-inside: avoid;}
.wb-unit .name {font-weight: 700; font-size: 1.02rem; margin-bottom: 0.2rem;}
.wb-counts {font-size: 0.92rem; margin-bottom: 0.5rem;}
.wb-grid {display: grid; grid-template-columns: repeat(20, 16px); gap: 3px;}
.wb-sq {width: 16px; height: 16px; border-radius: 2px; border: 1px solid rgba(0,0,0,0.15);
        -webkit-print-color-adjust: exact; print-color-adjust: exact;}
.wb-open {background: #2e9461;} .wb-clean {background: #e0a526;} .wb-occ {background: #9aa6b2;}

@media (max-width: 1100px) {.ff-kpis {grid-template-columns: repeat(3, minmax(0, 1fr));}}
@media print {
  [data-testid="stSidebar"], [data-testid="stHeader"], [data-testid="stToolbar"] {display: none !important;}
  .block-container {padding-top: 0 !important; max-width: 100% !important;}
  .stApp {background: #fff;}
}
</style>
"""


# --- Small HTML helpers (all dynamic text is escaped) ---------------------------------


def esc(value) -> str:
    return html.escape(str(value))


def section(title: str, note: str = ""):
    note_html = f"<span class='n'>{esc(note)}</span>" if note else ""
    st.markdown(f"<div class='ff-sec'><span class='t'>{esc(title)}</span>{note_html}</div>", unsafe_allow_html=True)


def pill(text: str, kind: str = "neutral") -> str:
    return f"<span class='ff-pill {kind}'>{esc(text)}</span>"


def banner(kind: str, title: str = "", body: str = "", items=None, raw: bool = False):
    """Status message with a colored left edge. `raw=True` means title/body are trusted HTML."""
    t = title if raw else esc(title)
    b = body if raw else esc(body)
    parts = [f"<div class='ff-banner {kind}'>"]
    if t:
        parts.append(f"<span class='bt'>{t}</span> ")
    if b:
        parts.append(b)
    if items:
        parts.append("<ul>" + "".join(f"<li>{esc(i)}</li>" for i in items) + "</ul>")
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def stacked_bar(parts) -> str:
    """parts: list of (css_class, count). Widths are shares of the total."""
    total = sum(n for _, n in parts) or 1
    return "<div class='ff-bar'>" + "".join(
        f"<i class='{c}' style='width:{100 * n / total:.2f}%'></i>" for c, n in parts if n
    ) + "</div>"


_card_count = [0]


def card():
    """White bordered panel (the key gives CSS a class to style it)."""
    _card_count[0] += 1
    return st.container(border=True, key=f"ffcard_{_card_count[0]}")


def silent_placeholder(what: str = "Predictions"):
    banner("neutral", "Hidden in silent mode.",
           f"{what} are logged for validation only during the Phase 1 pilot and are not shown to staff.")


HOUR_LABEL_EXPR = ("datum.value < 12 ? datum.value + ' AM' : "
                   "(datum.value == 12 ? '12 PM' : (datum.value - 12) + ' PM')")


def styled(chart):
    """Quiet, consistent chart styling: light grid, muted axes, no frame."""
    return (
        chart.configure(font=FONT, background="#ffffff")
        .configure_view(strokeWidth=0)
        .configure_axis(gridColor="#e8edf2", domainColor="#c3ccd6", tickColor="#c3ccd6",
                        labelColor="#44515e", titleColor="#44515e", labelFontSize=12, titleFontSize=12,
                        titleFontWeight=600)
        .configure_legend(labelColor="#44515e", labelFontSize=12, symbolStrokeWidth=2)
    )


def render_whiteboard(beds, ed):
    """Manual mode: printable whiteboard for downtime (no predictions)."""
    banner("crit", "Manual mode (whiteboard backup).",
           "Decision-support tools are off. Use this board, the last auto-printed board, and the paper log. "
           "Print with Ctrl/Cmd + P.")
    section("Beds by unit", f"Board as of {fe.REFERENCE_DATE:%A %b %d, %Y}, 9:00 AM")
    st.markdown("<div class='ff-legend' style='margin-bottom:0.6rem'>"
                "<span><span class='wb-sq wb-open' style='display:inline-block;vertical-align:middle'></span> Open</span>"
                "<span><span class='wb-sq wb-clean' style='display:inline-block;vertical-align:middle'></span> Needs cleaning</span>"
                "<span><span class='wb-sq wb-occ' style='display:inline-block;vertical-align:middle'></span> Occupied</span>"
                "</div>", unsafe_allow_html=True)
    cols = st.columns(2, gap="large")
    for i, row in enumerate(beds.itertuples()):
        squares = (["wb-open"] * row.open + ["wb-clean"] * row.needs_cleaning + ["wb-occ"] * row.occupied)
        grid = "".join(f"<div class='wb-sq {c}'></div>" for c in squares)
        cols[i % 2].markdown(
            f"<div class='wb-unit'><div class='name'>{esc(row.unit)} · {row.total_beds} beds</div>"
            f"<div class='wb-counts'><b>{row.open}</b> open · <b>{row.needs_cleaning}</b> needs cleaning · "
            f"<b>{row.occupied}</b> occupied</div><div class='wb-grid'>{grid}</div></div>",
            unsafe_allow_html=True,
        )
    section("ED boarding list")
    boarders = ed[ed["status"] == "Boarding"].sort_values("hours_boarding", ascending=False)
    st.dataframe(
        boarders[["patient_id", "bed", "age", "chief_complaint", "admit_unit", "hours_boarding"]],
        hide_index=True, width="stretch",
        column_config={"patient_id": "Patient", "bed": "ED bed", "age": "Age", "chief_complaint": "Complaint",
                       "admit_unit": "Admitted to",
                       "hours_boarding": st.column_config.NumberColumn("Hours boarding", format="%.1f")},
    )
    section("Manual protocol")
    st.markdown("\n".join(f"{i}. {step}" for i, step in enumerate(fe.MANUAL_PROTOCOL, start=1)))


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
                        "reason": f"Suggested bed type confirmed (rule: {rule})"})


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
# Forecast uses the ED list with physician overrides applied (display tables keep the model's view).
ed_forecast, physician_overrides = fe.apply_physician_overrides(ed, decision_log)
applied_alerts = [a for a in alerts if alert_status(decision_log, a) == "Applied to forecast"]


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
    st.markdown("<div class='ff-side'>Signed in as</div>", unsafe_allow_html=True)
    role = st.selectbox("Role", ROLES, index=0, label_visibility="collapsed")
    is_owner = role == DECISION_OWNER
    can_place = role in PLACEMENT_ROLES
    can_assign = role in ASSIGN_ROLES
    st.caption(ROLE_CAN[role])

    st.markdown("<div class='ff-side'>Forecast settings</div>", unsafe_allow_html=True)
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

    st.markdown("<div class='ff-side'>Operating mode</div>", unsafe_allow_html=True)
    silent = st.toggle("Silent mode (Phase 1 pilot)", key="silent_mode",
                       help="Hides all predictions, forecasts, tiers, and suggestions from staff.")
    manual = st.toggle("Manual mode (whiteboard backup)", key="manual_mode",
                       help="Replaces the app with a printable whiteboard for downtime.")

    staffing = fe.CURRENT_SHIFT_STAFFING
    st.markdown("<div class='ff-side'>ED staffing on shift</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='ff-legend' style='flex-direction:column;gap:0.15rem'>"
        + "".join(f"<span><b>{n}</b> {esc(r)}</span>" for r, n in staffing["staff"].items())
        + "</div>", unsafe_allow_html=True)
    st.caption(f"{staffing['shift']} · for context; not used in any calculation")

    st.markdown("<div class='ff-side'>System</div>", unsafe_allow_html=True)
    with st.expander("Note scanning (Claude API)"):
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

    with st.expander("Model performance & data"):
        st.metric("Admission at triage — AUC", f"{admission_model.auc:.2f}")
        st.metric("Likely home today — AUC", f"{discharge_model.auc:.2f}")
        st.caption("Tested on held-out synthetic data.")
        st.caption(
            "Both models are logistic regressions, chosen because they're explainable. "
            "Insurance, race/ethnicity, age group, and language are never model inputs; "
            "they're kept only for fairness checks."
        )
        st.caption(fe.DATA_SOURCES_NOTE)

st.markdown(APP_CSS, unsafe_allow_html=True)

# --- Header ------------------------------------------------------------------------
st.markdown(
    "<div class='ff-topbar'>"
    "<div class='ff-brand'><div class='ff-logo'>HMC</div><div>"
    "<div class='ff-title'>Halvian Medical Center</div>"
    "<div class='ff-sub'>ForeWard Flow · Patient flow &amp; bed management</div></div></div>"
    "<div class='ff-meta'>"
    f"<div><span>Date</span><b>{fe.REFERENCE_DATE:%a, %b %d, %Y}</b></div>"
    "<div><span>Board as of</span><b>9:00 AM</b></div>"
    f"<div><span>Shift</span><b>{esc(staffing['shift'])}</b></div>"
    f"<div><span>Signed in as</span><b>{esc(role)}</b></div>"
    "</div></div>"
    "<div class='ff-synth'>Fictional 320-bed hospital · synthetic data only, not real patients</div>",
    unsafe_allow_html=True,
)


def footer():
    st.markdown(f"<div class='ff-footer'>{esc(fe.FOOTER_TEXT)}</div>", unsafe_allow_html=True)


if manual:
    render_whiteboard(beds, ed)
    footer()
    st.stop()

if silent:
    st.markdown("<div class='ff-mode'><b>Silent mode · Phase 1 pilot.</b> Predictions are logged for "
                "validation only and hidden from staff.</div>", unsafe_allow_html=True)
    if st.session_state.get("silent_logged") != str(fe.REFERENCE_DATE.date()):
        med = fe.forecast_unit("Medicine", beds, ed_forecast, ranked, day_of_week, trend_level, applied_alerts)
        fe.log_silent_predictions(ed, med)
        st.session_state["silent_logged"] = str(fe.REFERENCE_DATE.date())

if "flash" in st.session_state:
    banner("crit", st.session_state.pop("flash"))

# --- Census strip (every tab) -------------------------------------------------------------
total_beds = int(beds["total_beds"].sum())
occupied = int(beds["occupied"].sum())
open_beds = int(beds["open"].sum())
to_clean = int(beds["needs_cleaning"].sum())
bm = fe.boarding_metrics(ed)
boarding_now = ed[ed["status"] == "Boarding"]
longest = float(boarding_now["hours_boarding"].max()) if len(boarding_now) else 0.0
kpis = [
    ("Hospital census", f"{occupied}/{total_beds}", f"{occupied / total_beds:.0%} occupied", ""),
    ("Open beds", f"{open_beds}", "ready for a patient now", ""),
    ("Rooms to clean", f"{to_clean}", "dirty or in cleaning", ""),
    ("ED boarders", f"{bm['patients']}", f"longest {longest:.1f} h", "crit" if longest > 4 else "warn"),
    ("Boarding cost today", f"${bm['cost']:,.0f}", f"{bm['total_hours']:.1f} h × ${fe.BOARDING_COST_PER_HOUR}/h", ""),
    ("ED waiting room", f"{fe.WAITING_ROOM_PATIENTS}", f"of {fe.WAITING_ROOM_SEATS} seats", ""),
]
st.markdown(
    "<div class='ff-kpis'>" + "".join(
        f"<div class='ff-kpi {k}'><div class='l'>{esc(l)}</div><div class='v'>{esc(v)}</div>"
        f"<div class='s'>{esc(s)}</div></div>" for l, v, s, k in kpis
    ) + "</div>", unsafe_allow_html=True,
)

units = list(fe.UNIT_CAPACITY)
unit_forecasts = {} if silent else {
    u: fe.forecast_unit(u, beds, ed_forecast, ranked, day_of_week, trend_level, applied_alerts) for u in units
}

tab_control, tab_ed, tab_barriers, tab_whatif, tab_equity = st.tabs(
    ["Command Center", "ED Admissions", "Barrier Flags", "What-If", "Equity & Oversight"]
)

# =====================================================================================
# Command Center
# =====================================================================================
with tab_control:
    # --- Inpatient bed board ------------------------------------------------------
    section("Inpatient beds by unit", "Live counts" + ("" if silent else " · outlook to 5 PM"))
    cards = []
    for row in beds.itertuples():
        outlook = ""
        f = unit_forecasts.get(row.unit)
        if f is not None:
            if f.runout_hour:
                outlook = f"<div class='ff-outlook crit'>Projected to run out by {fe.hour_label(f.runout_hour)}</div>"
            elif round(f.min_open) < 3:
                outlook = f"<div class='ff-outlook warn'>Tight: about {max(f.min_open, 0):.0f} open at lowest</div>"
            else:
                outlook = "<div class='ff-outlook ok'>Enough beds through 5 PM</div>"
        cards.append(
            f"<div class='ff-card'><div class='hd'><span class='name'>{esc(row.unit)}</span>"
            f"<span class='meta'>{row.total_beds} beds · {row.occupancy_pct:.0f}% full</span></div>"
            f"<div class='ff-big'>{row.open}<small>open</small></div>"
            + stacked_bar([("o", row.open), ("c", row.needs_cleaning), ("x", row.occupied)])
            + f"<div class='ff-legend'><span><span class='ff-dot o'></span><b>{row.open}</b> open</span>"
            f"<span><span class='ff-dot c'></span><b>{row.needs_cleaning}</b> cleaning</span>"
            f"<span><span class='ff-dot x'></span><b>{row.occupied}</b> occupied</span></div>{outlook}</div>"
        )
    st.markdown(f"<div class='ff-grid' style='grid-template-columns:repeat({len(cards)},minmax(0,1fr))'>"
                + "".join(cards) + "</div>", unsafe_allow_html=True)

    # --- ED rooms ------------------------------------------------------------------
    rooms = fe.ed_room_status(ed)
    treat = rooms.set_index("area").loc["Treatment rooms"]
    section("Emergency department", f"{treat['held_by_boarders']} of {treat['total']} treatment rooms "
                                    "held by admitted patients waiting for a bed")
    ed_cards = []
    for row in rooms.itertuples():
        legend = []
        if row.held_by_boarders:
            legend.append(f"<span><span class='ff-dot b'></span><b>{row.held_by_boarders}</b> boarders</span>")
        legend += [f"<span><span class='ff-dot x'></span><b>{row.other_patients}</b> other patients</span>",
                   f"<span><span class='ff-dot o'></span><b>{row.open}</b> open</span>"]
        ed_cards.append(
            f"<div class='ff-card'><div class='hd'><span class='name'>{esc(row.area)}</span>"
            f"<span class='meta'>{row.total} total</span></div>"
            f"<div class='ff-big'>{row.open}<small>open</small></div>"
            + stacked_bar([("o", row.open), ("b", row.held_by_boarders), ("x", row.other_patients)])
            + f"<div class='ff-legend'>{''.join(legend)}</div></div>"
        )
    ed_cards.append(
        f"<div class='ff-card'><div class='hd'><span class='name'>Waiting room</span>"
        f"<span class='meta'>{fe.WAITING_ROOM_SEATS} seats</span></div>"
        f"<div class='ff-big'>{fe.WAITING_ROOM_PATIENTS}<small>waiting</small></div>"
        + stacked_bar([("x", fe.WAITING_ROOM_PATIENTS), ("o", fe.WAITING_ROOM_SEATS - fe.WAITING_ROOM_PATIENTS)])
        + "<div class='ff-legend'><span>Not yet roomed</span></div></div>"
    )
    st.markdown(f"<div class='ff-grid' style='grid-template-columns:repeat({len(ed_cards)},minmax(0,1fr))'>"
                + "".join(ed_cards) + "</div>", unsafe_allow_html=True)

    if silent:
        section("Bed forecast")
        silent_placeholder("Forecasts, capacity tiers, and suggestions")
    else:
        # --- 8-hour forecast ------------------------------------------------------------
        section("8-hour bed forecast", "9 AM–5 PM")
        sel_col, _ = st.columns([1, 3])
        unit = sel_col.selectbox("Unit", units, index=units.index("Medicine"))
        forecast = unit_forecasts[unit]
        tier = fe.recommend_tier(forecast)
        tier_kind = {"Normal": "ok", "Level 1 — Tight": "warn", "Level 2 — Full": "crit",
                     "Level 3 — Crisis": "crit"}[tier]

        chart_col, status_col = st.columns([2.2, 1], gap="medium")
        with status_col:
            if forecast.runout_hour is not None:
                short = max(round(-forecast.min_open), 0)
                when = f"<div class='when crit'>{fe.hour_label(forecast.runout_hour)}</div>"
                lead = "Runs out of beds by"
                hs = forecast.head_start_hours
                rows_html = [
                    ("Short at worst", f"about {short} bed{'s' if short != 1 else ''} by "
                                       f"{fe.hour_label(forecast.worst_hour)}" if short else "under 1 bed"),
                    ("Head start", f"{hs} hour{'s' if hs != 1 else ''}" if hs else "none"),
                ]
            else:
                when = "<div class='when ok'>Through 5 PM</div>"
                lead = "Open beds last"
                rows_html = [("Lowest point", f"about {max(forecast.min_open, 0):.0f} open at "
                                              f"{fe.hour_label(forecast.worst_hour)}")]
            rows_html.append(("Admissions adjustment", f"+{forecast.admission_adjustment:.0%}"))
            st.markdown(
                f"<div class='ff-card ff-status'><div class='lbl'>{esc(unit)} · {lead}</div>{when}"
                + "".join(f"<div class='row'><span>{esc(a)}</span><b>{esc(b)}</b></div>" for a, b in rows_html)
                + f"<div class='row'><span>Recommended tier</span>{pill(tier, tier_kind)}</div></div>",
                unsafe_allow_html=True,
            )
        with chart_col:
            chart_df = forecast.table.assign(projected_open=lambda d: d["projected_open"].round(1))
            x = alt.X("hour:O", title=None, sort=None, axis=alt.Axis(labelExpr=HOUR_LABEL_EXPR, labelAngle=0))
            hover = alt.selection_point(fields=["hour"], nearest=True, on="pointerover", empty=False)
            y_min = min(float(chart_df["projected_open"].min()), 0) - 1
            y_scale = alt.Scale(domain=[y_min, max(float(chart_df["projected_open"].max()), 1) + 1], nice=False)
            shortage = alt.Chart(pd.DataFrame({"y": [y_min], "y2": [0]})).mark_rect(
                color="#fdecea", opacity=0.8).encode(y=alt.Y("y:Q", scale=y_scale), y2="y2:Q")
            line = alt.Chart(chart_df).mark_line(color=SERIES_BLUE, strokeWidth=2).encode(
                x=x, y=alt.Y("projected_open:Q", title="Projected open beds", scale=y_scale)
            )
            points = alt.Chart(chart_df).mark_point(filled=True, size=70, color=SERIES_BLUE).encode(
                x=x,
                y="projected_open:Q",
                opacity=alt.condition(hover, alt.value(1), alt.value(0.4)),
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
                align="left", dx=4, dy=10, color=CRITICAL_RED, fontSize=11
            ).encode(y="y:Q", text="label:N", x=alt.value(0))
            st.altair_chart(styled((shortage + zero + zero_label + line + points).properties(height=250)),
                            width="stretch")

        warning = forecast.warning()
        if warning:
            banner("crit", warning + ".", items=forecast.reasons())
        else:
            banner("ok", f"{unit} is projected to keep open beds through 5 PM.", items=forecast.reasons())
        unit_overrides = [o for o in physician_overrides if unit in (o["from"], o["to"])]
        if unit_overrides:
            banner("info", "Forecast updated by physician override.",
                   "; ".join(f"{o['patient_id']}: {o['from']} → {o['to']}" for o in unit_overrides))

        with st.expander("How this forecast is calculated · hourly table"):
            st.caption(
                "Projected open beds = open now + rooms cleaned + expected discharges "
                "(likely-home-today model) − expected admissions (ED patients from the admission "
                "model + new arrivals by hour)."
            )
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
                f = unit_forecasts[u]
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
        section("Capacity actions", f"Decision owner: {DECISION_OWNER}")
        tier_info = fe.CAPACITY_TIERS[tier]
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:0.6rem;margin-bottom:0.2rem'>"
            f"<span style='font-weight:700;font-size:1rem'>{esc(unit)}</span>{pill(tier, tier_kind)}"
            f"<span class='ff-small' style='margin:0'>{esc(fe.tier_rationale(forecast))}</span></div>",
            unsafe_allow_html=True,
        )
        if not is_owner:
            st.caption(f"View only. Actions are approved by the {DECISION_OWNER.lower()}.")

        context = f"{unit} · {tier} · {day_of_week} · {trend_level}"
        latest = {}
        for entry in decision_log[decision_log["context"] == context].to_dict("records"):
            latest[entry["item"]] = entry

        with card():
            st.text_input(
                "Reason for your decision (required to decline)",
                key="action_reason",
                disabled=not is_owner,
                placeholder="e.g. Discharge lounge already open since 8 AM",
            )
            for i, action in enumerate(tier_info["actions"]):
                a_col, s_col, ok_col, no_col = st.columns([5, 2.2, 1, 1], vertical_alignment="center")
                a_col.markdown(f"**{action}**")
                if action in latest:
                    e = latest[action]
                    kind = "ok" if e["decision"] == "Approved" else "neutral"
                    s_col.markdown(f"{pill(e['decision'], kind)} <span class='ff-small'>{esc(e['time'][11:16])}</span>",
                                   unsafe_allow_html=True)
                else:
                    s_col.markdown(pill("Awaiting decision", "warn"), unsafe_allow_html=True)
                ok_col.button(
                    "Approve", key=f"approve_{i}", disabled=not is_owner, type="primary", width="stretch",
                    on_click=decide_action, args=(role, "Approved", action, context),
                )
                no_col.button(
                    "Decline", key=f"decline_{i}", disabled=not is_owner, width="stretch",
                    on_click=decide_action, args=(role, "Declined", action, context),
                )

        with st.expander("All capacity tiers"):
            for name, info in fe.CAPACITY_TIERS.items():
                marker = " · recommended now" if name == tier else ""
                st.markdown(f"**{name}**{marker} — {info['when']}")
                st.markdown("\n".join(f"- {a}" for a in info["actions"]))

        # --- Public health & news alerts ------------------------------------------------------
        section("Public health & news alerts")
        ph_col, news_col = st.columns(2, gap="medium")
        with ph_col, card():
            st.markdown("**Public health surveillance**")
            st.caption(f"{feed['source']}, as of {feed['as_of']}.")
            loc, nat = feed["local"], feed["national"]
            c1, c2 = st.columns(2)
            c1.metric("Local flu/RSV level", loc["level"].replace(" flu/RSV", ""), loc["trend"], delta_color="inverse")
            c1.caption(f"Flu {loc['flu_ed_visits_pct']}% · RSV {loc['rsv_ed_visits_pct']}% of ED visits")
            c2.metric("National level", nat["level"], nat["trend"], delta_color="inverse")
            c2.caption(f"Flu {nat['flu_ed_visits_pct']}% · RSV {nat['rsv_ed_visits_pct']}% of ED visits")
            if trend_level == feed_level:
                banner("info", "Applied automatically.",
                       f"Local level {feed_level} adds +{fe.TREND_LEVELS[feed_level]:.0%} to expected admissions.")
            else:
                banner("warn", "Manual override.",
                       f"Forecast uses {trend_level} instead of the surveillance level ({feed_level}).")

        with news_col, card():
            st.markdown("**News alerts for house supervisor review**")
            st.caption("Simulated news. Alerts never change the forecast unless a person applies them.")
            for alert in alerts:
                status = alert_status(decision_log, alert)
                effect = ", ".join(f"{u} +{p:.0%}" for u, p in alert["effect"].items())
                if len(alert["effect"]) == len(fe.UNIT_CAPACITY):
                    effect = f"all units +{next(iter(alert['effect'].values())):.0%}"
                kind = {"Pending review": "warn", "Applied to forecast": "info"}.get(status, "neutral")
                st.markdown(
                    f"<div class='ff-banner {kind}'><div style='display:flex;justify-content:space-between;gap:0.5rem'>"
                    f"<span class='bt'>{esc(alert['headline'])}</span>{pill(status, kind)}</div>"
                    f"<div style='font-size:0.86rem;margin-top:0.2rem'>{esc(alert['detail'])}</div>"
                    f"<div class='ff-small'>{esc(alert['source'])} · If applied: {esc(effect)} expected admissions</div>"
                    "</div>", unsafe_allow_html=True,
                )
                if status == "Pending review":
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

    # --- Recent decisions ----------------------------------------------------------------------
    section("Recent decisions", "Full log, CSV download, and Clear log are under Equity & Oversight")
    if decision_log.empty:
        st.caption("No decisions yet.")
    else:
        st.dataframe(decision_log.iloc[::-1].head(5), hide_index=True, width="stretch",
                     column_config={c: c.capitalize() for c in decision_log.columns})

# =====================================================================================
# ED Admissions: admit prediction + bed type, confirmed by people
# =====================================================================================
with tab_ed:
    boarding = ed[ed["status"] == "Boarding"]
    section("ED Admissions", f"{len(ed)} patients in the ED · {len(boarding)} admitted and boarding · "
                             f"longest wait {boarding['hours_boarding'].max():.1f} h")
    banner("info", "Suggestions only.",
           "The physician confirms level of care and the bed manager assigns the room.")
    if not can_place:
        st.caption(f"Signed in as {role}: view only. Only the {' or '.join(PLACEMENT_ROLES).lower()} "
                   "can accept or override.")

    ed_log = decision_log[decision_log["context"] == "ED Admissions"]

    def place_status(pid):
        rows = ed_log[ed_log["item"].str.startswith(f"{pid} ")]
        return rows.iloc[-1] if len(rows) else None

    for row in ed.itertuples():
        pid = row.patient_id
        with card():
            info, model_col, bed_col = st.columns([3.2, 3, 3], gap="medium")
            with info:
                if row.status == "Boarding":
                    status_pill = pill("Boarding", "crit" if row.hours_boarding > 2 else "warn")
                    wait = (f"<div class='ff-small' style='color:var(--ff-crit);font-weight:600'>"
                            f"Boarding {row.hours_boarding:.1f} h · admitted to {esc(row.admit_unit)}</div>"
                            if row.hours_boarding > 2 else
                            f"<div class='ff-small'>Boarding {row.hours_boarding:.1f} h · "
                            f"admitted to {esc(row.admit_unit)}</div>")
                else:
                    status_pill = pill("In workup", "neutral")
                    wait = f"<div class='ff-small'>{row.hours_in_ed:.1f} h in ED</div>"
                st.markdown(
                    f"<div class='ff-pt-top'><span class='ff-bed'>{esc(row.bed)}</span>"
                    f"<span class='ff-pid'>{esc(pid)}</span>{status_pill}</div>"
                    f"<div class='ff-pt-main'>{row.age} y · {esc(row.chief_complaint)} · "
                    f"Triage acuity <b>{row.triage_acuity}</b></div>"
                    f"<div class='ff-small'>HR {row.heart_rate} · SpO₂ {row.o2_sat}% · "
                    f"Temp {row.temperature_c}°C · BP {row.systolic_bp}/{row.diastolic_bp}</div>{wait}",
                    unsafe_allow_html=True,
                )
            if silent:
                model_col.caption("Prediction hidden (silent mode)")
            else:
                with model_col:
                    reasons = [r for r in row.top_reasons.split("; ") if r]
                    factors = ("<ul class='ff-factors'>" + "".join(f"<li>{esc(r)}</li>" for r in reasons) + "</ul>"
                               if reasons else "<div class='ff-small'>No strong factors</div>")
                    st.markdown(
                        "<div class='ff-lbl'>Admission likelihood</div>"
                        f"<div class='ff-meter'><span class='num'>{row.admit_probability:.0%}</span>"
                        f"<div class='track'><div class='fill' style='width:{100 * row.admit_probability:.0f}%'>"
                        f"</div></div></div>{factors}",
                        unsafe_allow_html=True,
                    )
                with bed_col:
                    last = place_status(pid)
                    status_html = ""
                    if last is not None:
                        kind = "ok" if last["decision"] == "Accepted" else "info"
                        status_html = (f"<div style='margin-top:0.25rem'>{pill(last['decision'], kind)} "
                                       f"<span class='ff-small'>{esc(last['role'])} · {esc(last['time'][11:16])} · "
                                       f"{esc(last['item'].split(': ', 1)[-1])}</span></div>")
                    st.markdown(
                        "<div class='ff-lbl'>Suggested bed type</div>"
                        f"<div style='font-weight:700;font-size:1.05rem'>{esc(row.likely_bed)}</div>"
                        f"<div class='ff-small'>Rule: {esc(row.bed_rule)}</div>{status_html}",
                        unsafe_allow_html=True,
                    )
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
    section("Barrier Flags", "Inpatients ranked by how likely they are to go home today")
    if silent:
        silent_placeholder()
    else:
        if not can_assign:
            st.caption(f"Signed in as {role}: view only. Only the {' or '.join(ASSIGN_ROLES).lower()} "
                       "can assign barriers.")

        scan_col, info_col = st.columns([1, 3], vertical_alignment="center")
        scan_col.button("Scan nursing notes", key="scan_notes", type="primary", on_click=run_scan, width="stretch")
        scan = st.session_state.get("barrier_scan")
        with info_col:
            if scan is None:
                if api_key:
                    st.caption(f"Reads each note for what is holding up discharge. Will use: "
                               f"**{fe.SOURCE_CLAUDE}** ({model_name}, key from {key_source})")
                else:
                    st.caption(f"Reads each note for what is holding up discharge. Will use: "
                               f"**{fe.SOURCE_CACHED}** (no API key)")
            else:
                n_total = scan["n_claude"] + scan["n_cached"] + scan["n_offline"]
                why = "; ".join(scan["errors"])
                if scan["n_claude"] == n_total:
                    banner("ok", f"Source: {fe.SOURCE_CLAUDE}", f"· {scan['model']} · all {n_total} notes")
                elif scan["n_cached"] == n_total and not why:
                    banner("info", "Source: Claude (pre-computed run on synthetic notes).",
                           "In production this runs live via the API.")
                elif scan["n_offline"] == n_total:
                    reason = why or "no API key and no pre-computed file"
                    banner("warn", f"Source: {fe.SOURCE_OFFLINE} (keyword rules)",
                           f"for all {n_total} notes · {reason}")
                else:
                    parts = [f"{label} for {n} notes" for label, n in [
                        (fe.SOURCE_CLAUDE, scan["n_claude"]),
                        ("Claude (pre-computed run on synthetic notes)", scan["n_cached"]),
                        (fe.SOURCE_OFFLINE + " (keyword rules)", scan["n_offline"])] if n]
                    banner("warn", "Source:", ", ".join(parts) + (f" · live API: {why}" if why else ""))

        if scan is not None:
            summary = fe.barrier_summary(ranked, scan)
            t1, t2, t3 = st.columns(3)
            t1.metric("Patients with social barriers", int(summary["social"].sum()))
            t2.metric("Patients with medical barriers", int(summary["medical"].sum()))
            t3.metric("Patients scanned", len(summary))

            chart_df = fe.social_barrier_by_insurance(summary)
            chart_df["label"] = chart_df["insurance"] + " (n=" + chart_df["patients"].astype(str) + ")"
            with st.expander("Social barriers by insurance group (equity check)", expanded=False):
                bars = alt.Chart(chart_df).mark_bar(color=SOCIAL_ORANGE, cornerRadiusEnd=3, size=20).encode(
                    y=alt.Y("label:N", title=None, sort=None, axis=alt.Axis(labelLimit=220)),
                    x=alt.X("pct_social:Q", title="% of scanned patients with a social barrier",
                            scale=alt.Scale(domain=[0, 100])),
                    tooltip=[alt.Tooltip("insurance:N", title="Insurance"),
                             alt.Tooltip("patients:Q", title="Patients scanned"),
                             alt.Tooltip("with_social:Q", title="With a social barrier"),
                             alt.Tooltip("pct_social:Q", title="%", format=".0f")],
                )
                text = bars.mark_text(align="left", dx=4, color="#5b6876").encode(
                    text=alt.Text("pct_social:Q", format=".0f"))
                st.altair_chart(styled((bars + text).properties(height=32 * len(chart_df) + 30)), width="stretch")
                st.caption("Help is routed by documented need, never by insurance type. Clearing these barriers "
                           "early frees beds sooner and helps the groups that board longest.")

        bar_log = decision_log[decision_log["context"] == "Barrier Flags"]
        show = st.radio("Show", ["All", "Only patients with barriers", "Social only"], index=1,
                        horizontal=True, key="barrier_filter")
        if scan is None and show != "All":
            st.caption("Showing all patients until notes are scanned.")

        def keep(pid):
            if scan is None or show == "All":
                return True
            kinds = {b["type"] for b in scan["results"].get(pid, {}).get("barriers", [])}
            return bool(kinds) if show == "Only patients with barriers" else "Social" in kinds

        shown = 0
        for rank, row in enumerate(ranked.itertuples(), start=1):
            pid = row.patient_id
            if not keep(pid):
                continue
            shown += 1
            with card():
                head, prob = st.columns([3, 2], gap="medium")
                head.markdown(
                    f"<div class='ff-pt-top'><span class='ff-pid'>#{rank}</span>"
                    f"<span class='ff-bed'>{esc(row.unit)} {esc(row.room)}</span>"
                    f"<span class='ff-pid'>{esc(pid)} · {row.age} y</span></div>"
                    f"<div class='ff-small'><b>For:</b> {esc(row.reasons_for or '—')} &nbsp;·&nbsp; "
                    f"<b>Against:</b> {esc(row.reasons_against or '—')}</div>",
                    unsafe_allow_html=True,
                )
                prob.markdown(
                    "<div class='ff-lbl'>Likely home today</div>"
                    f"<div class='ff-meter'><span class='num'>{row.discharge_probability:.0%}</span>"
                    f"<div class='track'><div class='fill' style='width:{100 * row.discharge_probability:.0f}%'>"
                    "</div></div></div>",
                    unsafe_allow_html=True,
                )
                if scan is not None and pid in scan["results"]:
                    r = scan["results"][pid]
                    if not r["barriers"]:
                        st.caption(f"No barriers flagged · {r['source']}")
                    for j, b in enumerate(r["barriers"]):
                        kind = "social" if b["type"] == "Social" else "info"
                        item = f"{pid}: {b['barrier']}"
                        done = bar_log[bar_log["item"] == item]
                        flag_col, btn_col = st.columns([4, 2], vertical_alignment="center")
                        flag_col.markdown(
                            f"<div class='ff-flag'>{pill(b['type'], kind)} {esc(b['barrier'])} · "
                            f"owner: <b>{esc(b['owner'])}</b> <span class='src'>· {esc(r['source'])}</span></div>",
                            unsafe_allow_html=True,
                        )
                        if len(done):
                            e = done.iloc[-1]
                            btn_col.markdown(
                                f"{pill('Assigned', 'ok')} <span class='ff-small'>to {esc(b['owner'])} by "
                                f"{esc(e['role'])} · {esc(e['time'][11:16])}</span>", unsafe_allow_html=True)
                        else:
                            btn_col.button(f"Assign to {b['owner']}", key=f"assign_{pid}_{j}", disabled=not can_assign,
                                           on_click=assign_barrier, args=(role, pid, b, r["source"]), width="stretch")
                with st.expander("Nursing note"):
                    st.write(row.nurse_note)
        if scan is not None:
            st.caption(f"Showing {shown} of {len(ranked)} patients.")


# =====================================================================================
# What-If: beds ready earlier
# =====================================================================================
with tab_whatif:
    if silent:
        section("What-If")
        silent_placeholder()
    else:
        section("Medicine today: operational actions",
                "Same Medicine forecast as the Command Center, including applied alerts and physician overrides")
        with card():
            oc1, oc2 = st.columns([3, 2], gap="large")
            lounge = oc1.checkbox("Open discharge lounge (beds free about 2 hours earlier)", key="wi_lounge")
            clear_social = oc1.checkbox("Clear social barriers by 10 AM (from Barrier Flags)", key="wi_social")
            flex = oc2.slider("Flex beds", 0, 6, 0, key="wi_flex")
        scan_now = st.session_state.get("barrier_scan")
        sc = fe.medicine_scenario(beds, ed_forecast, ranked, discharge_model, day_of_week, trend_level,
                                  applied_alerts, lounge, clear_social, flex, scan_now)
        base_f, scen_f = sc["base"], sc["scenario"]

        def runout_text(f):
            return fe.hour_label(f.runout_hour) if f.runout_hour else "No run-out"

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Run-out, current plan", runout_text(base_f))
        m1.caption(f"Lowest point {base_f.min_open:+.1f} beds")
        m2.metric("Run-out, with actions", runout_text(scen_f))
        m2.caption(f"Lowest point {scen_f.min_open:+.1f} beds")
        m3.metric("Boarding hours saved today", f"{sc['boarding_hours_saved']:.1f}")
        m3.caption("Patient-hours without a Medicine bed, base minus scenario (9 AM–5 PM)")
        m4.metric("Saved today", f"${sc['dollars_saved']:,.0f}")
        m4.caption(f"{sc['boarding_hours_saved']:.1f} h × ${fe.BOARDING_COST_PER_HOUR}/h")

        both = pd.concat([
            base_f.table.assign(series="Base"),
            scen_f.table.assign(series="With actions"),
        ])[["hour", "time", "series", "projected_open"]].round(1)
        x_axis = alt.X("hour:O", title=None, sort=None, axis=alt.Axis(labelAngle=0, labelExpr=HOUR_LABEL_EXPR))
        sc_color = alt.Color("series:N", title=None, legend=alt.Legend(orient="top"),
                             scale=alt.Scale(domain=["Base", "With actions"], range=[SERIES_BLUE, SERIES_ORANGE]))
        sc_base = alt.Chart(both).encode(x=x_axis, y=alt.Y("projected_open:Q", title="Projected open Medicine beds"),
                                         color=sc_color)
        sc_lines = sc_base.mark_line(strokeWidth=2).encode(strokeDash=alt.StrokeDash(
            "series:N", legend=None, scale=alt.Scale(domain=["Base", "With actions"], range=[[1, 0], [6, 4]])))
        sc_pts = sc_base.mark_point(filled=True, size=55).encode(
            tooltip=[alt.Tooltip("time:N", title="Time"), alt.Tooltip("series:N", title="Forecast"),
                     alt.Tooltip("projected_open:Q", title="Projected open beds", format=".1f")])
        sc_zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=CRITICAL_RED, strokeDash=[4, 3]).encode(y="y:Q")
        with card():
            st.altair_chart(styled((sc_zero + sc_lines + sc_pts).properties(height=260)), width="stretch")
        notes = []
        if lounge:
            notes.append("discharge lounge pulls each discharge about 2 hours earlier")
        if clear_social:
            src = "the Barrier Flags scan" if scan_now else "documented social barriers (notes not scanned yet)"
            notes.append(f"{sc['medicine_social_cleared']} Medicine patients' social barriers cleared, from {src} "
                         f"(+{sc['extra_discharges_today']:.1f} expected discharges today)")
        if flex:
            notes.append(f"{flex} flex bed{'s' if flex != 1 else ''} open from 10 AM")
        st.caption("Actions: " + ("; ".join(notes) if notes else "none selected") + ".")

        section("Annual value: what if beds were ready earlier?",
                "Earlier discharges and faster cleaning mean admitted patients board in the ED for less time")

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

        st.markdown("**1 · Boarding cost avoided**")
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

        st.markdown("**2 · Fewer patients leave without being seen (estimate)**")
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1.5])
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
        banner(
            "warn", "This is an estimate, not a forecast.",
            f"Each ED room-hour freed from boarding serves about {pprh:.2f} more patients (233 arrivals/day ÷ "
            f"32 treatment rooms ÷ 24 h). Of those patients, the case's {fe.LWBS_RATE:.0%} "
            f"left-without-being-seen rate is assumed to be recovered, at ${fe.NET_REVENUE_PER_VISIT:,} average "
            "net revenue per ED visit.",
        )
        up = fe.upper_lwbs_estimate()
        st.markdown(
            f"**Upper estimate:** If the Month-18 goal is met (LWBS {fe.LWBS_RATE:.1%} → {fe.LWBS_GOAL_RATE:.1%}): "
            f"85,000 × {fe.LWBS_RATE - fe.LWBS_GOAL_RATE:.1%} = {up['visits']:,.0f} visits × "
            f"${fe.NET_REVENUE_PER_VISIT:,} ≈ ${up['revenue'] / 1e6:.2f}M/yr"
        )


# =====================================================================================
# Equity & Oversight
# =====================================================================================
with tab_equity:
    banner("info", "Fairness safeguard.",
           "Insurance, race, and language are never model inputs; they are used only to check fairness.")
    section("Halvian baseline", "From the Halvian case brief (not synthetic): ED performance by payer")
    base = fe.HALVIAN_BASELINE.copy()
    ref = base.set_index("group").loc["Commercial"]
    base["dtp_gap"] = base["door_to_provider_min"] - ref["door_to_provider_min"]
    base["lwbs_gap"] = (base["lwbs_pct"] - ref["lwbs_pct"]).round(1)
    is_ref = base["group"].isin(["Commercial", "Overall"])
    base["dtp_gap"] = [("—" if r else f"{g:+.0f} min") for r, g in zip(is_ref, base["dtp_gap"])]
    base["lwbs_gap"] = [("—" if r else f"{g:+.1f} pts") for r, g in zip(is_ref, base["lwbs_gap"])]

    overall = base.set_index("group").loc["Overall"]
    c1, c2, _ = st.columns([1, 1, 2])
    c1.metric("Door-to-provider, overall", f"{overall['door_to_provider_min']:.0f} min")
    c2.metric("Left without being seen, overall", f"{overall['lwbs_pct']:.1f}%")
    st.dataframe(
        base,
        hide_index=True,
        width="stretch",
        column_config={
            "group": "Payer",
            "door_to_provider_min": st.column_config.NumberColumn("Door-to-provider (min)", format="%d"),
            "dtp_gap": "Door-to-provider vs Commercial",
            "lwbs_pct": st.column_config.NumberColumn("Left without being seen (%)", format="%.1f"),
            "lwbs_gap": "LWBS vs Commercial",
        },
    )
    st.caption(
        "Uninsured patients wait 16 minutes longer than Commercial patients to see a provider "
        "and leave without being seen almost twice as often (5.8% vs 3.0%)."
    )

    # --- Boarding time by insurance --------------------------------------------------------
    section("Boarding time by insurance group", "Synthetic, last 30 days")
    gap = fe.boarding_gap_by_insurance(data["boarding"])
    overall_mean = data["boarding"]["boarding_minutes"].mean()
    th_col, _ = st.columns([1, 2])
    threshold = th_col.slider("Equity alert threshold: minutes above the overall average", 5, 60, 20, 5)
    over = gap[gap["vs_overall_minutes"] > threshold]
    if len(over):
        banner("crit", "Equity alert:", "; ".join(
            f"{r.insurance} patients board {r.vs_overall_minutes:.0f} min longer than average"
            for r in over.itertuples()
        ) + f" (threshold: {threshold} min).")
    else:
        banner("ok", f"No insurance group boards more than {threshold} min above the average.")
    gap_chart = gap.assign(label=gap["insurance"] + " (n=" + gap["patients"].astype(str) + ")",
                           over=gap["vs_overall_minutes"] > threshold)
    bars = alt.Chart(gap_chart).mark_bar(cornerRadiusEnd=3, size=20).encode(
        y=alt.Y("label:N", title=None, sort=None, axis=alt.Axis(labelLimit=220)),
        x=alt.X("mean_minutes:Q", title="Mean boarding time (minutes)"),
        color=alt.condition("datum.over", alt.value(CRITICAL_RED), alt.value(SERIES_BLUE)),
        tooltip=[alt.Tooltip("insurance:N", title="Insurance"),
                 alt.Tooltip("patients:Q", title="Boarded patients"),
                 alt.Tooltip("mean_minutes:Q", title="Mean (min)", format=".0f"),
                 alt.Tooltip("median_minutes:Q", title="Median (min)", format=".0f"),
                 alt.Tooltip("vs_overall_minutes:Q", title="vs average (min)", format="+.0f")],
    )
    labels = bars.mark_text(align="left", dx=4).encode(
        text=alt.Text("mean_minutes:Q", format=".0f"), color=alt.value("#5b6876"))
    avg = alt.Chart(pd.DataFrame({"x": [overall_mean], "t": [f"Average {overall_mean:.0f} min"]}))
    avg_rule = avg.mark_rule(strokeDash=[4, 3], color="#5b6876").encode(x="x:Q")
    avg_text = avg.mark_text(align="left", dx=4, dy=-8, color="#5b6876", fontSize=11).encode(
        x="x:Q", y=alt.value(0), text="t:N")
    with card():
        st.altair_chart(styled((bars + labels + avg_rule + avg_text).properties(height=32 * len(gap_chart) + 40)),
                        width="stretch")
    st.caption(f"Red bars exceed the average by more than {threshold} min. Overall median boarding: "
               f"{data['boarding']['boarding_minutes'].median() / 60:.1f} hours.")

    # --- Phase 1 bias test ----------------------------------------------------------------------
    section("Phase 1 bias test: admission model")
    fair = fe.fairness_auc(admission_model, data["ed_visits"])
    overall_auc = fair.attrs["overall_auc"]
    flagged = fair[fair["status"].str.startswith("Do not go live")]
    if len(flagged):
        banner("crit", "Do not go live: retrain and re-test.",
               "; ".join(f"{r.attribute} {r.group}: AUC {r.auc:.2f} ({r.gap:+.2f})" for r in flagged.itertuples())
               + f" — more than {fe.FAIRNESS_AUC_GAP} below overall ({overall_auc:.2f}).")
    else:
        banner("ok", f"Every group is within {fe.FAIRNESS_AUC_GAP} of overall AUC ({overall_auc:.2f}).")
    st.dataframe(
        fair, hide_index=True, width="stretch",
        column_config={
            "attribute": "Checked by", "group": "Group", "patients": "Test patients",
            "admitted": "Admitted",
            "auc": st.column_config.NumberColumn("AUC", format="%.3f"),
            "gap": st.column_config.NumberColumn("vs overall", format="%+.3f"),
            "status": "Result",
        },
    )
    st.caption(
        f"Held-out test set: {admission_model.n_test:,} synthetic ED visits, overall AUC {overall_auc:.3f}. "
        f"Groups with fewer than {fe.FAIRNESS_MIN_GROUP} patients or {fe.FAIRNESS_MIN_ADMITTED} admissions are "
        "marked \"Too few to judge\" because their AUC is too noisy."
    )

    if silent:
        section("Override rate & Trust Score")
        banner("neutral", "Hidden in silent mode.", "Staff see no suggestions to override during the pilot.")
    else:
        # --- Override rate ------------------------------------------------------------------------
        section("Override rate", "Share of ED bed-type decisions that overrode the suggestion")
        orate = fe.override_rate(decision_log)
        low, high = fe.OVERRIDE_BAND
        c1, c2 = st.columns([1, 2], vertical_alignment="center")
        c1.metric("Override rate", f"{orate['rate']:.0%}")
        with c2:
            banner("ok" if orate["status"] == "Healthy" else "warn", f"{orate['status']}",
                   f"· healthy band {low:.0%}–{high:.0%}. Source: {orate['source']}. Too high means staff don't "
                   "trust the tool; too low can mean rubber-stamping.")

        # --- Trust Score -----------------------------------------------------------------------------
        section("Trust Score", "Predicted vs. actual discharges, last 7 days")
        trust = fe.generate_discharge_trust()
        avg_miss = trust["miss"].mean()
        c1, c2, _ = st.columns([1, 1, 2])
        c1.metric("Average miss", f"{avg_miss:.1f} discharges/day")
        c2.metric("Average miss, % of actual", f"{(trust['miss'] / trust['actual']).mean():.1%}")
        long = trust.melt(id_vars=["date", "miss"], value_vars=["predicted", "actual"],
                          var_name="series", value_name="discharges")
        long["series"] = long["series"].map({"predicted": "Predicted", "actual": "Actual"})
        color = alt.Color("series:N", title=None, scale=alt.Scale(domain=["Predicted", "Actual"],
                                                                  range=[SERIES_BLUE, SERIES_ORANGE]),
                          legend=alt.Legend(orient="top"))
        base_t = alt.Chart(long).encode(x=alt.X("date:T", title=None, axis=alt.Axis(format="%a %b %d")),
                                        y=alt.Y("discharges:Q", title="Discharges", scale=alt.Scale(zero=False)),
                                        color=color)
        lines = base_t.mark_line(strokeWidth=2)
        pts = base_t.mark_point(filled=True, size=60).encode(
            tooltip=[alt.Tooltip("date:T", format="%a %b %d"), "series:N",
                     alt.Tooltip("discharges:Q", format=".0f"), alt.Tooltip("miss:Q", title="Miss", format=".1f")])
        with card():
            st.altair_chart(styled((lines + pts).properties(height=250)), width="stretch")
        st.caption("Synthetic: hospital-wide discharges predicted by the likely-home-today model vs. what happened.")

    # --- Pause / retire rules ------------------------------------------------------------------------
    with st.expander("Pause / retire rules"):
        for heading, items in fe.PAUSE_RETIRE_RULES.items():
            st.markdown(f"**{heading}**\n" + "\n".join(f"- {i}" for i in items))
        st.markdown("**Owners**\n" + "\n".join(f"- **{who}:** {what}" for who, what in fe.OVERSIGHT_OWNERS.items()))

    # --- Full decision log -------------------------------------------------------------------------------
    section("Decision log", f"Saved to data/decision_log.csv · only the {DECISION_OWNER.lower()} can clear it")
    log = decision_log
    if log.empty:
        st.caption("No decisions yet.")
    else:
        st.dataframe(log.iloc[::-1], hide_index=True, width="stretch",
                     column_config={c: c.capitalize() for c in log.columns})
        d_col, c_col, _ = st.columns([1, 1, 3])
        d_col.download_button(
            "Download log (CSV)", log.to_csv(index=False), "foreward_flow_decision_log.csv", "text/csv"
        )
        with c_col.popover("Clear log", disabled=not is_owner):
            st.markdown(f"Delete all {len(log)} entries? A single entry recording the clear is kept.")
            st.button("Yes, clear the log", key="confirm_clear_log", type="primary",
                      on_click=clear_log, args=(role,))

footer()
