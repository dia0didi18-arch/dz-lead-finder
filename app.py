"""DZ Lead Finder — B2B Prospecting & Lead Management Dashboard (Streamlit).

Run with:  streamlit run app.py
"""
import time
from collections import Counter
from datetime import datetime

import pandas as pd
import streamlit as st

from ai_engine import (
    PROVIDERS,
    AIAuthError,
    AIConfigError,
    AIError,
    AIQuotaError,
    fallback_analysis,
    make_client,
    score_lead,
    validate_api_key,
)
from scraper import ScraperError, compute_digital_health, run_audit, run_scrape, resolve_wilaya
from utils import (
    FOLLOWUP_DAYS,
    PRIORITIES,
    PRIORITY_RANK,
    STATUSES,
    KnownLeads,
    backfill_contacted_at,
    build_excel_bytes,
    followup_info,
    is_mobile,
    load_leads,
    load_pitchless,
    prepare_export,
    priority_badge,
    save_leads,
    update_status,
    whatsapp_launch_link,
)

st.set_page_config(page_title="DZ Lead Finder — Prospecting Dashboard", page_icon="📍", layout="wide")
st.title("📍 DZ Lead Finder — Prospecting Dashboard")
st.caption("Google Maps (wilaya-pinned) → strict +213 mobiles → digital health audit → AI-scored WhatsApp pitches")

st.session_state.setdefault("editor_ver", 0)
st.session_state.setdefault("due_filter", False)

PROVIDER_INFO = {
    "Google Gemini": (
        "https://aistudio.google.com/apikey",
        "New AI Studio keys start with AQ. (old AIza keys are being rejected); both are accepted here. "
        "Free tier: many more requests/day on Flash-Lite than on Flash.",
    ),
    "Groq (Free)": (
        "https://console.groq.com/keys",
        "Free plan: gpt-oss-120b / gpt-oss-20b / qwen, 8K tokens/min, 1,000 requests/day. "
        "The app paces itself to the token limit.",
    ),
    "OpenRouter (Free)": (
        "https://openrouter.ai/keys",
        "Free models: 20 requests/min and 50/day (1,000/day after buying $10 credits). Enable the free-model "
        "options at openrouter.ai/settings/privacy or every request fails.",
    ),
}
AI_DELAY_DEFAULTS = {"Google Gemini": 6.0, "Groq (Free)": 3.0, "OpenRouter (Free)": 4.0}
SAMPLE_LEAD = {
    "name": "مطعم الأمل", "keyword": "مطاعم", "city": "Blida", "address": "Blida",
    "rating": 4.2, "reviews": 60, "website": "", "site_active": "No website", "site_type": "None",
    "phone_type": "mobile", "has_website": False, "has_whatsapp": True, "health_score": 25,
    "health_weak_spots": ["Aucun site web", "Peu d'avis Google"], "maps_url": "https://maps.google.com/?q=sample",
}


# --------------------------------------------------------------------------
# Data preparation: health audit is DERIVED (not stored) so older databases
# keep working unchanged — has_website / health_score recompute on every load.
# --------------------------------------------------------------------------
def _derive_health(df: pd.DataFrame) -> pd.DataFrame:
    health = [
        compute_digital_health(w, s, r, rc, hw)
        for w, s, r, rc, hw in zip(
            df["website"], df["site_active"], df["rating"], df["reviews"], df["has_whatsapp"]
        )
    ]
    df["has_website"] = [h["has_website"] for h in health]
    df["is_unclaimed"] = [h["is_unclaimed"] for h in health]
    df["health_score"] = [h["health_score"] for h in health]
    df["health_weak_spots"] = ["; ".join(h["health_weak_spots"]) for h in health]
    return df


def _add_followup_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Derive 'followup_due' (still Contacted, 3+ days ago) and the table label from contacted_at."""
    now = datetime.now()
    info = [followup_info(s, c, now) for s, c in zip(df["status"], df["contacted_at"])]
    df["followup_due"] = [due for due, _ in info]
    df["follow_up"] = [
        f"⏰ Due ({d}d)" if due else (f"🕒 {d}d ago" if d is not None else "")
        for due, d in info
    ]
    return df


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure every expected column exists, then derive health, links, priority and follow-up state."""
    df = df.copy()
    for col in ("site_active", "site_type", "whatsapp_link", "scoring_reason", "custom_message_darija",
                "custom_message_franco_arabe", "custom_message_french", "custom_followup_darija",
                "website", "address", "phone", "phone_type", "maps_url", "contacted_at", "source"):
        if col not in df:
            df[col] = ""
        df[col] = df[col].fillna("")
    if "status" not in df:
        df["status"] = "New"
    df["status"] = df["status"].where(df["status"].isin(STATUSES), "New")
    df["source"] = df["source"].where(df["source"] != "", "maps")
    for col, default in (("lead_score", 0), ("rating", 0.0), ("reviews", 0)):
        df[col] = pd.to_numeric(df.get(col), errors="coerce").fillna(default)
    df["lead_score"] = df["lead_score"].astype(int)
    ptypes = ["mobile" if is_mobile(p) else t for p, t in zip(df["phone"], df["phone_type"])]
    df["has_whatsapp"] = [is_mobile(p) for p in df["phone"]]  # any verified +213 mobile is WhatsApp-ready
    _derive_health(df)

    # Rebuilt from phone + message so rows saved by older versions get current links too.
    df["whatsapp_launch"] = [
        whatsapp_launch_link(p or None, t, m) for p, t, m in zip(df["phone"], ptypes, df["custom_message_darija"])
    ]
    df["whatsapp_followup"] = [
        whatsapp_launch_link(p or None, t, m) for p, t, m in zip(df["phone"], ptypes, df["custom_followup_darija"])
    ]
    df["priority"] = [
        priority_badge(s, p, a) for s, p, a in zip(df["lead_score"], df["phone"], df["site_active"])
    ]
    # A lead with no pitch shows WHY (the AI error) instead of a silent empty cell.
    df["ai_note"] = [
        "" if m else "⚠️ " + str(r).replace("AI error: ", "")[:90]
        for r, m in zip(df["scoring_reason"], df["custom_message_darija"])
    ]
    return _add_followup_cols(df.reset_index(drop=True))


# --------------------------------------------------------------------------
# AI helpers (shared by Start, Retry and Test)
# --------------------------------------------------------------------------
def _open_client(api_key: str, provider: str):
    """Create the client and validate the key / discover usable models. Shows errors itself."""
    if not api_key.strip():
        st.error(f"Please enter your {provider} API key.")
        return None
    try:
        client = make_client(provider, api_key)
        with st.spinner(f"Checking {provider} key and available models..."):
            label = validate_api_key(client)
        st.toast(f"AI ready: {label}")
        return client
    except (AIAuthError, AIConfigError) as e:
        st.error(f"🔑 {e}")
    except AIError as e:
        st.error(f"AI setup problem: {e}")
    return None


def score_batch(client, leads, bar, lo, hi, offer, sender, pause) -> str:
    """Score leads in place. Returns a stop reason ('' if the whole batch ran).

    A fatal problem (bad key, account setting, daily cap) stops the AI calls but never loses the
    leads: the remaining ones keep an explanatory error and can be retried later.
    """
    stop = ""
    for i, lead in enumerate(leads, 1):
        if stop:
            lead.update(fallback_analysis(stop))
            continue
        bar.progress(lo + (hi - lo) * (i - 1) / len(leads), text=f"🤖 Writing pitch {i}/{len(leads)}: {lead['name']}")
        try:
            lead.update(score_lead(client, lead, offer, sender))
        except (AIAuthError, AIConfigError, AIQuotaError) as e:
            stop = str(e)
            lead.update(fallback_analysis(stop))
            continue
        time.sleep(pause)
    return stop


def ai_problem_text(leads, stop: str) -> str:
    """Human-readable list of why some leads have no pitch ('' when everything worked)."""
    failed = [l for l in leads if not l.get("custom_message_darija")]
    if not failed:
        return ""
    lines = []
    if stop:
        lines.append(f"⛔ {stop}")
    reasons = Counter(str(l.get("scoring_reason", "")).replace("AI error: ", "")[:220] for l in failed)
    lines += [f"• {r} (×{n})" for r, n in reasons.most_common(3) if r != stop]
    return (f"The AI could not write pitches for {len(failed)} lead(s):\n" + "\n".join(lines)
            + "\n\nFix the cause, then use **🔁 Retry AI** in the sidebar: no new scraping is needed.")


def _clean_row(rec: dict) -> dict:
    out = {}
    for k, v in rec.items():
        missing = v is None or (isinstance(v, float) and v != v)
        out[k] = (None if k == "rating" else "") if missing else v
    return out


# --------------------------------------------------------------------------
# Sidebar: inputs
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("🔎 Search (Google Maps)")
    keyword = st.text_input("Business keyword", placeholder="مطاعم / عيادات أسنان / مكتبات")
    city = st.text_input("City / Wilaya", placeholder="Blida, Oran, Alger...")
    wilaya = st.text_input("Wilaya (optional, pins the search)", placeholder="Blida",
                           help="Adding the wilaya keeps Google's local-IP redirection from widening "
                                "the search beyond your target area. Auto-resolved from the city when left blank.")
    max_leads = st.number_input("Max new leads", min_value=1, max_value=200, value=20, step=5)
    skip_saved = st.checkbox(
        "Skip previously saved leads", value=True,
        help="Leads already in leads.db (same phone, or same name + city) are skipped before "
             "they are opened, audited or sent to the AI.",
    )
    st.caption("Only Algerian mobiles (+2135 / +2136 / +2137) are kept; landlines are always discarded. "
               "Listings whose address is outside the requested area are dropped.")

    st.header("✍️ Sender")
    sender_name = st.text_input("Your Name / Agency Name", placeholder="Abderrahmane | SubZone")
    if not sender_name.strip():
        st.caption("⚠️ Empty: the pitches will be sent without a signature.")

    st.header("🤖 AI")
    provider = st.selectbox("AI Provider", PROVIDERS)
    key_url, key_note = PROVIDER_INFO[provider]
    api_key = st.text_input(f"{provider} API key", type="password", key=f"api_key_{provider}")
    st.caption(f"[Get a key]({key_url}) · {key_note}")
    offer = st.text_area(
        "What do you sell?",
        value="creation of websites and social media management for local businesses",
        height=80,
    )

    with st.expander("Advanced"):
        headless = st.checkbox("Headless browser", value=True,
                               help="Turn OFF to watch the browser or solve a CAPTCHA manually.")
        dmin, dmax = st.slider("Random delay between actions (s)", 1.0, 15.0, (3.0, 7.0), 0.5)
        ai_delay = st.slider("Pause between AI calls (s)", 0.0, 15.0, AI_DELAY_DEFAULTS[provider], 0.5,
                             key=f"ai_delay_{provider}",
                             help="Raise it if you hit free-tier rate limits (429).")

    start = st.button("🚀 Start Prospecting", type="primary")

    if st.button("🧪 Test AI (uses 1 request)"):
        test_client = _open_client(api_key, provider)
        if test_client:
            try:
                res = score_lead(test_client, SAMPLE_LEAD, offer.strip(), sender_name.strip())
                if res["custom_message_darija"]:
                    st.session_state["ai_test"] = (
                        "ok", f"✅ {provider} works. Sample score {res['lead_score']}/100.\n\n"
                              f"{res['custom_message_darija']}")
                else:
                    st.session_state["ai_test"] = ("error", f"{provider} failed: {res['scoring_reason']}")
            except (AIAuthError, AIConfigError, AIQuotaError) as e:
                st.session_state["ai_test"] = ("error", f"{provider} failed: {e}")

    pitchless_saved = len(load_pitchless())
    retry = st.button(f"🔁 Retry AI for {pitchless_saved} saved lead(s) without a pitch") if pitchless_saved else False

    if st.button("📂 Load saved leads (SQLite)"):
        saved = load_leads()
        if saved.empty:
            st.warning("The database is empty.")
        else:
            st.session_state["df"] = _prepare(saved)
            st.session_state.pop("run_summary", None)
            st.session_state["editor_ver"] += 1
            st.success(f"Loaded {len(saved)} leads.")


# --------------------------------------------------------------------------
# Pipelines
# --------------------------------------------------------------------------
def run_pipeline() -> None:
    st.session_state.pop("ai_problem", None)
    if not keyword.strip() or not city.strip():
        st.error("Please enter both a keyword and a city.")
        return

    # 1) Fail fast on a bad key / unavailable models (before a long scrape)
    client = _open_client(api_key, provider)
    if client is None:
        return

    resolved_wilaya = resolve_wilaya(wilaya.strip() or city)
    known = KnownLeads(enabled=skip_saved)
    stats: dict = {}
    badge = st.empty()
    if skip_saved:
        badge.info(f"⏭️ Skipped 0 existing leads from database ({known.loaded} saved leads indexed)")
    bar = st.progress(0.0, text="Starting...")

    # 2) Scrape (0% - 40%): wilaya-pinned query, dedup, strict mobile + area filter
    def on_progress(p: float, msg: str) -> None:
        bar.progress(min(max(p, 0.0), 1.0) * 0.4, text=f"🗺️ {msg}")
        if skip_saved:
            badge.info(f"⏭️ Skipped {stats.get('skipped', 0)} existing leads from database")

    try:
        leads = run_scrape(
            keyword.strip(), city.strip(), int(max_leads),
            progress=on_progress, headless=headless, delay_range=(dmin, max(dmin, dmax)),
            known=known, stats=stats, wilaya=resolved_wilaya,
        )
    except ScraperError as e:
        bar.empty()
        st.error(f"Scraping failed: {e}")
        return
    except Exception as e:
        bar.empty()
        st.error(f"Unexpected scraping error: {e}")
        return

    summary = (
        f"Visited {stats['visited']} listings in {resolved_wilaya or city} → **{len(leads)} new mobile leads** · "
        f"⏭️ Skipped {stats['skipped']} existing leads from database · "
        f"{stats['non_mobile']} without a valid mobile dropped · "
        f"{stats['out_of_area']} outside the area dropped · {stats['failed']} failed to load"
    )
    if skip_saved:
        badge.info(f"⏭️ Skipped {stats['skipped']} existing leads from database")
    for lead in leads:
        lead.update(keyword=keyword.strip(), city=city.strip(), status="New", source="maps")

    # 3) Digital Health Audit (40% - 50%), non-fatal
    bar.progress(0.42, text=f"🩺 Running the Digital Health Audit on {len(leads)} leads...")
    try:
        audits = run_audit(leads)
    except Exception:
        audits = [{
            "site_active": "Unknown", "site_type": "Unknown", "has_website": False,
            "is_unclaimed": False, "has_whatsapp": True, "health_score": 0,
            "health_weak_spots": ["Audit unavailable"],
        }] * len(leads)
    for lead, audit in zip(leads, audits):
        lead.update(audit)
    bar.progress(0.5, text="🩺 Audit done")

    # 4) AI scoring + pitches (50% - 100%). Never loses the scraped leads.
    stop = score_batch(client, leads, bar, 0.5, 1.0, offer.strip(), sender_name.strip(), ai_delay)
    bar.progress(1.0, text="✅ Done")

    problem = ai_problem_text(leads, stop)
    if problem:
        st.session_state["ai_problem"] = problem
        summary += f" · ⚠️ {sum(1 for l in leads if not l.get('custom_message_darija'))} without an AI pitch"

    df = _prepare(pd.DataFrame(leads))
    try:
        save_leads(df)
        st.toast("Leads saved to leads.db")
    except Exception as e:
        st.warning(f"Could not save to SQLite: {e}")
    st.session_state["df"] = df
    st.session_state["run_summary"] = summary
    st.session_state["editor_ver"] += 1


def run_retry() -> None:
    """Generate the missing pitches for leads already saved (no scraping, no new Maps requests)."""
    st.session_state.pop("ai_problem", None)
    client = _open_client(api_key, provider)
    if client is None:
        return
    pending = load_pitchless()
    if pending.empty:
        st.info("Every saved lead already has a pitch.")
        return
    leads = [_clean_row(r) for r in pending.to_dict("records")]
    bar = st.progress(0.0, text="Starting...")
    stop = score_batch(client, leads, bar, 0.0, 1.0, offer.strip(), sender_name.strip(), ai_delay)
    bar.progress(1.0, text="✅ Done")

    done = sum(1 for l in leads if l.get("custom_message_darija"))
    save_leads(pd.DataFrame(leads))  # upsert: keeps status and contacted_at
    problem = ai_problem_text(leads, stop)
    if problem:
        st.session_state["ai_problem"] = problem
    st.session_state["df"] = _prepare(load_leads())
    st.session_state["run_summary"] = f"🔁 Retry finished: {done}/{len(leads)} pitches written. Showing all saved leads."
    st.session_state["editor_ver"] += 1
    st.rerun()


if start:
    run_pipeline()
if retry:
    run_retry()


# --------------------------------------------------------------------------
# Callbacks: status editing (micro-CRM), follow-up filter, contact-date backfill
# --------------------------------------------------------------------------
def _on_status_edit() -> None:
    """Persist status changes made in the table, then reset the editor state."""
    key = f"leads_editor_{st.session_state['editor_ver']}"
    edits = st.session_state.get(key, {}).get("edited_rows", {})
    urls = st.session_state.get("_view_urls", [])
    base = st.session_state.get("df")
    for pos, changes in edits.items():
        pos = int(pos)
        if "status" not in changes or pos >= len(urls) or base is None:
            continue
        mask = base["maps_url"] == urls[pos]
        try:
            contacted_ts = update_status(urls[pos], changes["status"])  # sets contacted_at on 'Contacted'
            if contacted_ts:
                base.loc[mask, "contacted_at"] = contacted_ts
        except Exception as e:
            st.session_state["_status_error"] = str(e)
        base.loc[mask, "status"] = changes["status"]
    if base is not None:
        _add_followup_cols(base)
    st.session_state["editor_ver"] += 1  # new widget key -> no stale deltas on the next run


def _toggle_due_filter() -> None:
    st.session_state["due_filter"] = not st.session_state["due_filter"]


def _backfill_contacted() -> None:
    try:
        _, ts = backfill_contacted_at()
    except Exception as e:
        st.session_state["_status_error"] = str(e)
        return
    base = st.session_state.get("df")
    if base is not None:
        base.loc[(base["status"] == "Contacted") & (base["contacted_at"] == ""), "contacted_at"] = ts
        _add_followup_cols(base)
    st.session_state["editor_ver"] += 1


# --------------------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------------------
if test_result := st.session_state.get("ai_test"):
    (st.success if test_result[0] == "ok" else st.error)(test_result[1])
if ai_problem := st.session_state.get("ai_problem"):
    st.error(ai_problem)

df = st.session_state.get("df")
if df is None or df.empty:
    st.info("Fill in the sidebar and click **🚀 Start Prospecting** (or load saved leads).")
    st.stop()

if summary_text := st.session_state.get("run_summary"):
    st.info(summary_text)
if err := st.session_state.pop("_status_error", None):
    st.warning(f"Could not save to the database: {err}")

# ---- KPI row ---------------------------------------------------------------
st.subheader("📊 Pipeline Overview")
kpi = st.columns(6)
kpi[0].metric("Total leads", len(df))
status_counts = df["status"].value_counts()
kpi[1].metric("New", int(status_counts.get("New", 0)))
kpi[2].metric("Contacted", int(status_counts.get("Contacted", 0)))
kpi[3].metric("Closed / Won", int(status_counts.get("Closed", 0)))
kpi[4].metric("Avg. AI score", f"{df['lead_score'].mean():.0f}")
kpi[5].metric("Avg. digital health", f"{df['health_score'].mean():.0f}/100")
prio_counts = df["priority"].value_counts()
st.caption("  ·  ".join(f"{p}: {int(prio_counts.get(p, 0))}" for p in PRIORITIES))

# ---- Follow-up quick filter -------------------------------------------------
due_active = st.session_state["due_filter"]
due_count = int(df["followup_due"].sum())
DUE_LABEL = f"⏰ Due for Follow-up ({FOLLOWUP_DAYS}+ Days)"
q1, q2 = st.columns([2, 3])
q1.button(f"✖ Clear: {DUE_LABEL}" if due_active else DUE_LABEL,
          on_click=_toggle_due_filter, type="primary" if due_active else "secondary")
q2.caption(
    f"{due_count} lead(s) contacted {FOLLOWUP_DAYS}+ days ago and still 'Contacted'."
    + (" Status and priority filters are ignored while this filter is on." if due_active else "")
)
undated = int(((df["status"] == "Contacted") & (df["contacted_at"] == "")).sum())
if undated:
    st.button(
        f"Start the {FOLLOWUP_DAYS}-day clock now for {undated} 'Contacted' lead(s) without a date",
        on_click=_backfill_contacted,
        help="These leads were marked Contacted before dates were tracked, so they can't become due yet.",
    )

st.divider()

# ---- Filters -----------------------------------------------------------------
f1, f2, f3, f4 = st.columns(4)
min_score = f1.slider("Min AI score", 0, 100, 0)
min_rating = f2.slider("Min Google rating", 0.0, 5.0, 0.0, 0.1)
max_health = f3.slider("Max digital health (find weak businesses)", 0, 100, 100)
site_filter = f4.selectbox("Website", ["Any", "Own active site", "No own website"])
g1, g2, g3 = st.columns(3)
status_filter = g1.multiselect("Status", STATUSES, default=STATUSES)
priority_filter = g2.multiselect("Priority", PRIORITIES, default=PRIORITIES)
sort_by = g3.selectbox("Sort by", ["priority", "lead_score", "health_score", "rating", "reviews"])
ascending = st.checkbox("Ascending", value=False)

view = df[(df["lead_score"] >= min_score) & (df["rating"] >= min_rating) & (df["health_score"] <= max_health)]
if due_active:
    view = view[view["followup_due"]]
else:
    view = view[view["status"].isin(status_filter) & view["priority"].isin(priority_filter)]
if site_filter == "Own active site":
    view = view[view["has_website"] & (view["site_active"] == "Yes")]
elif site_filter == "No own website":
    view = view[~view["has_website"]]
if sort_by == "priority":  # High first (then best score) unless "Ascending" is ticked
    view = view.assign(_rank=view["priority"].map(PRIORITY_RANK)).sort_values(
        ["_rank", "lead_score"], ascending=[not ascending, ascending]
    ).drop(columns="_rank")
else:
    view = view.sort_values(sort_by, ascending=ascending)

st.caption("Change a lead's **Status** directly in the table, it is saved to the database instantly. "
           "Setting it to **Contacted** stamps the date and starts the follow-up clock. "
           "Tap **💬 WhatsApp** to open a prefilled chat.")

st.subheader(f"Leads ({len(view)})")
st.session_state["_view_urls"] = view["maps_url"].tolist()

# In follow-up mode the table swaps in the follow-up message and its WhatsApp link.
wa_col = "whatsapp_followup" if due_active else "whatsapp_launch"
msg_col = "custom_followup_darija" if due_active else "custom_message_darija"
table_cols = ["priority", "status", "follow_up", "health_score", "health_weak_spots",
              "name", "phone", "lead_score", "rating", wa_col, msg_col]
if (view["ai_note"] != "").any():
    table_cols.append("ai_note")
column_config = {
    "priority": "Priority",
    "status": st.column_config.SelectboxColumn("Status", options=STATUSES, required=True),
    "follow_up": "Follow-up",
    "health_score": st.column_config.ProgressColumn(
        "Digital Health", min_value=0, max_value=100, format="%d",
        help="0-100: website, rating, reviews, WhatsApp reachability, claimed listing.",
    ),
    "health_weak_spots": st.column_config.TextColumn("Weak spots", width="medium"),
    "name": "Name",
    "phone": "Phone",
    "lead_score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
    "rating": st.column_config.NumberColumn("Rating", format="%.1f ⭐"),
    wa_col: st.column_config.LinkColumn(
        "WhatsApp (follow-up)" if due_active else "WhatsApp",
        display_text="💬 WhatsApp: 3-day follow-up" if due_active else "💬 Send pitch",
    ),
    msg_col: st.column_config.TextColumn(
        "Follow-up (Darija)" if due_active else "AI Pitch (Darija)", width="large"
    ),
    "ai_note": st.column_config.TextColumn("AI problem", width="large"),
}
st.data_editor(
    view[table_cols],
    key=f"leads_editor_{st.session_state['editor_ver']}",
    on_change=_on_status_edit,
    hide_index=True,
    num_rows="fixed",
    disabled=[c for c in table_cols if c != "status"],
    column_config=column_config,
)

# ---- Lead detail: health card + wa.me quick-launch buttons -------------------
if not view.empty:
    st.subheader("Lead details")
    names = dict(zip(view["maps_url"], view["name"]))
    url = st.selectbox("Select a lead", list(names), format_func=lambda u: names[u])
    row = view[view["maps_url"] == url].iloc[0]
    due = bool(row["followup_due"])
    has_followup = bool(row["whatsapp_followup"] and row["custom_followup_darija"])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{row['name']}** — {row['priority']} — AI score **{int(row['lead_score'])}/100**")
        st.write(row["scoring_reason"])
        if row["address"]:
            st.write(f"📍 {row['address']}")
        st.write(f"🌐 {row['website'] or 'No website'} — active: **{row['site_active'] or 'Unknown'}** "
                 f"({row['site_type'] or 'n/a'})")
        st.write(f"⭐ {float(row['rating']):.1f} · 📝 {int(row['reviews'])} reviews")
        if row["contacted_at"]:
            st.write(f"📅 Status **{row['status']}** · contacted at {row['contacted_at']}")
        st.link_button("🗺️ Open in Google Maps", row["maps_url"])

        # Health card
        health = int(row["health_score"])
        h_color = "🟢" if health >= 60 else ("🟠" if health >= 30 else "🔴")
        st.markdown(f"#### {h_color} Digital Health: {health}/100")
        st.progress(health / 100)
        badges = []
        badges.append("✅ Website" if row["has_website"] else "❌ No website")
        badges.append("✅ WhatsApp line" if row["has_whatsapp"] else "❌ No WhatsApp line")
        badges.append("⚠️ Unclaimed listing" if row["is_unclaimed"] else "✅ Claimed listing")
        st.caption(" · ".join(badges))
        if row["health_weak_spots"]:
            st.warning("**Weak spots targeted by the AI pitch:** " + row["health_weak_spots"].replace("; ", " · "))

        # wa.me quick-launch buttons
        b1, b2 = st.columns(2)
        if row["whatsapp_launch"]:
            b1.link_button("💬 WhatsApp: send the pitch", row["whatsapp_launch"], type="primary")
        if has_followup:
            label = "💬 WhatsApp: 3-day follow-up" + (" ⏰ DUE" if due else "")
            b2.link_button(label, row["whatsapp_followup"],
                           type="primary" if due else "secondary")
        if due:
            st.warning(f"⏰ Contacted {row['follow_up'].split('(')[-1].rstrip(')')} ago: the follow-up is due.")
    with c2:
        st.caption("Pitch (Darija)")
        st.code(row["custom_message_darija"], language=None)
        st.caption("Pitch (Franco-Arabe)")
        st.code(row["custom_message_franco_arabe"], language=None)
        st.caption("Pitch (Business French)")
        st.code(row["custom_message_french"], language=None)
        st.caption("3-day follow-up (Darija)")
        st.code(row["custom_followup_darija"], language=None)

# ---- Export -------------------------------------------------------------------
st.subheader("Export")
st.caption("Both files have clickable WhatsApp columns (pitch and follow-up): open on PC or phone and tap the link.")
e1, e2 = st.columns(2)
e1.download_button(
    "⬇️ CSV", prepare_export(view).to_csv(index=False).encode("utf-8-sig"),  # utf-8-sig keeps Arabic readable in Excel
    file_name="leads.csv", mime="text/csv",
)
e2.download_button(
    "⬇️ Excel (clickable WhatsApp links)", build_excel_bytes(view), file_name="leads.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
