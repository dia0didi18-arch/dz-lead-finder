"""Helpers: phones, deduplication, priority, WhatsApp links, SQLite micro-CRM, Excel export."""
import io
import re
import sqlite3
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Optional, Set, Tuple
from urllib.parse import quote, unquote_plus

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill

DB_PATH = "leads.db"
STATUSES = ["New", "Contacted", "Interested", "Closed", "Not Interested"]
FOLLOWUP_DAYS = 3  # a lead still "Contacted" this many days later is due for the follow-up

# Strict Algerian mobile: +213 then 5, 6 or 7, then 8 digits. Everything else
# (+213 21, 23, 24, 25, 26, 27, 29, 3x, 4x ... landlines) is rejected.
MOBILE_E164_RE = re.compile(r"\+213[567]\d{8}")

COLUMNS = [
    "maps_url", "name", "phone", "phone_type", "address", "rating", "reviews",
    "website", "email", "bio", "source", "site_active", "site_type", "whatsapp_link", "whatsapp_launch",
    "lead_score", "scoring_reason", "custom_message_darija", "custom_message_french",
    "custom_followup_darija", "status", "contacted_at", "keyword", "city", "scraped_at",
]

# Columns added after v1: used to migrate an existing leads.db automatically.
MIGRATIONS = {
    "status": "TEXT DEFAULT 'New'",
    "site_active": "TEXT",
    "site_type": "TEXT",
    "whatsapp_launch": "TEXT",
    "custom_followup_darija": "TEXT",
    "contacted_at": "DATETIME",
    "email": "TEXT",
    "bio": "TEXT",
    "source": "TEXT DEFAULT 'maps'",
}


# --------------------------------------------------------------------------
# Phone handling
# --------------------------------------------------------------------------
def normalize_algerian_phone(raw: Optional[str]) -> Tuple[Optional[str], str]:
    """Return (E.164 number or None, type): 'mobile', 'landline', 'invalid' or 'missing'."""
    if not raw:
        return None, "missing"
    digits = re.sub(r"\D", "", str(raw))
    if not digits:
        return None, "missing"

    if digits.startswith("00213"):
        national = digits[5:]
    elif digits.startswith("213"):
        national = digits[3:]
    elif digits.startswith("0"):
        national = digits[1:]
    else:
        national = digits
    if national.startswith("0"):  # "+213 (0) 555..." case
        national = national[1:]

    if len(national) == 9 and national[0] in "567":
        return "+213" + national, "mobile"
    if len(national) == 8 and national[0] in "234":
        return "+213" + national, "landline"
    return None, "invalid"


def is_mobile(phone: Optional[str]) -> bool:
    return bool(phone and MOBILE_E164_RE.fullmatch(str(phone)))


def whatsapp_link(e164: Optional[str], phone_type: str) -> str:
    """Plain wa.me click-to-chat link (mobile numbers only)."""
    if e164 and phone_type == "mobile":
        return "https://wa.me/" + e164.lstrip("+")
    return ""


WA_SEND_URL = "https://api.whatsapp.com/send"
MAX_URL_LEN = 2000  # Excel hyperlinks break above ~2083 characters


def whatsapp_launch_link(e164: Optional[str], phone_type: str, message: str = "") -> str:
    """api.whatsapp.com link (works on WhatsApp Web, desktop and mobile) with the
    message pre-filled. Arabic text expands ~6x when URL-encoded, so a message that
    would push the URL past MAX_URL_LEN is trimmed and ends with an ellipsis."""
    if not (e164 and phone_type == "mobile"):
        return ""
    base = f"{WA_SEND_URL}?phone={e164.lstrip('+')}"
    msg = (message or "").strip()
    if not msg:
        return base
    url = f"{base}&text={quote(msg, safe='')}"
    while len(url) > MAX_URL_LEN and len(msg) > 20:
        msg = msg[: int(len(msg) * 0.9)].rsplit(" ", 1)[0].rstrip() + "…"
        url = f"{base}&text={quote(msg, safe='')}"
    return url


# --------------------------------------------------------------------------
# Lead priority
# --------------------------------------------------------------------------
PRIORITY_HIGH = "🔥 High Priority"
PRIORITY_MEDIUM = "⚡ Medium Priority"
PRIORITY_LOW = "💤 Low Priority"
PRIORITIES = [PRIORITY_HIGH, PRIORITY_MEDIUM, PRIORITY_LOW]
PRIORITY_RANK = {p: i for i, p in enumerate(PRIORITIES)}


def priority_badge(score, phone: Optional[str], site_active: Optional[str]) -> str:
    """High: score >= 80 AND verified mobile AND no active site (audit says No / No website).
    Medium: score 50-79 (and 80+ leads that miss one High condition). Low: score < 50."""
    try:
        score = int(score)
    except (TypeError, ValueError):
        score = 0
    no_active_site = str(site_active) in ("No", "No website")
    if score >= 80 and is_mobile(phone) and no_active_site:
        return PRIORITY_HIGH
    if score >= 50:
        return PRIORITY_MEDIUM
    return PRIORITY_LOW


# --------------------------------------------------------------------------
# Deduplication helpers
# --------------------------------------------------------------------------
def _norm(text) -> str:
    """Case-, accent- and punctuation-insensitive form (Arabic letters are kept)."""
    s = unicodedata.normalize("NFKD", str(text or "")).casefold()
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^\w]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def place_key(url: str) -> str:
    """Stable id of a Google Maps place, ignoring tracking parameters."""
    m = re.search(r"!1s(0x[0-9a-f]+:0x[0-9a-f]+)", url or "", re.I)
    if m:
        return m.group(1).lower()
    return (url or "").split("?")[0].split("/data=")[0]


def name_from_url(url: str) -> str:
    m = re.search(r"/maps/place/([^/@?]+)", url or "")
    return unquote_plus(m.group(1)) if m else ""


class KnownLeads:
    """Index of leads already stored in leads.db (by place id, phone, name + city).

    With enabled=False nothing is read from the database, but leads accepted during
    the current run are still remembered so the same business is never added twice.
    """

    def __init__(self, path: str = DB_PATH, enabled: bool = True):
        self.place_keys: Set[str] = set()
        self.phones: Set[str] = set()
        self.name_city: Set[Tuple[str, str]] = set()
        self.loaded = 0
        if enabled:
            try:
                # Every saved lead counts as known, even one whose AI pitch failed: re-searching must
                # not re-scrape the same places. Pitch-less leads are fixed with the app's
                # "Retry AI" button (no new scraping needed), see load_pitchless().
                with _db(path) as conn:
                    rows = conn.execute("SELECT maps_url, name, phone, city FROM leads").fetchall()
            except sqlite3.Error:
                rows = []
            for url, name, phone, city in rows:
                self.remember(name, city, phone, url)
            self.loaded = len(rows)

    def remember(self, name, city, phone, url) -> None:
        if url:
            self.place_keys.add(place_key(url))
        if phone:
            self.phones.add(str(phone))
        if name:
            self.name_city.add((_norm(name), _norm(city)))

    def seen_candidate(self, url: str, city: str) -> bool:
        """Cheap pre-check from the search result link, before opening the place page."""
        if place_key(url) in self.place_keys:
            return True
        name = name_from_url(url)
        return bool(name) and (_norm(name), _norm(city)) in self.name_city

    def seen_lead(self, name: str, city: str, phone: Optional[str]) -> bool:
        """Full check once the place page has been read."""
        if phone and phone in self.phones:
            return True
        return bool(name) and (_norm(name), _norm(city)) in self.name_city


# --------------------------------------------------------------------------
# SQLite storage
# --------------------------------------------------------------------------
@contextmanager
def _db(path: str = DB_PATH):
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS leads (
                maps_url TEXT PRIMARY KEY,
                name TEXT, phone TEXT, phone_type TEXT, address TEXT,
                rating REAL, reviews INTEGER, website TEXT,
                email TEXT, bio TEXT, source TEXT DEFAULT 'maps',
                site_active TEXT, site_type TEXT,
                whatsapp_link TEXT, whatsapp_launch TEXT,
                lead_score INTEGER, scoring_reason TEXT,
                custom_message_darija TEXT, custom_message_french TEXT,
                custom_followup_darija TEXT,
                status TEXT DEFAULT 'New',
                contacted_at DATETIME,
                keyword TEXT, city TEXT, scraped_at TEXT
            )"""
        )
        existing = {row[1] for row in conn.execute("PRAGMA table_info(leads)")}
        for col, ddl in MIGRATIONS.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE leads ADD COLUMN {col} {ddl}")
        yield conn
        conn.commit()
    finally:
        conn.close()


def save_leads(df: pd.DataFrame, path: str = DB_PATH) -> int:
    """Upsert by Google Maps URL. Re-saving a lead never resets its CRM status or contacted_at."""
    if df.empty:
        return 0
    data = df.reindex(columns=COLUMNS).astype(object)
    data = data.where(pd.notnull(data), None)
    data["status"] = data["status"].apply(lambda s: s if s in STATUSES else "New")
    rows = [tuple(r) for r in data.itertuples(index=False, name=None)]

    updatable = [c for c in COLUMNS if c not in ("maps_url", "status", "contacted_at")]
    sql = (
        f"INSERT INTO leads ({','.join(COLUMNS)}) VALUES ({','.join('?' * len(COLUMNS))}) "
        f"ON CONFLICT(maps_url) DO UPDATE SET "
        + ", ".join(f"{c}=excluded.{c}" for c in updatable)
    )
    with _db(path) as conn:
        conn.executemany(sql, rows)
    return len(rows)


def update_status(maps_url: str, status: str, path: str = DB_PATH, now: Optional[datetime] = None) -> Optional[str]:
    """Change a lead's status. When it changes TO 'Contacted', contacted_at is set to the
    current time (a no-op change such as Contacted -> Contacted keeps the original date,
    and moving to another status keeps it as history). Returns the timestamp it set, else None."""
    if status not in STATUSES:
        raise ValueError(f"Invalid status: {status}")
    ts = (now or datetime.now()).isoformat(sep=" ", timespec="seconds")
    with _db(path) as conn:
        row = conn.execute("SELECT status FROM leads WHERE maps_url = ?", (maps_url,)).fetchone()
        if row is not None and row[0] == status:
            return None
        if status == "Contacted":
            conn.execute("UPDATE leads SET status = ?, contacted_at = ? WHERE maps_url = ?", (status, ts, maps_url))
            return ts
        conn.execute("UPDATE leads SET status = ? WHERE maps_url = ?", (status, maps_url))
        return None


def _parse_ts(value) -> Optional[datetime]:
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text or text.lower() in ("nat", "none", "nan"):
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def followup_info(status, contacted_at, now: Optional[datetime] = None,
                  days: int = FOLLOWUP_DAYS) -> Tuple[bool, Optional[int]]:
    """Return (due, whole days since contact). due = status is still 'Contacted' AND the
    contact was `days` or more days ago. days_since is None without a status/date match."""
    if status != "Contacted":
        return False, None
    ts = _parse_ts(contacted_at)
    if ts is None:
        return False, None
    delta = (now or datetime.now()) - ts
    return delta >= timedelta(days=days), max(delta.days, 0)


def backfill_contacted_at(path: str = DB_PATH, now: Optional[datetime] = None) -> Tuple[int, str]:
    """Leads marked 'Contacted' before contacted_at existed have no date, so they could never
    become due. This starts their clock NOW. Returns (rows updated, timestamp used)."""
    ts = (now or datetime.now()).isoformat(sep=" ", timespec="seconds")
    with _db(path) as conn:
        cur = conn.execute(
            "UPDATE leads SET contacted_at = ? "
            "WHERE status = 'Contacted' AND (contacted_at IS NULL OR contacted_at = '')",
            (ts,),
        )
        return cur.rowcount, ts


def load_leads(path: str = DB_PATH) -> pd.DataFrame:
    with _db(path) as conn:
        return pd.read_sql_query("SELECT * FROM leads ORDER BY scraped_at DESC", conn)


def load_pitchless(path: str = DB_PATH) -> pd.DataFrame:
    """Saved leads that have no AI pitch yet (the AI failed or the quota ran out when they were found)."""
    with _db(path) as conn:
        return pd.read_sql_query(
            "SELECT * FROM leads WHERE COALESCE(custom_message_darija, '') = '' ORDER BY scraped_at DESC", conn
        )


# --------------------------------------------------------------------------
# Export (CSV / Excel with clickable WhatsApp links)
# --------------------------------------------------------------------------
EXPORT_COLUMNS = [
    ("priority", "Priority"), ("status", "Status"), ("contacted_at", "Contacted at"),
    ("name", "Name"), ("source", "Source"), ("phone", "Phone"), ("email", "Email"),
    ("wa_pitch", "WhatsApp (Pitch)"), ("wa_followup", "WhatsApp (3-day follow-up)"),
    ("lead_score", "Score"), ("rating", "Rating / Followers"), ("reviews", "Reviews"),
    ("site_active", "Active site"), ("site_type", "Link type"), ("website", "Website"),
    ("address", "Address"), ("bio", "Instagram bio"), ("scoring_reason", "Scoring reason"),
    ("custom_message_darija", "Pitch (Darija)"), ("custom_message_french", "Pitch (French)"),
    ("custom_followup_darija", "Follow-up 3d (Darija)"), ("maps_url", "Google Maps URL"),
    ("keyword", "Keyword"), ("city", "City"), ("scraped_at", "Scraped at"),
]
LINK_LABELS = {
    "WhatsApp (Pitch)": "💬 Open chat with pitch",
    "WhatsApp (3-day follow-up)": "💬 Open chat with follow-up",
}
_WIDE = {"Instagram bio": 45, "Pitch (Darija)": 60, "Pitch (French)": 60, "Follow-up 3d (Darija)": 55,
         "Scoring reason": 40, "Address": 40, "Website": 32, "Google Maps URL": 32,
         "WhatsApp (Pitch)": 26, "WhatsApp (3-day follow-up)": 28, "Priority": 20}
_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def prepare_export(df: pd.DataFrame) -> pd.DataFrame:
    """Flat export frame with fresh prefilled WhatsApp links (rebuilt from the stored
    phone and messages, so rows saved by older versions get the new link format too)."""
    d = df.copy()
    for col in ("phone", "phone_type", "custom_message_darija", "custom_followup_darija", "priority"):
        if col not in d:
            d[col] = ""
        d[col] = d[col].fillna("")
    ptypes = ["mobile" if is_mobile(p) else t for p, t in zip(d["phone"], d["phone_type"])]
    d["wa_pitch"] = [whatsapp_launch_link(p or None, t, m)
                     for p, t, m in zip(d["phone"], ptypes, d["custom_message_darija"])]
    d["wa_followup"] = [whatsapp_launch_link(p or None, t, m)
                        for p, t, m in zip(d["phone"], ptypes, d["custom_followup_darija"])]
    out = pd.DataFrame({label: d[key] if key in d else "" for key, label in EXPORT_COLUMNS})
    return out


def build_excel_bytes(df: pd.DataFrame) -> bytes:
    """Excel file where the WhatsApp columns are real, clickable hyperlinks
    (cell hyperlinks, not HYPERLINK() formulas, which are capped at 255 characters)."""
    out = prepare_export(df)
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda v: _ILLEGAL.sub("", v) if isinstance(v, str) else v)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        out.to_excel(writer, index=False, sheet_name="Leads")
        ws = writer.sheets["Leads"]

        header_fill = PatternFill("solid", fgColor="1F3A5F")
        link_cols = {}
        for idx, cell in enumerate(ws[1], 1):
            label = cell.value
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = header_fill
            cell.alignment = Alignment(vertical="center")
            ws.column_dimensions[cell.column_letter].width = _WIDE.get(label, 16)
            if label in LINK_LABELS:
                link_cols[idx] = label

        wrap = Alignment(wrap_text=True, vertical="top")
        link_font = Font(color="0563C1", underline="single")
        for row in ws.iter_rows(min_row=2):
            for idx, cell in enumerate(row, 1):
                if idx in link_cols:
                    url = cell.value
                    if isinstance(url, str) and url.startswith("http"):
                        cell.hyperlink = url
                        cell.value = LINK_LABELS[link_cols[idx]]
                        cell.font = link_font
                elif ws.cell(row=1, column=idx).value in _WIDE:
                    cell.alignment = wrap
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
    return buf.getvalue()
