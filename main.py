import os
import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import requests
from twilio.rest import Client

STATE_FILE = "state.json"
ET = ZoneInfo("America/New_York")

# ---------- ENV / CONFIG ----------
TWILIO_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
SPORTRADAR_KEY = os.environ["SPORTRADAR_API_KEY"]

FROM_WHATSAPP = os.environ.get("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
TO_WHATSAPP = f"whatsapp:{os.environ['MY_WHATSAPP_NUMBER']}"

TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"

MAX_BODY_CHARS = 1500
MAX_PLAYERS_PER_TEAM = int(os.environ.get("MAX_PLAYERS_PER_TEAM", "50"))

# ---- Impact Alerts (player props) ----
IMPACT_STATUSES_RAW = os.environ.get("IMPACT_STATUSES", "out,doubtful,questionable")
IMPACT_STATUSES = {x.strip().lower() for x in IMPACT_STATUSES_RAW.split(",") if x.strip()}
IMPACT_ONLY_CHANGES = os.environ.get("IMPACT_ONLY_CHANGES", "1") == "1"

# ---- Daily full summaries (spam control) ----
SUMMARY_TIMES_ET_RAW = os.environ.get("SUMMARY_TIMES_ET", "10:00,17:00").strip()
SUMMARY_TIMES_ET = [t.strip() for t in SUMMARY_TIMES_ET_RAW.split(",") if t.strip()]
SUMMARY_WINDOW_START_ET = int(os.environ.get("SUMMARY_WINDOW_START_ET", "0"))
SUMMARY_WINDOW_END_ET = int(os.environ.get("SUMMARY_WINDOW_END_ET", "24"))
FORCE_FULL = os.environ.get("FORCE_FULL", "0") == "1"

# ---- Pre-tip burst window ----
BURST_START_ET = os.environ.get("BURST_START_ET", "17:00").strip()   # 5:00 PM
BURST_END_ET = os.environ.get("BURST_END_ET", "22:30").strip()       # 10:30 PM
BURST_FULL_SUMMARY_EVERY_MIN = int(os.environ.get("BURST_FULL_SUMMARY_EVERY_MIN", "60"))
# If set to 0, burst won't send extra full summaries (only impact alerts)

twilio = Client(TWILIO_SID, TWILIO_TOKEN)


def _now_et():
    return datetime.now(ET)


def load_state():
    if not os.path.exists(STATE_FILE):
        return {"__meta__": {}, "players": {}}
    try:
        with open(STATE_FILE, "r") as f:
            raw = json.load(f)
        if isinstance(raw, dict) and "players" in raw:
            raw.setdefault("__meta__", {})
            return raw
        if isinstance(raw, dict):
            return {"__meta__": {}, "players": raw}
        return {"__meta__": {}, "players": {}}
    except Exception:
        return {"__meta__": {}, "players": {}}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def send_one(body: str):
    twilio.messages.create(from_=FROM_WHATSAPP, to=TO_WHATSAPP, body=body[:MAX_BODY_CHARS])


def send_chunked(full_text: str):
    if len(full_text) <= MAX_BODY_CHARS:
        send_one(full_text)
        return

    parts = []
    remaining = full_text

    while len(remaining) > MAX_BODY_CHARS:
        cut = remaining.rfind("\n", 0, MAX_BODY_CHARS)
        if cut < 200:
            cut = MAX_BODY_CHARS
        parts.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()

    if remaining:
        parts.append(remaining)

    total = len(parts)
    for i, p in enumerate(parts, start=1):
        header = f"(Part {i}/{total})\n"
        if len(header) + len(p) > MAX_BODY_CHARS:
            p = p[: MAX_BODY_CHARS - len(header)]
        send_one(header + p)


def fetch():
    url = "https://api.sportradar.com/nba/trial/v8/en/league/injuries.json"
    r = requests.get(url, params={"api_key": SPORTRADAR_KEY}, timeout=20)

    if r.status_code != 200:
        raise RuntimeError(f"Sportradar error {r.status_code}: {r.text[:300]}")

    content_type = (r.headers.get("Content-Type") or "").lower()
    if "json" not in content_type:
        raise RuntimeError(f"Unexpected content-type: {content_type}. Body: {r.text[:300]}")

    return r.json()


def parse_teams_and_injuries(data):
    injuries_by_team = {}
    flat_by_player = {}

    for team in data.get("teams", []):
        team_name = team.get("name") or team.get("market") or team.get("id", "TEAM")
        injuries_by_team.setdefault(team_name, [])

        for p in team.get("players", []):
            injuries = p.get("injuries") or []
            if not injuries:
                continue

            inj = injuries[-1]
            pid = p.get("id")
            if not pid:
                continue

            name = p.get("full_name") or f"{p.get('first_name','')} {p.get('last_name','')}".strip()
            status = (inj.get("status") or "Unknown").strip()
            detail = (inj.get("comment") or inj.get("description") or "").strip()

            player_obj = {"id": pid, "name": name, "team": team_name, "status": status, "detail": detail}
            injuries_by_team[team_name].append(player_obj)
            flat_by_player[pid] = {"name": name, "team": team_name, "status": status, "detail": detail}

    team_order = sorted(injuries_by_team.keys())
    for t in team_order:
        injuries_by_team[t].sort(key=lambda x: (x["name"] or ""))

    return team_order, injuries_by_team, flat_by_player


def status_in_scope(status: str) -> bool:
    return (status or "").strip().lower() in IMPACT_STATUSES


def build_impact_alerts(old_players, new_players):
    lines = []

    for pid, cur in new_players.items():
        if not status_in_scope(cur.get("status", "")):
            continue

        prev = old_players.get(pid)
        if IMPACT_ONLY_CHANGES:
            if prev is None:
                lines.append(
                    f"➕ {cur['name']} ({cur['team']}): {cur['status']}"
                    + (f" — {cur['detail']}" if cur.get("detail") else "")
                )
            else:
                if (prev.get("status"), prev.get("detail")) != (cur.get("status"), cur.get("detail")):
                    lines.append(
                        f"🚨 {cur['name']} ({cur['team']}): {prev.get('status')} → {cur.get('status')}"
                        + (f" — {cur.get('detail')}" if cur.get("detail") else "")
                    )
        else:
            lines.append(
                f"• {cur['name']} ({cur['team']}): {cur['status']}"
                + (f" — {cur['detail']}" if cur.get("detail") else "")
            )

    if IMPACT_ONLY_CHANGES:
        for pid, prev in old_players.items():
            if pid not in new_players and status_in_scope(prev.get("status", "")):
                lines.append(f"✅ {prev.get('name')} ({prev.get('team')}): cleared (was {prev.get('status')})")

    if not lines:
        return None

    return "🎯 Impact Alerts (player props)\n" + "\n".join(lines[:70])


def format_team_summary(team_order, injuries_by_team):
    lines = ["🏥 Full injury report (by team):"]
    for team in team_order:
        players = injuries_by_team.get(team, [])
        if not players:
            lines.append(f"\n{team}: No reported injuries")
            continue

        lines.append(f"\n{team}:")
        for p in players[:MAX_PLAYERS_PER_TEAM]:
            line = f"- {p['name']}: {p['status']}"
            if p.get("detail"):
                line += f" — {p['detail']}"
            lines.append(line)

        if len(players) > MAX_PLAYERS_PER_TEAM:
            lines.append(f"- …and {len(players) - MAX_PLAYERS_PER_TEAM} more")

    return "\n".join(lines)


def _time_to_minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _in_burst_window(now_et: datetime) -> bool:
    start = _time_to_minutes(BURST_START_ET)
    end = _time_to_minutes(BURST_END_ET)
    cur = now_et.hour * 60 + now_et.minute
    return start <= cur <= end


def should_send_full_summary(meta, now_et: datetime) -> bool:
    if FORCE_FULL:
        return True

    hour = now_et.hour
    if not (SUMMARY_WINDOW_START_ET <= hour < SUMMARY_WINDOW_END_ET):
        return False

    hhmm = now_et.strftime("%H:%M")
    if hhmm not in SUMMARY_TIMES_ET:
        return False

    sent = meta.get("sent_summaries", {})
    today = now_et.strftime("%Y-%m-%d")
    already = set(sent.get(today, []))
    return hhmm not in already


def mark_full_summary_sent(meta, now_et: datetime):
    hhmm = now_et.strftime("%H:%M")
    today = now_et.strftime("%Y-%m-%d")
    meta.setdefault("sent_summaries", {})
    meta["sent_summaries"].setdefault(today, [])
    if hhmm not in meta["sent_summaries"][today]:
        meta["sent_summaries"][today].append(hhmm)

    keys = sorted(meta["sent_summaries"].keys())
    if len(keys) > 7:
        for k in keys[:-7]:
            meta["sent_summaries"].pop(k, None)


def should_send_burst_summary(meta, now_et: datetime) -> bool:
    if BURST_FULL_SUMMARY_EVERY_MIN <= 0:
        return False
    if not _in_burst_window(now_et):
        return False

    # Send at most once per BURST_FULL_SUMMARY_EVERY_MIN
    key = "last_burst_summary_et"
    last = meta.get(key)
    cur_ts = int(now_et.timestamp())
    if not last:
        return True
    return (cur_ts - int(last)) >= (BURST_FULL_SUMMARY_EVERY_MIN * 60)


def mark_burst_summary_sent(meta, now_et: datetime):
    meta["last_burst_summary_et"] = int(now_et.timestamp())


def run():
    now_et = _now_et()
    ts_et = now_et.strftime("%Y-%m-%d %I:%M %p ET")

    if TEST_MODE:
        send_one(f"✅ NBA Injury Agent test: WhatsApp delivery is working. ({ts_et})")
        return

    state = load_state()
    meta = state.get("__meta__", {})
    old_players = state.get("players", {})

    data = fetch()
    team_order, injuries_by_team, new_players = parse_teams_and_injuries(data)

    # 1) Impact alerts (changes-only)
    impact_msg = build_impact_alerts(old_players, new_players)
    if impact_msg:
        send_chunked(f"{impact_msg}\n({ts_et})")

    # 2) Daily summaries at scheduled times
    if should_send_full_summary(meta, now_et):
        full_report = f"NBA Full Injury Summary ({ts_et})\n\n" + format_team_summary(team_order, injuries_by_team)
        send_chunked(full_report)
        mark_full_summary_sent(meta, now_et)

    # 3) Burst window summaries (optional extra)
    if should_send_burst_summary(meta, now_et):
        burst_report = f"⏱ Pre-tip Summary ({ts_et})\n\n" + format_team_summary(team_order, injuries_by_team)
        send_chunked(burst_report)
        mark_burst_summary_sent(meta, now_et)

    state["__meta__"] = meta
    state["players"] = new_players
    save_state(state)


if __name__ == "__main__":
    run()
