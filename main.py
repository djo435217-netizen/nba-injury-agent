import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo
import requests
from twilio.rest import Client

STATE_FILE = "state.json"
ET = ZoneInfo("America/New_York")

# ---------- REQUIRED ENV ----------
TWILIO_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
SPORTRADAR_KEY = os.environ["SPORTRADAR_API_KEY"]

# NEW: BallDontLie API key (GOAT tier is perfect)
BALLDONTLIE_API_KEY = os.environ.get("BALLDONTLIE_API_KEY", "").strip()

FROM_WHATSAPP = os.environ.get("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
TO_WHATSAPP = f"whatsapp:{os.environ['MY_WHATSAPP_NUMBER']}"

twilio = Client(TWILIO_SID, TWILIO_TOKEN)

# ---------- TUNABLE ENV ----------
TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
MAX_BODY_CHARS = 1500
MAX_PLAYERS_PER_TEAM = int(os.environ.get("MAX_PLAYERS_PER_TEAM", "50"))

# Impact alerts
IMPACT_STATUSES_RAW = os.environ.get("IMPACT_STATUSES", "out,doubtful,questionable")
IMPACT_STATUSES = {x.strip().lower() for x in IMPACT_STATUSES_RAW.split(",") if x.strip()}
IMPACT_ONLY_CHANGES = os.environ.get("IMPACT_ONLY_CHANGES", "1") == "1"

# Summaries
SUMMARY_TIMES_ET_RAW = os.environ.get("SUMMARY_TIMES_ET", "10:00,17:00").strip()
SUMMARY_TIMES_ET = [t.strip() for t in SUMMARY_TIMES_ET_RAW.split(",") if t.strip()]
SUMMARY_WINDOW_START_ET = int(os.environ.get("SUMMARY_WINDOW_START_ET", "0"))
SUMMARY_WINDOW_END_ET = int(os.environ.get("SUMMARY_WINDOW_END_ET", "24"))
FORCE_FULL = os.environ.get("FORCE_FULL", "0") == "1"

# Burst window
BURST_START_ET = os.environ.get("BURST_START_ET", "17:00").strip()
BURST_END_ET = os.environ.get("BURST_END_ET", "22:30").strip()
BURST_FULL_SUMMARY_EVERY_MIN = int(os.environ.get("BURST_FULL_SUMMARY_EVERY_MIN", "60"))

# Auto beneficiaries toggle (default ON if key exists)
AUTO_BENEFICIARIES = os.environ.get("AUTO_BENEFICIARIES", "1") == "1"
AUTO_BENEFICIARIES_TOPN = int(os.environ.get("AUTO_BENEFICIARIES_TOPN", "3"))
AUTO_BENEFICIARIES_GAMES = int(os.environ.get("AUTO_BENEFICIARIES_GAMES", "10"))

# Weighting for ranking (points-heavy since you bet points)
BEN_PTS_W = float(os.environ.get("BEN_PTS_W", "1.0"))
BEN_MIN_W = float(os.environ.get("BEN_MIN_W", "0.15"))

# Safety caps (avoid runaway API calls)
BDL_MAX_PAGES = int(os.environ.get("BDL_MAX_PAGES", "8"))
BDL_PER_PAGE = int(os.environ.get("BDL_PER_PAGE", "100"))  # max 100 per docs


# ---------- HELPERS ----------
def _now_et():
    return datetime.now(ET)


def _time_to_minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _in_burst_window(now_et: datetime) -> bool:
    start = _time_to_minutes(BURST_START_ET)
    end = _time_to_minutes(BURST_END_ET)
    cur = now_et.hour * 60 + now_et.minute
    return start <= cur <= end


def _season_year_for_ball_dont_lie(now_et: datetime) -> int:
    # NBA season year convention: Oct -> next spring, so Feb 2026 is season 2025
    return now_et.year if now_et.month >= 10 else now_et.year - 1


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


def fetch_sportradar_injuries():
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
        # Sportradar team usually has "name" like "Knicks"
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


def status_in_scope(status: str) -> bool:
    return (status or "").strip().lower() in IMPACT_STATUSES


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

    key = "last_burst_summary_et"
    last = meta.get(key)
    cur_ts = int(now_et.timestamp())
    if not last:
        return True
    return (cur_ts - int(last)) >= (BURST_FULL_SUMMARY_EVERY_MIN * 60)


def mark_burst_summary_sent(meta, now_et: datetime):
    meta["last_burst_summary_et"] = int(now_et.timestamp())


def action_tag(prev_status: str | None, cur_status: str, now_et: datetime) -> str:
    cs = (cur_status or "").strip().lower()
    ps = (prev_status or "").strip().lower() if prev_status else ""

    if ps == "questionable" and cs == "out":
        return "HIT NOW (Q→OUT)"
    if cs in {"out", "doubtful"} and _in_burst_window(now_et):
        return "HIT NOW"
    if not prev_status and cs == "questionable":
        return "WATCH"
    if cs == "questionable":
        return "WATCH"
    if cs == "doubtful":
        return "LEAN (monitor)"
    if cs == "out":
        return "ACTION"
    return ""


# ---------- BALLDONTLIE (AUTO BENEFICIARIES) ----------
_BDL_TEAMS_CACHE = None  # name->id
_BDL_HEADERS = None


def _bdl_headers():
    global _BDL_HEADERS
    if _BDL_HEADERS is None:
        _BDL_HEADERS = {"Authorization": BALLDONTLIE_API_KEY}
    return _BDL_HEADERS


def _bdl_get(path: str, params: dict | None = None, timeout: int = 20) -> dict:
    url = f"https://api.balldontlie.io{path}"
    r = requests.get(url, headers=_bdl_headers(), params=params or {}, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"BallDontLie error {r.status_code}: {r.text[:300]}")
    # BallDontLie returns JSON
    return r.json()


def _bdl_load_team_name_to_id():
    global _BDL_TEAMS_CACHE
    if _BDL_TEAMS_CACHE is not None:
        return _BDL_TEAMS_CACHE

    # docs show /v1/teams list endpoint
    data = _bdl_get("/v1/teams", params={"per_page": 100})
    teams = data.get("data") or []
    m = {}
    for t in teams:
        # key by short name like "Knicks"
        name = (t.get("name") or "").strip()
        if name:
            m[name] = int(t["id"])
    _BDL_TEAMS_CACHE = m
    return _BDL_TEAMS_CACHE


def _parse_minutes_to_float(min_str: str | None) -> float:
    # min can be "30" or "30:12" depending on feed; handle both
    if not min_str:
        return 0.0
    s = str(min_str)
    if ":" in s:
        try:
            mm, ss = s.split(":", 1)
            return float(mm) + float(ss) / 60.0
        except Exception:
            return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0


def _auto_boost_candidates(team_short_name: str, exclude_names_lower: set[str], now_et: datetime) -> list[str]:
    """
    Returns top N teammates ranked by last N games points + minutes.
    exclude_names_lower: set of lower-cased full names to exclude (injured players, etc.)
    """
    if not (AUTO_BENEFICIARIES and BALLDONTLIE_API_KEY):
        return []

    name_to_id = _bdl_load_team_name_to_id()
    team_id = name_to_id.get(team_short_name)
    if not team_id:
        return []

    # Active players for team
    # docs show /v1/players/active supports team_ids[] filters
    players = []
    cursor = None
    pages = 0

    while pages < 5:  # players list is small; keep it tight
        params = {"per_page": 100, "team_ids[]": team_id}
        if cursor is not None:
            params["cursor"] = cursor
        resp = _bdl_get("/v1/players/active", params=params)
        chunk = resp.get("data") or []
        players.extend(chunk)
        meta = resp.get("meta") or {}
        cursor = meta.get("next_cursor")
        pages += 1
        if not cursor:
            break

    # Build list of (player_id, full_name)
    roster = []
    for p in players:
        pid = p.get("id")
        fn = (p.get("first_name") or "").strip()
        ln = (p.get("last_name") or "").strip()
        full = (fn + " " + ln).strip()
        if not pid or not full:
            continue
        if full.lower() in exclude_names_lower:
            continue
        roster.append((int(pid), full))

    if not roster:
        return []

    # Pull stats in bulk for this roster for current season
    season_year = _season_year_for_ball_dont_lie(now_et)
    player_ids = [pid for pid, _ in roster]

    # We paginate /v1/stats until each player has enough games or pages cap reached
    games_by_player = {pid: [] for pid in player_ids}
    cursor = None
    pages = 0

    # NOTE: /v1/stats supports player_ids[] and seasons[] per docs
    while pages < BDL_MAX_PAGES:
        params = {
            "per_page": min(BDL_PER_PAGE, 100),
            "seasons[]": season_year,
        }
        # add player_ids[] list
        for pid in player_ids:
            params.setdefault("player_ids[]", [])
            params["player_ids[]"].append(pid)

        if cursor is not None:
            params["cursor"] = cursor

        resp = _bdl_get("/v1/stats", params=params)
        rows = resp.get("data") or []

        # Collect
        for row in rows:
            player_obj = row.get("player") or {}
            pid = player_obj.get("id")
            if not pid:
                continue
            pid = int(pid)
            if pid not in games_by_player:
                continue

            game = row.get("game") or {}
            date = game.get("date")  # "YYYY-MM-DD"
            pts = row.get("pts", 0) or 0
            mins = _parse_minutes_to_float(row.get("min"))

            if date:
                games_by_player[pid].append((date, float(pts), float(mins)))

        # Stop early if everyone has enough games
        done = True
        for pid in player_ids:
            if len(games_by_player[pid]) < AUTO_BENEFICIARIES_GAMES:
                done = False
                break
        if done:
            break

        meta = resp.get("meta") or {}
        cursor = meta.get("next_cursor")
        pages += 1
        if not cursor:
            break

    # Score each player by last N games
    scored = []
    for pid, full in roster:
        games = games_by_player.get(pid, [])
        if not games:
            continue
        games.sort(key=lambda x: x[0])  # sort by date
        last = games[-AUTO_BENEFICIARIES_GAMES:]
        if not last:
            continue

        pts_avg = sum(x[1] for x in last) / len(last)
        min_avg = sum(x[2] for x in last) / len(last)
        score = (BEN_PTS_W * pts_avg) + (BEN_MIN_W * min_avg)
        scored.append((score, pts_avg, min_avg, full))

    scored.sort(reverse=True, key=lambda x: x[0])
    top = [x[3] for x in scored[:AUTO_BENEFICIARIES_TOPN]]
    return top


# ---------- ALERT BUILD ----------
def build_impact_alerts(old_players, new_players, now_et: datetime):
    """
    League-wide betting-signal alerts + auto beneficiaries.
    """
    lines = []

    # Build exclusion list from current injury report (anyone in new_players)
    exclude_names_lower = {v.get("name", "").strip().lower() for v in new_players.values() if v.get("name")}

    def beneficiaries_text(team_short_name: str) -> str:
        try:
            picks = _auto_boost_candidates(team_short_name, exclude_names_lower, now_et)
            if not picks:
                return ""
            # Keep it tight for WhatsApp
            return " | Boost candidates: " + ", ".join(picks[:AUTO_BENEFICIARIES_TOPN])
        except Exception:
            # Never fail the whole run because BDL hiccuped
            return ""

    # NEW/CHANGED
    for pid, cur in new_players.items():
        if not status_in_scope(cur.get("status", "")):
            continue

        prev = old_players.get(pid)

        if IMPACT_ONLY_CHANGES:
            is_new = prev is None
            is_changed = (not is_new) and (
                (prev.get("status"), prev.get("detail")) != (cur.get("status"), cur.get("detail"))
            )
            if not (is_new or is_changed):
                continue

        tag = action_tag(prev.get("status") if prev else None, cur.get("status", ""), now_et)
        tag_txt = f"[{tag}] " if tag else ""
        ben_txt = beneficiaries_text(cur.get("team", ""))

        if prev is None:
            base = f"{tag_txt}{cur['name']} ({cur['team']}): {cur['status']}"
            if cur.get("detail"):
                base += f" — {cur['detail']}"
            lines.append("➕ " + base + ben_txt)
        else:
            msg = f"🚨 {tag_txt}{cur['name']} ({cur['team']}): {prev.get('status')} → {cur.get('status')}"
            if cur.get("detail"):
                msg += f" — {cur.get('detail')}"
            lines.append(msg + ben_txt)

    # CLEARED
    if IMPACT_ONLY_CHANGES:
        for pid, prev in old_players.items():
            if pid not in new_players and status_in_scope(prev.get("status", "")):
                lines.append(f"✅ {prev.get('name')} ({prev.get('team')}): cleared (was {prev.get('status')})")

    if not lines:
        return None

    header = "🎯 Impact Alerts (player points)"
    if AUTO_BENEFICIARIES and not BALLDONTLIE_API_KEY:
        header += "\n(Boost candidates disabled: missing BALLDONTLIE_API_KEY)"
    return header + "\n" + "\n".join(lines[:70])


def run():
    now_et = _now_et()
    ts_et = now_et.strftime("%Y-%m-%d %I:%M %p ET")

    if TEST_MODE:
        send_one(f"✅ NBA Injury Agent test: WhatsApp delivery is working. ({ts_et})")
        return

    state = load_state()
    meta = state.get("__meta__", {})
    old_players = state.get("players", {})

    data = fetch_sportradar_injuries()
    team_order, injuries_by_team, new_players = parse_teams_and_injuries(data)

    # 1) Impact alerts w/ auto beneficiaries
    impact_msg = build_impact_alerts(old_players, new_players, now_et)
    if impact_msg:
        send_chunked(f"{impact_msg}\n({ts_et})")

    # 2) Daily full summaries
    if should_send_full_summary(meta, now_et):
        full_report = f"NBA Full Injury Summary ({ts_et})\n\n" + format_team_summary(team_order, injuries_by_team)
        send_chunked(full_report)
        mark_full_summary_sent(meta, now_et)

    # 3) Burst window summaries
    if should_send_burst_summary(meta, now_et):
        burst_report = f"⏱ Pre-tip Summary ({ts_et})\n\n" + format_team_summary(team_order, injuries_by_team)
        send_chunked(burst_report)
        mark_burst_summary_sent(meta, now_et)

    state["__meta__"] = meta
    state["players"] = new_players
    save_state(state)


if __name__ == "__main__":
    run()
