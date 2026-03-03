import os
import json
import re
import time
import math
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from twilio.rest import Client

# =============================================================================
# NBA Props Agent (Points + 3PT Made)  — Injury-triggered + League-wide scan
# - Uses Sportradar injuries as "triggers"
# - Uses BallDontLie odds (/v2/odds/player_props) for prop lines
# - Uses BallDontLie stats (/v1/stats) for recent production + minutes
# - Supports multiple prop types (points, three_pointers_made) with alias fallback
# - Cooldown / resend guardrails to reduce repeats
# =============================================================================

STATE_FILE = "state.json"
ET = ZoneInfo("America/New_York")

# -------------------- REQUIRED ENV --------------------
TWILIO_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]

SPORTRADAR_KEY = os.environ["SPORTRADAR_API_KEY"]
BALLDONTLIE_API_KEY = os.environ["BALLDONTLIE_API_KEY"].strip()

FROM_WHATSAPP = os.environ.get("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
TO_WHATSAPP = f"whatsapp:{os.environ['MY_WHATSAPP_NUMBER']}"

twilio = Client(TWILIO_SID, TWILIO_TOKEN)

# -------------------- CONFIG (ENV) --------------------
TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
MAX_BODY_CHARS = 1500

IMPACT_STATUSES_RAW = os.environ.get("IMPACT_STATUSES", "out,doubtful,questionable").strip()
IMPACT_STATUSES = {x.strip().lower() for x in IMPACT_STATUSES_RAW.split(",") if x.strip()}
IMPACT_ONLY_CHANGES = os.environ.get("IMPACT_ONLY_CHANGES", "1") == "1"

# Vendors: allow comma-separated (we try each and also fallback to None)
BOOK_VENDOR_RAW = os.environ.get("BOOK_VENDORS", os.environ.get("BOOK_VENDOR", "fanduel")).strip().lower()
BOOK_VENDORS = [v.strip() for v in BOOK_VENDOR_RAW.split(",") if v.strip()]
if not BOOK_VENDORS:
    BOOK_VENDORS = ["fanduel"]

# Markets (prop types) we will compute / send
PROP_TYPES_RAW = os.environ.get("PROP_TYPES", "points,three_pointers_made").strip().lower()
PROP_TYPES = [p.strip() for p in PROP_TYPES_RAW.split(",") if p.strip()]

# For threes, BallDontLie sometimes uses different prop_type strings.
# We'll try these in order until we see rows.
THREES_PROP_ALIASES_RAW = os.environ.get(
    "THREES_PROP_ALIASES",
    "three_pointers_made,threes,fg3m,3pt_made,three_points_made,three_point_field_goals_made",
).strip().lower()
THREES_PROP_ALIASES = [x.strip() for x in THREES_PROP_ALIASES_RAW.split(",") if x.strip()]

# Multi-horizon windows
BASELINE_GAMES = int(os.environ.get("BASELINE_GAMES", "30"))
LOOKBACK_GAMES = int(os.environ.get("LOOKBACK_GAMES", "10"))
SHORT_GAMES = int(os.environ.get("SHORT_GAMES", "3"))

# Projection blend weights
W_BASE = float(os.environ.get("W_BASE", "0.45"))
W_L10 = float(os.environ.get("W_L10", "0.35"))
W_L3 = float(os.environ.get("W_L3", "0.10"))
W_LINE = float(os.environ.get("W_LINE", "0.10"))

# Output sizing (global + per-market)
MIN_PER_MARKET = int(os.environ.get("MIN_PER_MARKET", "2"))
MAX_PER_MARKET = int(os.environ.get("MAX_PER_MARKET", "6"))
MAX_BET_IDEAS = int(os.environ.get("MAX_BET_IDEAS", "10"))

# Default thresholds (points)
MIN_EDGE = float(os.environ.get("MIN_EDGE", "2.5"))
MIN_PROB = float(os.environ.get("MIN_PROB", "0.62"))

# Optional separate thresholds for threes (recommended lower)
MIN_EDGE_THREES = float(os.environ.get("MIN_EDGE_THREES", "0.7"))
MIN_PROB_THREES = float(os.environ.get("MIN_PROB_THREES", "0.56"))

STD_FLOOR = float(os.environ.get("STD_FLOOR", "5.0"))

# Guardrails
MIN_POINTS_LINE = float(os.environ.get("MIN_POINTS_LINE", "6.0"))
MAX_POINTS_LINE = float(os.environ.get("MAX_POINTS_LINE", "45.0"))
LINE_MIN_GAP = float(os.environ.get("LINE_MIN_GAP", "8.0"))
MIN_DELTA_FLOOR = float(os.environ.get("MIN_DELTA_FLOOR", "-3.0"))

# Injury vacancy requirements (for "trigger strength" / quality)
MIN_VAC_MIN = float(os.environ.get("MIN_VAC_MIN", "10.0"))
MIN_VAC_PTS = float(os.environ.get("MIN_VAC_PTS", "6.0"))

# Injury boost caps
BOOST_CAP_PTS = float(os.environ.get("BOOST_CAP_PTS", "5.5"))
BOOST_CAP_MIN = float(os.environ.get("BOOST_CAP_MIN", "6.0"))

# Burst window + pings
SEND_NO_EDGE_PING = os.environ.get("SEND_NO_EDGE_PING", "0") == "1"
BURST_START_ET = os.environ.get("BURST_START_ET", "17:00").strip()
BURST_END_ET = os.environ.get("BURST_END_ET", "23:45").strip()

# Slate Scan toggles
ENABLE_SLATE_SCAN = os.environ.get("ENABLE_SLATE_SCAN", "1") == "1"
SLATE_ONLY_IN_BURST = os.environ.get("SLATE_ONLY_IN_BURST", "0") == "1"
SLATE_SCAN_MAX_PLAYERS = int(os.environ.get("SLATE_SCAN_MAX_PLAYERS", "260"))

# Injury strict game matching (if 1, only generate injury edges for players that have a line in a "relevant" game)
STRICT_INJURY_GAME_MATCH = os.environ.get("STRICT_INJURY_GAME_MATCH", "1") == "1"

# Cooldown / resend rules
BET_COOLDOWN_MIN = int(os.environ.get("BET_COOLDOWN_MIN", "180"))
EDGE_JUMP_TO_RESEND = float(os.environ.get("EDGE_JUMP_TO_RESEND", "1.5"))

# Debug: print one sample prop row per type (comma-separated)
DEBUG_PROP_SAMPLE_TYPES_RAW = os.environ.get("DEBUG_PROP_SAMPLE_TYPES", "")
DEBUG_PROP_SAMPLE_TYPES = {x.strip().lower() for x in DEBUG_PROP_SAMPLE_TYPES_RAW.split(",") if x.strip()}
DEBUG_PRINTED_FOR = set()

# -------------------- UTILS --------------------
def _now_et() -> datetime:
    return datetime.now(ET)

def _season_year(now_et: datetime) -> int:
    return now_et.year if now_et.month >= 10 else now_et.year - 1

def _time_to_minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)

def _in_burst_window(now_et: datetime) -> bool:
    start = _time_to_minutes(BURST_START_ET)
    end = _time_to_minutes(BURST_END_ET)
    cur = now_et.hour * 60 + now_et.minute
    return start <= cur <= end

def _parse_minutes(min_str) -> float:
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

def _clean_name(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\.", "", s)
    s = re.sub(r"\s+", " ", s)
    s = s.replace("’", "'")
    return s.lower()

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _slice_last(games, n):
    if not games:
        return []
    return games[-min(len(games), n):]

def _role_trend(games):
    if not games:
        return 0.0, 0.0, 0.0, 0.0
    long_slice = _slice_last(games, LOOKBACK_GAMES)
    short_slice = _slice_last(games, SHORT_GAMES)

    stat_l, min_l, _ = avg_stat_min_std(long_slice)
    stat_s, min_s, _ = avg_stat_min_std(short_slice)
    rate_l = stat_l / max(min_l, 1e-6)
    rate_s = stat_s / max(min_s, 1e-6)
    return min_s, min_l, rate_s, rate_l

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"players": {}, "sent_bets": {}}
    try:
        with open(STATE_FILE, "r") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return {"players": {}, "sent_bets": {}}
        raw.setdefault("players", {})
        raw.setdefault("sent_bets", {})
        return raw
    except Exception:
        return {"players": {}, "sent_bets": {}}

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

def status_in_scope(status: str) -> bool:
    return (status or "").strip().lower() in IMPACT_STATUSES

# -------------------- MARKET CONFIG --------------------
def market_label(prop_type: str) -> str:
    if prop_type == "points":
        return "Points"
    if prop_type in ("three_pointers_made", "threes", "fg3m", "3pt_made"):
        return "3PT Made"
    return prop_type

def thresholds_for_market(prop_type: str) -> tuple[float, float]:
    # return (min_edge, min_prob)
    if prop_type == "three_pointers_made":
        return (MIN_EDGE_THREES, MIN_PROB_THREES)
    return (MIN_EDGE, MIN_PROB)

def line_range_for_market(prop_type: str) -> tuple[float, float]:
    if prop_type == "points":
        return (MIN_POINTS_LINE, MAX_POINTS_LINE)
    if prop_type == "three_pointers_made":
        # keep sane bounds
        return (0.5, 8.5)
    return (MIN_POINTS_LINE, MAX_POINTS_LINE)

def stat_key_candidates_for_market(prop_type: str) -> list[str]:
    # BallDontLie stats fields: points uses "pts".
    # 3PT made is usually "fg3m" (and sometimes "fg3m" only).
    if prop_type == "points":
        return ["pts"]
    if prop_type == "three_pointers_made":
        return ["fg3m", "fg3m"]  # keep duplicates harmless; just in case
    return ["pts"]

def prop_type_aliases(prop_type: str) -> list[str]:
    if prop_type == "three_pointers_made":
        return THREES_PROP_ALIASES or ["three_pointers_made"]
    return [prop_type]

# -------------------- SPORTRADAR --------------------
def fetch_sportradar_injuries():
    url = "https://api.sportradar.com/nba/trial/v8/en/league/injuries.json"
    r = requests.get(url, params={"api_key": SPORTRADAR_KEY}, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"Sportradar error {r.status_code}: {r.text[:300]}")
    ct = (r.headers.get("Content-Type") or "").lower()
    if "json" not in ct:
        raise RuntimeError(f"Unexpected content-type: {ct}. Body: {r.text[:300]}")
    return r.json()

def parse_injuries(data):
    flat_by_player = {}
    for team in data.get("teams", []):
        team_name = team.get("name") or team.get("market") or team.get("id", "TEAM")
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
            flat_by_player[pid] = {"name": name, "team": team_name, "status": status, "detail": detail}
    return flat_by_player

# -------------------- BALLDONTLIE --------------------
BDL_HEADERS = {"Authorization": BALLDONTLIE_API_KEY}
BDL_PREFIXES = ["/nba", ""]

BDL_MAX_RETRIES = int(os.environ.get("BDL_MAX_RETRIES", "5"))
BDL_RETRY_BASE_SEC = float(os.environ.get("BDL_RETRY_BASE_SEC", "1.5"))
BDL_PER_PAGE = int(os.environ.get("BDL_PER_PAGE", "100"))
BDL_MAX_PAGES = int(os.environ.get("BDL_MAX_PAGES", "10"))

TEAM_CACHE = None
PROPS_CACHE = {}  # (gid, vendor, prop_type_alias) -> rows
PLAYER_NAME_CACHE = {}  # pid -> "First Last"

def _bdl_get(path: str, params=None, timeout: int = 20) -> dict:
    last_err = None
    for pref in BDL_PREFIXES:
        url = f"https://api.balldontlie.io{pref}{path}"
        for attempt in range(BDL_MAX_RETRIES):
            try:
                r = requests.get(url, headers=BDL_HEADERS, params=params or {}, timeout=timeout)
                if r.status_code == 404:
                    last_err = f"404 {url}"
                    break
                if r.status_code in (429, 500, 502, 503, 504):
                    retry_after = r.headers.get("Retry-After")
                    sleep_s = float(retry_after) if retry_after else BDL_RETRY_BASE_SEC * (2 ** attempt)
                    last_err = f"{r.status_code} {r.text[:160]}"
                    time.sleep(min(sleep_s, 30.0))
                    continue
                if r.status_code != 200:
                    raise RuntimeError(f"BallDontLie error {r.status_code}: {r.text[:300]}")
                return r.json()
            except Exception as e:
                last_err = str(e)
                time.sleep(min(BDL_RETRY_BASE_SEC * (2 ** attempt), 30.0))
                continue
    raise RuntimeError(f"BallDontLie request failed for {path}. Last error: {last_err}")

def bdl_games_today_ids(now_et: datetime):
    today = now_et.strftime("%Y-%m-%d")
    resp = _bdl_get("/v1/games", params={"dates[]": [today], "per_page": 100})
    return [int(g["id"]) for g in (resp.get("data") or []) if g.get("id") is not None]

def bdl_team_name_to_id():
    global TEAM_CACHE
    if TEAM_CACHE is not None:
        return TEAM_CACHE
    data = _bdl_get("/v1/teams", params={"per_page": 100})
    m = {}
    for t in data.get("data", []):
        nm = (t.get("name") or "").strip()
        if nm and t.get("id") is not None:
            m[nm] = int(t["id"])
    TEAM_CACHE = m
    return TEAM_CACHE

def bdl_active_roster(team_short: str):
    team_map = bdl_team_name_to_id()
    team_id = team_map.get(team_short)
    if not team_id:
        return []
    players = []
    cursor = None
    pages = 0
    while pages < 5:
        params = {"per_page": 100, "team_ids[]": [team_id]}
        if cursor is not None:
            params["cursor"] = cursor
        resp = _bdl_get("/v1/players/active", params=params)
        players.extend(resp.get("data") or [])
        cursor = (resp.get("meta") or {}).get("next_cursor")
        pages += 1
        if not cursor:
            break
    out = []
    for p in players:
        team = p.get("team") or {}
        if (team.get("name") or "").strip() == team_short:
            out.append(p)
    return out

def bdl_find_player_id_on_team(team_short: str, full_name: str):
    roster = bdl_active_roster(team_short)
    if not roster:
        return None

    def strip_suffix(n: str) -> str:
        n = _clean_name(n)
        n = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", n).strip()
        n = re.sub(r"\s+", " ", n)
        return n

    t0 = strip_suffix(full_name)
    for p in roster:
        pid = p.get("id")
        nm = f"{p.get('first_name','')} {p.get('last_name','')}".strip()
        if pid and nm and strip_suffix(nm) == t0:
            return int(pid)
    return None

def _stat_from_row(row: dict, prop_type: str) -> float:
    # read stat field based on market
    keys = stat_key_candidates_for_market(prop_type)
    for k in keys:
        if k in row and row.get(k) is not None:
            try:
                return float(row.get(k) or 0)
            except Exception:
                pass
    return 0.0

def bdl_last_n_games_stats(player_ids, season: int, n: int, prop_type: str):
    """
    Returns: pid -> list[(date, stat_value, minutes)]
    Fills PLAYER_NAME_CACHE from stats endpoint.
    """
    out = {int(pid): [] for pid in player_ids}
    if not player_ids:
        return out

    cursor = None
    pages = 0
    while pages < BDL_MAX_PAGES:
        params = {"per_page": min(BDL_PER_PAGE, 100), "seasons[]": [season], "player_ids[]": player_ids}
        if cursor is not None:
            params["cursor"] = cursor

        resp = _bdl_get("/v1/stats", params=params)
        rows = resp.get("data") or []

        for row in rows:
            p = row.get("player") or {}
            pid = p.get("id")
            if pid is None:
                continue
            pid = int(pid)
            if pid not in out:
                continue

            fn = (p.get("first_name") or "").strip()
            ln = (p.get("last_name") or "").strip()
            if (fn or ln) and pid not in PLAYER_NAME_CACHE:
                PLAYER_NAME_CACHE[pid] = f"{fn} {ln}".strip()

            game = row.get("game") or {}
            date = game.get("date")
            stat_val = _stat_from_row(row, prop_type)
            mins = _parse_minutes(row.get("min"))
            if date:
                out[pid].append((date, stat_val, mins))

        if all(len(out[int(pid)]) >= n for pid in player_ids):
            break

        cursor = (resp.get("meta") or {}).get("next_cursor")
        pages += 1
        if not cursor:
            break

    for pid in list(out.keys()):
        g = out[pid]
        g.sort(key=lambda x: x[0])
        out[pid] = g[-n:]
    return out

def bdl_player_props(game_id: int, vendor: str | None, prop_type_alias: str):
    """
    Fetch odds rows for a given gid + vendor + prop_type alias.
    Cached.
    """
    key = (int(game_id), (vendor or ""), prop_type_alias)
    if key in PROPS_CACHE:
        return PROPS_CACHE[key]

    params = {"game_id": int(game_id), "prop_type": prop_type_alias}
    if vendor:
        params["vendors[]"] = [vendor]

    try:
        resp = _bdl_get("/v2/odds/player_props", params=params)
        rows = resp.get("data") or []
    except Exception:
        rows = []

    # Debug one sample row per requested market
    if prop_type_alias in DEBUG_PROP_SAMPLE_TYPES and rows and prop_type_alias not in DEBUG_PRINTED_FOR:
        print(f"[DEBUG] SAMPLE PROP ROW ({prop_type_alias}): {json.dumps(rows[0])[:2000]}")
        DEBUG_PRINTED_FOR.add(prop_type_alias)

    PROPS_CACHE[key] = rows
    return rows

def _pick_main_line(rows_for_player, prop_type: str):
    if not rows_for_player:
        return None

    lo, hi = line_range_for_market(prop_type)

    candidates = []
    for pp in rows_for_player:
        market = pp.get("market") or {}
        if (market.get("type") or "").lower() != "over_under":
            continue
        try:
            line = float(pp.get("line_value"))
        except Exception:
            continue
        if line < lo or line > hi:
            continue

        over = market.get("over_odds")
        under = market.get("under_odds")
        if isinstance(over, (int, float)) and isinstance(under, (int, float)):
            dist = abs(abs(float(over)) - 110.0) + abs(abs(float(under)) - 110.0)
        else:
            dist = None
        candidates.append((dist, line))

    if not candidates:
        return None

    with_dist = [c for c in candidates if c[0] is not None]
    if with_dist:
        with_dist.sort(key=lambda x: x[0])
        return with_dist[0][1]

    lines = sorted([c[1] for c in candidates])
    mid = len(lines) // 2
    return lines[mid] if len(lines) % 2 == 1 else 0.5 * (lines[mid - 1] + lines[mid])

def main_line_for_player(game_id: int, player_id: int, prop_type: str):
    """
    Find a "main" over/under line by:
      - trying all vendors + None
      - trying all prop_type aliases (for threes)
    """
    aliases = prop_type_aliases(prop_type)

    for alias in aliases:
        for v in BOOK_VENDORS + [None]:
            rows = bdl_player_props(game_id, v, alias)
            if not rows:
                continue
            rows_for_player = []
            for pp in rows:
                try:
                    if int(pp.get("player_id", -1)) != int(player_id):
                        continue
                except Exception:
                    continue
                rows_for_player.append(pp)

            line = _pick_main_line(rows_for_player, prop_type)
            if line is not None:
                return float(line), alias, (v or "MAIN")

    return None, None, None

# -------------------- STATS MATH --------------------
def avg_stat_min_std(games):
    # games: list[(date, stat, min)]
    if not games:
        return 0.0, 0.0, 0.0
    vals = [x[1] for x in games]
    mins = [x[2] for x in games]
    n = len(vals)
    avg_v = sum(vals) / n
    avg_m = sum(mins) / n
    var = sum((v - avg_v) ** 2 for v in vals) / max(n, 1)
    return avg_v, avg_m, math.sqrt(var)

def compute_projection_and_prob(games_all, line, prop_type: str, injury_boost_stat=0.0, injury_boost_min=0.0):
    base_slice = _slice_last(games_all, BASELINE_GAMES)
    l10_slice = _slice_last(games_all, LOOKBACK_GAMES)
    l3_slice = _slice_last(games_all, SHORT_GAMES)

    base_avg, _, base_std = avg_stat_min_std(base_slice)
    l10_avg, l10_min, l10_std = avg_stat_min_std(l10_slice)
    l3_avg, _, _ = avg_stat_min_std(l3_slice)

    sigma = max(STD_FLOOR, (l10_std if l10_std > 0 else base_std if base_std > 0 else STD_FLOOR))

    proj = (W_BASE * base_avg) + (W_L10 * l10_avg) + (W_L3 * l3_avg) + (W_LINE * line)

    # rate per minute for this stat
    rate = l10_avg / max(l10_min, 1e-6)

    # injury boosts
    proj += float(injury_boost_stat)
    proj += float(injury_boost_min) * rate * 0.20

    edge = proj - line
    z = (proj - line) / sigma
    prob_over = _norm_cdf(z)

    return proj, edge, prob_over, (base_avg, l10_avg, l3_avg, l10_min, sigma, rate)

# -------------------- INJURY ENGINE (GENERIC MARKET) --------------------
def build_injury_edges(team_short, injured_name, injured_status, exclude_names_lower, now_et, prop_type: str):
    """
    Returns list of ideas for one market (points or three_pointers_made)
    """
    season = _season_year(now_et)
    roster = bdl_active_roster(team_short)
    if not roster:
        return []

    roster_tuples = []
    for p in roster:
        pid = p.get("id")
        nm = f"{p.get('first_name','')} {p.get('last_name','')}".strip()
        if pid is None or not nm:
            continue
        if _clean_name(nm) in exclude_names_lower:
            continue
        roster_tuples.append((int(pid), nm))

    if not roster_tuples:
        return []

    injured_pid = bdl_find_player_id_on_team(team_short, injured_name)
    if not injured_pid:
        return []

    # injured player's recent production for this market + minutes
    inj_games = bdl_last_n_games_stats([injured_pid], season, BASELINE_GAMES, prop_type).get(injured_pid, [])
    inj_l10 = _slice_last(inj_games, LOOKBACK_GAMES)

    vac_stat_l10, vac_min_l10, _ = avg_stat_min_std(inj_l10)
    if len(inj_games) < 3:
        return []

    status = (injured_status or "").lower()
    STATUS_MULT = {"out": 1.0, "doubtful": 0.8, "questionable": 0.55}.get(status, 0.65)

    vac_stat = vac_stat_l10 * STATUS_MULT
    vac_min = vac_min_l10 * STATUS_MULT

    # keep original "quality filter" tied to points/minutes vacated
    # for threes, vac_stat is smaller — so don't kill it unfairly:
    if prop_type == "points":
        if not ((vac_min >= MIN_VAC_MIN) or (vac_stat >= MIN_VAC_PTS)):
            return []
        trigger_strength = min(100.0, (vac_min * 1.2 + vac_stat * 1.5))
    else:
        # threes: scale strength so it still ranks reasonably
        trigger_strength = min(100.0, (vac_min * 0.9 + vac_stat * 18.0))

    cand_ids = [pid for pid, _ in roster_tuples]
    stats = bdl_last_n_games_stats(cand_ids, season, BASELINE_GAMES, prop_type)

    game_ids = bdl_games_today_ids(now_et)
    if not game_ids:
        return []

    min_edge, min_prob = thresholds_for_market(prop_type)

    ideas = []
    for pid, nm in roster_tuples:
        games = stats.get(pid, [])
        if len(games) < 6:
            continue

        # role trend based on stat/min and minutes trend
        min_s, min_l, rate_s, rate_l = _role_trend(games)
        min_delta = min_s - min_l
        rate_delta = rate_s - rate_l

        l10_avg, l10_min, _ = avg_stat_min_std(_slice_last(games, LOOKBACK_GAMES))
        if l10_min < 10:
            continue

        # absorption score (how likely they soak up usage/role)
        absorption = 0.0
        if l10_min >= 28:
            absorption += 0.30
        if l10_min >= 34:
            absorption += 0.10
        if min_delta >= 2.0:
            absorption += 0.15
        if rate_delta > (0.05 if prop_type == "points" else 0.01):
            absorption += 0.10
        absorption = min(0.65, absorption)

        # find a line (optionally enforce "has a line in a relevant game")
        line = None
        use_gid = None
        used_alias = None
        used_vendor = None

        for gid in game_ids:
            line, used_alias, used_vendor = main_line_for_player(gid, pid, prop_type)
            if line is not None:
                use_gid = gid
                break

        if line is None:
            if STRICT_INJURY_GAME_MATCH:
                continue
            else:
                continue

        # points-only guardrail (avoid obvious misreads)
        if prop_type == "points":
            if (l10_avg - line) > LINE_MIN_GAP:
                continue

        # injury boost caps + scale
        # threes needs smaller cap scale
        if prop_type == "points":
            injury_boost_stat = min(BOOST_CAP_PTS, vac_stat * absorption * 0.65)
            injury_boost_min = min(BOOST_CAP_MIN, vac_min * absorption * 0.25)
        else:
            injury_boost_stat = min(1.0, vac_stat * absorption * 0.55)
            injury_boost_min = min(5.0, vac_min * absorption * 0.18)

        proj, edge, prob_over, aux = compute_projection_and_prob(
            games_all=games,
            line=line,
            prop_type=prop_type,
            injury_boost_stat=injury_boost_stat,
            injury_boost_min=injury_boost_min,
        )
        base_avg, l10_avg2, l3_avg, l10_min2, _, _ = aux

        if edge < min_edge or prob_over < min_prob:
            continue

        if min_delta < MIN_DELTA_FLOOR and edge < (min_edge + (1.5 if prop_type == "points" else 0.8)):
            continue

        why = (
            f"TriggerStrength {trigger_strength:.0f} | Absorb {absorption:.2f}. "
            f"{injured_name} {injured_status.upper()} vacates ~{vac_stat:.1f} {market_label(prop_type)} / {vac_min:.1f} min. "
            f"{nm} base(L{BASELINE_GAMES}) {base_avg:.1f}, L10 {l10_avg2:.1f}, L3 {l3_avg:.1f} "
            f"(mins L10 {l10_min2:.1f}). Role Δmin={min_delta:+.1f}, Δrate={rate_delta:+.2f}. "
            f"Proj {proj:.1f} vs {used_vendor} line {line:.1f} | edge +{edge:.1f} | P≈{prob_over*100:.0f}%. "
            f"[prop_type={used_alias}]"
        )

        ideas.append({
            "market": prop_type,
            "section": "injury",
            "player_name": nm,
            "player_id": pid,
            "line": float(line),
            "proj": float(proj),
            "edge": float(edge),
            "prob_over": float(prob_over),
            "trigger_strength": float(trigger_strength),
            "trigger": f"{injured_name} ({team_short}) {injured_status}",
            "why": why,
            "game_id": use_gid,
        })

    ideas.sort(key=lambda x: (x["trigger_strength"], x["edge"], x["prob_over"]), reverse=True)
    return ideas[:MAX_PER_MARKET]

# -------------------- SLATE SCAN ENGINE (GENERIC MARKET) --------------------
def slate_scan_edges(now_et, prop_type: str):
    if not ENABLE_SLATE_SCAN:
        return []
    if SLATE_ONLY_IN_BURST and (not _in_burst_window(now_et)):
        return []

    season = _season_year(now_et)
    game_ids = bdl_games_today_ids(now_et)
    if not game_ids:
        return []

    # Pull a set of players with "main" lines for this market
    player_to_best_line = {}  # pid -> (line, gid, used_alias, used_vendor)
    pulled = 0

    aliases = prop_type_aliases(prop_type)

    for gid in game_ids:
        # Try to pull one props blob from vendors/aliases; then group by pid.
        props_blob = None
        used_alias = None
        used_vendor = None

        for alias in aliases:
            for v in BOOK_VENDORS + [None]:
                rows = bdl_player_props(gid, v, alias)
                if rows:
                    props_blob = rows
                    used_alias = alias
                    used_vendor = (v or "MAIN")
                    break
            if props_blob:
                break

        if not props_blob:
            continue

        by_pid = {}
        for pp in props_blob:
            pid = pp.get("player_id")
            if pid is None:
                continue
            try:
                pid = int(pid)
            except Exception:
                continue
            by_pid.setdefault(pid, []).append(pp)

        for pid, rows in by_pid.items():
            if pid in player_to_best_line:
                continue
            line = _pick_main_line(rows, prop_type)
            if line is None:
                continue
            player_to_best_line[pid] = (float(line), int(gid), used_alias, used_vendor)
            pulled += 1
            if pulled >= SLATE_SCAN_MAX_PLAYERS:
                break

        if pulled >= SLATE_SCAN_MAX_PLAYERS:
            break

    if not player_to_best_line:
        return []

    pids = list(player_to_best_line.keys())
    stats = bdl_last_n_games_stats(pids, season, BASELINE_GAMES, prop_type)  # fills PLAYER_NAME_CACHE

    min_edge, min_prob = thresholds_for_market(prop_type)

    ideas = []
    for pid in pids:
        games = stats.get(pid, [])
        if len(games) < 8:
            continue

        line, gid, used_alias, used_vendor = player_to_best_line[pid]

        l10_avg, l10_min, _ = avg_stat_min_std(_slice_last(games, LOOKBACK_GAMES))
        if l10_min < 10:
            continue

        if prop_type == "points":
            if (l10_avg - line) > LINE_MIN_GAP:
                continue

        min_s, min_l, rate_s, rate_l = _role_trend(games)
        min_delta = min_s - min_l
        rate_delta = rate_s - rate_l

        proj, edge, prob_over, aux = compute_projection_and_prob(games_all=games, line=line, prop_type=prop_type)
        base_avg, l10_avg2, l3_avg, l10_min2, _, _ = aux

        if edge < min_edge or prob_over < min_prob:
            continue

        if min_delta < MIN_DELTA_FLOOR and edge < (min_edge + (2.0 if prop_type == "points" else 0.8)):
            continue

        name = PLAYER_NAME_CACHE.get(int(pid), f"Player {pid}")

        why = (
            f"SlateScan. base(L{BASELINE_GAMES}) {base_avg:.1f}, L10 {l10_avg2:.1f}, L3 {l3_avg:.1f} "
            f"(mins L10 {l10_min2:.1f}). Role Δmin={min_delta:+.1f}, Δrate={rate_delta:+.2f}. "
            f"Proj {proj:.1f} vs {used_vendor} line {line:.1f} | edge +{edge:.1f} | P≈{prob_over*100:.0f}%. "
            f"[prop_type={used_alias}]"
        )

        ideas.append({
            "market": prop_type,
            "section": "slate",
            "player_name": name,
            "player_id": pid,
            "line": float(line),
            "proj": float(proj),
            "edge": float(edge),
            "prob_over": float(prob_over),
            "trigger_strength": 0.0,
            "trigger": "No injury trigger (league-wide scan)",
            "why": why,
            "game_id": gid,
        })

    ideas.sort(key=lambda x: (x["edge"], x["prob_over"]), reverse=True)
    return ideas[:MAX_PER_MARKET]

# -------------------- COOLDOWN FILTER --------------------
def apply_cooldown(state, ideas, now_ts: int):
    sent = state.get("sent_bets", {}) or {}
    cooldown_sec = BET_COOLDOWN_MIN * 60

    kept = []
    for i in ideas:
        key = f"{i['market']}|{i['section']}|{int(i['player_id'])}|{i['line']:.1f}"
        prev = sent.get(key)

        if not prev:
            kept.append(i)
            continue

        last_ts = int(prev.get("ts", 0) or 0)
        last_edge = float(prev.get("edge", 0.0) or 0.0)
        last_line = float(prev.get("line", i["line"]) or i["line"])

        # resend if line moved meaningfully
        if abs(last_line - float(i["line"])) >= 0.5:
            kept.append(i)
            continue

        # resend if edge improved enough
        if (float(i["edge"]) - last_edge) >= EDGE_JUMP_TO_RESEND:
            kept.append(i)
            continue

        # resend if cooldown elapsed
        if (now_ts - last_ts) >= cooldown_sec:
            kept.append(i)

    return kept

def record_sent(state, ideas, now_ts: int):
    sent = state.get("sent_bets", {}) or {}
    for i in ideas:
        key = f"{i['market']}|{i['section']}|{int(i['player_id'])}|{i['line']:.1f}"
        sent[key] = {"ts": now_ts, "edge": float(i["edge"]), "line": float(i["line"])}
    state["sent_bets"] = sent

# -------------------- MAIN --------------------
def run():
    now_et = _now_et()
    ts_et = now_et.strftime("%Y-%m-%d %I:%M %p ET")
    now_ts = int(now_et.timestamp())

    print(
        f"[BOOT] ts={ts_et} TEST_MODE={int(TEST_MODE)} "
        f"PROP_TYPES={','.join(PROP_TYPES)} "
        f"MIN_PER_MARKET={MIN_PER_MARKET} MAX_PER_MARKET={MAX_PER_MARKET} MAX_BET_IDEAS={MAX_BET_IDEAS} "
        f"BOOK_VENDORS={','.join(BOOK_VENDORS)} ENABLE_SLATE_SCAN={int(ENABLE_SLATE_SCAN)} "
        f"STRICT_INJURY_GAME_MATCH={int(STRICT_INJURY_GAME_MATCH)}"
    )

    if TEST_MODE:
        send_one(f"✅ NBA props agent test OK ({ts_et})")
        return

    state = load_state()
    old_players = state.get("players", {})

    # Injuries
    sr = fetch_sportradar_injuries()
    new_players = parse_injuries(sr)

    exclude_names_lower = {_clean_name(v.get("name", "")) for v in new_players.values() if v.get("name")}

    # Build triggers (only those that pass change filter)
    active_triggers = []  # strings
    trigger_rows = []     # dicts w/ team/name/status

    for pid, cur in new_players.items():
        if not status_in_scope(cur.get("status", "")):
            continue

        prev = old_players.get(pid)
        if IMPACT_ONLY_CHANGES:
            is_new = prev is None
            is_changed = (not is_new) and ((prev.get("status"), prev.get("detail")) != (cur.get("status"), cur.get("detail")))
            if not (is_new or is_changed):
                continue

        team_short = cur.get("team", "")
        injured_name = cur.get("name", "")
        injured_status = (cur.get("status") or "").strip()

        active_triggers.append(f"{injured_name} ({team_short}) {injured_status}")
        trigger_rows.append({
            "team": team_short,
            "name": injured_name,
            "status": injured_status,
        })

    # Build ideas per market
    all_ideas = []

    for market in PROP_TYPES:
        # normalize market name internally
        prop_type = "three_pointers_made" if market in ("three_pointers_made", "threes", "fg3m", "3pt_made") else market

        # Injury-triggered
        injury_ideas_market = []
        for tr in trigger_rows:
            ideas = build_injury_edges(
                team_short=tr["team"],
                injured_name=tr["name"],
                injured_status=tr["status"],
                exclude_names_lower=exclude_names_lower | {_clean_name(tr["name"])},
                now_et=now_et,
                prop_type=prop_type,
            )
            if ideas:
                injury_ideas_market.extend(ideas)

        # Slate scan
        slate_ideas_market = slate_scan_edges(now_et, prop_type=prop_type)

        # combine + dedupe within market by (section, pid) keep best edge/prob
        combined_market = injury_ideas_market + slate_ideas_market
        best = {}
        for i in combined_market:
            k = (i["market"], i["section"], int(i["player_id"]))
            if (k not in best) or ((i["edge"], i["prob_over"]) > (best[k]["edge"], best[k]["prob_over"])):
                best[k] = i
        combined_market = list(best.values())

        all_ideas.extend(combined_market)

    # Apply cooldown across everything
    all_ideas = apply_cooldown(state, all_ideas, now_ts)

    # Re-split per market, per section, then select with caps
    output_chunks = []
    sent_final = []

    header = f"💰 FanDuel Props ({ts_et})"
    output_chunks.append(header)
    output_chunks.append("")

    # Show triggers once (if any)
    if active_triggers:
        # keep it short
        output_chunks.append("🚑 Injury-Triggered Plays:")
        output_chunks.append("Triggers:")
        for t in active_triggers[:8]:
            output_chunks.append(f"- {t}")
        if len(active_triggers) > 8:
            output_chunks.append(f"- …and {len(active_triggers)-8} more")
        output_chunks.append("")

    total_left = MAX_BET_IDEAS

    for market in PROP_TYPES:
        prop_type = "three_pointers_made" if market in ("three_pointers_made", "threes", "fg3m", "3pt_made") else market
        label = market_label(prop_type)

        market_ideas = [i for i in all_ideas if i["market"] == prop_type]
        if not market_ideas:
            continue

        injury = sorted(
            [i for i in market_ideas if i["section"] == "injury"],
            key=lambda x: (x["trigger_strength"], x["edge"], x["prob_over"]),
            reverse=True,
        )
        slate = sorted(
            [i for i in market_ideas if i["section"] == "slate"],
            key=lambda x: (x["edge"], x["prob_over"]),
            reverse=True,
        )

        # take up to MAX_PER_MARKET combined but ensure at least MIN_PER_MARKET if possible
        picked = []
        # prefer injury first, then slate
        for lst in (injury, slate):
            for it in lst:
                if len(picked) >= MAX_PER_MARKET:
                    break
                picked.append(it)
            if len(picked) >= MAX_PER_MARKET:
                break

        # if we still have less than MIN_PER_MARKET and there are remaining slate items, top up
        if len(picked) < MIN_PER_MARKET:
            rest = [x for x in slate if x not in picked]
            for it in rest:
                if len(picked) >= MIN_PER_MARKET:
                    break
                picked.append(it)

        if not picked:
            continue

        # global cap
        if total_left <= 0:
            break
        picked = picked[:total_left]
        total_left -= len(picked)

        output_chunks.append(f"🏷️ {label}")
        output_chunks.append("")

        # Group prints: injury then slate
        inj_picked = [x for x in picked if x["section"] == "injury"]
        slate_picked = [x for x in picked if x["section"] == "slate"]

        if inj_picked:
            for i in inj_picked:
                output_chunks.append(
                    f"• {i['player_name']} OVER {i['line']:.1f}  (edge +{i['edge']:.1f}, P≈{i['prob_over']*100:.0f}%)"
                )
                output_chunks.append(f"  Trigger: {i['trigger']}")
                output_chunks.append(f"  Why: {i['why']}")
                output_chunks.append("")
                sent_final.append(i)

        if slate_picked:
            output_chunks.append("🌎 League-Wide Slate Scan (no injury required):")
            output_chunks.append("")
            for i in slate_picked:
                output_chunks.append(
                    f"• {i['player_name']} OVER {i['line']:.1f}  (edge +{i['edge']:.1f}, P≈{i['prob_over']*100:.0f}%)"
                )
                output_chunks.append(f"  Why: {i['why']}")
                output_chunks.append("")
                sent_final.append(i)

    # If you asked for threes but none were found, tell you why (so it doesn't feel broken)
    if "three_pointers_made" in [("three_pointers_made" if m in ("three_pointers_made","threes","fg3m","3pt_made") else m) for m in PROP_TYPES]:
        has_threes = any(i["market"] == "three_pointers_made" for i in sent_final)
        if not has_threes:
            output_chunks.append("🧩 Note on 3PT Made:")
            output_chunks.append(
                "No 3PT lines matched your vendor(s) for this run. If you *know* FanDuel has 3PT props up, "
                "set DEBUG_PROP_SAMPLE_TYPES=three_pointers_made to print a sample row in logs so we can confirm the correct prop_type."
            )
            output_chunks.append("")

    text = "\n".join(output_chunks).strip()

    if sent_final:
        send_chunked(text)
        record_sent(state, sent_final, now_ts)
    else:
        if SEND_NO_EDGE_PING and _in_burst_window(now_et):
            send_one(f"🧠 No edges hit thresholds this run. ({ts_et})")

    state["players"] = new_players
    save_state(state)

if __name__ == "__main__":
    run()
