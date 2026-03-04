import os
import json
import re
import time
import math
from datetime import datetime
from zoneinfo import ZoneInfo
import requests
from twilio.rest import Client

# =========================================================
# Core
# =========================================================
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

# =========================================================
# CONFIG (ENV)
# =========================================================
TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
MAX_BODY_CHARS = 1500

# Runtime cap (seconds). If exceeded, script will stop work and send partial results.
RUN_MAX_SECONDS = int(os.environ.get("RUN_MAX_SECONDS", "170"))

# Injury scope
IMPACT_STATUSES_RAW = os.environ.get("IMPACT_STATUSES", "out,doubtful,questionable").strip()
IMPACT_STATUSES = {x.strip().lower() for x in IMPACT_STATUSES_RAW.split(",") if x.strip()}
IMPACT_ONLY_CHANGES = os.environ.get("IMPACT_ONLY_CHANGES", "1") == "1"

# Vendors
BOOK_VENDOR_RAW = os.environ.get("BOOK_VENDORS", os.environ.get("BOOK_VENDOR", "fanduel")).strip().lower()
BOOK_VENDORS = [v.strip() for v in BOOK_VENDOR_RAW.split(",") if v.strip()]

# Markets / prop types (accepts points,threes or points,three_pointers_made etc.)
PROP_TYPES_RAW = os.environ.get("PROP_TYPES", os.environ.get("MARKETS", "points,threes")).strip().lower()
PROP_TYPES = [x.strip() for x in PROP_TYPES_RAW.split(",") if x.strip()]

MIN_PER_MARKET = int(os.environ.get("MIN_PER_MARKET", "0"))
MAX_PER_MARKET = int(os.environ.get("MAX_PER_MARKET", "6"))
MAX_TOTAL_PLAYS = int(os.environ.get("MAX_TOTAL_PLAYS", os.environ.get("MAX_BET_IDEAS", "10")))

ENABLE_SLATE_SCAN = os.environ.get("ENABLE_SLATE_SCAN", "1") == "1"
ENABLE_INJURY_TRIGGERS = os.environ.get("ENABLE_INJURY_TRIGGERS", "1") == "1"

BURST_START_ET = os.environ.get("BURST_START_ET", "17:00").strip()
BURST_END_ET = os.environ.get("BURST_END_ET", "23:45").strip()
SLATE_ONLY_IN_BURST = os.environ.get("SLATE_ONLY_IN_BURST", "0") == "1"
SEND_NO_EDGE_PING = os.environ.get("SEND_NO_EDGE_PING", "0") == "1"

STRICT_INJURY_GAME_MATCH = os.environ.get("STRICT_INJURY_GAME_MATCH", "0") == "1"

# Projection horizons
BASELINE_GAMES = int(os.environ.get("BASELINE_GAMES", "30"))
LOOKBACK_GAMES = int(os.environ.get("LOOKBACK_GAMES", "10"))
SHORT_GAMES = int(os.environ.get("SHORT_GAMES", "3"))

# Projection blend weights
W_BASE = float(os.environ.get("W_BASE", "0.45"))
W_L10 = float(os.environ.get("W_L10", "0.35"))
W_L3 = float(os.environ.get("W_L3", "0.10"))
W_LINE = float(os.environ.get("W_LINE", "0.10"))

# Thresholds
MIN_EDGE = float(os.environ.get("MIN_EDGE", "2.5"))
MIN_PROB = float(os.environ.get("MIN_PROB", "0.62"))

# anti-trap controls
LINE_MIN_GAP = float(os.environ.get("LINE_MIN_GAP", "8.0"))
MIN_DELTA_FLOOR = float(os.environ.get("MIN_DELTA_FLOOR", "-3.0"))

# Injury vacancy requirements
MIN_VAC_MIN = float(os.environ.get("MIN_VAC_MIN", "10.0"))
MIN_VAC_PTS = float(os.environ.get("MIN_VAC_PTS", "6.0"))

# Injury boost caps
BOOST_CAP_PTS = float(os.environ.get("BOOST_CAP_PTS", "5.5"))
BOOST_CAP_MIN = float(os.environ.get("BOOST_CAP_MIN", "6.0"))

# Cooldown
BET_COOLDOWN_MIN = int(os.environ.get("BET_COOLDOWN_MIN", "180"))
EDGE_JUMP_TO_RESEND = float(os.environ.get("EDGE_JUMP_TO_RESEND", "1.5"))

# Value / plus-odds
VALUE_EDGE_MIN = float(os.environ.get("VALUE_EDGE_MIN", "0.00"))

# HARD caps so we don’t blow up runtime
SLATE_SCAN_MAX_PLAYERS = int(os.environ.get("SLATE_SCAN_MAX_PLAYERS", "260"))
MAX_INJURY_TEAMS_TO_PROCESS = int(os.environ.get("MAX_INJURY_TEAMS_TO_PROCESS", "25"))

# Debug sample printing
DEBUG_PROP_SAMPLE_TYPES_RAW = os.environ.get("DEBUG_PROP_SAMPLE_TYPES", "").strip().lower()
DEBUG_PROP_SAMPLE_TYPES = {x.strip() for x in DEBUG_PROP_SAMPLE_TYPES_RAW.split(",") if x.strip()}

# BDL request tuning
BDL_MAX_RETRIES = int(os.environ.get("BDL_MAX_RETRIES", "5"))
BDL_RETRY_BASE_SEC = float(os.environ.get("BDL_RETRY_BASE_SEC", "1.5"))
BDL_PER_PAGE = int(os.environ.get("BDL_PER_PAGE", "100"))
BDL_MAX_PAGES = int(os.environ.get("BDL_MAX_PAGES", "10"))

# =========================================================
# Deadline handling (graceful)
# =========================================================
_RUN_START_TS = time.time()
_DEADLINE_HIT = False

def time_left_s() -> float:
    return RUN_MAX_SECONDS - (time.time() - _RUN_START_TS)

def check_deadline(where: str = "") -> bool:
    global _DEADLINE_HIT
    if (time.time() - _RUN_START_TS) > RUN_MAX_SECONDS:
        _DEADLINE_HIT = True
        print(f"[DEADLINE] hit at {where}")
        return False
    return True

# =========================================================
# UTILS
# =========================================================
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

def avg_stat_min_std(games):
    if not games:
        return 0.0, 0.0, 0.0
    vals = [x[1] for x in games]
    mins = [x[2] for x in games]
    n = len(vals)
    v_avg = sum(vals) / n
    m_avg = sum(mins) / n
    var = sum((v - v_avg) ** 2 for v in vals) / max(n, 1)
    return v_avg, m_avg, math.sqrt(var)

def _role_trend(games):
    if not games:
        return 0.0, 0.0, 0.0, 0.0
    long_slice = _slice_last(games, LOOKBACK_GAMES)
    short_slice = _slice_last(games, SHORT_GAMES)

    v_l, min_l, _ = avg_stat_min_std(long_slice)
    v_s, min_s, _ = avg_stat_min_std(short_slice)

    rate_l = v_l / max(min_l, 1e-6)
    rate_s = v_s / max(min_s, 1e-6)
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

# =========================================================
# Market mapping
# =========================================================
def normalize_market(m: str) -> str:
    m = (m or "").strip().lower()
    alias = {
        "3pt": "threes",
        "3pts": "threes",
        "3pm": "threes",
        "3ptm": "threes",
        "three_pointers": "threes",
        "three_pointers_made": "threes",
        "threes_made": "threes",
        "3pt_made": "threes",
    }
    return alias.get(m, m)

def prop_type_api_for_market(m: str) -> str:
    m = normalize_market(m)
    if m == "points":
        return "points"
    if m == "threes":
        return "threes"
    return m

def stat_key_for_market(m: str) -> str:
    m = normalize_market(m)
    if m == "points":
        return "pts"
    if m == "threes":
        return "fg3m"
    return "pts"

def line_bounds_for_market(m: str):
    m = normalize_market(m)
    if m == "points":
        min_line = float(os.environ.get("MIN_POINTS_LINE", "6.0"))
        max_line = float(os.environ.get("MAX_POINTS_LINE", "45.0"))
        std_floor = float(os.environ.get("STD_FLOOR_POINTS", "5.0"))
        return min_line, max_line, std_floor
    if m == "threes":
        min_line = float(os.environ.get("MIN_THREES_LINE", "0.5"))
        max_line = float(os.environ.get("MAX_THREES_LINE", "6.5"))
        std_floor = float(os.environ.get("STD_FLOOR_THREES", "0.85"))
        return min_line, max_line, std_floor
    return 0.5, 50.0, float(os.environ.get("STD_FLOOR_FALLBACK", "5.0"))

# =========================================================
# SPORTRADAR Injuries
# =========================================================
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

# =========================================================
# BallDontLie API + caches
# =========================================================
BDL_HEADERS = {"Authorization": BALLDONTLIE_API_KEY}
BDL_PREFIXES = ["/nba", ""]

TEAM_CACHE = None
GAMES_CACHE = {}
PROPS_CACHE = {}            # (gid, vendorlabel, prop_type_api) -> list
PLAYER_NAME_CACHE = {}      # pid -> "First Last"
_DEBUG_PRINTED = set()
ROSTER_CACHE = {}           # team_name -> list[players]
FIND_PID_CACHE = {}         # (team, clean_name) -> pid or None

def _bdl_get(path: str, params=None, timeout: int = 20) -> dict:
    last_err = None
    for pref in BDL_PREFIXES:
        url = f"https://api.balldontlie.io{pref}{path}"
        for attempt in range(BDL_MAX_RETRIES):
            if not check_deadline(f"_bdl_get {path}"):
                raise RuntimeError("deadline")
            try:
                r = requests.get(url, headers=BDL_HEADERS, params=params or {}, timeout=timeout)
                if r.status_code == 404:
                    last_err = f"404 {url}"
                    break
                if r.status_code in (429, 500, 502, 503, 504):
                    retry_after = r.headers.get("Retry-After")
                    sleep_s = float(retry_after) if retry_after else BDL_RETRY_BASE_SEC * (2 ** attempt)
                    last_err = f"{r.status_code} {r.text[:120]}"
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
    if today in GAMES_CACHE:
        return GAMES_CACHE[today]
    resp = _bdl_get("/v1/games", params={"dates[]": [today], "per_page": 100})
    ids = [int(g["id"]) for g in (resp.get("data") or []) if g.get("id") is not None]
    GAMES_CACHE[today] = ids
    return ids

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
    if team_short in ROSTER_CACHE:
        return ROSTER_CACHE[team_short]

    team_map = bdl_team_name_to_id()
    team_id = team_map.get(team_short)
    if not team_id:
        ROSTER_CACHE[team_short] = []
        return []

    players = []
    cursor = None
    pages = 0
    while pages < 5:
        if not check_deadline("bdl_active_roster"):
            break
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

    ROSTER_CACHE[team_short] = out
    return out

def bdl_find_player_id_on_team(team_short: str, full_name: str):
    key = (team_short, _clean_name(full_name))
    if key in FIND_PID_CACHE:
        return FIND_PID_CACHE[key]

    roster = bdl_active_roster(team_short)
    if not roster:
        FIND_PID_CACHE[key] = None
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
            FIND_PID_CACHE[key] = int(pid)
            return int(pid)

    FIND_PID_CACHE[key] = None
    return None

def bdl_last_n_games_stats(player_ids, season: int, n: int, stat_key: str):
    out = {int(pid): [] for pid in player_ids}
    if not player_ids:
        return out

    cursor = None
    pages = 0
    while pages < BDL_MAX_PAGES:
        if not check_deadline("bdl_last_n_games_stats"):
            break

        params = {
            "per_page": min(BDL_PER_PAGE, 100),
            "seasons[]": [season],
            "player_ids[]": player_ids
        }
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

            val = row.get(stat_key, None)
            if val is None:
                val = 0
            try:
                stat_val = float(val or 0)
            except Exception:
                stat_val = 0.0

            mins = _parse_minutes(row.get("min"))
            if date:
                out[pid].append((date, stat_val, mins))

        cursor = (resp.get("meta") or {}).get("next_cursor")
        pages += 1
        if not cursor:
            break

    for pid in list(out.keys()):
        g = out[pid]
        g.sort(key=lambda x: x[0])
        out[pid] = g[-n:]
    return out

# =========================================================
# Odds / Lines
# =========================================================
def bdl_player_props(game_id: int, vendor: str | None, prop_type_api: str):
    vendor_label = vendor or "NO_VENDOR"
    key = (int(game_id), vendor_label, prop_type_api)
    if key in PROPS_CACHE:
        return PROPS_CACHE[key]

    params = {"game_id": int(game_id), "prop_type": prop_type_api}
    if vendor:
        params["vendors[]"] = [vendor]

    try:
        resp = _bdl_get("/v2/odds/player_props", params=params)
        props = resp.get("data") or []
    except Exception:
        props = []

    dkey = (prop_type_api, vendor_label)
    if prop_type_api in DEBUG_PROP_SAMPLE_TYPES and (dkey not in _DEBUG_PRINTED) and props:
        print(f"[DEBUG] SAMPLE PROP ROW ({prop_type_api}, vendor={vendor_label}): {json.dumps(props[0])[:2000]}")
        _DEBUG_PRINTED.add(dkey)

    PROPS_CACHE[key] = props
    return props

def american_to_implied_prob(odds: float) -> float | None:
    try:
        o = float(odds)
    except Exception:
        return None
    if o == 0:
        return None
    if o < 0:
        return (-o) / ((-o) + 100.0)
    return 100.0 / (o + 100.0)

def build_today_lines_map(game_ids, prop_type_api: str, min_line: float, max_line: float, cap_players: int):
    pid_best = {}  # pid -> (dist, line, gid, vendor, over_odds, under_odds)
    pulled = 0

    for gid in game_ids:
        if not check_deadline("build_today_lines_map"):
            break

        for v in BOOK_VENDORS + [None]:
            vendor_label = v or "NO_VENDOR"
            props = bdl_player_props(gid, v, prop_type_api)
            if not props:
                continue

            by_pid = {}
            for pp in props:
                pid = pp.get("player_id")
                if pid is None:
                    continue
                try:
                    pid = int(pid)
                except Exception:
                    continue
                by_pid.setdefault(pid, []).append(pp)

            for pid, rows in by_pid.items():
                best_dist = None
                best_line = None
                best_over = None
                best_under = None

                for pp in rows:
                    market = pp.get("market") or {}
                    if (market.get("type") or "").lower() != "over_under":
                        continue
                    try:
                        line = float(pp.get("line_value"))
                    except Exception:
                        continue
                    if line < min_line or line > max_line:
                        continue

                    over = market.get("over_odds")
                    under = market.get("under_odds")
                    if isinstance(over, (int, float)) and isinstance(under, (int, float)):
                        dist = abs(abs(float(over)) - 110.0) + abs(abs(float(under)) - 110.0)
                    else:
                        dist = 9999.0

                    if (best_dist is None) or (dist < best_dist):
                        best_dist = dist
                        best_line = line
                        best_over = over
                        best_under = under

                if best_line is None:
                    continue

                cur = pid_best.get(pid)
                if (cur is None) or (best_dist < cur[0]):
                    pid_best[pid] = (best_dist, float(best_line), int(gid), vendor_label, best_over, best_under)
                    pulled += 1

                if pulled >= cap_players:
                    break

            if pulled >= cap_players:
                break

        if pulled >= cap_players:
            break

    return {pid: (line, gid, vendor, over, under) for pid, (dist, line, gid, vendor, over, under) in pid_best.items()}

# =========================================================
# Projection core
# =========================================================
def compute_projection_and_prob(games_all, line, std_floor: float, injury_boost_val=0.0, injury_boost_min=0.0):
    base_slice = _slice_last(games_all, BASELINE_GAMES)
    l10_slice = _slice_last(games_all, LOOKBACK_GAMES)
    l3_slice = _slice_last(games_all, SHORT_GAMES)

    base_avg, _, base_std = avg_stat_min_std(base_slice)
    l10_avg, l10_min, l10_std = avg_stat_min_std(l10_slice)
    l3_avg, _, _ = avg_stat_min_std(l3_slice)

    sigma = max(std_floor, (l10_std if l10_std > 0 else base_std if base_std > 0 else std_floor))
    proj = (W_BASE * base_avg) + (W_L10 * l10_avg) + (W_L3 * l3_avg) + (W_LINE * line)

    rate = l10_avg / max(l10_min, 1e-6)
    proj += injury_boost_val
    proj += (injury_boost_min * rate * 0.20)

    edge = proj - line
    z = (proj - line) / sigma
    prob_over = _norm_cdf(z)
    return proj, edge, prob_over, (base_avg, l10_avg, l3_avg, l10_min, sigma, rate)

# =========================================================
# Injury Engine (NO BIG STATS CALLS INSIDE)
# =========================================================
def build_injury_edges(team_short, injured_name, injured_status, exclude_names_lower, now_et,
                       market: str, today_lines_map: dict, stats_all: dict):
    if not ENABLE_INJURY_TRIGGERS:
        return []

    season = _season_year(now_et)
    prop_type_api = prop_type_api_for_market(market)
    stat_key = stat_key_for_market(market)
    min_line, max_line, std_floor = line_bounds_for_market(market)

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

    # Small stats pull for the injured player ONLY (cheap)
    inj_games = bdl_last_n_games_stats([injured_pid], season, BASELINE_GAMES, stat_key).get(injured_pid, [])
    inj_l10_avg, inj_l10_min, _ = avg_stat_min_std(_slice_last(inj_games, LOOKBACK_GAMES))
    if len(inj_games) < 3:
        return []

    status = (injured_status or "").lower()
    STATUS_MULT = {"out": 1.0, "doubtful": 0.8, "questionable": 0.55}.get(status, 0.65)

    vac_val = inj_l10_avg * STATUS_MULT
    vac_min = inj_l10_min * STATUS_MULT

    if market == "points":
        if not ((vac_min >= MIN_VAC_MIN) or (vac_val >= MIN_VAC_PTS)):
            return []
    else:
        if vac_min < (MIN_VAC_MIN * 0.8):
            return []

    trigger_strength = min(100.0, (vac_min * 1.2 + (vac_val * 10.0 if market == "threes" else vac_val * 1.5)))

    ideas = []
    for pid, nm in roster_tuples:
        if not check_deadline("injury candidates"):
            break

        games = stats_all.get(int(pid), [])
        if len(games) < 6:
            continue

        l10_avg, l10_min, _ = avg_stat_min_std(_slice_last(games, LOOKBACK_GAMES))
        if l10_min < 10:
            continue

        tup = today_lines_map.get(int(pid))
        if not tup:
            continue
        line, use_gid, use_vendor, over_odds, under_odds = tup
        if line < min_line or line > max_line:
            continue

        if (l10_avg - line) > LINE_MIN_GAP:
            continue

        min_s, min_l, rate_s, rate_l = _role_trend(games)
        min_delta = min_s - min_l
        rate_delta = rate_s - rate_l

        absorption = 0.0
        if l10_min >= 28:
            absorption += 0.30
        if l10_min >= 34:
            absorption += 0.10
        if min_delta >= 2.0:
            absorption += 0.15
        if rate_delta > (0.05 if market == "points" else 0.02):
            absorption += 0.10
        absorption = min(0.65, absorption)

        injury_boost_val = min(BOOST_CAP_PTS if market == "points" else 1.2, vac_val * absorption * (0.65 if market == "points" else 0.45))
        injury_boost_min = min(BOOST_CAP_MIN, vac_min * absorption * 0.25)

        proj, edge, prob_over, aux = compute_projection_and_prob(
            games_all=games,
            line=float(line),
            std_floor=std_floor,
            injury_boost_val=injury_boost_val,
            injury_boost_min=injury_boost_min
        )
        base_avg, l10_avg2, l3_avg, l10_min2, _, _ = aux

        if edge < MIN_EDGE or prob_over < MIN_PROB:
            continue
        if min_delta < MIN_DELTA_FLOOR and edge < (MIN_EDGE + 1.5):
            continue

        implied = None
        if isinstance(over_odds, (int, float)):
            implied = american_to_implied_prob(over_odds)
        if (VALUE_EDGE_MIN > 0) and (implied is not None):
            if (prob_over - implied) < VALUE_EDGE_MIN:
                continue

        why = (
            f"TriggerStrength {trigger_strength:.0f} | Absorb {absorption:.2f}. "
            f"{injured_name} {injured_status.upper()} vacates ~{vac_val:.1f} {market.title()} / {vac_min:.1f} min. "
            f"{nm} base(L{BASELINE_GAMES}) {base_avg:.1f}, L10 {l10_avg2:.1f}, L3 {l3_avg:.1f} "
            f"(mins L10 {l10_min2:.1f}). Role Δmin={min_delta:+.1f}, Δrate={rate_delta:+.2f}. "
            f"Proj {proj:.1f} vs {use_vendor.lower()} line {line:.1f} | edge +{edge:.1f} | P≈{prob_over*100:.0f}%."
            f" [prop_type={prop_type_api}]"
        )

        ideas.append({
            "section": "injury",
            "market": market,
            "prop_type": prop_type_api,
            "player_name": nm,
            "player_id": int(pid),
            "line": float(line),
            "proj": float(proj),
            "edge": float(edge),
            "prob_over": float(prob_over),
            "trigger_strength": float(trigger_strength),
            "trigger": f"{injured_name} ({team_short}) {injured_status}",
            "why": why,
            "game_id": int(use_gid),
            "vendor": use_vendor,
        })

    ideas.sort(key=lambda x: (x["trigger_strength"], x["edge"], x["prob_over"]), reverse=True)
    return ideas

# =========================================================
# Slate Scan (uses pre-fetched stats)
# =========================================================
def slate_scan_edges(now_et, market: str, today_lines_map: dict, stats_all: dict):
    if not ENABLE_SLATE_SCAN:
        return []
    if SLATE_ONLY_IN_BURST and (not _in_burst_window(now_et)):
        return []

    prop_type_api = prop_type_api_for_market(market)
    min_line, max_line, std_floor = line_bounds_for_market(market)

    ideas = []
    for pid, tup in today_lines_map.items():
        if not check_deadline("slate scan"):
            break

        games = stats_all.get(int(pid), [])
        if len(games) < 8:
            continue

        line, gid, vendor, over_odds, under_odds = tup
        if line < min_line or line > max_line:
            continue

        l10_avg, l10_min, _ = avg_stat_min_std(_slice_last(games, LOOKBACK_GAMES))
        if l10_min < 10:
            continue
        if (l10_avg - line) > LINE_MIN_GAP:
            continue

        min_s, min_l, rate_s, rate_l = _role_trend(games)
        min_delta = min_s - min_l
        rate_delta = rate_s - rate_l

        proj, edge, prob_over, aux = compute_projection_and_prob(games_all=games, line=line, std_floor=std_floor)
        base_avg, l10_avg2, l3_avg, l10_min2, _, _ = aux

        if edge < MIN_EDGE or prob_over < MIN_PROB:
            continue
        if min_delta < MIN_DELTA_FLOOR and edge < (MIN_EDGE + 2.0):
            continue

        implied = None
        if isinstance(over_odds, (int, float)):
            implied = american_to_implied_prob(over_odds)
        if (VALUE_EDGE_MIN > 0) and (implied is not None):
            if (prob_over - implied) < VALUE_EDGE_MIN:
                continue

        name = PLAYER_NAME_CACHE.get(int(pid), f"Player {pid}")
        why = (
            f"SlateScan. base(L{BASELINE_GAMES}) {base_avg:.1f}, L10 {l10_avg2:.1f}, L3 {l3_avg:.1f} "
            f"(mins L10 {l10_min2:.1f}). Role Δmin={min_delta:+.1f}, Δrate={rate_delta:+.2f}. "
            f"Proj {proj:.1f} vs {vendor.lower()} line {line:.1f} | edge +{edge:.1f} | P≈{prob_over*100:.0f}%."
            f" [prop_type={prop_type_api}]"
        )

        ideas.append({
            "section": "slate",
            "market": market,
            "prop_type": prop_type_api,
            "player_name": name,
            "player_id": int(pid),
            "line": float(line),
            "proj": float(proj),
            "edge": float(edge),
            "prob_over": float(prob_over),
            "trigger_strength": 0.0,
            "trigger": "No injury trigger (league-wide scan)",
            "why": why,
            "game_id": int(gid),
            "vendor": vendor,
        })

    ideas.sort(key=lambda x: (x["edge"], x["prob_over"]), reverse=True)
    return ideas

# =========================================================
# Cooldown
# =========================================================
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

        if abs(last_line - float(i["line"])) >= 0.5:
            kept.append(i)
            continue

        if (float(i["edge"]) - last_edge) >= EDGE_JUMP_TO_RESEND:
            kept.append(i)
            continue

        if (now_ts - last_ts) >= cooldown_sec:
            kept.append(i)

    return kept

def record_sent(state, ideas, now_ts: int):
    sent = state.get("sent_bets", {}) or {}
    for i in ideas:
        key = f"{i['market']}|{i['section']}|{int(i['player_id'])}|{i['line']:.1f}"
        sent[key] = {"ts": now_ts, "edge": float(i["edge"]), "line": float(i["line"])}
    state["sent_bets"] = sent

# =========================================================
# MAIN
# =========================================================
def run():
    now_et = _now_et()
    ts_et = now_et.strftime("%Y-%m-%d %I:%M %p ET")
    now_ts = int(now_et.timestamp())

    mkts = [normalize_market(x) for x in PROP_TYPES]
    mkts = [m for m in mkts if m in ("points", "threes")]

    print(
        f"[BOOT] ts={ts_et} TEST_MODE={int(TEST_MODE)} "
        f"PROP_TYPES={','.join(mkts)} MIN_PER_MARKET={MIN_PER_MARKET} MAX_PER_MARKET={MAX_PER_MARKET} "
        f"MAX_TOTAL_PLAYS={MAX_TOTAL_PLAYS} BOOK_VENDORS={','.join(BOOK_VENDORS)} "
        f"ENABLE_SLATE_SCAN={int(ENABLE_SLATE_SCAN)} ENABLE_INJURY_TRIGGERS={int(ENABLE_INJURY_TRIGGERS)} "
        f"STRICT_INJURY_GAME_MATCH={int(STRICT_INJURY_GAME_MATCH)}"
    )

    if TEST_MODE:
        send_one(f"✅ NBA props agent test OK ({ts_et})")
        return

    state = load_state()
    old_players = state.get("players", {})

    game_ids = bdl_games_today_ids(now_et)
    if not game_ids:
        if SEND_NO_EDGE_PING and _in_burst_window(now_et):
            send_one(f"🧠 No games found today. ({ts_et})")
        return

    # Injuries
    sr = fetch_sportradar_injuries()
    new_players = parse_injuries(sr)
    exclude_names_lower = {_clean_name(v.get("name", "")) for v in new_players.values() if v.get("name")}

    season = _season_year(now_et)

    # Build line maps + prefetch stats ONCE PER MARKET
    today_lines_map_by_market = {}
    stats_by_market = {}

    for m in mkts:
        if not check_deadline("market prebuild"):
            break

        prop_type_api = prop_type_api_for_market(m)
        stat_key = stat_key_for_market(m)
        min_line, max_line, _std_floor = line_bounds_for_market(m)

        today_lines = build_today_lines_map(
            game_ids=game_ids,
            prop_type_api=prop_type_api,
            min_line=min_line,
            max_line=max_line,
            cap_players=SLATE_SCAN_MAX_PLAYERS
        )
        today_lines_map_by_market[m] = today_lines

        # Prefetch stats for ONLY the pids we have lines for (single batch call)
        pids = list(today_lines.keys())
        stats_by_market[m] = bdl_last_n_games_stats(pids, season, BASELINE_GAMES, stat_key)

    # Collect ideas
    all_ideas = []
    triggers_out = []

    # Pre-filter injuries we’ll consider so we don’t loop 100+ and blow time
    injury_candidates = []
    if ENABLE_INJURY_TRIGGERS:
        for pid, cur in new_players.items():
            if not status_in_scope(cur.get("status", "")):
                continue

            prev = old_players.get(pid)
            if IMPACT_ONLY_CHANGES:
                is_new = prev is None
                is_changed = (not is_new) and ((prev.get("status"), prev.get("detail")) != (cur.get("status"), cur.get("detail")))
                if not (is_new or is_changed):
                    continue

            injury_candidates.append((pid, cur))
            if len(injury_candidates) >= MAX_INJURY_TEAMS_TO_PROCESS:
                break

    for market in mkts:
        if not check_deadline("market loop"):
            break

        today_lines_map = today_lines_map_by_market.get(market, {}) or {}
        stats_all = stats_by_market.get(market, {}) or {}

        injury_ideas = []
        triggers_this_market = []

        if ENABLE_INJURY_TRIGGERS:
            for _, cur in injury_candidates:
                if not check_deadline("injury loop"):
                    break

                team_short = cur.get("team", "")
                injured_name = cur.get("name", "")
                injured_status = (cur.get("status") or "").strip()

                ideas = build_injury_edges(
                    team_short=team_short,
                    injured_name=injured_name,
                    injured_status=injured_status,
                    exclude_names_lower=exclude_names_lower | {_clean_name(injured_name)},
                    now_et=now_et,
                    market=market,
                    today_lines_map=today_lines_map,
                    stats_all=stats_all
                )

                if ideas:
                    triggers_this_market.append(f"{injured_name} ({team_short}) {injured_status}")
                    injury_ideas.extend(ideas)

        slate_ideas = slate_scan_edges(now_et, market=market, today_lines_map=today_lines_map, stats_all=stats_all)

        combined = injury_ideas + slate_ideas

        # best per (market, pid, line)
        best = {}
        for i in combined:
            k = (i["market"], int(i["player_id"]), float(i["line"]))
            cur = best.get(k)
            if (cur is None) or ((i["edge"], i["prob_over"]) > (cur["edge"], cur["prob_over"])):
                best[k] = i
        combined = list(best.values())

        combined = apply_cooldown(state, combined, now_ts)

        inj_sorted = sorted(
            [i for i in combined if i["section"] == "injury"],
            key=lambda x: (x["trigger_strength"], x["edge"], x["prob_over"]),
            reverse=True
        )
        slate_sorted = sorted(
            [i for i in combined if i["section"] == "slate"],
            key=lambda x: (x["edge"], x["prob_over"]),
            reverse=True
        )

        market_out = []
        market_out.extend(inj_sorted[:MAX_PER_MARKET])
        if len(market_out) < MIN_PER_MARKET:
            need = MIN_PER_MARKET - len(market_out)
            market_out.extend(slate_sorted[:need])

        remaining = MAX_PER_MARKET - len(market_out)
        if remaining > 0:
            already = {(int(x["player_id"]), float(x["line"])) for x in market_out}
            add = []
            for s in slate_sorted:
                k2 = (int(s["player_id"]), float(s["line"]))
                if k2 in already:
                    continue
                add.append(s)
                if len(add) >= remaining:
                    break
            market_out.extend(add)

        if triggers_this_market:
            triggers_out.extend(triggers_this_market)

        all_ideas.extend(market_out)

    all_ideas.sort(key=lambda x: (x["edge"], x["prob_over"]), reverse=True)
    final_out = all_ideas[:MAX_TOTAL_PLAYS]

    if final_out:
        msg = [f"💰 FanDuel Props ({ts_et})", ""]
        if _DEADLINE_HIT:
            msg.append("⚠️ Note: runtime cap hit; sent partial results this run.")
            msg.append("")

        by_market = {}
        for i in final_out:
            by_market.setdefault(i["market"], []).append(i)

        for market in ("points", "threes"):
            if market not in by_market:
                continue

            msg.append(f"🏷️ {'Points' if market == 'points' else '3PT Made'}")
            msg.append("")

            inj = [i for i in by_market[market] if i["section"] == "injury"]
            slate = [i for i in by_market[market] if i["section"] == "slate"]

            if inj:
                msg.append("🚑 Injury-Triggered Plays:")
                if triggers_out and market == "points":
                    msg.append("Triggers:")
                    for t in triggers_out[:8]:
                        msg.append(f"- {t}")
                    if len(triggers_out) > 8:
                        msg.append(f"- …and {len(triggers_out)-8} more")
                msg.append("")
                for i in inj:
                    msg.append(f"• {i['player_name']} OVER {i['line']:.1f}  (edge +{i['edge']:.1f}, P≈{i['prob_over']*100:.0f}%)")
                    msg.append(f"  Trigger: {i['trigger']}")
                    msg.append(f"  Why: {i['why']}")
                    msg.append("")

            if slate:
                msg.append("🌎 League-Wide Slate Scan (no injury required):")
                msg.append("")
                for i in slate:
                    msg.append(f"• {i['player_name']} OVER {i['line']:.1f}  (edge +{i['edge']:.1f}, P≈{i['prob_over']*100:.0f}%)")
                    msg.append(f"  Why: {i['why']}")
                    msg.append("")

            msg.append("")

        send_chunked("\n".join(msg).strip())
        record_sent(state, final_out, now_ts)
    else:
        if SEND_NO_EDGE_PING and _in_burst_window(now_et):
            send_one(f"🧠 No edges ≥ {MIN_EDGE:.1f} and P ≥ {MIN_PROB:.2f} this run. ({ts_et})")

    state["players"] = new_players
    save_state(state)

if __name__ == "__main__":
    run()
