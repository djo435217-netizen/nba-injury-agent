import os
import json
import re
import time
import math
from datetime import datetime
from zoneinfo import ZoneInfo
import requests
from twilio.rest import Client

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

BOOK_VENDOR_RAW = os.environ.get("BOOK_VENDOR", "fanduel").strip().lower()
BOOK_VENDORS = [v.strip() for v in BOOK_VENDOR_RAW.split(",") if v.strip()]

MARKETS_RAW = os.environ.get("MARKETS", "points").strip().lower()
MARKETS = [m.strip() for m in MARKETS_RAW.split(",") if m.strip()]  # points,threes

ENABLE_THREES = os.environ.get("ENABLE_THREES", "1") == "1"

# Multi-horizon windows
BASELINE_GAMES = int(os.environ.get("BASELINE_GAMES", "30"))
LOOKBACK_GAMES = int(os.environ.get("LOOKBACK_GAMES", "10"))
SHORT_GAMES = int(os.environ.get("SHORT_GAMES", "3"))

# Projection blend weights (apply to both markets)
W_BASE = float(os.environ.get("W_BASE", "0.45"))
W_L10 = float(os.environ.get("W_L10", "0.35"))
W_L3 = float(os.environ.get("W_L3", "0.10"))
W_LINE = float(os.environ.get("W_LINE", "0.10"))

# Output sizing
INJURY_TOPN = int(os.environ.get("INJURY_TOPN", "6"))
SLATE_TOPN = int(os.environ.get("SLATE_TOPN", "6"))
MAX_BET_IDEAS = int(os.environ.get("MAX_BET_IDEAS", "12"))

# Points thresholds
MIN_EDGE = float(os.environ.get("MIN_EDGE", "2.5"))
MIN_PROB = float(os.environ.get("MIN_PROB", "0.62"))
STD_FLOOR = float(os.environ.get("STD_FLOOR", "5.0"))

# Threes thresholds (separate)
MIN_EDGE_THREES = float(os.environ.get("MIN_EDGE_THREES", "0.8"))
MIN_PROB_THREES = float(os.environ.get("MIN_PROB_THREES", "0.62"))
STD_FLOOR_THREES = float(os.environ.get("STD_FLOOR_THREES", "1.1"))

# Guardrails (points)
MIN_POINTS_LINE = float(os.environ.get("MIN_POINTS_LINE", "6.0"))
MAX_POINTS_LINE = float(os.environ.get("MAX_POINTS_LINE", "45.0"))
LINE_MIN_GAP = float(os.environ.get("LINE_MIN_GAP", "8.0"))
MIN_DELTA_FLOOR = float(os.environ.get("MIN_DELTA_FLOOR", "-3.0"))

# Guardrails (threes)
MIN_THREES_LINE = float(os.environ.get("MIN_THREES_LINE", "0.5"))
MAX_THREES_LINE = float(os.environ.get("MAX_THREES_LINE", "6.5"))
LINE_MIN_GAP_THREES = float(os.environ.get("LINE_MIN_GAP_THREES", "2.5"))

# Injury vacancy requirements
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
SLATE_ONLY_IN_BURST = os.environ.get("SLATE_ONLY_IN_BURST", "1") == "1"
SLATE_SCAN_MAX_PLAYERS = int(os.environ.get("SLATE_SCAN_MAX_PLAYERS", "260"))

# Cooldown
BET_COOLDOWN_MIN = int(os.environ.get("BET_COOLDOWN_MIN", "180"))
EDGE_JUMP_TO_RESEND = float(os.environ.get("EDGE_JUMP_TO_RESEND", "1.5"))

# Debug
DEBUG_PROP_SAMPLE = os.environ.get("DEBUG_PROP_SAMPLE", "0") == "1"

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

def _role_trend_from_pts(games):
    """
    games rows: (date, pts, min, fg3m, fg3a)
    """
    if not games:
        return 0.0, 0.0, 0.0, 0.0
    long_slice = _slice_last(games, LOOKBACK_GAMES)
    short_slice = _slice_last(games, SHORT_GAMES)

    pts_l = sum(r[1] for r in long_slice) / max(len(long_slice), 1)
    min_l = sum(r[2] for r in long_slice) / max(len(long_slice), 1)
    pts_s = sum(r[1] for r in short_slice) / max(len(short_slice), 1)
    min_s = sum(r[2] for r in short_slice) / max(len(short_slice), 1)

    ppm_l = pts_l / max(min_l, 1e-6)
    ppm_s = pts_s / max(min_s, 1e-6)
    return min_s, min_l, ppm_s, ppm_l

def _role_trend_from_threes(games):
    if not games:
        return 0.0, 0.0, 0.0, 0.0
    long_slice = _slice_last(games, LOOKBACK_GAMES)
    short_slice = _slice_last(games, SHORT_GAMES)

    th_l = sum(r[3] for r in long_slice) / max(len(long_slice), 1)
    min_l = sum(r[2] for r in long_slice) / max(len(long_slice), 1)
    th_s = sum(r[3] for r in short_slice) / max(len(short_slice), 1)
    min_s = sum(r[2] for r in short_slice) / max(len(short_slice), 1)

    tpm_l = th_l / max(min_l, 1e-6)
    tpm_s = th_s / max(min_s, 1e-6)
    return min_s, min_l, tpm_s, tpm_l

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
PROPS_CACHE = {}  # (game_id, vendor, market_key, resolved_prop_type) -> props list
DEBUG_PRINTED_ONCE = set()

PLAYER_NAME_CACHE = {}  # pid -> "First Last"

# For threes, BDL prop_type string varies. We’ll try these until we get data.
THREES_PROP_TYPE_CANDIDATES = [
    "three_pointers",
    "three_pointers_made",
    "threes",
    "fg3m",
    "3pt_made",
]

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

def bdl_last_n_games_stats(player_ids, season: int, n: int):
    """
    Returns: pid -> list of tuples (date, pts, min, fg3m, fg3a)
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
            pts = float(row.get("pts", 0) or 0)
            mins = _parse_minutes(row.get("min"))

            fg3m = float(row.get("fg3m", 0) or 0)
            fg3a = float(row.get("fg3a", 0) or 0)

            if date:
                out[pid].append((date, pts, mins, fg3m, fg3a))

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

# -------------- PROPS FETCH (points + threes) --------------
def _props_cache_key(game_id: int, vendor: str | None, market_key: str, prop_type: str):
    return (int(game_id), vendor or "", market_key, prop_type)

def bdl_player_props(game_id: int, vendor: str | None, market_key: str):
    """
    market_key: 'points' or 'threes'
    returns: props list
    """
    if market_key == "points":
        prop_types_to_try = ["points"]
    else:
        if not ENABLE_THREES:
            return []
        prop_types_to_try = THREES_PROP_TYPE_CANDIDATES

    for prop_type in prop_types_to_try:
        key = _props_cache_key(game_id, vendor, market_key, prop_type)
        if key in PROPS_CACHE:
            props = PROPS_CACHE[key]
            if props:
                return props
            continue

        params = {"game_id": int(game_id), "prop_type": prop_type}
        if vendor:
            params["vendors[]"] = [vendor]

        try:
            resp = _bdl_get("/v2/odds/player_props", params=params)
            props = resp.get("data") or []
        except Exception:
            props = []

        PROPS_CACHE[key] = props

        if DEBUG_PROP_SAMPLE and props and (market_key not in DEBUG_PRINTED_ONCE):
            print(f"[DEBUG] SAMPLE {market_key.upper()} PROP ROW ({prop_type}):", json.dumps(props[0])[:2000])
            DEBUG_PRINTED_ONCE.add(market_key)

        if props:
            return props

    return []

def _pick_main_line(rows_for_player, market_key: str):
    if not rows_for_player:
        return None

    if market_key == "points":
        min_line, max_line = MIN_POINTS_LINE, MAX_POINTS_LINE
    else:
        min_line, max_line = MIN_THREES_LINE, MAX_THREES_LINE

    candidates = []
    for pp in rows_for_player:
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

def line_for_player(game_id: int, player_id: int, market_key: str):
    for v in BOOK_VENDORS + [None]:
        props = bdl_player_props(game_id, v, market_key)
        if not props:
            continue

        rows = []
        for pp in props:
            try:
                if int(pp.get("player_id", -1)) != int(player_id):
                    continue
            except Exception:
                continue
            rows.append(pp)

        line = _pick_main_line(rows, market_key)
        if line is not None:
            return float(line)

    return None

# -------------------- PROJECTION CORE --------------------
def _avg_std(values):
    if not values:
        return 0.0, 0.0
    n = len(values)
    mu = sum(values) / n
    var = sum((x - mu) ** 2 for x in values) / max(n, 1)
    return mu, math.sqrt(var)

def compute_projection_and_prob(games_all, line, market_key: str, injury_boost_pts=0.0, injury_boost_min=0.0):
    """
    games rows: (date, pts, min, fg3m, fg3a)
    """
    base_slice = _slice_last(games_all, BASELINE_GAMES)
    l10_slice = _slice_last(games_all, LOOKBACK_GAMES)
    l3_slice = _slice_last(games_all, SHORT_GAMES)

    if market_key == "points":
        base_vals = [r[1] for r in base_slice]
        l10_vals = [r[1] for r in l10_slice]
        l3_vals = [r[1] for r in l3_slice]
        mins_l10 = [r[2] for r in l10_slice]
        std_floor = STD_FLOOR
        min_gap = LINE_MIN_GAP
        min_edge = MIN_EDGE
        min_prob = MIN_PROB
    else:
        base_vals = [r[3] for r in base_slice]
        l10_vals = [r[3] for r in l10_slice]
        l3_vals = [r[3] for r in l3_slice]
        mins_l10 = [r[2] for r in l10_slice]
        std_floor = STD_FLOOR_THREES
        min_gap = LINE_MIN_GAP_THREES
        min_edge = MIN_EDGE_THREES
        min_prob = MIN_PROB_THREES

    base_avg, base_std = _avg_std(base_vals)
    l10_avg, l10_std = _avg_std(l10_vals)
    l3_avg, _ = _avg_std(l3_vals)
    l10_min = sum(mins_l10) / max(len(mins_l10), 1)

    sigma = max(std_floor, l10_std if l10_std > 0 else base_std if base_std > 0 else std_floor)

    proj = (W_BASE * base_avg) + (W_L10 * l10_avg) + (W_L3 * l3_avg) + (W_LINE * line)

    # injury minutes boost scales with "per-minute" for that market
    per_min = l10_avg / max(l10_min, 1e-6)

    proj += injury_boost_pts
    proj += (injury_boost_min * per_min * 0.20)

    edge = proj - line
    z = (proj - line) / sigma
    prob_over = _norm_cdf(z)

    # extra guardrail: if L10 is WAY above line, skip “obvious” repeats
    if (l10_avg - line) > min_gap:
        return None

    if edge < min_edge or prob_over < min_prob:
        return None

    return {
        "proj": float(proj),
        "edge": float(edge),
        "prob_over": float(prob_over),
        "base_avg": float(base_avg),
        "l10_avg": float(l10_avg),
        "l3_avg": float(l3_avg),
        "l10_min": float(l10_min),
        "sigma": float(sigma),
        "per_min": float(per_min),
    }

# -------------------- INJURY ENGINE --------------------
def build_injury_edges(team_short, injured_name, injured_status, exclude_names_lower, now_et, market_key: str):
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

    inj_games = bdl_last_n_games_stats([injured_pid], season, BASELINE_GAMES).get(injured_pid, [])
    if len(inj_games) < 3:
        return []

    # Vacancy (still based on pts+min), used only for distributing “impact”:
    ip10 = sum(r[1] for r in _slice_last(inj_games, LOOKBACK_GAMES)) / max(len(_slice_last(inj_games, LOOKBACK_GAMES)), 1)
    im10 = sum(r[2] for r in _slice_last(inj_games, LOOKBACK_GAMES)) / max(len(_slice_last(inj_games, LOOKBACK_GAMES)), 1)

    status = (injured_status or "").lower()
    STATUS_MULT = {"out": 1.0, "doubtful": 0.8, "questionable": 0.55}.get(status, 0.65)

    vac_pts = ip10 * STATUS_MULT
    vac_min = im10 * STATUS_MULT
    if not ((vac_min >= MIN_VAC_MIN) or (vac_pts >= MIN_VAC_PTS)):
        return []

    trigger_strength = min(100.0, (vac_min * 1.2 + vac_pts * 1.5))

    cand_ids = [pid for pid, _ in roster_tuples]
    stats = bdl_last_n_games_stats(cand_ids, season, BASELINE_GAMES)
    game_ids = bdl_games_today_ids(now_et)
    if not game_ids:
        return []

    ideas = []
    for pid, nm in roster_tuples:
        games = stats.get(pid, [])
        if len(games) < 8:
            continue

        # role trend differs per market
        if market_key == "points":
            min_s, min_l, r_s, r_l = _role_trend_from_pts(games)
        else:
            min_s, min_l, r_s, r_l = _role_trend_from_threes(games)

        min_delta = min_s - min_l
        rate_delta = r_s - r_l

        # absorption = who likely benefits from minutes/usage
        l10_min = sum(r[2] for r in _slice_last(games, LOOKBACK_GAMES)) / max(len(_slice_last(games, LOOKBACK_GAMES)), 1)

        absorption = 0.0
        if l10_min >= 28:
            absorption += 0.30
        if l10_min >= 34:
            absorption += 0.10
        if min_delta >= 2.0:
            absorption += 0.15
        if rate_delta > (0.05 if market_key == "points" else 0.015):
            absorption += 0.10
        absorption = min(0.65, absorption)

        # find line for today
        line = None
        use_gid = None
        for gid in game_ids:
            line = line_for_player(gid, pid, market_key)
            if line is not None:
                use_gid = gid
                break
        if line is None:
            continue

        injury_boost_pts = min(BOOST_CAP_PTS, vac_pts * absorption * (0.65 if market_key == "points" else 0.25))
        injury_boost_min = min(BOOST_CAP_MIN, vac_min * absorption * 0.25)

        out = compute_projection_and_prob(
            games_all=games,
            line=line,
            market_key=market_key,
            injury_boost_pts=injury_boost_pts,
            injury_boost_min=injury_boost_min,
        )
        if out is None:
            continue

        # extra filter for weak trends + low edge
        if min_delta < MIN_DELTA_FLOOR and out["edge"] < ((MIN_EDGE if market_key == "points" else MIN_EDGE_THREES) + 1.5):
            continue

        why = (
            f"TriggerStrength {trigger_strength:.0f} | Absorb {absorption:.2f}. "
            f"{injured_name} {injured_status.upper()} vacates ~{vac_pts:.1f} pts / {vac_min:.1f} min. "
            f"{nm} base(L{BASELINE_GAMES}) {out['base_avg']:.1f}, L10 {out['l10_avg']:.1f}, L3 {out['l3_avg']:.1f} "
            f"(mins L10 {out['l10_min']:.1f}). Role Δmin={min_delta:+.1f}, Δrate={rate_delta:+.2f}. "
            f"Proj {out['proj']:.1f} vs MAIN line {line:.1f} | edge +{out['edge']:.1f} | P≈{out['prob_over']*100:.0f}%."
        )

        ideas.append({
            "section": "injury",
            "market": market_key,
            "player_name": nm,
            "player_id": pid,
            "line": float(line),
            "proj": float(out["proj"]),
            "edge": float(out["edge"]),
            "prob_over": float(out["prob_over"]),
            "trigger_strength": float(trigger_strength),
            "trigger": f"{injured_name} ({team_short}) {injured_status}",
            "why": why,
            "game_id": use_gid,
        })

    ideas.sort(key=lambda x: (x["trigger_strength"], x["edge"], x["prob_over"]), reverse=True)
    return ideas[:INJURY_TOPN]

# -------------------- SLATE SCAN ENGINE --------------------
def slate_scan_edges(now_et, market_key: str):
    if not ENABLE_SLATE_SCAN:
        return []
    if SLATE_ONLY_IN_BURST and (not _in_burst_window(now_et)):
        return []
    if market_key == "threes" and not ENABLE_THREES:
        return []

    season = _season_year(now_et)
    game_ids = bdl_games_today_ids(now_et)
    if not game_ids:
        return []

    player_to_best_line = {}  # pid -> (line, gid)
    pulled = 0

    for gid in game_ids:
        props = []
        for v in BOOK_VENDORS + [None]:
            props = bdl_player_props(gid, v, market_key)
            if props:
                break
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
            if pid in player_to_best_line:
                continue
            line = _pick_main_line(rows, market_key)
            if line is None:
                continue
            player_to_best_line[pid] = (float(line), int(gid))
            pulled += 1
            if pulled >= SLATE_SCAN_MAX_PLAYERS:
                break
        if pulled >= SLATE_SCAN_MAX_PLAYERS:
            break

    if not player_to_best_line:
        return []

    pids = list(player_to_best_line.keys())
    stats = bdl_last_n_games_stats(pids, season, BASELINE_GAMES)  # fills name cache

    ideas = []
    for pid in pids:
        games = stats.get(pid, [])
        if len(games) < 8:
            continue

        line, gid = player_to_best_line[pid]

        # market-specific trends
        if market_key == "points":
            min_s, min_l, r_s, r_l = _role_trend_from_pts(games)
        else:
            min_s, min_l, r_s, r_l = _role_trend_from_threes(games)

        min_delta = min_s - min_l
        rate_delta = r_s - r_l

        out = compute_projection_and_prob(games_all=games, line=line, market_key=market_key)
        if out is None:
            continue

        if min_delta < MIN_DELTA_FLOOR and out["edge"] < ((MIN_EDGE if market_key == "points" else MIN_EDGE_THREES) + 2.0):
            continue

        name = PLAYER_NAME_CACHE.get(int(pid), f"Player {pid}")

        why = (
            f"SlateScan. base(L{BASELINE_GAMES}) {out['base_avg']:.1f}, L10 {out['l10_avg']:.1f}, L3 {out['l3_avg']:.1f} "
            f"(mins L10 {out['l10_min']:.1f}). Role Δmin={min_delta:+.1f}, Δrate={rate_delta:+.2f}. "
            f"Proj {out['proj']:.1f} vs MAIN line {line:.1f} | edge +{out['edge']:.1f} | P≈{out['prob_over']*100:.0f}%."
        )

        ideas.append({
            "section": "slate",
            "market": market_key,
            "player_name": name,
            "player_id": pid,
            "line": float(line),
            "proj": float(out["proj"]),
            "edge": float(out["edge"]),
            "prob_over": float(out["prob_over"]),
            "trigger_strength": 0.0,
            "trigger": "No injury trigger (league-wide scan)",
            "why": why,
            "game_id": gid,
        })

    ideas.sort(key=lambda x: (x["edge"], x["prob_over"]), reverse=True)
    return ideas[:SLATE_TOPN]

# -------------------- COOLDOWN FILTER --------------------
def apply_cooldown(state, ideas, now_ts: int):
    sent = state.get("sent_bets", {}) or {}
    cooldown_sec = BET_COOLDOWN_MIN * 60

    kept = []
    for i in ideas:
        key = f"{i['section']}|{i['market']}|{int(i['player_id'])}|{i['line']:.1f}"
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
        key = f"{i['section']}|{i['market']}|{int(i['player_id'])}|{i['line']:.1f}"
        sent[key] = {"ts": now_ts, "edge": float(i["edge"]), "line": float(i["line"])}
    state["sent_bets"] = sent

# -------------------- MAIN --------------------
def run():
    now_et = _now_et()
    ts_et = now_et.strftime("%Y-%m-%d %I:%M %p ET")
    now_ts = int(now_et.timestamp())

    print(
        f"[BOOT] ts={ts_et} TEST_MODE={int(TEST_MODE)} "
        f"MARKETS={','.join(MARKETS)} BOOK_VENDORS={','.join(BOOK_VENDORS)} "
        f"ENABLE_SLATE_SCAN={int(ENABLE_SLATE_SCAN)} ENABLE_THREES={int(ENABLE_THREES)}"
    )

    if TEST_MODE:
        send_one(f"✅ NBA betting agent test OK ({ts_et})")
        return

    state = load_state()
    old_players = state.get("players", {})

    sr = fetch_sportradar_injuries()
    new_players = parse_injuries(sr)

    exclude_names_lower = {_clean_name(v.get("name", "")) for v in new_players.values() if v.get("name")}

    triggers = []
    ideas_all = []

    # ----- Injury-triggered edges (for each enabled market) -----
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

        any_added = False
        for market_key in MARKETS:
            if market_key == "threes" and not ENABLE_THREES:
                continue
            if market_key not in ("points", "threes"):
                continue

            added = build_injury_edges(
                team_short=team_short,
                injured_name=injured_name,
                injured_status=injured_status,
                exclude_names_lower=exclude_names_lower | {_clean_name(injured_name)},
                now_et=now_et,
                market_key=market_key,
            )
            if added:
                ideas_all.extend(added)
                any_added = True

        if any_added:
            triggers.append(f"{injured_name} ({team_short}) {injured_status}")

    # ----- Slate scan edges -----
    for market_key in MARKETS:
        if market_key == "threes" and not ENABLE_THREES:
            continue
        if market_key not in ("points", "threes"):
            continue
        ideas_all.extend(slate_scan_edges(now_et, market_key))

    # Dedupe by (section, market, player)
    best = {}
    for i in ideas_all:
        k = (i["section"], i["market"], int(i["player_id"]))
        if (k not in best) or ((i["edge"], i["prob_over"]) > (best[k]["edge"], best[k]["prob_over"])):
            best[k] = i
    ideas_all = list(best.values())

    # cooldown
    ideas_all = apply_cooldown(state, ideas_all, now_ts)

    # Slice outputs per section + market
    out = []
    for market_key in ("points", "threes"):
        if market_key == "threes" and not ENABLE_THREES:
            continue
        if market_key not in MARKETS:
            continue

        inj = [i for i in ideas_all if i["section"] == "injury" and i["market"] == market_key]
        slt = [i for i in ideas_all if i["section"] == "slate" and i["market"] == market_key]

        inj.sort(key=lambda x: (x["trigger_strength"], x["edge"], x["prob_over"]), reverse=True)
        slt.sort(key=lambda x: (x["edge"], x["prob_over"]), reverse=True)

        out.extend(inj[:INJURY_TOPN])
        out.extend(slt[:SLATE_TOPN])

    # final cap
    out.sort(key=lambda x: (x["edge"], x["prob_over"]), reverse=True)
    final_out = out[:MAX_BET_IDEAS]

    if final_out:
        def market_title(m):
            return "Points" if m == "points" else "3PT Made"

        msg = [f"💰 FanDuel Props ({ts_et})", ""]

        # group by market
        for market_key in ("points", "threes"):
            group = [i for i in final_out if i["market"] == market_key]
            if not group:
                continue

            msg.append(f"🏷️ {market_title(market_key)}")
            msg.append("")

            inj_group = [i for i in group if i["section"] == "injury"]
            slt_group = [i for i in group if i["section"] == "slate"]

            if inj_group:
                msg.append("🚑 Injury-Triggered Plays:")
                if triggers:
                    msg.append("Triggers:")
                    for t in triggers[:8]:
                        msg.append(f"- {t}")
                    if len(triggers) > 8:
                        msg.append(f"- …and {len(triggers)-8} more")
                msg.append("")
                for i in inj_group:
                    msg.append(f"• {i['player_name']} OVER {i['line']:.1f}  (edge +{i['edge']:.1f}, P≈{i['prob_over']*100:.0f}%)")
                    msg.append(f"  Trigger: {i['trigger']}")
                    msg.append(f"  Why: {i['why']}")
                    msg.append("")

            if slt_group:
                msg.append("🌎 League-Wide Slate Scan (no injury required):")
                msg.append("")
                for i in slt_group:
                    msg.append(f"• {i['player_name']} OVER {i['line']:.1f}  (edge +{i['edge']:.1f}, P≈{i['prob_over']*100:.0f}%)")
                    msg.append(f"  Why: {i['why']}")
                    msg.append("")

            msg.append("")

        send_chunked("\n".join(msg).strip())
        record_sent(state, final_out, now_ts)
    else:
        if SEND_NO_EDGE_PING and _in_burst_window(now_et):
            send_one(f"🧠 No edges this run. ({ts_et})")

    state["players"] = new_players
    save_state(state)

if __name__ == "__main__":
    run()
