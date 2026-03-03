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

BOOK_VENDOR_RAW = os.environ.get("BOOK_VENDORS", os.environ.get("BOOK_VENDOR", "fanduel")).strip().lower()
BOOK_VENDORS = [v.strip() for v in BOOK_VENDOR_RAW.split(",") if v.strip()]

# Which prop types to run (e.g., "points,threes")
PROP_TYPES_RAW = os.environ.get("PROP_TYPES", "points").strip().lower()
PROP_TYPES = [p.strip() for p in PROP_TYPES_RAW.split(",") if p.strip()]

# Multi-horizon windows
BASELINE_GAMES = int(os.environ.get("BASELINE_GAMES", "30"))
LOOKBACK_GAMES = int(os.environ.get("LOOKBACK_GAMES", "10"))
SHORT_GAMES = int(os.environ.get("SHORT_GAMES", "3"))

# Projection blend weights
W_BASE = float(os.environ.get("W_BASE", "0.45"))
W_L10 = float(os.environ.get("W_L10", "0.35"))
W_L3 = float(os.environ.get("W_L3", "0.10"))
W_LINE = float(os.environ.get("W_LINE", "0.10"))

# Output sizing
MIN_PER_MARKET = int(os.environ.get("MIN_PER_MARKET", "0"))
MAX_PER_MARKET = int(os.environ.get("MAX_PER_MARKET", "6"))
MAX_BET_IDEAS = int(os.environ.get("MAX_BET_IDEAS", "10"))

# Thresholds (global)
MIN_EDGE = float(os.environ.get("MIN_EDGE", "2.5"))
MIN_PROB = float(os.environ.get("MIN_PROB", "0.62"))
STD_FLOOR = float(os.environ.get("STD_FLOOR", "5.0"))

# Optional separate thresholds for threes (if unset, uses global)
MIN_EDGE_THREES = float(os.environ.get("MIN_EDGE_THREES", str(MIN_EDGE)))
MIN_PROB_THREES = float(os.environ.get("MIN_PROB_THREES", str(MIN_PROB)))

# Guardrails (points + generally safe for most markets)
MIN_POINTS_LINE = float(os.environ.get("MIN_POINTS_LINE", "0.5"))
MAX_POINTS_LINE = float(os.environ.get("MAX_POINTS_LINE", "60.0"))
LINE_MIN_GAP = float(os.environ.get("LINE_MIN_GAP", "8.0"))
MIN_DELTA_FLOOR = float(os.environ.get("MIN_DELTA_FLOOR", "-3.0"))

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
SLATE_ONLY_IN_BURST = os.environ.get("SLATE_ONLY_IN_BURST", "0") == "1"
SLATE_SCAN_MAX_PLAYERS = int(os.environ.get("SLATE_SCAN_MAX_PLAYERS", "260"))

# Cooldown
BET_COOLDOWN_MIN = int(os.environ.get("BET_COOLDOWN_MIN", "180"))
EDGE_JUMP_TO_RESEND = float(os.environ.get("EDGE_JUMP_TO_RESEND", "1.5"))

# Strict injury/game matching toggle (your script prints this already)
STRICT_INJURY_GAME_MATCH = os.environ.get("STRICT_INJURY_GAME_MATCH", "1") == "1"

# Debug: print sample prop rows for specific prop types (comma list)
DEBUG_PROP_SAMPLE_TYPES_RAW = os.environ.get("DEBUG_PROP_SAMPLE_TYPES", "").strip().lower()
DEBUG_PROP_SAMPLE_TYPES = {x.strip() for x in DEBUG_PROP_SAMPLE_TYPES_RAW.split(",") if x.strip()}


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

def avg_pts_min_std(games):
    if not games:
        return 0.0, 0.0, 0.0
    pts = [x[1] for x in games]
    mins = [x[2] for x in games]
    n = len(pts)
    pts_avg = sum(pts) / n
    min_avg = sum(mins) / n
    var = sum((p - pts_avg) ** 2 for p in pts) / max(n, 1)
    return pts_avg, min_avg, math.sqrt(var)

def _slice_last(games, n):
    if not games:
        return []
    return games[-min(len(games), n):]

def _role_trend(games):
    if not games:
        return 0.0, 0.0, 0.0, 0.0
    long_slice = _slice_last(games, LOOKBACK_GAMES)
    short_slice = _slice_last(games, SHORT_GAMES)

    pts_l, min_l, _ = avg_pts_min_std(long_slice)
    pts_s, min_s, _ = avg_pts_min_std(short_slice)
    rate_l = pts_l / max(min_l, 1e-6)
    rate_s = pts_s / max(min_s, 1e-6)
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
PROPS_CACHE = {}
PLAYER_NAME_CACHE = {}  # pid -> name
_DEBUG_PRINTED = set()  # (prop_type, vendor) printed already

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
            stat_val = float(row.get("pts", 0) or 0)  # default points
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

def _stat_from_row(row: dict, prop_type: str) -> float:
    """
    Return the historical stat value for a given prop_type from a BDL stats row.
    Supports: points, threes.
    """
    prop_type = (prop_type or "").lower()
    if prop_type in ("points", "pts"):
        return float(row.get("pts", 0) or 0)
    if prop_type in ("threes", "3pt", "3pm", "fg3m", "three_pointers_made"):
        # BDL stat key is usually "fg3m" for 3PT made
        return float(row.get("fg3m", 0) or 0)
    # fallback to points
    return float(row.get("pts", 0) or 0)

def bdl_last_n_games_stats_for_prop(player_ids, season: int, n: int, prop_type: str):
    """
    Same as bdl_last_n_games_stats but uses prop_type-specific stat extraction (points/threes).
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

def bdl_player_props(game_id: int, vendor: str | None, prop_type: str):
    """
    Pull props for a game/vendor/prop_type. Cached.
    """
    key = (int(game_id), vendor or "", prop_type)
    if key in PROPS_CACHE:
        return PROPS_CACHE[key]

    params = {"game_id": int(game_id), "prop_type": prop_type}
    if vendor:
        params["vendors[]"] = [vendor]

    try:
        resp = _bdl_get("/v2/odds/player_props", params=params)
        props = resp.get("data") or []
    except Exception:
        props = []

    # debug sample row (one per prop_type+vendor)
    if prop_type in DEBUG_PROP_SAMPLE_TYPES and props and (prop_type, vendor or "NO_VENDOR") not in _DEBUG_PRINTED:
        print(f"[DEBUG] SAMPLE PROP ROW ({prop_type}, vendor={vendor or 'NO_VENDOR'}): {json.dumps(props[0])[:2000]}")
        _DEBUG_PRINTED.add((prop_type, vendor or "NO_VENDOR"))

    PROPS_CACHE[key] = props
    return props

def _pick_main_line(rows_for_player):
    """
    Choose 'main' line (closest to -110/-110 if present), otherwise median.
    Works for both points and threes.
    """
    if not rows_for_player:
        return None
    candidates = []
    for pp in rows_for_player:
        market = pp.get("market") or {}
        if (market.get("type") or "").lower() != "over_under":
            continue
        try:
            line = float(pp.get("line_value"))
        except Exception:
            continue
        if line < MIN_POINTS_LINE or line > MAX_POINTS_LINE:
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

def line_for_player(game_id: int, player_id: int, prop_type: str):
    """
    Return a best/main line for player. Tries:
      - vendor-filtered for each vendor
      - then NO vendor filter fallback
    NOTE: This vendorless fallback is especially important for threes.
    """
    # 1) vendor-filtered tries
    for v in BOOK_VENDORS:
        props = bdl_player_props(game_id, v, prop_type)
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
        line = _pick_main_line(rows)
        if line is not None:
            return float(line), v

    # 2) vendorless fallback
    props = bdl_player_props(game_id, None, prop_type)
    if props:
        rows = []
        for pp in props:
            try:
                if int(pp.get("player_id", -1)) != int(player_id):
                    continue
            except Exception:
                continue
            rows.append(pp)
        line = _pick_main_line(rows)
        if line is not None:
            return float(line), "MAIN"

    return None, None


# -------------------- PROJECTION CORE --------------------
def compute_projection_and_prob(games_all, line, prop_type: str, injury_boost_stat=0.0, injury_boost_min=0.0):
    """
    games_all = list of (date, stat_value, minutes)
    For threes, stat_value = fg3m.
    """
    base_slice = _slice_last(games_all, BASELINE_GAMES)
    l10_slice = _slice_last(games_all, LOOKBACK_GAMES)
    l3_slice = _slice_last(games_all, SHORT_GAMES)

    base_avg, _, base_std = avg_pts_min_std(base_slice)
    l10_avg, l10_min, l10_std = avg_pts_min_std(l10_slice)
    l3_avg, _, _ = avg_pts_min_std(l3_slice)

    sigma = max(STD_FLOOR, (l10_std if l10_std > 0 else base_std if base_std > 0 else STD_FLOOR))
    proj = (W_BASE * base_avg) + (W_L10 * l10_avg) + (W_L3 * l3_avg) + (W_LINE * line)

    rate = l10_avg / max(l10_min, 1e-6)
    proj += injury_boost_stat
    proj += (injury_boost_min * rate * 0.20)

    edge = proj - line
    z = (proj - line) / sigma
    prob_over = _norm_cdf(z)
    return proj, edge, prob_over, (base_avg, l10_avg, l3_avg, l10_min, sigma, rate)


# -------------------- INJURY ENGINE --------------------
def build_injury_edges(team_short, injured_name, injured_status, exclude_names_lower, now_et, prop_type: str):
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

    inj_games = bdl_last_n_games_stats_for_prop([injured_pid], season, BASELINE_GAMES, prop_type).get(injured_pid, [])
    ip10, im10, _ = avg_pts_min_std(_slice_last(inj_games, LOOKBACK_GAMES))
    if len(inj_games) < 3:
        return []

    status = (injured_status or "").lower()
    STATUS_MULT = {"out": 1.0, "doubtful": 0.8, "questionable": 0.55}.get(status, 0.65)

    vac_stat = ip10 * STATUS_MULT
    vac_min = im10 * STATUS_MULT
    if not ((vac_min >= MIN_VAC_MIN) or (vac_stat >= MIN_VAC_PTS)):
        return []

    trigger_strength = min(100.0, (vac_min * 1.2 + vac_stat * 1.5))

    cand_ids = [pid for pid, _ in roster_tuples]
    stats = bdl_last_n_games_stats_for_prop(cand_ids, season, BASELINE_GAMES, prop_type)

    game_ids = bdl_games_today_ids(now_et)
    if not game_ids:
        return []

    # thresholds per market
    if prop_type == "threes":
        min_edge_use = MIN_EDGE_THREES
        min_prob_use = MIN_PROB_THREES
        stat_label = "3PT Made"
    else:
        min_edge_use = MIN_EDGE
        min_prob_use = MIN_PROB
        stat_label = "Points"

    ideas = []
    for pid, nm in roster_tuples:
        games = stats.get(pid, [])
        if len(games) < 6:
            continue

        min_s, min_l, rate_s, rate_l = _role_trend(games)
        min_delta = min_s - min_l
        rate_delta = rate_s - rate_l

        l10_avg, l10_min, _ = avg_pts_min_std(_slice_last(games, LOOKBACK_GAMES))
        if l10_min < 10:
            continue

        absorption = 0.0
        if l10_min >= 28:
            absorption += 0.30
        if l10_min >= 34:
            absorption += 0.10
        if min_delta >= 2.0:
            absorption += 0.15
        if rate_delta > 0.05:
            absorption += 0.10
        absorption = min(0.65, absorption)

        line = None
        use_gid = None
        used_vendor = None
        for gid in game_ids:
            line, used_vendor = line_for_player(gid, pid, prop_type)
            if line is not None:
                use_gid = gid
                break
        if line is None:
            continue

        # Guardrail: avoid cases where L10 average dwarfs line by a huge gap (often mis-mapped markets/lines)
        if (l10_avg - line) > LINE_MIN_GAP:
            continue

        injury_boost_stat = min(BOOST_CAP_PTS, vac_stat * absorption * 0.65)
        injury_boost_min = min(BOOST_CAP_MIN, vac_min * absorption * 0.25)

        proj, edge, prob_over, aux = compute_projection_and_prob(
            games_all=games,
            line=line,
            prop_type=prop_type,
            injury_boost_stat=injury_boost_stat,
            injury_boost_min=injury_boost_min
        )
        base_avg, l10_avg2, l3_avg, l10_min2, _, _ = aux

        if edge < min_edge_use or prob_over < min_prob_use:
            continue
        if min_delta < MIN_DELTA_FLOOR and edge < (min_edge_use + 1.5):
            continue

        why = (
            f"TriggerStrength {trigger_strength:.0f} | Absorb {absorption:.2f}. "
            f"{injured_name} {injured_status.upper()} vacates ~{vac_stat:.1f} {stat_label} / {vac_min:.1f} min. "
            f"{nm} base(L{BASELINE_GAMES}) {base_avg:.1f}, L10 {l10_avg2:.1f}, L3 {l3_avg:.1f} "
            f"(mins L10 {l10_min2:.1f}). Role Δmin={min_delta:+.1f}, Δrate={rate_delta:+.2f}. "
            f"Proj {proj:.1f} vs {used_vendor} line {line:.1f} | edge +{edge:.1f} | P≈{prob_over*100:.0f}%. "
            f"[prop_type={prop_type}]"
        )

        ideas.append({
            "section": "injury",
            "prop_type": prop_type,
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
            "vendor": used_vendor,
        })

    ideas.sort(key=lambda x: (x["trigger_strength"], x["edge"], x["prob_over"]), reverse=True)
    return ideas[:MAX_PER_MARKET]


# -------------------- SLATE SCAN ENGINE --------------------
def slate_scan_edges(now_et, prop_type: str):
    if not ENABLE_SLATE_SCAN:
        return []
    if SLATE_ONLY_IN_BURST and (not _in_burst_window(now_et)):
        return []

    season = _season_year(now_et)
    game_ids = bdl_games_today_ids(now_et)
    if not game_ids:
        return []

    player_to_best_line = {}  # pid -> (line, gid, vendor)
    pulled = 0

    for gid in game_ids:
        # Try vendor-filtered first, but also allow vendorless fallback.
        props = []
        used_vendor = None

        for v in BOOK_VENDORS:
            props = bdl_player_props(gid, v, prop_type)
            if props:
                used_vendor = v
                break

        if not props:
            props = bdl_player_props(gid, None, prop_type)
            if props:
                used_vendor = "MAIN"

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
            line = _pick_main_line(rows)
            if line is None:
                continue
            player_to_best_line[pid] = (float(line), int(gid), used_vendor)
            pulled += 1
            if pulled >= SLATE_SCAN_MAX_PLAYERS:
                break
        if pulled >= SLATE_SCAN_MAX_PLAYERS:
            break

    if not player_to_best_line:
        return []

    pids = list(player_to_best_line.keys())
    stats = bdl_last_n_games_stats_for_prop(pids, season, BASELINE_GAMES, prop_type)  # fills PLAYER_NAME_CACHE

    if prop_type == "threes":
        min_edge_use = MIN_EDGE_THREES
        min_prob_use = MIN_PROB_THREES
        stat_label = "3PT Made"
    else:
        min_edge_use = MIN_EDGE
        min_prob_use = MIN_PROB
        stat_label = "Points"

    ideas = []
    for pid in pids:
        games = stats.get(pid, [])
        if len(games) < 8:
            continue

        line, gid, used_vendor = player_to_best_line[pid]

        l10_avg, l10_min, _ = avg_pts_min_std(_slice_last(games, LOOKBACK_GAMES))
        if l10_min < 10:
            continue
        if (l10_avg - line) > LINE_MIN_GAP:
            continue

        min_s, min_l, rate_s, rate_l = _role_trend(games)
        min_delta = min_s - min_l
        rate_delta = rate_s - rate_l

        proj, edge, prob_over, aux = compute_projection_and_prob(games_all=games, line=line, prop_type=prop_type)
        base_avg, l10_avg2, l3_avg, l10_min2, _, _ = aux

        if edge < min_edge_use or prob_over < min_prob_use:
            continue
        if min_delta < MIN_DELTA_FLOOR and edge < (min_edge_use + 2.0):
            continue

        name = PLAYER_NAME_CACHE.get(int(pid), f"Player {pid}")

        why = (
            f"SlateScan. base(L{BASELINE_GAMES}) {base_avg:.1f}, L10 {l10_avg2:.1f}, L3 {l3_avg:.1f} "
            f"(mins L10 {l10_min2:.1f}). Role Δmin={min_delta:+.1f}, Δrate={rate_delta:+.2f}. "
            f"Proj {proj:.1f} vs {used_vendor} line {line:.1f} | edge +{edge:.1f} | P≈{prob_over*100:.0f}%. "
            f"[prop_type={prop_type}]"
        )

        ideas.append({
            "section": "slate",
            "prop_type": prop_type,
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
            "vendor": used_vendor,
        })

    ideas.sort(key=lambda x: (x["edge"], x["prob_over"]), reverse=True)
    return ideas[:MAX_PER_MARKET]


# -------------------- COOLDOWN FILTER --------------------
def apply_cooldown(state, ideas, now_ts: int):
    sent = state.get("sent_bets", {}) or {}
    cooldown_sec = BET_COOLDOWN_MIN * 60

    kept = []
    for i in ideas:
        key = f"{i['section']}|{i['prop_type']}|{int(i['player_id'])}|{i['line']:.1f}"
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
        key = f"{i['section']}|{i['prop_type']}|{int(i['player_id'])}|{i['line']:.1f}"
        sent[key] = {"ts": now_ts, "edge": float(i["edge"]), "line": float(i["line"])}
    state["sent_bets"] = sent


# -------------------- MAIN --------------------
def run():
    now_et = _now_et()
    ts_et = now_et.strftime("%Y-%m-%d %I:%M %p ET")
    now_ts = int(now_et.timestamp())

    print(
        f"[BOOT] ts={ts_et} TEST_MODE={int(TEST_MODE)} "
        f"PROP_TYPES={','.join(PROP_TYPES)} MIN_PER_MARKET={MIN_PER_MARKET} MAX_PER_MARKET={MAX_PER_MARKET} "
        f"MAX_BET_IDEAS={MAX_BET_IDEAS} BOOK_VENDORS={','.join(BOOK_VENDORS)} "
        f"ENABLE_SLATE_SCAN={int(ENABLE_SLATE_SCAN)} STRICT_INJURY_GAME_MATCH={int(STRICT_INJURY_GAME_MATCH)}"
    )

    if TEST_MODE:
        send_one(f"✅ NBA betting agent test OK ({ts_et})")
        return

    state = load_state()
    old_players = state.get("players", {})

    sr = fetch_sportradar_injuries()
    new_players = parse_injuries(sr)

    exclude_names_lower = {_clean_name(v.get("name", "")) for v in new_players.values() if v.get("name")}

    # Build triggers once (injury changes only)
    triggers = []
    injury_triggers = []
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

        triggers.append(f"{injured_name} ({team_short}) {injured_status}")
        injury_triggers.append((team_short, injured_name, injured_status))

    all_ideas = []

    for prop_type in PROP_TYPES:
        # Injury ideas
        injury_ideas = []
        for (team_short, injured_name, injured_status) in injury_triggers:
            ideas = build_injury_edges(
                team_short=team_short,
                injured_name=injured_name,
                injured_status=injured_status,
                exclude_names_lower=exclude_names_lower | {_clean_name(injured_name)},
                now_et=now_et,
                prop_type=prop_type
            )
            injury_ideas.extend(ideas)

        # Slate ideas
        slate_ideas = slate_scan_edges(now_et, prop_type=prop_type)

        # Dedup per player within this prop_type (keep best edge/prob)
        best = {}
        for i in (injury_ideas + slate_ideas):
            k = (i["section"], i["prop_type"], int(i["player_id"]))
            if (k not in best) or ((i["edge"], i["prob_over"]) > (best[k]["edge"], best[k]["prob_over"])):
                best[k] = i

        merged = list(best.values())
        all_ideas.extend(merged)

    # Apply cooldown across everything
    all_ideas = apply_cooldown(state, all_ideas, now_ts)

    # Organize output by market
    out_by_market = {p: {"injury": [], "slate": []} for p in PROP_TYPES}
    for i in all_ideas:
        pt = i["prop_type"]
        if pt not in out_by_market:
            continue
        out_by_market[pt][i["section"]].append(i)

    # Sort and cap per market
    for pt in list(out_by_market.keys()):
        out_by_market[pt]["injury"].sort(key=lambda x: (x["trigger_strength"], x["edge"], x["prob_over"]), reverse=True)
        out_by_market[pt]["slate"].sort(key=lambda x: (x["edge"], x["prob_over"]), reverse=True)

        out_by_market[pt]["injury"] = out_by_market[pt]["injury"][:MAX_PER_MARKET]
        out_by_market[pt]["slate"] = out_by_market[pt]["slate"][:MAX_PER_MARKET]

        # Apply MIN_PER_MARKET gating (but never suppress other markets)
        if MIN_PER_MARKET > 0:
            total_here = len(out_by_market[pt]["injury"]) + len(out_by_market[pt]["slate"])
            if total_here < MIN_PER_MARKET:
                out_by_market[pt]["injury"] = []
                out_by_market[pt]["slate"] = []

    # Build final message with overall cap
    msg = [f"💰 FanDuel Props ({ts_et})", ""]

    any_output = False
    sent_list = []

    for pt in PROP_TYPES:
        inj = out_by_market[pt]["injury"]
        slt = out_by_market[pt]["slate"]
        if not inj and not slt:
            continue

        any_output = True
        header = "Points" if pt == "points" else ("3PT Made" if pt == "threes" else pt)
        msg.append(f"🏷️ {header}")
        msg.append("")

        if inj:
            msg.append("🚑 Injury-Triggered Plays:")
            if triggers:
                msg.append("Triggers:")
                for t in triggers[:8]:
                    msg.append(f"- {t}")
                if len(triggers) > 8:
                    msg.append(f"- …and {len(triggers)-8} more")
            msg.append("")
            for i in inj:
                msg.append(f"• {i['player_name']} OVER {i['line']:.1f}  (edge +{i['edge']:.1f}, P≈{i['prob_over']*100:.0f}%)")
                msg.append(f"  Trigger: {i['trigger']}")
                msg.append(f"  Why: {i['why']}")
                msg.append("")
                sent_list.append(i)

        if slt:
            msg.append("🌎 League-Wide Slate Scan (no injury required):")
            msg.append("")
            for i in slt:
                msg.append(f"• {i['player_name']} OVER {i['line']:.1f}  (edge +{i['edge']:.1f}, P≈{i['prob_over']*100:.0f}%)")
                msg.append(f"  Why: {i['why']}")
                msg.append("")
                sent_list.append(i)

        # Overall cap enforcement at message level (stop adding if exceeded)
        if len(sent_list) >= MAX_BET_IDEAS:
            break

    # Trim sent_list to max ideas and record cooldown
    sent_list = sent_list[:MAX_BET_IDEAS]

    if any_output and sent_list:
        send_chunked("\n".join(msg).strip())
        record_sent(state, sent_list, now_ts)
    else:
        if SEND_NO_EDGE_PING and _in_burst_window(now_et):
            send_one(f"🧠 No edges met thresholds this run. ({ts_et})")

    state["players"] = new_players
    save_state(state)

if __name__ == "__main__":
    run()
