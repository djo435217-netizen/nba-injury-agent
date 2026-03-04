import os
import json
import re
import time
import math
from datetime import datetime
from zoneinfo import ZoneInfo
import requests
from twilio.rest import Client

# ==================== FILES / TZ ====================
STATE_FILE = "state.json"
ET = ZoneInfo("America/New_York")

# ==================== REQUIRED ENV ====================
TWILIO_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
SPORTRADAR_KEY = os.environ.get("SPORTRADAR_API_KEY", "").strip()
BALLDONTLIE_API_KEY = os.environ["BALLDONTLIE_API_KEY"].strip()

FROM_WHATSAPP = os.environ.get("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
TO_WHATSAPP = f"whatsapp:{os.environ['MY_WHATSAPP_NUMBER']}"

twilio = Client(TWILIO_SID, TWILIO_TOKEN)

# ==================== CONFIG (ENV) ====================
TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
MAX_BODY_CHARS = int(os.environ.get("MAX_BODY_CHARS", "1500"))

# Injury filtering
IMPACT_STATUSES_RAW = os.environ.get("IMPACT_STATUSES", "out,doubtful,questionable").strip()
IMPACT_STATUSES = {x.strip().lower() for x in IMPACT_STATUSES_RAW.split(",") if x.strip()}
IMPACT_ONLY_CHANGES = os.environ.get("IMPACT_ONLY_CHANGES", "1") == "1"

ENABLE_INJURY_TRIGGERS = os.environ.get("ENABLE_INJURY_TRIGGERS", "1") == "1"
ENABLE_SLATE_SCAN = os.environ.get("ENABLE_SLATE_SCAN", "1") == "1"

# Vendors + prop types
BOOK_VENDOR_RAW = os.environ.get("BOOK_VENDORS", os.environ.get("BOOK_VENDOR", "fanduel")).strip().lower()
BOOK_VENDORS = [v.strip() for v in BOOK_VENDOR_RAW.split(",") if v.strip()]

PROP_TYPES_RAW = os.environ.get("PROP_TYPES", os.environ.get("PROP_TYPE", "points")).strip().lower()
PROP_TYPES = [p.strip() for p in PROP_TYPES_RAW.split(",") if p.strip()]

# Output sizing
MIN_PER_MARKET = int(os.environ.get("MIN_PER_MARKET", "0"))
MAX_PER_MARKET = int(os.environ.get("MAX_PER_MARKET", "6"))
MAX_TOTAL_PLAYS = int(os.environ.get("MAX_TOTAL_PLAYS", os.environ.get("MAX_BET_IDEAS", "10")))

# Burst window
BURST_START_ET = os.environ.get("BURST_START_ET", "17:00").strip()
BURST_END_ET = os.environ.get("BURST_END_ET", "23:45").strip()
SEND_NO_EDGE_PING = os.environ.get("SEND_NO_EDGE_PING", "0") == "1"

SLATE_ONLY_IN_BURST = os.environ.get("SLATE_ONLY_IN_BURST", "0") == "1"

# Multi-horizon windows
BASELINE_GAMES = int(os.environ.get("BASELINE_GAMES", "30"))
LOOKBACK_GAMES = int(os.environ.get("LOOKBACK_GAMES", "10"))
SHORT_GAMES = int(os.environ.get("SHORT_GAMES", "3"))
MIN_SAMPLE_GAMES = int(os.environ.get("MIN_SAMPLE_GAMES", "10"))

# Projection weights (your “3 accuracy vars”)
W_L30 = float(os.environ.get("PROJECTION_WEIGHT_L30", os.environ.get("W_BASE", "0.50")))
W_L10 = float(os.environ.get("PROJECTION_WEIGHT_L10", os.environ.get("W_L10", "0.35")))
W_L3 = float(os.environ.get("PROJECTION_WEIGHT_L3", os.environ.get("W_L3", "0.15")))
W_LINE = float(os.environ.get("W_LINE", "0.00"))  # keep small unless you want anchoring

# Thresholds
MIN_EDGE = float(os.environ.get("MIN_EDGE", "2.5"))
MIN_PROB = float(os.environ.get("MIN_PROB", "0.62"))
STD_FLOOR = float(os.environ.get("STD_FLOOR", "0.75"))  # lower for threes; sigma is computed from game log
LINE_MIN_GAP = float(os.environ.get("LINE_MIN_GAP", "8.0"))
MIN_DELTA_FLOOR = float(os.environ.get("MIN_DELTA_FLOOR", "-3.0"))

# Injury vacancy requirements / caps
MIN_VAC_MIN = float(os.environ.get("MIN_VAC_MIN", "10.0"))
MIN_VAC_PTS = float(os.environ.get("MIN_VAC_PTS", "6.0"))
BOOST_CAP_PTS = float(os.environ.get("BOOST_CAP_PTS", "5.5"))
BOOST_CAP_MIN = float(os.environ.get("BOOST_CAP_MIN", "6.0"))

# EV / value / steam / consensus
VALUE_EDGE_MIN = float(os.environ.get("VALUE_EDGE_MIN", "0.05"))          # model_prob - market_prob
EV_MIN = float(os.environ.get("EV_MIN", "0.03"))                          # expected value
PLUS_ODDS_MIN = int(os.environ.get("PLUS_ODDS_MIN", "100"))               # +100 or more
PLUS_ODDS_TOPN = int(os.environ.get("PLUS_ODDS_TOPN", "3"))
STEAM_MIN_SCORE = int(os.environ.get("STEAM_MIN_SCORE", "1"))
MIN_VENDORS_FOR_CONSENSUS = int(os.environ.get("MIN_VENDORS_FOR_CONSENSUS", "2"))

# Matching rule (you had this toggle earlier)
STRICT_INJURY_GAME_MATCH = os.environ.get("STRICT_INJURY_GAME_MATCH", "0") == "1"

# Debug samples
DEBUG_PROP_SAMPLE_TYPES_RAW = os.environ.get("DEBUG_PROP_SAMPLE_TYPES", "").strip().lower()
DEBUG_PROP_SAMPLE_TYPES = {x.strip() for x in DEBUG_PROP_SAMPLE_TYPES_RAW.split(",") if x.strip()}

# ==================== UTILS ====================
def _now_et() -> datetime:
    return datetime.now(ET)

def _time_to_minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)

def _in_burst_window(now_et: datetime) -> bool:
    start = _time_to_minutes(BURST_START_ET)
    end = _time_to_minutes(BURST_END_ET)
    cur = now_et.hour * 60 + now_et.minute
    return start <= cur <= end

def _season_year(now_et: datetime) -> int:
    return now_et.year if now_et.month >= 10 else now_et.year - 1

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

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"players": {}, "sent_bets": {}, "line_state": {}}
    try:
        with open(STATE_FILE, "r") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return {"players": {}, "sent_bets": {}, "line_state": {}}
        raw.setdefault("players", {})
        raw.setdefault("sent_bets", {})
        raw.setdefault("line_state", {})
        return raw
    except Exception:
        return {"players": {}, "sent_bets": {}, "line_state": {}}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)

def send_one(body: str):
    # Don’t crash cron if Twilio errors (sandbox rejoin issues etc.)
    try:
        twilio.messages.create(from_=FROM_WHATSAPP, to=TO_WHATSAPP, body=body[:MAX_BODY_CHARS])
    except Exception as e:
        print(f"[TWILIO_ERROR] {type(e).__name__}: {e}")

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

# ==================== ODDS MATH (VIG-FREE + EV) ====================
def implied_prob_from_american(odds: float) -> float | None:
    try:
        o = float(odds)
    except Exception:
        return None
    if o == 0:
        return None
    if o > 0:
        return 100.0 / (o + 100.0)
    return (-o) / ((-o) + 100.0)

def vig_free_probs(over_odds, under_odds) -> tuple[float | None, float | None]:
    po = implied_prob_from_american(over_odds)
    pu = implied_prob_from_american(under_odds)
    if po is None or pu is None:
        return None, None
    s = po + pu
    if s <= 0:
        return None, None
    return po / s, pu / s

def payout_b_from_american(odds: float) -> float | None:
    """Return net profit per $1 stake (b in Kelly/EV), e.g. +120 => 1.2, -150 => 0.666.."""
    try:
        o = float(odds)
    except Exception:
        return None
    if o == 0:
        return None
    if o > 0:
        return o / 100.0
    return 100.0 / (-o)

def ev_per_dollar(model_prob: float, over_odds: float) -> float | None:
    b = payout_b_from_american(over_odds)
    if b is None:
        return None
    p = float(model_prob)
    return p * b - (1.0 - p)

# ==================== SPORTRADAR ====================
def fetch_sportradar_injuries():
    if not SPORTRADAR_KEY:
        print("[WARN] SPORTRADAR_KEY not set; injuries list will be empty.")
        return {"teams": []}
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

# ==================== BALLDONTLIE ====================
BDL_HEADERS = {"Authorization": BALLDONTLIE_API_KEY}
BDL_PREFIXES = ["/nba", ""]
BDL_MAX_RETRIES = int(os.environ.get("BDL_MAX_RETRIES", "5"))
BDL_RETRY_BASE_SEC = float(os.environ.get("BDL_RETRY_BASE_SEC", "1.5"))
BDL_PER_PAGE = int(os.environ.get("BDL_PER_PAGE", "100"))
BDL_MAX_PAGES = int(os.environ.get("BDL_MAX_PAGES", "10"))

TEAM_CACHE = None
PROPS_CACHE = {}  # (gid, vendor, prop_type) -> rows
DEBUG_PRINTED = set()  # (prop_type, vendor) printed once

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

def bdl_games_today(now_et: datetime):
    today = now_et.strftime("%Y-%m-%d")
    resp = _bdl_get("/v1/games", params={"dates[]": [today], "per_page": 100})
    return resp.get("data") or []

def bdl_games_today_ids(now_et: datetime):
    return [int(g["id"]) for g in bdl_games_today(now_et) if g.get("id") is not None]

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

# Map prop_type -> stats field (BDL stats keys)
STAT_FIELD = {
    "points": "pts",
    "pts": "pts",
    "threes": "fg3m",
    "three_pointers_made": "fg3m",
    "3pm": "fg3m",
}

def _stat_value_from_row(row: dict, prop_type: str) -> float:
    key = STAT_FIELD.get(prop_type, "pts")
    v = row.get(key, 0) or 0
    try:
        return float(v)
    except Exception:
        return 0.0

def bdl_last_n_games_stats(player_ids, season: int, n: int, prop_type: str):
    """
    Returns dict: pid -> list[(date, stat_value, minutes)]
    We fetch from /v1/stats and extract pts or fg3m depending on prop_type.
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
            statv = _stat_value_from_row(row, prop_type)
            mins = _parse_minutes(row.get("min"))
            if date:
                out[pid].append((date, statv, mins))

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
    key = (int(game_id), (vendor or ""), prop_type)
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

    # Debug sample row in logs
    if prop_type in DEBUG_PROP_SAMPLE_TYPES:
        dk = (prop_type, vendor or "NO_VENDOR")
        if dk not in DEBUG_PRINTED and props:
            print(f"[DEBUG] SAMPLE PROP ROW ({prop_type}, vendor={vendor or 'NO_VENDOR'}): {json.dumps(props[0])}")
            DEBUG_PRINTED.add(dk)

    PROPS_CACHE[key] = props
    return props

def _pick_main_line_over_under(rows_for_player):
    """
    Choose a "main" line among over/under markets:
    prefer closest juice to -110/-110, else median.
    """
    if not rows_for_player:
        return None, None, None  # line, over_odds, under_odds

    candidates = []
    for pp in rows_for_player:
        market = pp.get("market") or {}
        if (market.get("type") or "").lower() != "over_under":
            continue
        try:
            line = float(pp.get("line_value"))
        except Exception:
            continue

        over = market.get("over_odds")
        under = market.get("under_odds")
        if isinstance(over, (int, float)) and isinstance(under, (int, float)):
            dist = abs(abs(float(over)) - 110.0) + abs(abs(float(under)) - 110.0)
        else:
            dist = None

        candidates.append((dist, line, over, under))

    if not candidates:
        return None, None, None

    with_dist = [c for c in candidates if c[0] is not None]
    if with_dist:
        with_dist.sort(key=lambda x: x[0])
        _, line, over, under = with_dist[0]
        return float(line), over, under

    lines = sorted([c[1] for c in candidates])
    mid = len(lines) // 2
    chosen = lines[mid] if len(lines) % 2 == 1 else 0.5 * (lines[mid - 1] + lines[mid])
    # no odds info in median fallback
    return float(chosen), None, None

def lines_for_player_across_vendors(game_id: int, player_id: int, prop_type: str):
    """
    Returns dict vendor -> (line, over_odds, under_odds) for over_under market
    """
    out = {}
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
        line, over, under = _pick_main_line_over_under(rows)
        if line is not None:
            out[v] = (float(line), over, under)
    return out

def consensus_line(lines_by_vendor: dict):
    """
    Median line across vendors (requires MIN_VENDORS_FOR_CONSENSUS).
    """
    lines = [x[0] for x in lines_by_vendor.values() if x and x[0] is not None]
    if len(lines) < MIN_VENDORS_FOR_CONSENSUS:
        return None
    lines.sort()
    mid = len(lines) // 2
    return lines[mid] if len(lines) % 2 == 1 else 0.5 * (lines[mid - 1] + lines[mid])

# ==================== PROJECTION CORE ====================
def avg_stat_min_std(games):
    """
    games: list[(date, stat_value, minutes)]
    returns (avg_stat, avg_minutes, std_stat)
    std computed on stat values
    """
    if not games:
        return 0.0, 0.0, 0.0
    vals = [x[1] for x in games]
    mins = [x[2] for x in games]
    n = len(vals)
    avg_v = sum(vals) / n
    avg_m = sum(mins) / n
    var = sum((v - avg_v) ** 2 for v in vals) / max(n, 1)
    return avg_v, avg_m, math.sqrt(var)

def role_trend(games, lookback=LOOKBACK_GAMES, short=SHORT_GAMES):
    """
    returns (min_delta, rate_delta, mins_short, mins_long, rate_short, rate_long)
    """
    if not games:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    long_slice = _slice_last(games, lookback)
    short_slice = _slice_last(games, short)
    v_l, m_l, _ = avg_stat_min_std(long_slice)
    v_s, m_s, _ = avg_stat_min_std(short_slice)
    rate_l = v_l / max(m_l, 1e-6)
    rate_s = v_s / max(m_s, 1e-6)
    return (m_s - m_l), (rate_s - rate_l), m_s, m_l, rate_s, rate_l

def compute_projection_prob_ev(
    games_all,
    line: float,
    over_odds: float | None,
    under_odds: float | None,
    injury_boost_stat: float = 0.0,
    injury_boost_min: float = 0.0,
):
    """
    Uses multi-horizon blend + sigma from recent samples.
    Returns projection + edge + model_prob_over + market_prob_over + value_edge + EV
    """
    base_slice = _slice_last(games_all, BASELINE_GAMES)
    l10_slice = _slice_last(games_all, LOOKBACK_GAMES)
    l3_slice = _slice_last(games_all, SHORT_GAMES)

    l30_avg, _, l30_std = avg_stat_min_std(base_slice)
    l10_avg, l10_min, l10_std = avg_stat_min_std(l10_slice)
    l3_avg, _, _ = avg_stat_min_std(l3_slice)

    # sigma from l10 if possible else l30
    sigma = l10_std if l10_std and l10_std > 0 else l30_std
    sigma = max(STD_FLOOR, sigma if sigma and sigma > 0 else STD_FLOOR)

    proj = (W_L30 * l30_avg) + (W_L10 * l10_avg) + (W_L3 * l3_avg) + (W_LINE * float(line))

    # minutes-scaled injury bump using current rate
    rate = l10_avg / max(l10_min, 1e-6)
    proj += injury_boost_stat
    proj += (injury_boost_min * rate * 0.20)

    edge = proj - float(line)
    z = (proj - float(line)) / sigma
    model_p_over = _norm_cdf(z)

    market_p_over, _ = vig_free_probs(over_odds, under_odds)
    value_edge = None
    ev = None
    if market_p_over is not None:
        value_edge = model_p_over - market_p_over
    if over_odds is not None:
        ev = ev_per_dollar(model_p_over, over_odds)

    aux = {
        "l30": l30_avg,
        "l10": l10_avg,
        "l3": l3_avg,
        "mins_l10": l10_min,
        "sigma": sigma,
        "rate": rate,
    }
    return proj, edge, model_p_over, market_p_over, value_edge, ev, aux

# ==================== STEAM TRACKING ====================
def steam_score(state, key: str, line: float, over_odds, ts_iso: str):
    """
    Score line movement: +1 if line moved up, +1 if over got more expensive, -1 for opposite.
    """
    ls = state.get("line_state", {}) or {}
    prev = ls.get(key)
    score = 0

    def _odds_to_float(x):
        try:
            return float(x)
        except Exception:
            return None

    if prev:
        prev_line = prev.get("line")
        prev_over = _odds_to_float(prev.get("over_odds"))
        cur_over = _odds_to_float(over_odds)

        try:
            if prev_line is not None and float(line) > float(prev_line):
                score += 1
            if prev_line is not None and float(line) < float(prev_line):
                score -= 1
        except Exception:
            pass

        # “More expensive” for over means odds move toward more negative (e.g., -110 -> -130)
        if prev_over is not None and cur_over is not None:
            if cur_over < prev_over:
                score += 1
            if cur_over > prev_over:
                score -= 1

    # store current
    ls[key] = {"line": float(line), "over_odds": over_odds, "ts": ts_iso}
    state["line_state"] = ls
    return score

# ==================== INJURY ENGINE ====================
def build_injury_edges_for_prop(prop_type: str, team_short: str, injured_name: str, injured_status: str,
                                exclude_names_lower: set[str], now_et: datetime, state: dict):
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

    # Vacancy computed on POINTS (stronger trigger), but we boost each prop using absorption+rate.
    inj_games_pts = bdl_last_n_games_stats([injured_pid], season, BASELINE_GAMES, "points").get(injured_pid, [])
    ip10, im10, _ = avg_stat_min_std(_slice_last(inj_games_pts, LOOKBACK_GAMES))
    if len(inj_games_pts) < 3:
        return []

    status = (injured_status or "").lower()
    STATUS_MULT = {"out": 1.0, "doubtful": 0.8, "questionable": 0.55}.get(status, 0.65)

    vac_pts = ip10 * STATUS_MULT
    vac_min = im10 * STATUS_MULT
    if not ((vac_min >= MIN_VAC_MIN) or (vac_pts >= MIN_VAC_PTS)):
        return []

    trigger_strength = min(100.0, (vac_min * 1.2 + vac_pts * 1.5))

    game_ids = bdl_games_today_ids(now_et)
    if not game_ids:
        return []

    ideas = []

    # Pull stats for this prop type for roster
    cand_ids = [pid for pid, _ in roster_tuples]
    stats = bdl_last_n_games_stats(cand_ids, season, BASELINE_GAMES, prop_type)

    for pid, nm in roster_tuples:
        games = stats.get(pid, [])
        if len(games) < max(MIN_SAMPLE_GAMES, 8):
            continue

        min_delta, rate_delta, _, _, _, _ = role_trend(games)

        l10_avg, l10_min, _ = avg_stat_min_std(_slice_last(games, LOOKBACK_GAMES))
        if l10_min < 10:
            continue

        # Absorption heuristic
        absorption = 0.0
        if l10_min >= 28:
            absorption += 0.25
        if l10_min >= 34:
            absorption += 0.10
        if min_delta >= 2.0:
            absorption += 0.15
        if rate_delta > 0.05:
            absorption += 0.10
        absorption = min(0.65, absorption)

        # Find a game line (and odds) from ANY vendor; then compute consensus
        chosen_gid = None
        chosen_vendor = None
        chosen_line = None
        chosen_over = None
        chosen_under = None
        chosen_consensus = None
        lines_by_vendor = None

        for gid in game_ids:
            lv = lines_for_player_across_vendors(gid, pid, prop_type)
            if not lv:
                continue
            cons = consensus_line(lv)
            # pick a “primary” vendor line preference (fanduel first if present)
            for v in BOOK_VENDORS:
                if v in lv:
                    line, over, under = lv[v]
                    chosen_gid = int(gid)
                    chosen_vendor = v
                    chosen_line = float(line)
                    chosen_over = over
                    chosen_under = under
                    chosen_consensus = cons
                    lines_by_vendor = lv
                    break
            if chosen_gid is not None:
                break

        if chosen_gid is None or chosen_line is None:
            continue

        # Consensus sanity (avoid outlier vendor lines if we have enough books)
        if chosen_consensus is not None:
            gap = abs(chosen_line - float(chosen_consensus))
            # tighter tolerance for threes
            tol = 0.5 if prop_type in ("threes", "three_pointers_made", "3pm") else 1.0
            if gap > tol:
                continue

        # avoid pure “l10 >> line” traps
        if (l10_avg - chosen_line) > LINE_MIN_GAP:
            continue

        # Injury boosts: pts vacancy maps to any prop weaker, but still useful
        injury_boost_stat = min(BOOST_CAP_PTS, vac_pts * absorption * 0.40)
        injury_boost_min = min(BOOST_CAP_MIN, vac_min * absorption * 0.25)

        proj, edge, model_p, market_p, value_edge, ev, aux = compute_projection_prob_ev(
            games_all=games,
            line=chosen_line,
            over_odds=chosen_over,
            under_odds=chosen_under,
            injury_boost_stat=injury_boost_stat,
            injury_boost_min=injury_boost_min,
        )

        if edge < MIN_EDGE or model_p < MIN_PROB:
            continue

        # Market-based filters (if odds available)
        if market_p is not None and value_edge is not None and value_edge < VALUE_EDGE_MIN:
            continue
        if ev is not None and ev < EV_MIN:
            continue

        # Steam filter
        ts_iso = now_et.astimezone(ET).isoformat()
        steam_key = f"{prop_type}|{chosen_vendor}|{chosen_gid}|{pid}"
        sscore = steam_score(state, steam_key, chosen_line, chosen_over, ts_iso)
        if sscore < STEAM_MIN_SCORE:
            # If you want to allow “no steam but huge edge”, relax here.
            continue

        # Role guardrail
        if min_delta < MIN_DELTA_FLOOR and edge < (MIN_EDGE + 1.5):
            continue

        why = (
            f"TriggerStrength {trigger_strength:.0f} | Absorb {absorption:.2f} | Steam {sscore}. "
            f"{injured_name} {injured_status.upper()} vacates ~{vac_pts:.1f} Points / {vac_min:.1f} min. "
            f"{nm} base(L{BASELINE_GAMES}) {aux['l30']:.1f}, L10 {aux['l10']:.1f}, L3 {aux['l3']:.1f} "
            f"(mins L10 {aux['mins_l10']:.1f}). Role Δmin={min_delta:+.1f}, Δrate={rate_delta:+.2f}. "
            f"Proj {proj:.1f} vs {chosen_vendor} line {chosen_line:.1f} | edge +{edge:.1f} | P≈{model_p*100:.0f}%"
        )
        if market_p is not None:
            why += f" | mkt≈{market_p*100:.0f}%"
        if ev is not None:
            why += f" | EV≈{ev:+.2f}"

        ideas.append({
            "section": "injury",
            "prop_type": prop_type,
            "player_name": nm,
            "player_id": int(pid),
            "line": float(chosen_line),
            "proj": float(proj),
            "edge": float(edge),
            "prob_over": float(model_p),
            "market_prob": float(market_p) if market_p is not None else None,
            "value_edge": float(value_edge) if value_edge is not None else None,
            "ev": float(ev) if ev is not None else None,
            "trigger_strength": float(trigger_strength),
            "trigger": f"{injured_name} ({team_short}) {injured_status}",
            "why": why + f". [prop_type={prop_type}]",
        })

    ideas.sort(key=lambda x: (x["trigger_strength"], x["edge"], x["prob_over"]), reverse=True)
    return ideas

# ==================== LEAGUE-WIDE SLATE SCAN ====================
def slate_scan_edges_for_prop(prop_type: str, now_et: datetime, state: dict):
    if not ENABLE_SLATE_SCAN:
        return []
    if SLATE_ONLY_IN_BURST and (not _in_burst_window(now_et)):
        return []

    season = _season_year(now_et)
    games = bdl_games_today(now_et)
    if not games:
        return []

    ideas = []
    for g in games:
        gid = g.get("id")
        if gid is None:
            continue
        gid = int(gid)

        # Fetch props for each vendor once
        # We gather lines per player across vendors.
        player_lines = {}  # pid -> {vendor: (line, over, under)}

        for v in BOOK_VENDORS:
            rows = bdl_player_props(gid, v, prop_type)
            if not rows:
                continue
            by_pid = {}
            for pp in rows:
                pid = pp.get("player_id")
                if pid is None:
                    continue
                try:
                    pid = int(pid)
                except Exception:
                    continue
                by_pid.setdefault(pid, []).append(pp)
            for pid, plist in by_pid.items():
                line, over, under = _pick_main_line_over_under(plist)
                if line is None:
                    continue
                player_lines.setdefault(pid, {})[v] = (float(line), over, under)

        if not player_lines:
            continue

        pids = list(player_lines.keys())
        stats = bdl_last_n_games_stats(pids, season, BASELINE_GAMES, prop_type)

        for pid in pids:
            games_all = stats.get(pid, [])
            if len(games_all) < max(MIN_SAMPLE_GAMES, 8):
                continue

            lv = player_lines.get(pid, {})
            if not lv:
                continue

            cons = consensus_line(lv)
            if cons is None:
                # if you want 1-book scan, lower MIN_VENDORS_FOR_CONSENSUS
                continue

            # prefer fanduel line if present else first vendor
            primary_vendor = "fanduel" if "fanduel" in lv else next(iter(lv.keys()))
            line, over, under = lv[primary_vendor]

            # avoid outlier primary line vs consensus
            gap = abs(float(line) - float(cons))
            tol = 0.5 if prop_type in ("threes", "three_pointers_made", "3pm") else 1.0
            if gap > tol:
                continue

            min_delta, rate_delta, _, _, _, _ = role_trend(games_all)

            l10_avg, l10_min, _ = avg_stat_min_std(_slice_last(games_all, LOOKBACK_GAMES))
            if l10_min < 10:
                continue
            if (l10_avg - float(line)) > LINE_MIN_GAP:
                continue

            proj, edge, model_p, market_p, value_edge, ev, aux = compute_projection_prob_ev(
                games_all=games_all,
                line=float(line),
                over_odds=over,
                under_odds=under,
                injury_boost_stat=0.0,
                injury_boost_min=0.0,
            )

            if edge < MIN_EDGE or model_p < MIN_PROB:
                continue
            if market_p is not None and value_edge is not None and value_edge < VALUE_EDGE_MIN:
                continue
            if ev is not None and ev < EV_MIN:
                continue

            ts_iso = now_et.astimezone(ET).isoformat()
            steam_key = f"{prop_type}|{primary_vendor}|{gid}|{pid}"
            sscore = steam_score(state, steam_key, float(line), over, ts_iso)
            if sscore < STEAM_MIN_SCORE:
                continue

            if min_delta < MIN_DELTA_FLOOR and edge < (MIN_EDGE + 2.0):
                continue

            name = PLAYER_NAME_CACHE.get(int(pid), f"Player {pid}")
            why = (
                f"SlateScan | Steam {sscore}. base(L{BASELINE_GAMES}) {aux['l30']:.1f}, L10 {aux['l10']:.1f}, L3 {aux['l3']:.1f} "
                f"(mins L10 {aux['mins_l10']:.1f}). Role Δmin={min_delta:+.1f}, Δrate={rate_delta:+.2f}. "
                f"Proj {proj:.1f} vs {primary_vendor} line {float(line):.1f} | edge +{edge:.1f} | P≈{model_p*100:.0f}%"
            )
            if market_p is not None:
                why += f" | mkt≈{market_p*100:.0f}%"
            if ev is not None:
                why += f" | EV≈{ev:+.2f}"

            ideas.append({
                "section": "slate",
                "prop_type": prop_type,
                "player_name": name,
                "player_id": int(pid),
                "line": float(line),
                "proj": float(proj),
                "edge": float(edge),
                "prob_over": float(model_p),
                "market_prob": float(market_p) if market_p is not None else None,
                "value_edge": float(value_edge) if value_edge is not None else None,
                "ev": float(ev) if ev is not None else None,
                "trigger_strength": 0.0,
                "trigger": "No injury trigger (league-wide scan)",
                "why": why + f". [prop_type={prop_type}]",
            })

    ideas.sort(key=lambda x: (x["edge"], x["prob_over"]), reverse=True)
    return ideas

# ==================== PLUS-ODDS BUCKET ====================
def pick_plus_odds(ideas):
    """
    Return top PLUS_ODDS_TOPN ideas where over_odds >= PLUS_ODDS_MIN AND EV/value already passed.
    We infer “plus odds” from EV calc inputs only if we had odds.
    Since we don’t store raw odds per idea, we approximate:
    - If market_prob present, it had odds; but we need actual odds to check plus money.
    For reliability: we’ll scan the idea['why'] string for 'EV' only is not enough.
    So: we do plus-odds only for plays whose why contains 'EV≈' AND also vendor odds were +.
    To do it correctly, you can store odds in idea later; for now we keep it simple:
    We'll skip this bucket unless you add STORE_ODDS=1 later.
    """
    # Best practice: store over_odds in idea. We didn’t to keep message clean.
    # So this is a placeholder bucket; you can enable storing odds if you want.
    return []

# ==================== MAIN ====================
def run():
    now_et = _now_et()
    ts_et = now_et.strftime("%Y-%m-%d %I:%M %p ET")

    print(
        f"[BOOT] ts={ts_et} TEST_MODE={int(TEST_MODE)} PROP_TYPES={','.join(PROP_TYPES)} "
        f"MIN_PER_MARKET={MIN_PER_MARKET} MAX_PER_MARKET={MAX_PER_MARKET} MAX_TOTAL_PLAYS={MAX_TOTAL_PLAYS} "
        f"BOOK_VENDORS={','.join(BOOK_VENDORS)} ENABLE_SLATE_SCAN={int(ENABLE_SLATE_SCAN)} "
        f"ENABLE_INJURY_TRIGGERS={int(ENABLE_INJURY_TRIGGERS)} STRICT_INJURY_GAME_MATCH={int(STRICT_INJURY_GAME_MATCH)}"
    )

    if TEST_MODE:
        send_one(f"✅ NBA betting agent test OK ({ts_et})")
        return

    state = load_state()
    old_players = state.get("players", {}) or {}

    # Injuries
    new_players = {}
    if ENABLE_INJURY_TRIGGERS:
        try:
            sr = fetch_sportradar_injuries()
            new_players = parse_injuries(sr)
        except Exception as e:
            print(f"[WARN] Sportradar fetch failed: {type(e).__name__}: {e}")
            new_players = {}

    exclude_names_lower = {_clean_name(v.get("name", "")) for v in new_players.values() if v.get("name")}

    triggers = []
    injury_ideas_all = []

    if ENABLE_INJURY_TRIGGERS and new_players:
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

            # Build edges per prop type
            for prop_type in PROP_TYPES:
                pt = "threes" if prop_type in ("threes", "three_pointers_made", "3pm") else "points"
                ideas = build_injury_edges_for_prop(
                    prop_type=pt,
                    team_short=team_short,
                    injured_name=injured_name,
                    injured_status=injured_status,
                    exclude_names_lower=exclude_names_lower | {_clean_name(injured_name)},
                    now_et=now_et,
                    state=state,
                )
                if ideas:
                    injury_ideas_all.extend(ideas)

            if team_short and injured_name:
                triggers.append(f"{injured_name} ({team_short}) {injured_status}")

    # Slate scan per prop type
    slate_ideas_all = []
    if ENABLE_SLATE_SCAN:
        for prop_type in PROP_TYPES:
            pt = "threes" if prop_type in ("threes", "three_pointers_made", "3pm") else "points"
            try:
                slate_ideas_all.extend(slate_scan_edges_for_prop(pt, now_et, state))
            except Exception as e:
                print(f"[WARN] Slate scan failed for {pt}: {type(e).__name__}: {e}")

    # Combine + dedupe (section, prop_type, player_id)
    combined = injury_ideas_all + slate_ideas_all
    best = {}
    for i in combined:
        k = (i["section"], i["prop_type"], int(i["player_id"]))
        if (k not in best) or ((i["edge"], i["prob_over"]) > (best[k]["edge"], best[k]["prob_over"])):
            best[k] = i
    combined = list(best.values())

    # Sort and allocate per market
    out_by_prop = {pt: [] for pt in set([("threes" if p in ("threes","three_pointers_made","3pm") else "points") for p in PROP_TYPES])}

    # Prefer injury then slate within prop
    for pt in out_by_prop.keys():
        inj = sorted([x for x in combined if x["prop_type"] == pt and x["section"] == "injury"],
                     key=lambda x: (x["trigger_strength"], x["edge"], x["prob_over"]), reverse=True)
        slt = sorted([x for x in combined if x["prop_type"] == pt and x["section"] == "slate"],
                     key=lambda x: (x["edge"], x["prob_over"]), reverse=True)

        # Build a final list with some injury emphasis but not required
        chosen = []
        chosen.extend(inj[:MAX_PER_MARKET])
        # fill remainder with slate
        for x in slt:
            if len(chosen) >= MAX_PER_MARKET:
                break
            # don’t duplicate same player already in injury list
            if any(int(y["player_id"]) == int(x["player_id"]) for y in chosen):
                continue
            chosen.append(x)

        # Respect MIN_PER_MARKET
        if len(chosen) >= max(0, MIN_PER_MARKET):
            out_by_prop[pt] = chosen

    # Flatten respecting MAX_TOTAL_PLAYS
    # Interleave props: points first then threes
    order = ["points", "threes"]
    final_out = []
    for pt in order:
        for x in out_by_prop.get(pt, []):
            if len(final_out) >= MAX_TOTAL_PLAYS:
                break
            final_out.append(x)
        if len(final_out) >= MAX_TOTAL_PLAYS:
            break

    if not final_out:
        if SEND_NO_EDGE_PING and _in_burst_window(now_et):
            send_one(f"🧠 No edges met filters (MIN_EDGE {MIN_EDGE}, MIN_PROB {MIN_PROB}, VALUE_EDGE_MIN {VALUE_EDGE_MIN}, EV_MIN {EV_MIN}). ({ts_et})")
        # still persist line_state + injuries
        state["players"] = new_players
        save_state(state)
        return

    # Build message
    msg = [f"💰 FanDuel Props (filtered by value+EV+steam) ({ts_et})", ""]

    # Triggers summary
    if triggers:
        msg.append("🚑 Injury inputs (may or may not produce plays):")
        msg.append("Triggers:")
        for t in triggers[:10]:
            msg.append(f"- {t}")
        if len(triggers) > 10:
            msg.append(f"- …and {len(triggers)-10} more")
        msg.append("")

    # Sections per prop
    prop_titles = {"points": "🏷️ Points", "threes": "🏷️ 3PT Made"}
    for pt in order:
        bucket = out_by_prop.get(pt, [])
        if not bucket:
            continue

        msg.append(prop_titles.get(pt, f"🏷️ {pt}"))
        msg.append("")

        inj = [x for x in bucket if x["section"] == "injury"]
        slt = [x for x in bucket if x["section"] == "slate"]

        if inj:
            msg.append("🚑 Injury-Triggered Plays:")
            msg.append("")
            for x in inj:
                msg.append(f"• {x['player_name']} OVER {x['line']:.1f}  (edge +{x['edge']:.1f}, P≈{x['prob_over']*100:.0f}%)")
                msg.append(f"  Trigger: {x['trigger']}")
                msg.append(f"  Why: {x['why']}")
                msg.append("")

        if slt:
            msg.append("🌎 League-Wide Slate Scan (no injury required):")
            msg.append("")
            for x in slt:
                msg.append(f"• {x['player_name']} OVER {x['line']:.1f}  (edge +{x['edge']:.1f}, P≈{x['prob_over']*100:.0f}%)")
                msg.append(f"  Why: {x['why']}")
                msg.append("")

        msg.append("")

    # Note if threes requested but no lines passed filters
    if ("threes" in [("threes" if p in ("threes","three_pointers_made","3pm") else "points") for p in PROP_TYPES]) and not out_by_prop.get("threes"):
        msg.append("🧩 Note on 3PT Made:")
        msg.append("No 3PT plays passed filters this run (value+EV+steam+consensus).")
        msg.append("If you want more 3PT volume, lower VALUE_EDGE_MIN / EV_MIN or set STEAM_MIN_SCORE=0.")
        msg.append("")

    send_chunked("\n".join(msg).strip())

    # Persist injury state and line_state updates
    state["players"] = new_players
    save_state(state)

if __name__ == "__main__":
    run()
