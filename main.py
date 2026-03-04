import os
import json
import re
import time
import math
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

STATE_FILE = "state.json"
ET = ZoneInfo("America/New_York")

# -------------------- REQUIRED ENV --------------------
TWILIO_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
BALLDONTLIE_API_KEY = os.environ["BALLDONTLIE_API_KEY"].strip()

# Sportradar optional (script runs without, but injuries section will be empty)
SPORTRADAR_KEY = os.environ.get("SPORTRADAR_API_KEY", "").strip()

FROM_WHATSAPP = os.environ.get("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
TO_WHATSAPP = f"whatsapp:{os.environ['MY_WHATSAPP_NUMBER']}"

twilio = Client(TWILIO_SID, TWILIO_TOKEN)

# -------------------- CONFIG (ENV) --------------------
TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
MAX_BODY_CHARS = 1500

# vendors list
BOOK_VENDOR_RAW = os.environ.get("BOOK_VENDORS", os.environ.get("BOOK_VENDOR", "fanduel")).strip().lower()
BOOK_VENDORS = [v.strip() for v in BOOK_VENDOR_RAW.split(",") if v.strip()]

# prop types list
PROP_TYPES_RAW = os.environ.get("PROP_TYPES", "points,threes").strip().lower()
PROP_TYPES = [p.strip() for p in PROP_TYPES_RAW.split(",") if p.strip()]

# Multi-horizon windows
BASELINE_GAMES = int(os.environ.get("BASELINE_GAMES", "30"))
LOOKBACK_GAMES = int(os.environ.get("LOOKBACK_GAMES", "10"))
MID_GAMES = int(os.environ.get("MID_GAMES", "5"))
SHORT_GAMES = int(os.environ.get("SHORT_GAMES", "3"))

# Blend weights (must sum ~1.0-ish; doesn't need to be exact)
W_BASE = float(os.environ.get("W_BASE", "0.40"))
W_L10 = float(os.environ.get("W_L10", "0.30"))
W_L5 = float(os.environ.get("W_L5", "0.20"))
W_L3 = float(os.environ.get("W_L3", "0.10"))

# Thresholds (quality gates)
MIN_EDGE_PROB = float(os.environ.get("MIN_EDGE_PROB", "0.05"))   # 5% prob edge vs de-vig fair prob
MIN_MODEL_PROB = float(os.environ.get("MIN_MODEL_PROB", "0.62")) # model must like it
STD_FLOOR = float(os.environ.get("STD_FLOOR", "4.5"))            # volatility floor

# Output sizing
MIN_PER_MARKET = int(os.environ.get("MIN_PER_MARKET", "0"))
MAX_PER_MARKET = int(os.environ.get("MAX_PER_MARKET", "6"))
MAX_TOTAL_PLAYS = int(os.environ.get("MAX_TOTAL_PLAYS", "10"))

# Diversification (prevents “4 sixers”)
MAX_PER_TEAM = int(os.environ.get("MAX_PER_TEAM", "2"))
MAX_PER_GAME = int(os.environ.get("MAX_PER_GAME", "3"))

# Lines guardrails
MIN_LINE_POINTS = float(os.environ.get("MIN_LINE_POINTS", "6.0"))
MAX_LINE_POINTS = float(os.environ.get("MAX_LINE_POINTS", "45.0"))
MIN_LINE_THREES = float(os.environ.get("MIN_LINE_THREES", "0.5"))
MAX_LINE_THREES = float(os.environ.get("MAX_LINE_THREES", "6.5"))

# Burst window
BURST_START_ET = os.environ.get("BURST_START_ET", "17:00").strip()
BURST_END_ET = os.environ.get("BURST_END_ET", "23:45").strip()

# Injury controls
IMPACT_STATUSES_RAW = os.environ.get("IMPACT_STATUSES", "out,doubtful,questionable").strip().lower()
IMPACT_STATUSES = {x.strip() for x in IMPACT_STATUSES_RAW.split(",") if x.strip()}
IMPACT_ONLY_CHANGES = os.environ.get("IMPACT_ONLY_CHANGES", "1") == "1"
ENABLE_INJURY_TRIGGERS = os.environ.get("ENABLE_INJURY_TRIGGERS", "1") == "1"
STRICT_INJURY_GAME_MATCH = os.environ.get("STRICT_INJURY_GAME_MATCH", "0") == "1"  # optional, default off

# Slate scan
ENABLE_SLATE_SCAN = os.environ.get("ENABLE_SLATE_SCAN", "1") == "1"
SLATE_ONLY_IN_BURST = os.environ.get("SLATE_ONLY_IN_BURST", "0") == "1"  # optional: 0 = scan always
SLATE_SCAN_MAX_PLAYERS = int(os.environ.get("SLATE_SCAN_MAX_PLAYERS", "260"))

# Cooldown
BET_COOLDOWN_MIN = int(os.environ.get("BET_COOLDOWN_MIN", "180"))
EDGEPROB_JUMP_TO_RESEND = float(os.environ.get("EDGEPROB_JUMP_TO_RESEND", "0.03"))

# Debug samples
DEBUG_PROP_SAMPLE_TYPES_RAW = os.environ.get("DEBUG_PROP_SAMPLE_TYPES", "").strip().lower()
DEBUG_PROP_SAMPLE_TYPES = {x.strip() for x in DEBUG_PROP_SAMPLE_TYPES_RAW.split(",") if x.strip()}
DEBUG_PRINTED = set()

# -------------------- UTILS --------------------
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

def _avg_std(vals):
    if not vals:
        return 0.0, 0.0
    n = len(vals)
    mu = sum(vals) / n
    var = sum((x - mu) ** 2 for x in vals) / max(n, 1)
    return mu, math.sqrt(var)

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"players": {}, "sent": {}}
    try:
        with open(STATE_FILE, "r") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return {"players": {}, "sent": {}}
        raw.setdefault("players", {})
        raw.setdefault("sent", {})
        return raw
    except Exception:
        return {"players": {}, "sent": {}}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)

def send_one(body: str):
    try:
        twilio.messages.create(from_=FROM_WHATSAPP, to=TO_WHATSAPP, body=body[:MAX_BODY_CHARS])
    except TwilioRestException as e:
        print(f"[TWILIO_ERROR] status={getattr(e,'status',None)} code={getattr(e,'code',None)} msg={str(e)[:500]}")
        return
    except Exception as e:
        print(f"[SEND_ERROR] {type(e).__name__}: {str(e)[:500]}")
        return

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

# -------------------- SPORTRADAR --------------------
def fetch_sportradar_injuries():
    if not SPORTRADAR_KEY:
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

def status_in_scope(status: str) -> bool:
    return (status or "").strip().lower() in IMPACT_STATUSES

# -------------------- BALLDONTLIE --------------------
BDL_HEADERS = {"Authorization": BALLDONTLIE_API_KEY}
BDL_PREFIXES = ["/nba", ""]

BDL_MAX_RETRIES = int(os.environ.get("BDL_MAX_RETRIES", "5"))
BDL_RETRY_BASE_SEC = float(os.environ.get("BDL_RETRY_BASE_SEC", "1.5"))
BDL_PER_PAGE = int(os.environ.get("BDL_PER_PAGE", "100"))
BDL_MAX_PAGES = int(os.environ.get("BDL_MAX_PAGES", "12"))

TEAM_CACHE = None
PROPS_CACHE = {}     # (gid, vendor, prop_type) -> rows
PLAYER_NAME_CACHE = {}  # pid -> name
PLAYER_TEAM_CACHE = {}  # pid -> team name (best-effort)

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
    games = resp.get("data") or []
    out = []
    for g in games:
        try:
            gid = int(g.get("id"))
        except Exception:
            continue
        out.append({
            "id": gid,
            "home_team": (g.get("home_team") or {}).get("name", ""),
            "visitor_team": (g.get("visitor_team") or {}).get("name", ""),
        })
    return out

def bdl_games_today_ids(now_et: datetime):
    return [g["id"] for g in bdl_games_today(now_et)]

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

def bdl_active_roster(team_name: str):
    team_map = bdl_team_name_to_id()
    team_id = team_map.get(team_name)
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
        if (team.get("name") or "").strip() == team_name:
            out.append(p)
    return out

def bdl_last_n_games_stats(player_ids, season: int, n: int):
    """
    Returns dict pid -> list of tuples:
      (date, pts, min, fg3m)
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
            if (fn or ln):
                PLAYER_NAME_CACHE[pid] = f"{fn} {ln}".strip()

            team = (p.get("team") or {}).get("name")
            if team:
                PLAYER_TEAM_CACHE[pid] = team

            game = row.get("game") or {}
            date = game.get("date")
            mins = _parse_minutes(row.get("min"))
            pts = float(row.get("pts", 0) or 0)
            fg3m = float(row.get("fg3m", 0) or 0)
            if date:
                out[pid].append((date, pts, mins, fg3m))

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

    # optional debug sample
    if prop_type in DEBUG_PROP_SAMPLE_TYPES and props:
        sample_key = f"{prop_type}|{vendor or 'NO_VENDOR'}"
        if sample_key not in DEBUG_PRINTED:
            print(f"[DEBUG] SAMPLE PROP ROW ({prop_type}, vendor={vendor or 'NO_VENDOR'}): {json.dumps(props[0])[:2000]}")
            DEBUG_PRINTED.add(sample_key)

    PROPS_CACHE[key] = props
    return props

def _odds_to_prob_american(odds):
    """
    Converts American odds to implied probability (WITH vig).
    """
    try:
        o = float(odds)
    except Exception:
        return None
    if o == 0:
        return None
    if o > 0:
        return 100.0 / (o + 100.0)
    return (-o) / ((-o) + 100.0)

def _devig_two_way(p_over, p_under):
    """
    De-vig by normalizing two implied probabilities.
    """
    if p_over is None or p_under is None:
        return None, None
    s = p_over + p_under
    if s <= 0:
        return None, None
    return p_over / s, p_under / s

def _pick_main_line_and_fair_prob(rows_for_player):
    """
    Picks the "main" over/under line for a player:
    - only over_under markets
    - chooses line with odds closest to -110/-110 if available, else median line
    Returns: (line, fair_prob_over, book_over_odds, book_under_odds)
    """
    candidates = []
    for pp in rows_for_player:
        market = pp.get("market") or {}
        if (market.get("type") or "").lower() != "over_under":
            continue
        try:
            line = float(pp.get("line_value"))
        except Exception:
            continue
        over_odds = market.get("over_odds")
        under_odds = market.get("under_odds")
        p_over = _odds_to_prob_american(over_odds)
        p_under = _odds_to_prob_american(under_odds)
        fair_over, _ = _devig_two_way(p_over, p_under)
        # distance to "balanced" odds for main line selection
        dist = None
        if isinstance(over_odds, (int, float)) and isinstance(under_odds, (int, float)):
            dist = abs(abs(float(over_odds)) - 110.0) + abs(abs(float(under_odds)) - 110.0)
        candidates.append((dist, line, fair_over, over_odds, under_odds))

    if not candidates:
        return None, None, None, None

    with_dist = [c for c in candidates if c[0] is not None]
    if with_dist:
        with_dist.sort(key=lambda x: x[0])
        _, line, fair, oo, uo = with_dist[0]
        return line, fair, oo, uo

    # fallback median line
    candidates.sort(key=lambda x: x[1])
    mid = len(candidates) // 2
    _, line, fair, oo, uo = candidates[mid]
    return line, fair, oo, uo

def get_best_line_for_player(game_id: int, player_id: int, prop_type: str):
    """
    Tries each vendor then no-vendor until it finds a main line and de-vig fair prob.
    Returns dict with: line, fair_prob_over, vendor_used, over_odds, under_odds
    """
    for v in BOOK_VENDORS + [None]:
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
        line, fair, oo, uo = _pick_main_line_and_fair_prob(rows)
        if line is not None and fair is not None:
            return {
                "line": float(line),
                "fair_prob_over": float(fair),
                "vendor_used": (v or "NO_VENDOR"),
                "over_odds": oo,
                "under_odds": uo,
            }
    return None

# -------------------- FEATURE MAP BY MARKET --------------------
# What stat do we model for each prop_type?
PROP_TO_STAT = {
    "points": "pts",
    "threes": "fg3m",
    # If you add more later, map here (rebounds -> reb, assists -> ast, etc)
}

def _line_bounds(prop_type: str):
    if prop_type == "points":
        return MIN_LINE_POINTS, MAX_LINE_POINTS
    if prop_type == "threes":
        return MIN_LINE_THREES, MAX_LINE_THREES
    return 0.0, 1e9

# -------------------- MODEL: build projection + model prob --------------------
def compute_model_prob(games_all, line: float, prop_type: str, injury_boost: float = 0.0):
    """
    games_all: list of (date, pts, min, fg3m)
    """
    stat_key = PROP_TO_STAT.get(prop_type)
    if not stat_key:
        return None

    base = _slice_last(games_all, BASELINE_GAMES)
    l10 = _slice_last(games_all, LOOKBACK_GAMES)
    l5 = _slice_last(games_all, MID_GAMES)
    l3 = _slice_last(games_all, SHORT_GAMES)

    def pick_stat(rows):
        if stat_key == "pts":
            return [r[1] for r in rows]
        if stat_key == "fg3m":
            return [r[3] for r in rows]
        return []

    base_vals = pick_stat(base)
    l10_vals = pick_stat(l10)
    l5_vals = pick_stat(l5)
    l3_vals = pick_stat(l3)

    base_mu, base_sd = _avg_std(base_vals)
    l10_mu, l10_sd = _avg_std(l10_vals)
    l5_mu, l5_sd = _avg_std(l5_vals)
    l3_mu, l3_sd = _avg_std(l3_vals)

    # minutes/rate trend used as a small adjustment
    l10_mins = [r[2] for r in l10]
    l3_mins = [r[2] for r in l3]
    min10 = sum(l10_mins)/len(l10_mins) if l10_mins else 0.0
    min3 = sum(l3_mins)/len(l3_mins) if l3_mins else 0.0
    min_delta = (min3 - min10)

    proj = (W_BASE * base_mu) + (W_L10 * l10_mu) + (W_L5 * l5_mu) + (W_L3 * l3_mu)
    proj += injury_boost

    # tiny minutes trend bump (bounded)
    proj += max(-1.0, min(1.0, 0.08 * min_delta))

    # volatility: use best available, floor it
    sigma = max(
        STD_FLOOR,
        l10_sd if l10_sd > 0 else base_sd if base_sd > 0 else STD_FLOOR
    )

    z = (proj - line) / sigma
    p_over = _norm_cdf(z)

    return {
        "proj": float(proj),
        "sigma": float(sigma),
        "p_over": float(p_over),
        "base_mu": float(base_mu),
        "l10_mu": float(l10_mu),
        "l5_mu": float(l5_mu),
        "l3_mu": float(l3_mu),
        "min10": float(min10),
        "min3": float(min3),
        "min_delta": float(min_delta),
    }

# -------------------- INJURY ENGINE (adds “trigger strength” but edge still odds-based) --------------------
def injury_triggers(now_et, state):
    if not ENABLE_INJURY_TRIGGERS:
        return [], {}, []

    old_players = state.get("players", {}) or {}
    sr = fetch_sportradar_injuries()
    new_players = parse_injuries(sr)

    triggers = []
    trigger_meta = {}  # (team_name) -> list of injured names/status
    trigger_items = [] # list of dicts for messaging

    for pid, cur in new_players.items():
        if not status_in_scope(cur.get("status", "")):
            continue

        prev = old_players.get(pid)
        if IMPACT_ONLY_CHANGES:
            is_new = prev is None
            is_changed = (not is_new) and ((prev.get("status"), prev.get("detail")) != (cur.get("status"), cur.get("detail")))
            if not (is_new or is_changed):
                continue

        team = cur.get("team", "")
        name = cur.get("name", "")
        status = (cur.get("status") or "").strip()

        triggers.append(f"{name} ({team}) {status}")
        trigger_meta.setdefault(team, []).append((name, status))
        trigger_items.append({"team": team, "name": name, "status": status})

    state["players"] = new_players
    return triggers, trigger_meta, trigger_items

def _status_mult(status: str) -> float:
    s = (status or "").strip().lower()
    return {"out": 1.0, "doubtful": 0.8, "questionable": 0.55}.get(s, 0.65)

# -------------------- EDGE BUILDERS --------------------
def build_edges_for_games(now_et, trigger_items):
    """
    Computes both:
      - injury-triggered edges (players on triggered teams)
      - league-wide scan edges
    across all configured PROP_TYPES.
    """
    season = _season_year(now_et)
    games_today = bdl_games_today(now_et)
    game_ids = [g["id"] for g in games_today]
    if not game_ids:
        return []

    # For injury-triggered: precompute which teams are triggered today
    triggered_teams = {t["team"] for t in trigger_items}

    edges = []

    # Slate: gather prop lines for many players by market
    # We do a “pull props then compute stats only for those players”
    for prop_type in PROP_TYPES:
        # gather player->(line, fair_prob, vendor, game_id)
        player_lines = {}
        pulled = 0

        for g in games_today:
            gid = g["id"]

            # try vendors then NO_VENDOR until we get a payload
            props = []
            vendor_used = None
            for v in BOOK_VENDORS + [None]:
                props = bdl_player_props(gid, v, prop_type)
                if props:
                    vendor_used = v
                    break
            if not props:
                continue

            # group by player_id
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
                if pid in player_lines:
                    continue
                main = _pick_main_line_and_fair_prob(rows)
                line, fair_over, oo, uo = main
                if line is None or fair_over is None:
                    continue

                lo, hi = _line_bounds(prop_type)
                if not (lo <= float(line) <= hi):
                    continue

                player_lines[pid] = {
                    "line": float(line),
                    "fair_prob_over": float(fair_over),
                    "vendor": (vendor_used or "NO_VENDOR"),
                    "over_odds": oo,
                    "under_odds": uo,
                    "game_id": int(gid),
                    "home_team": g["home_team"],
                    "visitor_team": g["visitor_team"],
                }
                pulled += 1
                if pulled >= SLATE_SCAN_MAX_PLAYERS:
                    break
            if pulled >= SLATE_SCAN_MAX_PLAYERS:
                break

        if not player_lines:
            continue

        # stats for all these players
        pids = list(player_lines.keys())
        stats = bdl_last_n_games_stats(pids, season, BASELINE_GAMES)

        for pid in pids:
            games = stats.get(pid, [])
            if len(games) < 8:
                continue

            line_meta = player_lines[pid]
            line = float(line_meta["line"])
            fair = float(line_meta["fair_prob_over"])
            gid = int(line_meta["game_id"])

            # injury boost is *only* applied if player's team is triggered
            # (we approximate by PLAYER_TEAM_CACHE once filled)
            injury_boost = 0.0
            trigger_str = 0.0
            trigger_txt = "No injury trigger (league-wide scan)"

            team_guess = PLAYER_TEAM_CACHE.get(pid)
            if team_guess and team_guess in triggered_teams:
                # small boost proportional to number/strength of outs on that team (bounded)
                team_inj = [x for x in trigger_items if x["team"] == team_guess]
                if team_inj:
                    strength = 0.0
                    for it in team_inj:
                        strength += 20.0 * _status_mult(it["status"])
                    trigger_str = min(100.0, strength)
                    trigger_txt = f"Injury trigger: {team_guess} ({len(team_inj)} absences)"
                    # boost is intentionally small because we now rely on odds edge, not just boost math
                    injury_boost = min(1.2, 0.012 * trigger_str)

                if STRICT_INJURY_GAME_MATCH:
                    # if strict match, only keep if player is in a game featuring that team name
                    ht = (line_meta["home_team"] or "")
                    vt = (line_meta["visitor_team"] or "")
                    if team_guess not in (ht, vt):
                        continue

            model = compute_model_prob(games, line, prop_type, injury_boost=injury_boost)
            if not model:
                continue

            p_over = float(model["p_over"])
            proj = float(model["proj"])
            sigma = float(model["sigma"])

            # primary edge metric: probability edge vs de-vig fair prob
            edge_prob = p_over - fair

            if p_over < MIN_MODEL_PROB:
                continue
            if edge_prob < MIN_EDGE_PROB:
                continue

            name = PLAYER_NAME_CACHE.get(pid, f"Player {pid}")
            why = (
                f"Book fair P(over)≈{fair*100:.0f}% (de-vig). Model P(over)≈{p_over*100:.0f}% "
                f"(σ≈{sigma:.1f}). base {model['base_mu']:.1f}, L10 {model['l10_mu']:.1f}, "
                f"L5 {model['l5_mu']:.1f}, L3 {model['l3_mu']:.1f}. "
                f"mins L10 {model['min10']:.1f}→L3 {model['min3']:.1f} (Δ{model['min_delta']:+.1f}). "
                f"Proj {proj:.1f} vs line {line:.1f}. EdgeProb +{edge_prob*100:.1f}%."
            )

            edges.append({
                "prop_type": prop_type,
                "player_id": int(pid),
                "player_name": name,
                "team": team_guess or "",
                "game_id": gid,
                "line": line,
                "proj": proj,
                "model_prob": p_over,
                "book_fair_prob": fair,
                "edge_prob": edge_prob,
                "vendor": line_meta["vendor"],
                "trigger_strength": trigger_str,
                "trigger": trigger_txt,
                "why": why,
            })

    # Sort globally: strongest edge_prob first, then model_prob, then trigger_strength
    edges.sort(key=lambda x: (x["edge_prob"], x["model_prob"], x["trigger_strength"]), reverse=True)
    return edges

# -------------------- COOLDOWN + DIVERSIFICATION --------------------
def apply_cooldown_and_diversify(state, edges, now_ts):
    sent = state.get("sent", {}) or {}
    cooldown_sec = BET_COOLDOWN_MIN * 60

    # cooldown filter
    kept = []
    for e in edges:
        key = f"{e['prop_type']}|{e['player_id']}|{e['line']:.1f}"
        prev = sent.get(key)
        if not prev:
            kept.append(e)
            continue

        last_ts = int(prev.get("ts", 0) or 0)
        last_edge = float(prev.get("edge_prob", 0.0) or 0.0)

        if (e["edge_prob"] - last_edge) >= EDGEPROB_JUMP_TO_RESEND:
            kept.append(e)
            continue

        if (now_ts - last_ts) >= cooldown_sec:
            kept.append(e)

    # diversification by team and game
    team_ct = {}
    game_ct = {}
    out = []
    for e in kept:
        t = e.get("team") or ""
        g = e.get("game_id")
        if t:
            if team_ct.get(t, 0) >= MAX_PER_TEAM:
                continue
        if g is not None:
            if game_ct.get(g, 0) >= MAX_PER_GAME:
                continue
        out.append(e)
        if t:
            team_ct[t] = team_ct.get(t, 0) + 1
        if g is not None:
            game_ct[g] = game_ct.get(g, 0) + 1
    return out

def record_sent(state, plays, now_ts):
    sent = state.get("sent", {}) or {}
    for e in plays:
        key = f"{e['prop_type']}|{e['player_id']}|{e['line']:.1f}"
        sent[key] = {"ts": int(now_ts), "edge_prob": float(e["edge_prob"])}
    state["sent"] = sent

# -------------------- FORMATTING --------------------
DISPLAY_LABEL = {
    "points": "Points",
    "threes": "3PT Made",
}

def market_label(prop_type: str) -> str:
    return DISPLAY_LABEL.get(prop_type, prop_type)

# -------------------- MAIN --------------------
def run():
    now_et = _now_et()
    ts_et = now_et.strftime("%Y-%m-%d %I:%M %p ET")
    now_ts = int(now_et.timestamp())

    print(
        f"[BOOT] ts={ts_et} TEST_MODE={int(TEST_MODE)} "
        f"PROP_TYPES={','.join(PROP_TYPES)} MIN_PER_MARKET={MIN_PER_MARKET} MAX_PER_MARKET={MAX_PER_MARKET} "
        f"MAX_TOTAL_PLAYS={MAX_TOTAL_PLAYS} BOOK_VENDORS={','.join(BOOK_VENDORS)} "
        f"ENABLE_SLATE_SCAN={int(ENABLE_SLATE_SCAN)} ENABLE_INJURY_TRIGGERS={int(ENABLE_INJURY_TRIGGERS)} "
        f"STRICT_INJURY_GAME_MATCH={int(STRICT_INJURY_GAME_MATCH)}"
    )

    if TEST_MODE:
        send_one(f"✅ NBA props agent test OK ({ts_et})")
        return

    state = load_state()

    triggers, trigger_meta, trigger_items = injury_triggers(now_et, state)

    # If slate scan is off and injuries off -> nothing to do
    if not ENABLE_SLATE_SCAN and not ENABLE_INJURY_TRIGGERS:
        print("[INFO] Both slate scan and injury triggers disabled.")
        save_state(state)
        return

    if SLATE_ONLY_IN_BURST and not _in_burst_window(now_et):
        print("[INFO] SLATE_ONLY_IN_BURST is on and we are outside burst; skipping.")
        save_state(state)
        return

    edges = build_edges_for_games(now_et, trigger_items)

    # Cooldown + diversify
    edges = apply_cooldown_and_diversify(state, edges, now_ts)

    # Now allocate per market
    by_market = {m: [] for m in PROP_TYPES}
    for e in edges:
        by_market.setdefault(e["prop_type"], []).append(e)

    picks = []
    # pick top per market, then globally cap
    for m in PROP_TYPES:
        rows = by_market.get(m, [])
        rows.sort(key=lambda x: (x["edge_prob"], x["model_prob"], x["trigger_strength"]), reverse=True)
        picks.extend(rows[:MAX_PER_MARKET])

    # Ensure minimum per market if desired (optional)
    if MIN_PER_MARKET > 0:
        for m in PROP_TYPES:
            if len([p for p in picks if p["prop_type"] == m]) < MIN_PER_MARKET:
                # add more from that market if available
                rows = by_market.get(m, [])
                rows.sort(key=lambda x: (x["edge_prob"], x["model_prob"], x["trigger_strength"]), reverse=True)
                need = MIN_PER_MARKET - len([p for p in picks if p["prop_type"] == m])
                existing_keys = {(p["prop_type"], p["player_id"], p["line"]) for p in picks}
                for r in rows:
                    k = (r["prop_type"], r["player_id"], r["line"])
                    if k in existing_keys:
                        continue
                    picks.append(r)
                    existing_keys.add(k)
                    need -= 1
                    if need <= 0:
                        break

    # Global cap: keep best overall
    picks.sort(key=lambda x: (x["edge_prob"], x["model_prob"], x["trigger_strength"]), reverse=True)
    picks = picks[:MAX_TOTAL_PLAYS]

    if not picks:
        # Silent when no picks; if you want a ping, add env + message here
        print("[INFO] No qualified plays this run.")
        save_state(state)
        return

    # message build
    msg = [f"💰 FanDuel Props (data edge) ({ts_et})", ""]

    if triggers:
        msg.append("🚑 Injury feed changes (used as a boost only, not the whole model):")
        msg.append("Triggers:")
        for t in triggers[:10]:
            msg.append(f"- {t}")
        if len(triggers) > 10:
            msg.append(f"- …and {len(triggers)-10} more")
        msg.append("")

    # group by market
    for m in PROP_TYPES:
        rows = [p for p in picks if p["prop_type"] == m]
        if not rows:
            continue
        msg.append(f"🏷️ {market_label(m)}")
        msg.append("")
        for p in rows:
            msg.append(
                f"• {p['player_name']} OVER {p['line']:.1f}  "
                f"(EdgeProb +{p['edge_prob']*100:.1f}%, ModelP {p['model_prob']*100:.0f}%)"
            )
            if p["trigger_strength"] > 0:
                msg.append(f"  Trigger: {p['trigger']} | Strength {p['trigger_strength']:.0f}")
            msg.append(f"  Why: {p['why']} [vendor={p['vendor']}]")
            msg.append("")

    send_chunked("\n".join(msg).strip())
    record_sent(state, picks, now_ts)
    save_state(state)

if __name__ == "__main__":
    run()
