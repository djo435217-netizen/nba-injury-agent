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

# Vendors: allow multiple (fanduel,fanatics,main, etc)
BOOK_VENDOR_RAW = os.environ.get("BOOK_VENDOR", "fanduel").strip().lower()
BOOK_VENDORS = [v.strip() for v in BOOK_VENDOR_RAW.split(",") if v.strip()]

# Markets you want in one run (points + threes)
PROP_TYPES_RAW = os.environ.get("PROP_TYPES", "points,three_pointers_made").strip().lower()
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
MIN_PER_MARKET = int(os.environ.get("MIN_PER_MARKET", "2"))
MAX_PER_MARKET = int(os.environ.get("MAX_PER_MARKET", "6"))
MAX_BET_IDEAS = int(os.environ.get("MAX_BET_IDEAS", "10"))

# Thresholds (shared across markets by default)
MIN_EDGE = float(os.environ.get("MIN_EDGE", "2.5"))
MIN_PROB = float(os.environ.get("MIN_PROB", "0.62"))
STD_FLOOR = float(os.environ.get("STD_FLOOR", "1.0"))  # for 3PT made, keep low; internal floor applied per market too

# Guardrails
MIN_LINE = float(os.environ.get("MIN_LINE", "0.5"))
MAX_LINE = float(os.environ.get("MAX_LINE", "60.0"))
LINE_MIN_GAP = float(os.environ.get("LINE_MIN_GAP", "8.0"))   # avoid obvious “avg >> line” garbage
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
SLATE_ONLY_IN_BURST = os.environ.get("SLATE_ONLY_IN_BURST", "1") == "1"
SLATE_SCAN_MAX_PLAYERS = int(os.environ.get("SLATE_SCAN_MAX_PLAYERS", "260"))

# Cooldown
BET_COOLDOWN_MIN = int(os.environ.get("BET_COOLDOWN_MIN", "180"))
EDGE_JUMP_TO_RESEND = float(os.environ.get("EDGE_JUMP_TO_RESEND", "1.5"))

# Hard safety: do not mislabel injury-triggered plays
STRICT_INJURY_GAME_MATCH = os.environ.get("STRICT_INJURY_GAME_MATCH", "1") == "1"

# Debug
DEBUG_POINTS_SAMPLE = os.environ.get("DEBUG_POINTS_SAMPLE", "0") == "1"
DEBUG_GAMES_MAP = os.environ.get("DEBUG_GAMES_MAP", "0") == "1"


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

def avg_val_min_std(games):
    """
    games tuples are (date, value, minutes) for the selected stat (pts or fg3m).
    """
    if not games:
        return 0.0, 0.0, 0.0
    vals = [x[1] for x in games]
    mins = [x[2] for x in games]
    n = len(vals)
    v_avg = sum(vals) / n
    min_avg = sum(mins) / n
    var = sum((v - v_avg) ** 2 for v in vals) / max(n, 1)
    return v_avg, min_avg, math.sqrt(var)

def _slice_last(games, n):
    if not games:
        return []
    return games[-min(len(games), n):]

def _role_trend(games):
    if not games:
        return 0.0, 0.0, 0.0, 0.0
    long_slice = _slice_last(games, LOOKBACK_GAMES)
    short_slice = _slice_last(games, SHORT_GAMES)

    v_l, min_l, _ = avg_val_min_std(long_slice)
    v_s, min_s, _ = avg_val_min_std(short_slice)
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
DEBUG_PRINTED_ONCE = False

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

def build_games_map(now_et: datetime):
    """
    game_id -> set(team_ids involved)
    Tries multiple shapes, so it won't break if BDL fields differ.
    """
    games = bdl_games_today(now_et)
    out = {}
    for g in games:
        gid = g.get("id")
        if gid is None:
            continue
        teams = set()

        # common BDL shapes
        for k in ("home_team", "visitor_team", "home", "visitor"):
            t = g.get(k) or {}
            tid = t.get("id") if isinstance(t, dict) else None
            if tid is not None:
                try:
                    teams.add(int(tid))
                except Exception:
                    pass

        # sometimes nested like team_ids
        if not teams:
            for k in ("home_team_id", "visitor_team_id"):
                tid = g.get(k)
                if tid is not None:
                    try:
                        teams.add(int(tid))
                    except Exception:
                        pass

        out[int(gid)] = teams

    if DEBUG_GAMES_MAP:
        sample = list(out.items())[:3]
        print("[DEBUG] games_map sample:", sample)

    return out

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

def bdl_find_player_id_on_team(team_name: str, full_name: str):
    roster = bdl_active_roster(team_name)
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

def _stat_key_for_prop(prop_type: str) -> str:
    # BDL stats fields commonly include:
    # - points: "pts"
    # - threes made: often "fg3m"
    p = (prop_type or "").lower()
    if p in ("points", "pts"):
        return "pts"
    if p in ("three_pointers_made", "threes", "3pt", "3pm", "fg3m"):
        return "fg3m"
    # default: points so we don't silently break
    return "pts"

def bdl_last_n_games_stats(player_ids, season: int, n: int, stat_key: str):
    """
    Returns {pid: [(date, value, minutes), ...]} for chosen stat_key.
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

            raw_val = row.get(stat_key, 0)
            try:
                val = float(raw_val or 0)
            except Exception:
                val = 0.0

            mins = _parse_minutes(row.get("min"))

            if date:
                out[pid].append((date, val, mins))

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
    Pull props rows for a given prop_type (points / three_pointers_made).
    Cache: (game_id, vendor, prop_type)
    """
    global DEBUG_PRINTED_ONCE
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

    if DEBUG_POINTS_SAMPLE and (not DEBUG_PRINTED_ONCE) and props:
        print(f"[DEBUG] SAMPLE PROP ROW ({prop_type}):", json.dumps(props[0])[:2000])
        DEBUG_PRINTED_ONCE = True

    PROPS_CACHE[key] = props
    return props

def _pick_main_line(rows_for_player):
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
        if line < MIN_LINE or line > MAX_LINE:
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
    Finds “main” line for player for selected prop_type,
    trying vendors in order, then fallback to no-vendor.
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
        line = _pick_main_line(rows)
        if line is not None:
            return float(line)
    return None


# -------------------- PROJECTION CORE --------------------
def compute_projection_and_prob(games_all, line, injury_boost_val=0.0, injury_boost_min=0.0, std_floor_override=None):
    base_slice = _slice_last(games_all, BASELINE_GAMES)
    l10_slice = _slice_last(games_all, LOOKBACK_GAMES)
    l3_slice = _slice_last(games_all, SHORT_GAMES)

    base_avg, _, base_std = avg_val_min_std(base_slice)
    l10_avg, l10_min, l10_std = avg_val_min_std(l10_slice)
    l3_avg, _, _ = avg_val_min_std(l3_slice)

    floor = STD_FLOOR if std_floor_override is None else float(std_floor_override)
    sigma = max(floor, (l10_std if l10_std > 0 else base_std if base_std > 0 else floor))

    proj = (W_BASE * base_avg) + (W_L10 * l10_avg) + (W_L3 * l3_avg) + (W_LINE * line)

    rate = l10_avg / max(l10_min, 1e-6)  # value per minute
    proj += injury_boost_val
    proj += (injury_boost_min * rate * 0.20)

    edge = proj - line
    z = (proj - line) / sigma
    prob_over = _norm_cdf(z)
    return proj, edge, prob_over, (base_avg, l10_avg, l3_avg, l10_min, sigma, rate)


# -------------------- INJURY ENGINE --------------------
def build_injury_edges(team_name, injured_name, injured_status, exclude_names_lower, now_et, prop_type, games_map):
    """
    IMPORTANT upgrade:
    - We only label a play as injury-triggered if:
        (a) the candidate player is on the injured team (team_id match), AND
        (b) the chosen game_id includes the injured team (game team_ids include team_id)
      If we cannot verify team_id, we either soft-skip (strict) or allow (non-strict).
    """
    season = _season_year(now_et)
    team_map = bdl_team_name_to_id()
    injured_team_id = team_map.get(team_name)

    roster = bdl_active_roster(team_name)
    if not roster:
        return []

    roster_tuples = []
    for p in roster:
        pid = p.get("id")
        nm = f"{p.get('first_name','')} {p.get('last_name','')}".strip()
        t = p.get("team") or {}
        tid = t.get("id")
        if pid is None or not nm:
            continue
        if _clean_name(nm) in exclude_names_lower:
            continue
        try:
            tid_int = int(tid) if tid is not None else None
        except Exception:
            tid_int = None
        roster_tuples.append((int(pid), nm, tid_int))

    if not roster_tuples:
        return []

    injured_pid = bdl_find_player_id_on_team(team_name, injured_name)
    if not injured_pid:
        return []

    # For vacancy strength, we use POINTS + MINUTES only (still the best signal for role shift),
    # even if prop_type is threes. That’s intentional.
    inj_games_pts = bdl_last_n_games_stats([injured_pid], season, BASELINE_GAMES, stat_key="pts").get(injured_pid, [])
    ip10, im10, _ = avg_val_min_std(_slice_last(inj_games_pts, LOOKBACK_GAMES))
    if len(inj_games_pts) < 3:
        return []

    status = (injured_status or "").lower()
    STATUS_MULT = {"out": 1.0, "doubtful": 0.8, "questionable": 0.55}.get(status, 0.65)

    vac_pts = ip10 * STATUS_MULT
    vac_min = im10 * STATUS_MULT
    if not ((vac_min >= MIN_VAC_MIN) or (vac_pts >= MIN_VAC_PTS)):
        return []

    trigger_strength = min(100.0, (vac_min * 1.2 + vac_pts * 1.5))

    # Candidate stats for THIS MARKET
    stat_key = _stat_key_for_prop(prop_type)
    cand_ids = [pid for pid, _, _ in roster_tuples]
    stats = bdl_last_n_games_stats(cand_ids, season, BASELINE_GAMES, stat_key=stat_key)

    game_ids = bdl_games_today_ids(now_et)
    if not game_ids:
        return []

    # Market-specific std floor (threes are lower variance scale)
    std_floor = 1.0 if stat_key == "fg3m" else 5.0

    ideas = []
    for pid, nm, player_team_id in roster_tuples:
        games = stats.get(pid, [])
        if len(games) < 8:
            continue

        min_s, min_l, rate_s, rate_l = _role_trend(games)
        min_delta = min_s - min_l
        rate_delta = rate_s - rate_l

        l10_avg, l10_min, _ = avg_val_min_std(_slice_last(games, LOOKBACK_GAMES))
        if l10_min < 10:
            continue

        # absorption score (same as your working logic)
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
        for gid in game_ids:
            line = line_for_player(gid, pid, prop_type)
            if line is not None:
                use_gid = int(gid)
                break
        if line is None:
            continue

        # Injury-trigger verification (best “don’t break” safety)
        verified = True
        if injured_team_id is not None:
            if player_team_id is not None and int(player_team_id) != int(injured_team_id):
                verified = False
            teams_in_game = games_map.get(int(use_gid), set())
            if teams_in_game and int(injured_team_id) not in teams_in_game:
                verified = False
        else:
            # can't verify injured team id
            verified = (not STRICT_INJURY_GAME_MATCH)

        if not verified:
            # do NOT include in injury list; prevents bad labeling
            continue

        # avoid obvious “avg >> line” junk
        if (l10_avg - line) > LINE_MIN_GAP:
            continue

        # injury boost in "value" units for the market:
        # - for points, boost is pts; for threes, we convert vac_pts to small bump (still minutes-driven)
        if stat_key == "pts":
            injury_boost_val = min(BOOST_CAP_PTS, vac_pts * absorption * 0.65)
        else:
            # threes: most of the signal is minutes; keep value bump small
            injury_boost_val = min(1.2, (vac_pts / 25.0) * absorption)

        injury_boost_min = min(BOOST_CAP_MIN, vac_min * absorption * 0.25)

        proj, edge, prob_over, aux = compute_projection_and_prob(
            games_all=games,
            line=line,
            injury_boost_val=injury_boost_val,
            injury_boost_min=injury_boost_min,
            std_floor_override=std_floor
        )
        base_avg, l10_avg2, l3_avg, l10_min2, _, _ = aux

        if edge < MIN_EDGE or prob_over < MIN_PROB:
            continue
        if min_delta < MIN_DELTA_FLOOR and edge < (MIN_EDGE + 1.5):
            continue

        why = (
            f"TriggerStrength {trigger_strength:.0f} | Absorb {absorption:.2f}. "
            f"{injured_name} {injured_status.upper()} vacates ~{vac_pts:.1f} Points / {vac_min:.1f} min. "
            f"{nm} base(L{BASELINE_GAMES}) {base_avg:.1f}, L10 {l10_avg2:.1f}, L3 {l3_avg:.1f} "
            f"(mins L10 {l10_min2:.1f}). Role Δmin={min_delta:+.1f}, Δrate={rate_delta:+.2f}. "
            f"Proj {proj:.1f} vs MAIN line {line:.1f} | edge +{edge:.1f} | P≈{prob_over*100:.0f}%."
        )

        ideas.append({
            "section": "injury",
            "market": prop_type,
            "player_name": nm,
            "player_id": pid,
            "line": float(line),
            "proj": float(proj),
            "edge": float(edge),
            "prob_over": float(prob_over),
            "trigger_strength": float(trigger_strength),
            "trigger": f"{injured_name} ({team_name}) {injured_status}",
            "why": why,
            "game_id": use_gid,
        })

    ideas.sort(key=lambda x: (x["trigger_strength"], x["edge"], x["prob_over"]), reverse=True)
    return ideas


# -------------------- SLATE SCAN ENGINE --------------------
def slate_scan_edges(now_et, prop_type):
    if not ENABLE_SLATE_SCAN:
        return []
    if SLATE_ONLY_IN_BURST and (not _in_burst_window(now_et)):
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
            props = bdl_player_props(gid, v, prop_type)
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
            line = _pick_main_line(rows)
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

    # stats for this market
    stat_key = _stat_key_for_prop(prop_type)
    pids = list(player_to_best_line.keys())
    stats = bdl_last_n_games_stats(pids, season, BASELINE_GAMES, stat_key=stat_key)  # fills PLAYER_NAME_CACHE

    std_floor = 1.0 if stat_key == "fg3m" else 5.0

    ideas = []
    for pid in pids:
        games = stats.get(pid, [])
        if len(games) < 10:
            continue

        line, gid = player_to_best_line[pid]

        l10_avg, l10_min, _ = avg_val_min_std(_slice_last(games, LOOKBACK_GAMES))
        if l10_min < 10:
            continue
        if (l10_avg - line) > LINE_MIN_GAP:
            continue

        min_s, min_l, rate_s, rate_l = _role_trend(games)
        min_delta = min_s - min_l
        rate_delta = rate_s - rate_l

        proj, edge, prob_over, aux = compute_projection_and_prob(
            games_all=games,
            line=line,
            std_floor_override=std_floor
        )
        base_avg, l10_avg2, l3_avg, l10_min2, _, _ = aux

        if edge < MIN_EDGE or prob_over < MIN_PROB:
            continue
        if min_delta < MIN_DELTA_FLOOR and edge < (MIN_EDGE + 2.0):
            continue

        name = PLAYER_NAME_CACHE.get(int(pid), f"Player {pid}")

        why = (
            f"SlateScan. base(L{BASELINE_GAMES}) {base_avg:.1f}, L10 {l10_avg2:.1f}, L3 {l3_avg:.1f} "
            f"(mins L10 {l10_min2:.1f}). Role Δmin={min_delta:+.1f}, Δrate={rate_delta:+.2f}. "
            f"Proj {proj:.1f} vs MAIN line {line:.1f} | edge +{edge:.1f} | P≈{prob_over*100:.0f}%."
        )

        ideas.append({
            "section": "slate",
            "market": prop_type,
            "player_name": name,
            "player_id": int(pid),
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
    return ideas


# -------------------- COOLDOWN FILTER --------------------
def apply_cooldown(state, ideas, now_ts: int):
    sent = state.get("sent_bets", {}) or {}
    cooldown_sec = BET_COOLDOWN_MIN * 60

    kept = []
    for i in ideas:
        # include market so points/threes are tracked separately
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
        f"PROP_TYPES={','.join(PROP_TYPES)} "
        f"MIN_PER_MARKET={MIN_PER_MARKET} MAX_PER_MARKET={MAX_PER_MARKET} MAX_BET_IDEAS={MAX_BET_IDEAS} "
        f"BOOK_VENDORS={','.join(BOOK_VENDORS)} ENABLE_SLATE_SCAN={int(ENABLE_SLATE_SCAN)} "
        f"STRICT_INJURY_GAME_MATCH={int(STRICT_INJURY_GAME_MATCH)}"
    )

    if TEST_MODE:
        send_one(f"✅ NBA betting agent test OK ({ts_et})")
        return

    state = load_state()
    old_players = state.get("players", {})

    # Build today games_map once (used to prevent mislabeling injury-trigger plays)
    games_map = build_games_map(now_et)

    # injuries
    sr = fetch_sportradar_injuries()
    new_players = parse_injuries(sr)

    exclude_names_lower = {_clean_name(v.get("name", "")) for v in new_players.values() if v.get("name")}

    # For each market, compute injury + slate, then format a section
    all_outputs = []
    anything_sent = False

    for market in PROP_TYPES:
        triggers = []
        injury_ideas = []

        for pid, cur in new_players.items():
            if not status_in_scope(cur.get("status", "")):
                continue

            prev = old_players.get(pid)
            if IMPACT_ONLY_CHANGES:
                is_new = prev is None
                is_changed = (not is_new) and ((prev.get("status"), prev.get("detail")) != (cur.get("status"), cur.get("detail")))
                if not (is_new or is_changed):
                    continue

            team_name = cur.get("team", "")
            injured_name = cur.get("name", "")
            injured_status = (cur.get("status") or "").strip()

            ideas = build_injury_edges(
                team_name=team_name,
                injured_name=injured_name,
                injured_status=injured_status,
                exclude_names_lower=exclude_names_lower | {_clean_name(injured_name)},
                now_et=now_et,
                prop_type=market,
                games_map=games_map
            )

            if ideas:
                triggers.append(f"{injured_name} ({team_name}) {injured_status}")
                injury_ideas.extend(ideas)

        slate_ideas = slate_scan_edges(now_et, market)

        combined = injury_ideas + slate_ideas

        # de-dupe within market (keep best edge for a player/section)
        best = {}
        for i in combined:
            k = (i["section"], int(i["player_id"]))
            if (k not in best) or ((i["edge"], i["prob_over"]) > (best[k]["edge"], best[k]["prob_over"])):
                best[k] = i
        combined = list(best.values())

        combined = apply_cooldown(state, combined, now_ts)

        # pick top per section
        injury_out = sorted(
            [i for i in combined if i["section"] == "injury"],
            key=lambda x: (x["trigger_strength"], x["edge"], x["prob_over"]),
            reverse=True
        )[:MAX_PER_MARKET]

        slate_out = sorted(
            [i for i in combined if i["section"] == "slate"],
            key=lambda x: (x["edge"], x["prob_over"]),
            reverse=True
        )[:MAX_PER_MARKET]

        # enforce minimum per market if possible
        picked = injury_out + slate_out
        picked = picked[:MAX_PER_MARKET * 2]

        if len(picked) < MIN_PER_MARKET:
            # if we don't have enough, still allow sending nothing for this market
            pass

        # Build text block for this market
        if picked:
            anything_sent = True
            market_title = "Points" if market == "points" else "3PT Made" if market == "three_pointers_made" else market
            block = [f"🏷️ {market_title}", ""]

            if injury_out:
                block.append("🚑 Injury-Triggered Plays:")
                if triggers:
                    block.append("Triggers:")
                    for t in triggers[:8]:
                        block.append(f"- {t}")
                    if len(triggers) > 8:
                        block.append(f"- …and {len(triggers)-8} more")
                block.append("")
                for i in injury_out:
                    block.append(f"• {i['player_name']} OVER {i['line']:.1f}  (edge +{i['edge']:.1f}, P≈{i['prob_over']*100:.0f}%)")
                    block.append(f"  Trigger: {i['trigger']}")
                    block.append(f"  Why: {i['why']}")
                    block.append("")

            if slate_out:
                block.append("🌎 League-Wide Slate Scan (no injury required):")
                block.append("")
                for i in slate_out:
                    block.append(f"• {i['player_name']} OVER {i['line']:.1f}  (edge +{i['edge']:.1f}, P≈{i['prob_over']*100:.0f}%)")
                    block.append(f"  Why: {i['why']}")
                    block.append("")

            all_outputs.append("\n".join(block).strip())

            # record cooldown only for the plays we actually selected for this market
            record_sent(state, injury_out + slate_out, now_ts)

    if anything_sent:
        header = [f"💰 FanDuel Props ({ts_et})", ""]
        full_msg = "\n\n".join(header + all_outputs)
        # keep overall message under chunker
        send_chunked(full_msg.strip())
    else:
        if SEND_NO_EDGE_PING and _in_burst_window(now_et):
            send_one(f"🧠 No edges ≥ {MIN_EDGE:.1f} and P ≥ {MIN_PROB:.2f} this run. ({ts_et})")

    # Save injury state so “only changes” works
    state["players"] = new_players
    save_state(state)


if __name__ == "__main__":
    run()
