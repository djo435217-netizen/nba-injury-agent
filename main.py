import os
import json
import re
import time
import math
from datetime import datetime, timezone
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

# -------------------- CONFIG --------------------
TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
MAX_BODY_CHARS = int(os.environ.get("MAX_BODY_CHARS", "1500"))

BOOK_VENDOR_RAW = os.environ.get("BOOK_VENDORS", os.environ.get("BOOK_VENDOR", "fanduel")).strip().lower()
BOOK_VENDORS = [v.strip() for v in BOOK_VENDOR_RAW.split(",") if v.strip()]

PROP_TYPES_RAW = os.environ.get("PROP_TYPES", "points").strip().lower()
PROP_TYPES = [p.strip() for p in PROP_TYPES_RAW.split(",") if p.strip()]

ENABLE_INJURY_TRIGGERS = os.environ.get("ENABLE_INJURY_TRIGGERS", "1") == "1"
ENABLE_SLATE_SCAN = os.environ.get("ENABLE_SLATE_SCAN", "1") == "1"

# Consensus + Steam + EV + Value-edge
MIN_VENDORS_FOR_CONSENSUS = int(os.environ.get("MIN_VENDORS_FOR_CONSENSUS", "1"))

ENABLE_STEAM = os.environ.get("ENABLE_STEAM", "0") == "1"
STEAM_MIN_SCORE = float(os.environ.get("STEAM_MIN_SCORE", "1.0"))
STEAM_MAX_AGE_MIN = int(os.environ.get("STEAM_MAX_AGE_MIN", "240"))

EV_MIN = float(os.environ.get("EV_MIN", "0.00"))
VALUE_EDGE_MIN = float(os.environ.get("VALUE_EDGE_MIN", "0.00"))

# Output sizing
MIN_PER_MARKET = int(os.environ.get("MIN_PER_MARKET", "0"))
MAX_PER_MARKET = int(os.environ.get("MAX_PER_MARKET", "6"))
MAX_TOTAL_PLAYS = int(os.environ.get("MAX_TOTAL_PLAYS", "10"))

# Exposure caps (NEW)
MAX_PLAYS_PER_TEAM = int(os.environ.get("MAX_PLAYS_PER_TEAM", "1"))
MAX_PLAYS_PER_GAME = int(os.environ.get("MAX_PLAYS_PER_GAME", "2"))

# Market respect / sharp vendor requirement (NEW)
MIN_SHARP_VENDORS = int(os.environ.get("MIN_SHARP_VENDORS", "0"))
SHARP_VENDORS_RAW = os.environ.get("SHARP_VENDORS", "draftkings,caesars,betmgm,bet365,circa,superbook").strip().lower()
SHARP_VENDORS = {v.strip() for v in SHARP_VENDORS_RAW.split(",") if v.strip()}

# Windows
BASELINE_GAMES = int(os.environ.get("BASELINE_GAMES", "30"))
LOOKBACK_GAMES = int(os.environ.get("LOOKBACK_GAMES", "10"))
SHORT_GAMES = int(os.environ.get("SHORT_GAMES", "3"))

# Model thresholds
MIN_EDGE = float(os.environ.get("MIN_EDGE", "2.5"))
MIN_PROB = float(os.environ.get("MIN_PROB", "0.62"))
STD_FLOOR = float(os.environ.get("STD_FLOOR", "5.0"))

# Guardrails
MIN_L10_MIN = float(os.environ.get("MIN_L10_MIN", "10"))
LINE_MIN_GAP = float(os.environ.get("LINE_MIN_GAP", "8.0"))

# Injury vacancy requirements
IMPACT_STATUSES_RAW = os.environ.get("IMPACT_STATUSES", "out,doubtful,questionable").strip()
IMPACT_STATUSES = {x.strip().lower() for x in IMPACT_STATUSES_RAW.split(",") if x.strip()}
IMPACT_ONLY_CHANGES = os.environ.get("IMPACT_ONLY_CHANGES", "1") == "1"

MIN_VAC_MIN = float(os.environ.get("MIN_VAC_MIN", "10.0"))
MIN_VAC_STAT = float(os.environ.get("MIN_VAC_PTS", os.environ.get("MIN_VAC_STAT", "6.0")))
BOOST_CAP_RATE = float(os.environ.get("BOOST_CAP_RATE", "0.20"))
BOOST_CAP_STAT = float(os.environ.get("BOOST_CAP_STAT", "5.5"))
BOOST_CAP_MIN = float(os.environ.get("BOOST_CAP_MIN", "6.0"))

# Cooldown
BET_COOLDOWN_MIN = int(os.environ.get("BET_COOLDOWN_MIN", "180"))
EDGE_JUMP_TO_RESEND = float(os.environ.get("EDGE_JUMP_TO_RESEND", "1.5"))

# Runtime guard
RUN_MAX_SECONDS = int(os.environ.get("RUN_MAX_SECONDS", "170"))
STAT_BATCH_SIZE = int(os.environ.get("STAT_BATCH_SIZE", "90"))

DEBUG_PROP_SAMPLE_TYPES = os.environ.get("DEBUG_PROP_SAMPLE_TYPES", "0").strip().lower()

# Plus-odds bucket
PLUS_ODDS_MIN = float(os.environ.get("PLUS_ODDS_MIN", "100"))
PLUS_ODDS_TOPN = int(os.environ.get("PLUS_ODDS_TOPN", "3"))

# SYNTH LADDER (longshots)
ENABLE_SYNTH_LADDER = os.environ.get("ENABLE_SYNTH_LADDER", "1") == "1"
SYNTH_LADDER_MIN_ODDS = float(os.environ.get("SYNTH_LADDER_MIN_ODDS", "250"))
SYNTH_LADDER_MAX_ODDS = float(os.environ.get("SYNTH_LADDER_MAX_ODDS", "450"))
SYNTH_LADDER_RUNG_STEPS_RAW = os.environ.get("SYNTH_LADDER_RUNG_STEPS", "3,4,5,6")
SYNTH_LADDER_RUNG_STEPS = [int(x.strip()) for x in SYNTH_LADDER_RUNG_STEPS_RAW.split(",") if x.strip().isdigit()]
SYNTH_LADDER_TOPN = int(os.environ.get("SYNTH_LADDER_TOPN", "6"))
SYNTH_LADDER_MIN_L10_MIN = float(os.environ.get("SYNTH_LADDER_MIN_L10_MIN", "18"))

# Threes beta-binomial controls (NEW)
THREES_USE_BETA_BINOM = os.environ.get("THREES_USE_BETA_BINOM", "1") == "1"
THREES_ATTEMPT_LOOKBACK = int(os.environ.get("THREES_ATTEMPT_LOOKBACK", "10"))
THREES_MIN_AVG_ATTEMPTS = float(os.environ.get("THREES_MIN_AVG_ATTEMPTS", "2.5"))

# LineupExperts / FantasyNerds style (NEW; optional + safe)
ENABLE_LINEUPEXPERTS = os.environ.get("ENABLE_LINEUPEXPERTS", "1") == "1"
LINEUPEXPERTS_API_KEY = os.environ.get("LINEUPEXPERTS_API_KEY", "").strip()

LINEUPEXPERTS_BASE_URL = os.environ.get("LINEUPEXPERTS_BASE_URL", "https://api.fantasynerds.com/v1").strip().rstrip("/")
LINEUPEXPERTS_INJURIES_PATH = os.environ.get("LINEUPEXPERTS_INJURIES_PATH", "/nba/injuries").strip()
LINEUPEXPERTS_NEWS_PATH = os.environ.get("LINEUPEXPERTS_NEWS_PATH", "/nba/player-news").strip()
LINEUPEXPERTS_LINEUPS_PATH = os.environ.get("LINEUPEXPERTS_LINEUPS_PATH", "/nba/starting-lineups").strip()

NEWS_MAX_AGE_MIN = int(os.environ.get("NEWS_MAX_AGE_MIN", "360"))
NEWS_START_KEYWORDS = ("will start", "starting", "named starter", "enters starting", "starting lineup")
NEWS_LIMIT_KEYWORDS = ("minutes restriction", "will be limited", "limit", "cap")

# -------------------- RUNTIME DEADLINE --------------------
RUN_START = time.time()

def deadline_exceeded() -> bool:
    return (time.time() - RUN_START) > RUN_MAX_SECONDS

# -------------------- UTILS --------------------
def _now_et() -> datetime:
    return datetime.now(ET)

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

def american_to_prob(odds: float) -> float:
    o = float(odds)
    if o > 0:
        return 100.0 / (o + 100.0)
    return (-o) / ((-o) + 100.0)

def american_to_payout(odds: float) -> float:
    o = float(odds)
    if o > 0:
        return o / 100.0
    return 100.0 / (-o)

def ev_per_dollar(p_win: float, odds: float) -> float:
    b = american_to_payout(odds)
    return p_win * b - (1.0 - p_win)

def prob_to_american(p: float) -> float:
    p = max(1e-9, min(1 - 1e-9, float(p)))
    dec = 1.0 / p
    if dec >= 2.0:
        return (dec - 1.0) * 100.0   # +odds
    return -100.0 / (dec - 1.0)      # -odds

def avg_stat_min_std(games):
    # games items may be: (date, val, mins, extra_attempts, team_id)
    if not games:
        return 0.0, 0.0, 0.0
    vals = [x[1] for x in games]
    mins = [x[2] for x in games]
    n = len(vals)
    v_avg = sum(vals) / n
    m_avg = sum(mins) / n
    var = sum((v - v_avg) ** 2 for v in vals) / max(n, 1)
    return v_avg, m_avg, math.sqrt(var)

def _slice_last(games, n):
    if not games:
        return []
    return games[-min(len(games), n):]

def _role_trend(games, long_n=LOOKBACK_GAMES, short_n=SHORT_GAMES):
    if not games:
        return 0.0, 0.0, 0.0, 0.0
    long_slice = _slice_last(games, long_n)
    short_slice = _slice_last(games, short_n)
    v_l, m_l, _ = avg_stat_min_std(long_slice)
    v_s, m_s, _ = avg_stat_min_std(short_slice)
    rate_l = v_l / max(m_l, 1e-6)
    rate_s = v_s / max(m_s, 1e-6)
    return m_s, m_l, rate_s, rate_l

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"players": {}, "sent_bets": {}, "market": {}}
    try:
        with open(STATE_FILE, "r") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return {"players": {}, "sent_bets": {}, "market": {}}
        raw.setdefault("players", {})
        raw.setdefault("sent_bets", {})
        raw.setdefault("market", {})
        return raw
    except Exception:
        return {"players": {}, "sent_bets": {}, "market": {}}

def save_state(state):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, sort_keys=True)
    except Exception:
        pass

def send_one(body: str):
    try:
        twilio.messages.create(from_=FROM_WHATSAPP, to=TO_WHATSAPP, body=body[:MAX_BODY_CHARS])
    except Exception as e:
        print(f"[WARN] Twilio send failed: {type(e).__name__}: {e}")

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

def _chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def _parse_iso_dt_maybe(s: str):
    if not s:
        return None
    try:
        # handle "2026-03-04T13:31:24.792Z" etc
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except Exception:
        return None

# -------------------- SPORTRADAR (injuries) --------------------
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

# -------------------- LINEUPEXPERTS (FantasyNerds style) --------------------
def _le_get(path: str, params=None, timeout: int = 20):
    if not ENABLE_LINEUPEXPERTS or not LINEUPEXPERTS_API_KEY:
        return None
    url = f"{LINEUPEXPERTS_BASE_URL}{path}"
    qp = dict(params or {})
    qp["apikey"] = LINEUPEXPERTS_API_KEY
    try:
        r = requests.get(url, params=qp, timeout=timeout)
        if r.status_code != 200:
            print(f"[WARN] LineupExperts HTTP {r.status_code} for {path}: {r.text[:120]}")
            return None
        return r.json()
    except Exception as e:
        print(f"[WARN] LineupExperts error for {path}: {type(e).__name__}: {e}")
        return None

def le_fetch_injuries_map():
    """
    Returns name_key -> dict(status, team, detail)
    We intentionally key by cleaned player name (since provider IDs vary).
    """
    data = _le_get(LINEUPEXPERTS_INJURIES_PATH)
    if not data:
        return {}

    # Try common shapes: {"Injuries":[...]} or list[...]
    rows = data.get("Injuries") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return {}

    out = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        name = (r.get("PlayerName") or r.get("player") or r.get("name") or "").strip()
        if not name:
            continue
        status = (r.get("Status") or r.get("status") or "").strip().lower()
        # normalize to your statuses
        if "out" in status:
            st = "out"
        elif "doubt" in status:
            st = "doubtful"
        elif "ques" in status or "prob" in status:
            st = "questionable"
        else:
            continue
        team = (r.get("Team") or r.get("team") or "").strip()
        detail = (r.get("Injury") or r.get("injury") or r.get("Notes") or r.get("notes") or "").strip()
        out[_clean_name(name)] = {"status": st, "team": team, "detail": detail}
    return out

def le_fetch_news_boosts():
    """
    Returns name_key -> {"start_boost":bool, "limit_flag":bool, "age_min":float}
    """
    data = _le_get(LINEUPEXPERTS_NEWS_PATH)
    if not data:
        return {}

    rows = data.get("PlayerNews") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return {}

    out = {}
    now_utc = datetime.now(timezone.utc)
    for r in rows:
        if not isinstance(r, dict):
            continue
        name = (r.get("PlayerName") or r.get("player") or r.get("name") or "").strip()
        if not name:
            continue
        txt = (r.get("News") or r.get("news") or r.get("Report") or r.get("report") or "").strip()
        if not txt:
            continue

        dt = _parse_iso_dt_maybe(r.get("Updated") or r.get("updated") or r.get("Date") or r.get("date") or "")
        age_min = None
        if dt is not None:
            try:
                dt_utc = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
                age_min = (now_utc - dt_utc).total_seconds() / 60.0
            except Exception:
                age_min = None

        # If we can't parse recency, keep but treat as old
        if age_min is not None and age_min > NEWS_MAX_AGE_MIN:
            continue

        lo = txt.lower()
        start_boost = any(k in lo for k in NEWS_START_KEYWORDS)
        limit_flag = any(k in lo for k in NEWS_LIMIT_KEYWORDS)

        if not (start_boost or limit_flag):
            continue

        out[_clean_name(name)] = {"start_boost": start_boost, "limit_flag": limit_flag, "age_min": float(age_min) if age_min is not None else None}
    return out

# -------------------- BALLDONTLIE --------------------
BDL_HEADERS = {"Authorization": BALLDONTLIE_API_KEY}
BDL_PREFIXES = ["/nba", ""]

BDL_MAX_RETRIES = int(os.environ.get("BDL_MAX_RETRIES", "5"))
BDL_RETRY_BASE_SEC = float(os.environ.get("BDL_RETRY_BASE_SEC", "1.5"))
BDL_PER_PAGE = int(os.environ.get("BDL_PER_PAGE", "100"))
BDL_MAX_PAGES = int(os.environ.get("BDL_MAX_PAGES", "10"))

TEAM_CACHE = None
PLAYER_NAME_CACHE = {}   # pid -> "First Last"
PLAYER_TEAM_CACHE = {}   # pid -> team_id (from stats)
PROPS_CACHE = {}         # (gid, vendor, prop_type) -> list[rows]

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
                    sleep_s = float(retry_after) if retry_after else (BDL_RETRY_BASE_SEC * (2 ** attempt))
                    last_err = f"{r.status_code} {r.text[:120]}"
                    time.sleep(min(sleep_s, 20.0))
                    continue
                if r.status_code != 200:
                    raise RuntimeError(f"BallDontLie error {r.status_code}: {r.text[:300]}")
                return r.json()
            except Exception as e:
                last_err = str(e)
                time.sleep(min(BDL_RETRY_BASE_SEC * (2 ** attempt), 20.0))
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
    while pages < 5 and (not deadline_exceeded()):
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

def bdl_last_n_games_stats(player_ids, season: int, n: int, stat_key: str):
    """
    Returns dict pid -> list[(date, val, mins, extra_attempts, team_id)]
    extra_attempts is fg3a when stat_key is fg3m, else None.
    """
    out = {int(pid): [] for pid in player_ids}
    if not player_ids:
        return out

    cursor = None
    pages = 0
    while pages < BDL_MAX_PAGES and (not deadline_exceeded()):
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

            team = row.get("team") or {}
            team_id = team.get("id")
            if team_id is not None:
                try:
                    PLAYER_TEAM_CACHE[pid] = int(team_id)
                except Exception:
                    pass

            game = row.get("game") or {}
            date = game.get("date")
            val = float(row.get(stat_key, 0) or 0)
            mins = _parse_minutes(row.get("min"))

            extra_attempts = None
            if stat_key == "fg3m":
                try:
                    extra_attempts = float(row.get("fg3a", 0) or 0)
                except Exception:
                    extra_attempts = None

            if date:
                out[pid].append((date, val, mins, extra_attempts, PLAYER_TEAM_CACHE.get(pid)))

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

def bdl_fetch_props_for_game(game_id: int, vendor: str | None, prop_type: str):
    key = (int(game_id), (vendor or "NO_VENDOR"), prop_type)
    if key in PROPS_CACHE:
        return PROPS_CACHE[key]

    params = {"game_id": int(game_id), "prop_type": prop_type}
    if vendor:
        params["vendors[]"] = [vendor]

    try:
        resp = _bdl_get("/v2/odds/player_props", params=params)
        props = resp.get("data") or []
    except Exception as e:
        print(f"[WARN] props fetch failed gid={game_id} vendor={vendor} prop_type={prop_type}: {e}")
        props = []

    if DEBUG_PROP_SAMPLE_TYPES and (prop_type in DEBUG_PROP_SAMPLE_TYPES.split(",")) and props:
        print(f"[DEBUG] SAMPLE PROP ROW ({prop_type}, vendor={vendor or 'NO_VENDOR'}): {json.dumps(props[0])[:2000]}")

    PROPS_CACHE[key] = props
    return props

# -------------------- PROP TYPES / STAT KEYS --------------------
STAT_KEY_BY_PROP = {
    "points": "pts",
    "threes": "fg3m",
    "three_pointers_made": "fg3m",
}

# -------------------- PROP COLLECTION (FAST) --------------------
def build_today_props(now_et: datetime):
    """
    One pass: fetch props for all games today for each prop_type.
    Returns:
      lines_map[prop_type][pid] -> list of dict rows for over_under
    """
    game_ids = bdl_games_today_ids(now_et)
    lines_map = {pt: {} for pt in PROP_TYPES}

    for gid in game_ids:
        if deadline_exceeded():
            break

        for pt in PROP_TYPES:
            if deadline_exceeded():
                break

            merged = []
            for v in BOOK_VENDORS + [None]:
                if deadline_exceeded():
                    break
                props = bdl_fetch_props_for_game(gid, v, pt)
                if not props:
                    continue
                merged.extend(props)

            if not merged:
                continue

            for pp in merged:
                try:
                    pid = int(pp.get("player_id"))
                except Exception:
                    continue

                market = pp.get("market") or {}
                mtype = (market.get("type") or "").lower()
                if mtype != "over_under":
                    continue

                try:
                    line = float(pp.get("line_value"))
                except Exception:
                    continue

                over_odds = market.get("over_odds")
                under_odds = market.get("under_odds")
                if not isinstance(over_odds, (int, float)) or not isinstance(under_odds, (int, float)):
                    continue

                row = {
                    "pid": pid,
                    "gid": int(pp.get("game_id")) if pp.get("game_id") is not None else int(gid),
                    "vendor": (pp.get("vendor") or (v or "no_vendor")).strip().lower(),
                    "prop_type": (pp.get("prop_type") or pt),
                    "line": float(line),
                    "over_odds": float(over_odds),
                    "under_odds": float(under_odds),
                    "updated_at": pp.get("updated_at"),
                }
                lines_map.setdefault(pt, {}).setdefault(pid, []).append(row)

    return lines_map

# -------------------- CONSENSUS + OFFER PICKING --------------------
def _round_to_half(x: float) -> float:
    return round(float(x) * 2.0) / 2.0

def consensus_line(rows):
    """
    Returns:
      (cons_line, n_vendors, vendors_set, sharp_vendor_count)
    """
    if not rows:
        return None, 0, set(), 0

    by_vendor = {}
    for r in rows:
        v = str(r.get("vendor") or "").strip().lower()
        if not v:
            continue
        try:
            line = float(r["line"])
        except Exception:
            continue
        by_vendor.setdefault(v, _round_to_half(line))

    vendors = set(by_vendor.keys())
    lines = sorted(by_vendor.values())
    n = len(lines)
    if n == 0:
        return None, 0, vendors, 0

    sharp_cnt = sum(1 for v in vendors if v in SHARP_VENDORS)

    mid = n // 2
    if n % 2 == 1:
        return float(lines[mid]), n, vendors, sharp_cnt
    return float(0.5 * (lines[mid - 1] + lines[mid])), n, vendors, sharp_cnt

def best_offer_near_consensus(rows, cons_line: float):
    if not rows or cons_line is None:
        return None

    cons_line = float(cons_line)
    exact = [r for r in rows if abs(float(r.get("line", 9e9)) - cons_line) < 1e-9]
    if exact:
        pool = exact
    else:
        near = [r for r in rows if abs(float(r.get("line", 9e9)) - cons_line) <= 0.5 + 1e-9]
        pool = near if near else rows

    # prefer best over odds, then closest line
    def score(r):
        try:
            return (float(r["over_odds"]), -abs(float(r["line"]) - cons_line))
        except Exception:
            return (-1e9, -1e9)

    pool = [r for r in pool if isinstance(r.get("over_odds"), (int, float))]
    if not pool:
        return None
    pool.sort(key=score, reverse=True)
    return pool[0]

# -------------------- PROJECTION CORE (POINTS) --------------------
def compute_projection_and_prob(games_all, line, w_base=0.45, w_l10=0.35, w_l3=0.10, w_line=0.10):
    base_slice = _slice_last(games_all, BASELINE_GAMES)
    l10_slice = _slice_last(games_all, LOOKBACK_GAMES)
    l3_slice = _slice_last(games_all, SHORT_GAMES)

    base_avg, _, base_std = avg_stat_min_std(base_slice)
    l10_avg, l10_min, l10_std = avg_stat_min_std(l10_slice)
    l3_avg, _, _ = avg_stat_min_std(l3_slice)

    sigma = max(STD_FLOOR, (l10_std if l10_std > 0 else base_std if base_std > 0 else STD_FLOOR))
    proj = (w_base * base_avg) + (w_l10 * l10_avg) + (w_l3 * l3_avg) + (w_line * float(line))

    edge = proj - float(line)
    z = (proj - float(line)) / max(sigma, 1e-6)
    prob_over = _norm_cdf(z)
    return proj, edge, prob_over, (base_avg, l10_avg, l3_avg, l10_min, sigma)

# -------------------- THREES MODEL (BETA-BINOMIAL) --------------------
def _log_beta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

def _log_choose(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

def beta_binom_tail_prob(n, a, b, k_min):
    """
    P(X >= k_min) where X ~ BetaBinomial(n, a, b)
    """
    if n <= 0:
        return 0.0
    k_min = int(k_min)
    if k_min <= 0:
        return 1.0
    if k_min > n:
        return 0.0

    lb_ab = _log_beta(a, b)
    s = 0.0
    for k in range(k_min, n + 1):
        lp = _log_choose(n, k) + _log_beta(k + a, n - k + b) - lb_ab
        s += math.exp(lp)
    return max(0.0, min(1.0, s))

def threes_prob_over_beta_binom(games_all, line):
    """
    Uses last THREES_ATTEMPT_LOOKBACK games.
    games tuple: (date, fg3m, mins, fg3a, team_id)
    """
    sl = _slice_last(games_all, THREES_ATTEMPT_LOOKBACK)
    makes = []
    atts = []
    mins = []
    for _, m, mn, a, _tid in sl:
        if a is None:
            continue
        try:
            a = float(a)
            m = float(m)
        except Exception:
            continue
        if a < 0:
            continue
        makes.append(m)
        atts.append(a)
        mins.append(mn)

    if len(atts) < 5:
        return None

    avg_att = sum(atts) / len(atts)
    if avg_att < THREES_MIN_AVG_ATTEMPTS:
        return None

    # Empirical mean/var of make% (smoothed)
    p_list = []
    for m, a in zip(makes, atts):
        if a <= 0:
            continue
        p_list.append(m / a)

    if len(p_list) < 5:
        return None

    mu = sum(p_list) / len(p_list)
    var = sum((p - mu) ** 2 for p in p_list) / max(1, len(p_list))

    # Convert mean/var -> beta(a,b) with a modest prior strength
    # If var is tiny, fall back to prior-strength
    prior_strength = 20.0
    if var <= 1e-6 or mu <= 1e-6 or mu >= 1 - 1e-6:
        a = max(1.0, mu * prior_strength)
        b = max(1.0, (1 - mu) * prior_strength)
    else:
        # k = mu*(1-mu)/var - 1
        k = (mu * (1 - mu) / max(var, 1e-6)) - 1.0
        if k <= 0:
            k = prior_strength
        a = max(1.0, mu * k)
        b = max(1.0, (1 - mu) * k)

    n_att = max(1, int(round(avg_att)))
    k_min = int(math.floor(line + 1e-9)) + 1  # over 2.5 => need 3
    p_over = beta_binom_tail_prob(n_att, a, b, k_min)

    proj = n_att * (a / (a + b))
    edge = proj - float(line)
    return {
        "p_over": float(p_over),
        "proj": float(proj),
        "edge": float(edge),
        "avg_att": float(avg_att),
        "n_att": int(n_att),
        "a": float(a),
        "b": float(b),
    }

# -------------------- STEAM DETECTION --------------------
def steam_score(prev, cur):
    try:
        prev_line = float(prev.get("line"))
        cur_line = float(cur.get("line"))
        prev_over = float(prev.get("over_odds"))
        cur_over = float(cur.get("over_odds"))
    except Exception:
        return 0.0

    line_move = (prev_line - cur_line)  # + means better for over
    odds_move = (cur_over - prev_over) / 50.0
    score = (line_move / 0.5) * 0.9 + odds_move * 0.6
    return float(score)

def market_key(prop_type: str, pid: int) -> str:
    return f"{prop_type}|{int(pid)}"

def remember_market(state, prop_type, pid, offer, cons_line, n_cons, now_ts):
    mk = market_key(prop_type, pid)
    state.setdefault("market", {})
    state["market"][mk] = {
        "ts": int(now_ts),
        "line": float(cons_line) if cons_line is not None else float(offer["line"]),
        "over_odds": float(offer["over_odds"]),
        "under_odds": float(offer["under_odds"]),
        "vendor": str(offer.get("vendor", "")),
        "n_cons": int(n_cons),
    }

def get_prev_market(state, prop_type, pid, now_ts):
    mk = market_key(prop_type, pid)
    prev = (state.get("market") or {}).get(mk)
    if not prev:
        return None
    age_min = (int(now_ts) - int(prev.get("ts", 0))) / 60.0
    if age_min > STEAM_MAX_AGE_MIN:
        return None
    return prev

# -------------------- ENGINES --------------------
def build_injury_edges(team_short, injured_name, injured_status, exclude_names_lower, now_et, prop_type, lines_map_for_prop, state, now_ts, news_boosts):
    if deadline_exceeded():
        return []

    season = _season_year(now_et)
    stat_key = STAT_KEY_BY_PROP.get(prop_type, "pts")

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

    injured_pid = bdl_find_player_id_on_team(team_short, injured_name)
    if not injured_pid:
        return []

    inj_games = bdl_last_n_games_stats([injured_pid], season, BASELINE_GAMES, stat_key).get(injured_pid, [])
    ip10, im10, _ = avg_stat_min_std(_slice_last(inj_games, LOOKBACK_GAMES))
    if len(inj_games) < 3:
        return []

    status = (injured_status or "").lower()
    status_mult = {"out": 1.0, "doubtful": 0.8, "questionable": 0.55}.get(status, 0.65)

    vac_stat = ip10 * status_mult
    vac_min = im10 * status_mult
    if not ((vac_min >= MIN_VAC_MIN) or (vac_stat >= MIN_VAC_STAT)):
        return []

    trigger_strength = min(100.0, (vac_min * 1.2 + vac_stat * 1.5))

    cand_ids = [pid for pid, _ in roster_tuples]
    stats_all = {}
    for chunk_ids in _chunk(cand_ids, STAT_BATCH_SIZE):
        if deadline_exceeded():
            break
        stats_all.update(bdl_last_n_games_stats(chunk_ids, season, BASELINE_GAMES, stat_key))

    ideas = []
    for pid, nm in roster_tuples:
        if deadline_exceeded():
            break

        games = stats_all.get(pid, [])
        if len(games) < 8:
            continue

        v10, m10, _ = avg_stat_min_std(_slice_last(games, LOOKBACK_GAMES))
        if m10 < MIN_L10_MIN:
            continue

        rows = (lines_map_for_prop or {}).get(pid, [])
        cons, n_cons, vendors_set, sharp_cnt = consensus_line(rows)

        if cons is None or n_cons < MIN_VENDORS_FOR_CONSENSUS:
            continue
        if MIN_SHARP_VENDORS > 0 and sharp_cnt < MIN_SHARP_VENDORS:
            continue

        offer = best_offer_near_consensus(rows, cons)
        if not offer:
            continue

        line = float(cons)
        if (v10 - line) > LINE_MIN_GAP:
            continue

        min_s, min_l, rate_s, rate_l = _role_trend(games)
        min_delta = min_s - min_l
        rate_delta = rate_s - rate_l

        absorption = 0.0
        if m10 >= 28:
            absorption += 0.30
        if m10 >= 34:
            absorption += 0.10
        if min_delta >= 2.0:
            absorption += 0.15
        if rate_delta > 0.05:
            absorption += 0.10
        absorption = min(0.65, absorption)

        injury_boost_stat = min(BOOST_CAP_STAT, vac_stat * absorption * 0.65)
        injury_boost_min = min(BOOST_CAP_MIN, vac_min * absorption * 0.25)

        # ---------- model core ----------
        if prop_type in ("threes", "three_pointers_made") and THREES_USE_BETA_BINOM:
            bb = threes_prob_over_beta_binom(games, line)
            if not bb:
                continue
            proj = bb["proj"]
            edge = bb["edge"]
            prob_over = bb["p_over"]
            sigma = None
            base_avg = l10_avg = l3_avg = l10_min = None
        else:
            proj, edge, prob_over, aux = compute_projection_and_prob(games_all=games, line=line)
            base_avg, l10_avg, l3_avg, l10_min, sigma = aux

            # apply injury boost after base projection (points model)
            rate = l10_avg / max(l10_min, 1e-6)
            proj = proj + injury_boost_stat + (injury_boost_min * rate * BOOST_CAP_RATE)
            edge = proj - line
            z = (proj - line) / max(sigma, 1e-6)
            prob_over = _norm_cdf(z)

        # ---------- news adjustments ----------
        nb = news_boosts.get(_clean_name(nm))
        news_tags = []
        if nb:
            if nb.get("start_boost"):
                # small bump (won't explode your model)
                proj += 0.6 if prop_type == "points" else 0.15
                edge = proj - line
                # nudge prob a bit (small + bounded)
                prob_over = min(0.99, prob_over + 0.02)
                news_tags.append("news:start")
            if nb.get("limit_flag"):
                proj *= 0.92
                edge = proj - line
                prob_over = max(0.01, prob_over - 0.03)
                news_tags.append("news:limit")

        if edge < MIN_EDGE or prob_over < MIN_PROB:
            continue

        p_over = american_to_prob(float(offer["over_odds"]))
        p_under = american_to_prob(float(offer["under_odds"]))
        p_market = p_over / max(p_over + p_under, 1e-9)

        value_edge = prob_over - p_market
        if value_edge < VALUE_EDGE_MIN:
            continue

        ev = ev_per_dollar(prob_over, float(offer["over_odds"]))
        if ev < EV_MIN:
            continue

        steam = 0.0
        if ENABLE_STEAM:
            prev = get_prev_market(state, prop_type, pid, now_ts)
            if prev:
                cur = {"line": line, "over_odds": offer["over_odds"], "under_odds": offer["under_odds"], "ts": now_ts}
                steam = steam_score(prev, cur)

        # for caps
        team_id = PLAYER_TEAM_CACHE.get(pid)
        gid = int(offer.get("gid") or offer.get("game_id") or offer.get("gid") or 0) or int(offer.get("gid", 0) or 0)
        try:
            gid = int(offer.get("gid", offer.get("game_id", offer.get("gid", 0))) or offer.get("gid") or offer.get("game_id") or offer.get("gid") or 0)
        except Exception:
            gid = int(offer.get("gid") or offer.get("game_id") or 0)

        # offer row already has gid, but make sure:
        gid = int(offer.get("gid", offer.get("game_id", 0)) or 0)

        why_core = (
            f"TriggerStrength {trigger_strength:.0f} | Absorb {absorption:.2f}. "
            f"{injured_name} {injured_status.upper()} vacates ~{vac_stat:.1f} {prop_type.title()} / {vac_min:.1f} min. "
        )

        if prop_type in ("threes", "three_pointers_made") and THREES_USE_BETA_BINOM:
            why_model = (
                f"{nm} beta-binomial(3PA) model. "
                f"Proj {proj:.2f} vs CONS {line:.1f} (n={n_cons}, sharp={sharp_cnt}) | "
                f"offer {offer['vendor']} {offer['line']:.1f} ({int(offer['over_odds']):+d}). "
            )
        else:
            why_model = (
                f"{nm} base(L{BASELINE_GAMES}) {base_avg:.1f}, L10 {l10_avg:.1f}, L3 {l3_avg:.1f} "
                f"(mins L10 {l10_min:.1f}). Role Δmin={min_delta:+.1f}, Δrate={rate_delta:+.2f}. "
                f"Proj {proj:.1f} vs CONS {line:.1f} (n={n_cons}, sharp={sharp_cnt}) | "
                f"offer {offer['vendor']} {offer['line']:.1f} ({int(offer['over_odds']):+d}) "
            )

        why_tail = (
            f"| edge +{edge:.1f} | P≈{prob_over*100:.0f}% (mkt≈{p_market*100:.0f}%, val_edge≈{value_edge:+.2f}) "
            f"| EV≈{ev:+.2f}/$1 | steam={steam:.1f}"
        )

        if news_tags:
            why_tail += f" | {' '.join(news_tags)}"

        ideas.append({
            "section": "injury",
            "prop_type": prop_type,
            "player_name": nm,
            "player_id": int(pid),
            "cons_line": float(line),
            "line": float(offer["line"]),
            "proj": float(proj),
            "edge": float(edge),
            "prob_over": float(prob_over),
            "market_prob": float(p_market),
            "value_edge": float(value_edge),
            "ev": float(ev),
            "vendor": str(offer["vendor"]),
            "over_odds": float(offer["over_odds"]),
            "under_odds": float(offer["under_odds"]),
            "n_cons": int(n_cons),
            "sharp_cnt": int(sharp_cnt),
            "steam": float(steam),
            "trigger_strength": float(trigger_strength),
            "trigger": f"{injured_name} ({team_short}) {injured_status}",
            "why": (why_core + why_model + why_tail + "."),
            "team_id": int(team_id) if team_id is not None else None,
            "gid": int(gid) if gid else None,
        })

        remember_market(state, prop_type, pid, offer, line, n_cons, now_ts)

    ideas.sort(key=lambda x: (x["trigger_strength"], x["ev"], x["value_edge"], x["edge"], x["prob_over"]), reverse=True)
    return ideas

def slate_scan_edges(now_et, prop_type, lines_map_for_prop, state, now_ts, news_boosts):
    if not ENABLE_SLATE_SCAN or deadline_exceeded():
        return []

    season = _season_year(now_et)
    stat_key = STAT_KEY_BY_PROP.get(prop_type, "pts")

    pids = list((lines_map_for_prop or {}).keys())
    if not pids:
        return []

    stats_all = {}
    for chunk_ids in _chunk(pids, STAT_BATCH_SIZE):
        if deadline_exceeded():
            break
        stats_all.update(bdl_last_n_games_stats(chunk_ids, season, BASELINE_GAMES, stat_key))

    ideas = []
    for pid in pids:
        if deadline_exceeded():
            break

        games = stats_all.get(int(pid), [])
        if len(games) < 8:
            continue

        rows = (lines_map_for_prop or {}).get(int(pid), [])
        cons, n_cons, vendors_set, sharp_cnt = consensus_line(rows)

        if cons is None or n_cons < MIN_VENDORS_FOR_CONSENSUS:
            continue
        if MIN_SHARP_VENDORS > 0 and sharp_cnt < MIN_SHARP_VENDORS:
            continue

        offer = best_offer_near_consensus(rows, cons)
        if not offer:
            continue

        v10, m10, _ = avg_stat_min_std(_slice_last(games, LOOKBACK_GAMES))
        if m10 < MIN_L10_MIN:
            continue

        line = float(cons)
        if (v10 - line) > LINE_MIN_GAP:
            continue

        min_s, min_l, rate_s, rate_l = _role_trend(games)
        min_delta = min_s - min_l
        rate_delta = rate_s - rate_l

        # ---------- model core ----------
        if prop_type in ("threes", "three_pointers_made") and THREES_USE_BETA_BINOM:
            bb = threes_prob_over_beta_binom(games, line)
            if not bb:
                continue
            proj = bb["proj"]
            edge = bb["edge"]
            prob_over = bb["p_over"]
            base_avg = l10_avg = l3_avg = l10_min = None
        else:
            proj, edge, prob_over, aux = compute_projection_and_prob(games_all=games, line=line)
            base_avg, l10_avg, l3_avg, l10_min, sigma = aux

        # ---------- news adjustments ----------
        name = PLAYER_NAME_CACHE.get(int(pid), f"Player {pid}")
        nb = news_boosts.get(_clean_name(name))
        news_tags = []
        if nb:
            if nb.get("start_boost"):
                proj += 0.6 if prop_type == "points" else 0.15
                edge = proj - line
                prob_over = min(0.99, prob_over + 0.02)
                news_tags.append("news:start")
            if nb.get("limit_flag"):
                proj *= 0.92
                edge = proj - line
                prob_over = max(0.01, prob_over - 0.03)
                news_tags.append("news:limit")

        if edge < MIN_EDGE or prob_over < MIN_PROB:
            continue

        p_over = american_to_prob(float(offer["over_odds"]))
        p_under = american_to_prob(float(offer["under_odds"]))
        p_market = p_over / max(p_over + p_under, 1e-9)

        value_edge = prob_over - p_market
        if value_edge < VALUE_EDGE_MIN:
            continue

        ev = ev_per_dollar(prob_over, float(offer["over_odds"]))
        if ev < EV_MIN:
            continue

        steam = 0.0
        if ENABLE_STEAM:
            prev = get_prev_market(state, prop_type, pid, now_ts)
            if prev:
                cur = {"line": line, "over_odds": offer["over_odds"], "under_odds": offer["under_odds"], "ts": now_ts}
                steam = steam_score(prev, cur)

        team_id = PLAYER_TEAM_CACHE.get(int(pid))
        gid = int(offer.get("gid", offer.get("game_id", 0)) or 0)

        if prop_type in ("threes", "three_pointers_made") and THREES_USE_BETA_BINOM:
            why_model = (
                f"SlateScan. beta-binomial(3PA) threes model. "
                f"Proj {proj:.2f} vs CONS {line:.1f} (n={n_cons}, sharp={sharp_cnt}) | "
                f"offer {offer['vendor']} {offer['line']:.1f} ({int(offer['over_odds']):+d}) "
            )
        else:
            why_model = (
                f"SlateScan. base(L{BASELINE_GAMES}) {base_avg:.1f}, L10 {l10_avg:.1f}, L3 {l3_avg:.1f} "
                f"(mins L10 {l10_min:.1f}). Role Δmin={min_delta:+.1f}, Δrate={rate_delta:+.2f}. "
                f"Proj {proj:.1f} vs CONS {line:.1f} (n={n_cons}, sharp={sharp_cnt}) | "
                f"offer {offer['vendor']} {offer['line']:.1f} ({int(offer['over_odds']):+d}) "
            )

        why_tail = (
            f"| edge +{edge:.1f} | P≈{prob_over*100:.0f}% (mkt≈{p_market*100:.0f}%, val_edge≈{value_edge:+.2f}) "
            f"| EV≈{ev:+.2f}/$1 | steam={steam:.1f}"
        )
        if news_tags:
            why_tail += f" | {' '.join(news_tags)}"

        ideas.append({
            "section": "slate",
            "prop_type": prop_type,
            "player_name": name,
            "player_id": int(pid),
            "cons_line": float(line),
            "line": float(offer["line"]),
            "proj": float(proj),
            "edge": float(edge),
            "prob_over": float(prob_over),
            "market_prob": float(p_market),
            "value_edge": float(value_edge),
            "ev": float(ev),
            "vendor": str(offer["vendor"]),
            "over_odds": float(offer["over_odds"]),
            "under_odds": float(offer["under_odds"]),
            "n_cons": int(n_cons),
            "sharp_cnt": int(sharp_cnt),
            "steam": float(steam),
            "trigger_strength": 0.0,
            "trigger": "No injury trigger (league-wide scan)",
            "why": (why_model + why_tail + "."),
            "team_id": int(team_id) if team_id is not None else None,
            "gid": int(gid) if gid else None,
        })

        remember_market(state, prop_type, pid, offer, line, n_cons, now_ts)

    ideas.sort(key=lambda x: (x["ev"], x["value_edge"], x["edge"], x["prob_over"]), reverse=True)
    return ideas

# -------------------- SYNTH LADDER (LONGSHOTS) --------------------
def synth_ladder_from_model(points_ideas, stats_cache_by_pid):
    if not ENABLE_SYNTH_LADDER or not points_ideas:
        return []

    out = []
    for idea in points_ideas:
        if deadline_exceeded():
            break
        if idea.get("prop_type") != "points":
            continue

        pid = int(idea["player_id"])
        games = stats_cache_by_pid.get(pid, [])
        if len(games) < 8:
            continue

        _, m10, _ = avg_stat_min_std(_slice_last(games, LOOKBACK_GAMES))
        if m10 < SYNTH_LADDER_MIN_L10_MIN:
            continue

        proj = float(idea["proj"])
        base_slice = _slice_last(games, BASELINE_GAMES)
        l10_slice = _slice_last(games, LOOKBACK_GAMES)
        _, _, base_std = avg_stat_min_std(base_slice)
        _, _, l10_std = avg_stat_min_std(l10_slice)
        sigma = max(STD_FLOOR, (l10_std if l10_std > 0 else base_std if base_std > 0 else STD_FLOOR))

        base_line = float(idea.get("cons_line", idea.get("line", 0.0)))
        base_rung = int(math.ceil(base_line))

        for step in SYNTH_LADDER_RUNG_STEPS:
            rung = int(base_rung + step)
            z = (proj - rung) / max(sigma, 1e-6)
            p = _norm_cdf(z)
            imp_odds = prob_to_american(p)

            if imp_odds < 0:
                continue
            if not (SYNTH_LADDER_MIN_ODDS <= imp_odds <= SYNTH_LADDER_MAX_ODDS):
                continue

            out.append({
                "section": "synth_ladder",
                "prop_type": "points_ladder",
                "player_name": idea["player_name"],
                "player_id": pid,
                "rung": rung,
                "model_prob": float(p),
                "model_implied_odds": float(imp_odds),
                "proj": float(proj),
                "sigma": float(sigma),
                "why": (
                    f"Model ladder idea around +{int(SYNTH_LADDER_MIN_ODDS)}. "
                    f"Proj≈{proj:.1f}, sigma≈{sigma:.1f}. "
                    f"P({rung}+ pts)≈{p*100:.1f}% → model-implied ≈ {int(imp_odds):+d}. "
                    f"Compare FanDuel ladder price to this."
                )
            })

    out.sort(key=lambda x: (x["model_prob"], -x["model_implied_odds"]), reverse=True)
    return out[:SYNTH_LADDER_TOPN]

# -------------------- COOLDOWN FILTER --------------------
def apply_cooldown(state, ideas, now_ts: int):
    sent = state.get("sent_bets", {}) or {}
    cooldown_sec = BET_COOLDOWN_MIN * 60

    kept = []
    for i in ideas:
        sec = i.get("section", "")
        if sec == "synth_ladder":
            key = f"synth_ladder|{int(i['player_id'])}|{int(i['rung'])}"
        else:
            key = f"{i['prop_type']}|{i['section']}|{int(i['player_id'])}|{float(i.get('cons_line', i.get('line', 0.0))):.1f}"

        prev = sent.get(key)
        if not prev:
            kept.append(i)
            continue

        last_ts = int(prev.get("ts", 0) or 0)
        last_edge = float(prev.get("edge", 0.0) or 0.0)

        if "edge" in i and (float(i.get("edge", 0.0)) - last_edge) >= EDGE_JUMP_TO_RESEND:
            kept.append(i)
            continue

        if (now_ts - last_ts) >= cooldown_sec:
            kept.append(i)

    return kept

def record_sent(state, ideas, now_ts: int):
    sent = state.get("sent_bets", {}) or {}
    for i in ideas:
        sec = i.get("section", "")
        if sec == "synth_ladder":
            key = f"synth_ladder|{int(i['player_id'])}|{int(i['rung'])}"
            sent[key] = {"ts": now_ts, "edge": 0.0}
        else:
            key = f"{i['prop_type']}|{i['section']}|{int(i['player_id'])}|{float(i.get('cons_line', i.get('line', 0.0))):.1f}"
            sent[key] = {"ts": now_ts, "edge": float(i.get("edge", 0.0))}
    state["sent_bets"] = sent

# -------------------- EXPOSURE CAPS (NEW) --------------------
def apply_exposure_caps(ideas):
    """
    Enforces:
      - MAX_PLAYS_PER_TEAM
      - MAX_PLAYS_PER_GAME
    Only applies if we have team_id/gid; otherwise it won't incorrectly drop plays.
    """
    if not ideas:
        return ideas

    team_ct = {}
    game_ct = {}
    out = []

    for i in ideas:
        tid = i.get("team_id")
        gid = i.get("gid")

        if gid is not None:
            gk = int(gid)
            if MAX_PLAYS_PER_GAME > 0 and game_ct.get(gk, 0) >= MAX_PLAYS_PER_GAME:
                continue

        if tid is not None:
            tk = int(tid)
            if MAX_PLAYS_PER_TEAM > 0 and team_ct.get(tk, 0) >= MAX_PLAYS_PER_TEAM:
                continue

        out.append(i)

        if gid is not None:
            game_ct[int(gid)] = game_ct.get(int(gid), 0) + 1
        if tid is not None:
            team_ct[int(tid)] = team_ct.get(int(tid), 0) + 1

    return out

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
        f"MIN_VENDORS_FOR_CONSENSENSUS={MIN_VENDORS_FOR_CONSENSUS} "
        f"MIN_SHARP_VENDORS={MIN_SHARP_VENDORS} ENABLE_STEAM={int(ENABLE_STEAM)} "
        f"MAX_PLAYS_PER_TEAM={MAX_PLAYS_PER_TEAM} MAX_PLAYS_PER_GAME={MAX_PLAYS_PER_GAME} "
        f"THREES_BETA_BINOM={int(THREES_USE_BETA_BINOM)} LINEUPEXPERTS={int(ENABLE_LINEUPEXPERTS and bool(LINEUPEXPERTS_API_KEY))}"
    )

    if TEST_MODE:
        send_one(f"✅ NBA betting agent test OK ({ts_et})")
        return

    state = load_state()
    old_players = state.get("players", {})

    # Build today props
    lines_map = build_today_props(now_et)

    # LineupExperts add-ons (safe; never hard-fail)
    le_inj = le_fetch_injuries_map()
    le_news = le_fetch_news_boosts()

    # Sportradar injuries
    new_players = {}
    triggers = []
    injury_ideas_all = []

    if ENABLE_INJURY_TRIGGERS and (not deadline_exceeded()):
        try:
            sr = fetch_sportradar_injuries()
            new_players = parse_injuries(sr)
        except Exception as e:
            print(f"[WARN] Sportradar injuries failed: {e}")
            new_players = {}

        # merge in lineup experts injuries as additional triggers (by name only)
        # NOTE: we do NOT try to manufacture player IDs; we just use it to widen trigger list safely.
        # We'll add "synthetic" keys like "LE:<name>" to old/new tracking to avoid spam.
        if le_inj:
            for name_key, info in le_inj.items():
                fake_pid = f"LE:{name_key}"
                new_players[fake_pid] = {
                    "name": name_key.title(),
                    "team": info.get("team", ""),
                    "status": info.get("status", ""),
                    "detail": info.get("detail", ""),
                }

        exclude_names_lower = {_clean_name(v.get("name", "")) for v in new_players.values() if v.get("name")}

        for pid, cur in new_players.items():
            if deadline_exceeded():
                break

            st = cur.get("status", "")
            if not status_in_scope(st):
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

            got_any = False
            for pt in PROP_TYPES:
                if deadline_exceeded():
                    break
                ideas = build_injury_edges(
                    team_short=team_short,
                    injured_name=injured_name,
                    injured_status=injured_status,
                    exclude_names_lower=exclude_names_lower | {_clean_name(injured_name)},
                    now_et=now_et,
                    prop_type=pt,
                    lines_map_for_prop=lines_map.get(pt, {}),
                    state=state,
                    now_ts=now_ts,
                    news_boosts=le_news
                )
                if ideas:
                    got_any = True
                    injury_ideas_all.extend(ideas)

            if got_any:
                triggers.append(f"{injured_name} ({team_short}) {injured_status}")

    # Slate scan
    slate_ideas_all = []
    if ENABLE_SLATE_SCAN and (not deadline_exceeded()):
        for pt in PROP_TYPES:
            if deadline_exceeded():
                break
            slate_ideas_all.extend(
                slate_scan_edges(now_et, pt, lines_map.get(pt, {}), state=state, now_ts=now_ts, news_boosts=le_news)
            )

    # Combine + dedupe best per player/market
    combined = injury_ideas_all + slate_ideas_all
    best = {}
    for i in combined:
        k = (i["prop_type"], int(i["player_id"]))
        score = (
            float(i.get("ev", 0.0)),
            float(i.get("value_edge", 0.0)),
            float(i.get("edge", 0.0)),
            float(i.get("prob_over", 0.0)),
        )
        if (k not in best) or (score > best[k][0]):
            best[k] = (score, i)

    combined = [v[1] for v in best.values()]
    combined = apply_cooldown(state, combined, now_ts)

    # Per market ordering + limits
    out_by_market = {}
    for pt in PROP_TYPES:
        inj = [x for x in combined if x["prop_type"] == pt and x["section"] == "injury"]
        slt = [x for x in combined if x["prop_type"] == pt and x["section"] == "slate"]

        inj.sort(key=lambda x: (x["trigger_strength"], x["ev"], x["value_edge"], x["edge"], x["prob_over"]), reverse=True)
        slt.sort(key=lambda x: (x["ev"], x["value_edge"], x["edge"], x["prob_over"]), reverse=True)

        picks = inj + slt
        if MIN_PER_MARKET > 0:
            picks = picks[:max(MIN_PER_MARKET, MAX_PER_MARKET)]
        picks = picks[:MAX_PER_MARKET]
        out_by_market[pt] = picks

    # Flatten + global cap
    final_out = []
    for pt in PROP_TYPES:
        final_out.extend(out_by_market.get(pt, []))
    final_out = final_out[:MAX_TOTAL_PLAYS * 3]  # build bigger list before exposure caps

    # Apply exposure caps (NEW)
    final_out = apply_exposure_caps(final_out)
    final_out = final_out[:MAX_TOTAL_PLAYS]

    # Plus odds bucket
    plus_bucket = []
    for x in final_out:
        try:
            if float(x.get("over_odds", -999)) >= PLUS_ODDS_MIN:
                plus_bucket.append(x)
        except Exception:
            pass
    plus_bucket.sort(key=lambda x: (x["ev"], x["value_edge"], x["prob_over"]), reverse=True)
    plus_bucket = plus_bucket[:PLUS_ODDS_TOPN]

    # Synth ladder from best points ideas
    synth_ladder = []
    if ENABLE_SYNTH_LADDER and ("points" in PROP_TYPES) and (not deadline_exceeded()):
        pids = list({int(x["player_id"]) for x in final_out if x.get("prop_type") == "points"})
        stats_cache = {}
        season = _season_year(now_et)
        for chunk_ids in _chunk(pids, STAT_BATCH_SIZE):
            if deadline_exceeded():
                break
            stats_cache.update(bdl_last_n_games_stats(chunk_ids, season, BASELINE_GAMES, "pts"))
        points_ideas = [x for x in final_out if x.get("prop_type") == "points"]
        synth_ladder = synth_ladder_from_model(points_ideas, stats_cache)
        synth_ladder = apply_cooldown(state, synth_ladder, now_ts)

    # Message
    if final_out or synth_ladder:
        msg = [f"💰 FanDuel Props ({ts_et})", ""]

        if triggers:
            msg.append("🚑 Injury-Triggered Plays:")
            msg.append("Triggers:")
            for t in triggers[:8]:
                msg.append(f"- {t}")
            if len(triggers) > 8:
                msg.append(f"- …and {len(triggers)-8} more")
            msg.append("")

        # Build market sections
        for pt in PROP_TYPES:
            picks = [x for x in final_out if x.get("prop_type") == pt]
            if not picks:
                continue

            label = "Points" if pt == "points" else ("3PT Made" if pt in ("threes", "three_pointers_made") else pt)
            msg.append(f"🏷️ {label}")
            msg.append("")

            inj = [x for x in picks if x["section"] == "injury"]
            slt = [x for x in picks if x["section"] == "slate"]

            if inj:
                msg.append("🚑 Injury-Triggered Plays:")
                msg.append("")
                for i in inj:
                    fire = " 🔥" if i.get("ev", 0) >= 0.25 else ""
                    msg.append(
                        f"• {i['player_name']} OVER {i['cons_line']:.1f}  "
                        f"(edge +{i['edge']:.1f}, P≈{i['prob_over']*100:.0f}%, EV≈{i['ev']:+.2f}/$1){fire}"
                    )
                    msg.append(f"  Trigger: {i['trigger']}")
                    msg.append(f"  Why: {i['why']}")
                    msg.append("")
            if slt:
                msg.append("🌎 League-Wide Slate Scan (no injury required):")
                msg.append("")
                for i in slt:
                    fire = " 🔥" if i.get("ev", 0) >= 0.25 else ""
                    msg.append(
                        f"• {i['player_name']} OVER {i['cons_line']:.1f}  "
                        f"(edge +{i['edge']:.1f}, P≈{i['prob_over']*100:.0f}%, EV≈{i['ev']:+.2f}/$1){fire}"
                    )
                    msg.append(f"  Why: {i['why']}")
                    msg.append("")
            msg.append("")

        if plus_bucket:
            msg.append("💎 Plus-odds value bucket:")
            msg.append("")
            for i in plus_bucket:
                msg.append(
                    f"• {i['player_name']} OVER {i['cons_line']:.1f}  "
                    f"(offer {i['vendor']} {int(i['over_odds']):+d}, P≈{i['prob_over']*100:.0f}%, EV≈{i['ev']:+.2f}/$1)"
                )
            msg.append("")

        if synth_ladder:
            msg.append(f"🎯 Points Ladder (model-implied around +{int(SYNTH_LADDER_MIN_ODDS)}):")
            msg.append("")
            for i in synth_ladder:
                msg.append(
                    f"• {i['player_name']} {int(i['rung'])}+ pts  "
                    f"(model P≈{i['model_prob']*100:.1f}%, model-implied {int(i['model_implied_odds']):+d})"
                )
                msg.append(f"  Why: {i['why']}")
                msg.append("")

        send_chunked("\n".join(msg).strip())

        record_sent(state, final_out, now_ts)
        record_sent(state, synth_ladder, now_ts)
    else:
        print("[INFO] No plays cleared thresholds this run.")

    state["players"] = new_players
    save_state(state)

if __name__ == "__main__":
    run()
