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

# -------------------- CONFIG --------------------
TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
MAX_BODY_CHARS = int(os.environ.get("MAX_BODY_CHARS", "1500"))

BOOK_VENDOR_RAW = os.environ.get("BOOK_VENDORS", os.environ.get("BOOK_VENDOR", "fanduel")).strip().lower()
BOOK_VENDORS = [v.strip() for v in BOOK_VENDOR_RAW.split(",") if v.strip()]

PROP_TYPES_RAW = os.environ.get("PROP_TYPES", "points").strip().lower()
PROP_TYPES = [p.strip() for p in PROP_TYPES_RAW.split(",") if p.strip()]

ENABLE_INJURY_TRIGGERS = os.environ.get("ENABLE_INJURY_TRIGGERS", "1") == "1"
ENABLE_SLATE_SCAN = os.environ.get("ENABLE_SLATE_SCAN", "1") == "1"

# Consensus + Steam + EV + Plus odds
MIN_VENDORS_FOR_CONSENSUS = int(os.environ.get("MIN_VENDORS_FOR_CONSENSUS", "1"))
MIN_SHARP_VENDORS = int(os.environ.get("MIN_SHARP_VENDORS", "0"))
SHARP_VENDORS_RAW = os.environ.get("SHARP_VENDORS", "draftkings,caesars,betmgm,bet365,fanatics")
SHARP_VENDORS = {x.strip().lower() for x in SHARP_VENDORS_RAW.split(",") if x.strip()}

ENABLE_STEAM = os.environ.get("ENABLE_STEAM", "0") == "1"
STEAM_MIN_SCORE = float(os.environ.get("STEAM_MIN_SCORE", "1.0"))
STEAM_MAX_AGE_MIN = int(os.environ.get("STEAM_MAX_AGE_MIN", "240"))

# Exposure caps (quick win)
MAX_PLAYS_PER_TEAM = int(os.environ.get("MAX_PLAYS_PER_TEAM", "2"))
MAX_PLAYS_PER_GAME = int(os.environ.get("MAX_PLAYS_PER_GAME", "2"))

# Output sizing
MIN_PER_MARKET = int(os.environ.get("MIN_PER_MARKET", "0"))
MAX_PER_MARKET = int(os.environ.get("MAX_PER_MARKET", "6"))
MAX_TOTAL_PLAYS = int(os.environ.get("MAX_TOTAL_PLAYS", "10"))

# Windows
BASELINE_GAMES = int(os.environ.get("BASELINE_GAMES", "30"))
LOOKBACK_GAMES = int(os.environ.get("LOOKBACK_GAMES", "10"))
SHORT_GAMES = int(os.environ.get("SHORT_GAMES", "3"))

# Model thresholds
MIN_EDGE = float(os.environ.get("MIN_EDGE", "2.5"))
MIN_PROB = float(os.environ.get("MIN_PROB", "0.62"))
STD_FLOOR = float(os.environ.get("STD_FLOOR", "5.0"))

# EV filter
EV_MIN = float(os.environ.get("EV_MIN", "0.00"))
VALUE_EDGE_MIN = float(os.environ.get("VALUE_EDGE_MIN", "0.00"))  # model_prob - vigfree_market_prob

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
STAT_BATCH_SIZE = int(os.environ.get("STAT_BATCH_SIZE", "80"))  # keep smaller to avoid timeouts

DEBUG_PROP_SAMPLE_TYPES = os.environ.get("DEBUG_PROP_SAMPLE_TYPES", "0").strip().lower()

# Plus-odds bucket
PLUS_ODDS_MIN = float(os.environ.get("PLUS_ODDS_MIN", "100"))
PLUS_ODDS_TOPN = int(os.environ.get("PLUS_ODDS_TOPN", "3"))

# Synth ladder (model-implied only)
ENABLE_SYNTH_LADDER = os.environ.get("ENABLE_SYNTH_LADDER", "1") == "1"
SYNTH_LADDER_MIN_ODDS = float(os.environ.get("SYNTH_LADDER_MIN_ODDS", "250"))
SYNTH_LADDER_MAX_ODDS = float(os.environ.get("SYNTH_LADDER_MAX_ODDS", "500"))
SYNTH_LADDER_RUNG_STEPS_RAW = os.environ.get("SYNTH_LADDER_RUNG_STEPS", "3,4,5,6")
SYNTH_LADDER_RUNG_STEPS = [int(x.strip()) for x in SYNTH_LADDER_RUNG_STEPS_RAW.split(",") if x.strip().isdigit()]
SYNTH_LADDER_TOPN = int(os.environ.get("SYNTH_LADDER_TOPN", "6"))
SYNTH_LADDER_MIN_L10_MIN = float(os.environ.get("SYNTH_LADDER_MIN_L10_MIN", "18"))

# Threes model upgrade
THREES_BETA_BINOM = os.environ.get("THREES_BETA_BINOM", "1") == "1"
THREES_MIN_L10_3PA = float(os.environ.get("THREES_MIN_L10_3PA", "3.0"))
THREES_ALPHA0 = float(os.environ.get("THREES_ALPHA0", "1.0"))
THREES_BETA0 = float(os.environ.get("THREES_BETA0", "1.0"))

# LineupExperts
ENABLE_LINEUPEXPERTS = os.environ.get("ENABLE_LINEUPEXPERTS", os.environ.get("LINEUPEXPERTS", "0")) == "1"
LINEUPEXPERTS_API_KEY = os.environ.get("LINEUPEXPERTS_API_KEY", "").strip()
LINEUPEXPERTS_PREMIUM = os.environ.get("LINEUPEXPERTS_PREMIUM", "1") == "1"
LINEUPEXPERTS_BASE = os.environ.get("LINEUPEXPERTS_BASE", "https://api.lineupexperts.com/v1").rstrip("/")

NEWS_MAX_AGE_HOURS = float(os.environ.get("NEWS_MAX_AGE_HOURS", "36"))
NEWS_PENALTY = float(os.environ.get("NEWS_PENALTY", "0.20"))  # subtract from EV-score if bad news
NEWS_BOOST = float(os.environ.get("NEWS_BOOST", "0.08"))      # add if strong positive news
NEWS_REQUIRE_OK = os.environ.get("NEWS_REQUIRE_OK", "0") == "1"  # if 1, drop plays with strong negative news

# -------------------- RUNTIME DEADLINE --------------------
RUN_START = time.time()

def deadline_exceeded() -> bool:
    return (time.time() - RUN_START) > RUN_MAX_SECONDS

def check_deadline(where=""):
    if deadline_exceeded():
        raise RuntimeError(f"[DEADLINE] Script exceeded {RUN_MAX_SECONDS}s at {where or 'unknown'}")

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
        return round((dec - 1.0) * 100.0, 0)   # +odds
    return round(-100.0 / (dec - 1.0), 0)      # -odds

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
        return {"players": {}, "sent_bets": {}, "market": {}, "le_players_cache": {}}
    try:
        with open(STATE_FILE, "r") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return {"players": {}, "sent_bets": {}, "market": {}, "le_players_cache": {}}
        raw.setdefault("players", {})
        raw.setdefault("sent_bets", {})
        raw.setdefault("market", {})
        raw.setdefault("le_players_cache", {})
        return raw
    except Exception:
        return {"players": {}, "sent_bets": {}, "market": {}, "le_players_cache": {}}

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

def _iso_to_dt(s: str):
    try:
        # often comes as "2026-03-05T04:23:43.220Z"
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

# -------------------- BALLDONTLIE --------------------
BDL_HEADERS = {"Authorization": BALLDONTLIE_API_KEY}
BDL_PREFIXES = ["/nba", ""]

BDL_MAX_RETRIES = int(os.environ.get("BDL_MAX_RETRIES", "5"))
BDL_RETRY_BASE_SEC = float(os.environ.get("BDL_RETRY_BASE_SEC", "1.5"))
BDL_PER_PAGE = int(os.environ.get("BDL_PER_PAGE", "100"))
BDL_MAX_PAGES = int(os.environ.get("BDL_MAX_PAGES", "10"))

TEAM_CACHE = None
PLAYER_NAME_CACHE = {}     # pid -> "First Last"
PLAYER_TEAM_CACHE = {}     # pid -> team_name (string)
PROPS_CACHE = {}           # (gid, vendor, prop_type) -> list[rows]
GAMES_TODAY_CACHE = None   # {gid: {"home":..., "visitor":..., "home_name":..., "visitor_name":...}}

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

def bdl_games_today(now_et: datetime):
    global GAMES_TODAY_CACHE
    if GAMES_TODAY_CACHE is not None:
        return GAMES_TODAY_CACHE
    today = now_et.strftime("%Y-%m-%d")
    resp = _bdl_get("/v1/games", params={"dates[]": [today], "per_page": 100})
    out = {}
    for g in (resp.get("data") or []):
        try:
            gid = int(g["id"])
        except Exception:
            continue
        home = g.get("home_team") or {}
        vis = g.get("visitor_team") or {}
        out[gid] = {
            "home_name": (home.get("name") or "").strip(),
            "visitor_name": (vis.get("name") or "").strip(),
            "home_id": home.get("id"),
            "visitor_id": vis.get("id"),
        }
    GAMES_TODAY_CACHE = out
    return out

def bdl_games_today_ids(now_et: datetime):
    return list(bdl_games_today(now_et).keys())

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

def bdl_player_by_id(pid: int):
    # Used for exposure team cap (only for shortlisted players)
    if pid in PLAYER_TEAM_CACHE and pid in PLAYER_NAME_CACHE:
        return {"team_name": PLAYER_TEAM_CACHE[pid], "full_name": PLAYER_NAME_CACHE[pid]}
    try:
        resp = _bdl_get(f"/v1/players/{int(pid)}", params={})
        p = resp.get("data") or {}
        team = p.get("team") or {}
        fn = (p.get("first_name") or "").strip()
        ln = (p.get("last_name") or "").strip()
        nm = f"{fn} {ln}".strip() or f"Player {pid}"
        tname = (team.get("name") or "").strip()
        if nm:
            PLAYER_NAME_CACHE[pid] = nm
        if tname:
            PLAYER_TEAM_CACHE[pid] = tname
        return {"team_name": tname, "full_name": nm}
    except Exception:
        return {"team_name": "", "full_name": PLAYER_NAME_CACHE.get(pid, f"Player {pid}")}

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

            game = row.get("game") or {}
            date = game.get("date")
            val = float(row.get(stat_key, 0) or 0)
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

def bdl_last_n_games_threes(player_ids, season: int, n: int):
    """
    For threes beta-binomial: capture (date, fg3m, fg3a, min)
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

            game = row.get("game") or {}
            date = game.get("date")
            makes = float(row.get("fg3m", 0) or 0)
            atts = float(row.get("fg3a", 0) or 0)
            mins = _parse_minutes(row.get("min"))
            if date:
                out[pid].append((date, makes, atts, mins))

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
                    "vendor": (pp.get("vendor") or (v or "no_vendor")),
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
    Returns: (median_line, n_vendors_used, n_sharp_used)
    Uses one line per vendor; checks sharp-vendor presence.
    """
    if not rows:
        return None, 0, 0

    by_vendor = {}
    sharp_used = 0
    for r in rows:
        v = str(r.get("vendor") or "").strip().lower()
        if not v:
            continue
        try:
            line = float(r["line"])
        except Exception:
            continue
        by_vendor[v] = _round_to_half(line)

    if not by_vendor:
        return None, 0, 0

    for v in by_vendor.keys():
        if v in SHARP_VENDORS:
            sharp_used += 1

    lines = sorted(by_vendor.values())
    n = len(lines)
    mid = n // 2
    if n % 2 == 1:
        return float(lines[mid]), n, sharp_used
    return float(0.5 * (lines[mid - 1] + lines[mid])), n, sharp_used

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

    def score(r):
        # prefer better over odds, and closer to consensus
        try:
            return (float(r["over_odds"]), -abs(float(r["line"]) - cons_line))
        except Exception:
            return (-1e9, -1e9)

    pool = [r for r in pool if isinstance(r.get("over_odds"), (int, float))]
    if not pool:
        return None
    pool.sort(key=score, reverse=True)
    return pool[0]

# -------------------- PROJECTION CORE --------------------
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

# -------------------- THREES: BETA-BINOMIAL --------------------
def _log_beta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

def beta_binom_pmf(k, n, a, b):
    # C(n,k) * B(k+a, n-k+b) / B(a,b)
    return math.exp(
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        + _log_beta(k + a, n - k + b) - _log_beta(a, b)
    )

def beta_binom_tail_prob_over(line, n, a, b):
    # P(X > line) where line may be .5 etc
    k0 = int(math.floor(float(line) + 1e-9)) + 1
    if k0 <= 0:
        return 1.0
    if k0 > n:
        return 0.0
    s = 0.0
    for k in range(k0, n + 1):
        s += beta_binom_pmf(k, n, a, b)
    return float(max(0.0, min(1.0, s)))

def threes_prob_over_beta_binom(threes_games, line):
    """
    threes_games: list of (date, fg3m, fg3a, min)
    returns: (proj_makes, prob_over, sigma_like, att_mean_l10)
    """
    if not threes_games:
        return 0.0, 0.0, STD_FLOOR, 0.0

    l10 = threes_games[-min(len(threes_games), LOOKBACK_GAMES):]
    makes = [float(x[1]) for x in l10]
    atts = [float(x[2]) for x in l10]
    mins = [float(x[3]) for x in l10]

    att_mean = sum(atts) / max(1, len(atts))
    if att_mean <= 0:
        return 0.0, 0.0, STD_FLOOR, att_mean

    # posterior for make%
    sum_m = sum(makes)
    sum_a = sum(atts)
    a = THREES_ALPHA0 + max(0.0, sum_m)
    b = THREES_BETA0 + max(0.0, (sum_a - sum_m))

    n = int(max(1, round(att_mean)))
    p_over = beta_binom_tail_prob_over(line=line, n=n, a=a, b=b)

    # projection: expected makes = E[p] * n, where E[p]=a/(a+b)
    p_mean = a / (a + b)
    proj = p_mean * n

    # "sigma-like" just for messaging
    sigma_like = max(1.0, math.sqrt(n * p_mean * (1 - p_mean)))
    return float(proj), float(p_over), float(sigma_like), float(att_mean)

# -------------------- STEAM DETECTION --------------------
def steam_score(prev, cur):
    try:
        prev_line = float(prev.get("line"))
        cur_line = float(cur.get("line"))
        prev_over = float(prev.get("over_odds"))
        cur_over = float(cur.get("over_odds"))
    except Exception:
        return 0.0

    line_move = (prev_line - cur_line)  # + means good for over
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

# -------------------- LINEUPEXPERTS (NEWS) --------------------
def le_get(path: str, params: dict, timeout=15):
    if not ENABLE_LINEUPEXPERTS or not LINEUPEXPERTS_API_KEY:
        return None
    url = f"{LINEUPEXPERTS_BASE}/{path.lstrip('/')}"
    q = dict(params or {})
    q["key"] = LINEUPEXPERTS_API_KEY
    try:
        r = requests.get(url, params=q, timeout=timeout)
        if r.status_code != 200:
            print(f"[WARN] LineupExperts HTTP {r.status_code} for /{path}: {r.text[:220]}")
            return None
        return r.json()
    except Exception as e:
        print(f"[WARN] LineupExperts error for /{path}: {type(e).__name__}: {e}")
        return None

def le_players_map_cached(state, now_ts: int):
    """
    Builds a mapping clean_name -> lineupExpertsPlayerId.
    Cached in state to avoid fetching every run. Refresh every 7 days.
    """
    cache = state.get("le_players_cache") or {}
    last = int(cache.get("ts", 0) or 0)
    if cache.get("map") and (now_ts - last) < 7 * 24 * 3600:
        return cache["map"]

    data = le_get("nba-Players", params={}, timeout=25)
    mp = {}
    if isinstance(data, list):
        for p in data:
            fn = str(p.get("FirstName", "") or "").strip()
            ln = str(p.get("LastName", "") or "").strip()
            pid = p.get("PlayerID")
            if not pid:
                continue
            nm = _clean_name(f"{fn} {ln}".strip())
            if nm:
                mp[nm] = int(pid)

    state["le_players_cache"] = {"ts": int(now_ts), "map": mp}
    return mp

def le_news_by_players(le_player_ids):
    """
    Returns dict: le_pid -> best_story (dict)
    """
    if not le_player_ids:
        return {}
    # API expects comma-separated "players"
    path = "nba-NewsByPlayerPremium" if LINEUPEXPERTS_PREMIUM else "nba-NewsByPlayer"
    params = {"players": ",".join(str(int(x)) for x in le_player_ids[:50])}
    data = le_get(path, params=params, timeout=20)
    out = {}
    if not isinstance(data, list):
        return out

    # pick most recent per player
    for item in data:
        pid = item.get("PlayerID")
        if not pid:
            continue
        pid = int(pid)
        # try to parse Updated or Created
        dt = _iso_to_dt(str(item.get("Updated", "") or item.get("Created", "") or ""))
        if not dt:
            # if no dt, still keep first
            dt = datetime.now().astimezone()
        prev = out.get(pid)
        if (prev is None) or (dt > prev["_dt"]):
            it = dict(item)
            it["_dt"] = dt
            out[pid] = it
    return out

def score_news(text: str):
    """
    crude but useful. returns (score, label)
    negative => minutes risk / bench / rest
    positive => starting / increased role
    """
    t = (text or "").lower()

    neg = [
        "minutes restriction", "minute restriction", "will be limited", "limited workload",
        "rest", "out", "doubtful", "won't play", "will not play", "inactive",
        "return to bench", "coming off the bench", "bench role",
        "questionable", "game-time decision", "gtd",
    ]
    pos = [
        "will start", "expected to start", "in the starting lineup", "named starter",
        "increased role", "bigger role", "more minutes", "starting at",
    ]

    s = 0.0
    for w in neg:
        if w in t:
            s -= 1.0
    for w in pos:
        if w in t:
            s += 1.0

    if s <= -2:
        return -1.0, "bad"
    if s < 0:
        return -0.5, "caution"
    if s >= 2:
        return +1.0, "good"
    if s > 0:
        return +0.5, "good"
    return 0.0, "neutral"

# -------------------- ENGINES --------------------
def build_injury_edges(team_short, injured_name, injured_status, exclude_names_lower, now_et, prop_type, lines_map_for_prop, state, now_ts):
    if deadline_exceeded():
        return []

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

    injured_pid = bdl_find_player_id_on_team(team_short, injured_name)
    if not injured_pid:
        return []

    # injury "vacancy" stat uses points even if prop_type=threes (stable signal)
    inj_games_pts = bdl_last_n_games_stats([injured_pid], season, BASELINE_GAMES, "pts").get(injured_pid, [])
    ip10, im10, _ = avg_stat_min_std(_slice_last(inj_games_pts, LOOKBACK_GAMES))
    if len(inj_games_pts) < 3:
        return []

    status = (injured_status or "").lower()
    status_mult = {"out": 1.0, "doubtful": 0.8, "questionable": 0.55}.get(status, 0.65)

    vac_stat = ip10 * status_mult
    vac_min = im10 * status_mult
    if not ((vac_min >= MIN_VAC_MIN) or (vac_stat >= MIN_VAC_STAT)):
        return []

    trigger_strength = min(100.0, (vac_min * 1.2 + vac_stat * 1.5))

    cand_ids = [pid for pid, _ in roster_tuples]

    ideas = []
    # stats batch pulling differs for threes if beta-binomial enabled
    if prop_type == "threes" and THREES_BETA_BINOM:
        stats_all = {}
        for chunk_ids in _chunk(cand_ids, STAT_BATCH_SIZE):
            if deadline_exceeded():
                break
            stats_all.update(bdl_last_n_games_threes(chunk_ids, season, BASELINE_GAMES))
        for pid, nm in roster_tuples:
            if deadline_exceeded():
                break
            games = stats_all.get(pid, [])
            if len(games) < 8:
                continue

            # l10 guardrail attempts
            l10 = games[-min(len(games), LOOKBACK_GAMES):]
            l10_3pa = sum(float(x[2]) for x in l10) / max(1, len(l10))
            l10_min = sum(float(x[3]) for x in l10) / max(1, len(l10))
            if l10_min < MIN_L10_MIN or l10_3pa < THREES_MIN_L10_3PA:
                continue

            rows = (lines_map_for_prop or {}).get(pid, [])
            cons, n_cons, sharp_used = consensus_line(rows)
            if cons is None or n_cons < MIN_VENDORS_FOR_CONSENSUS:
                continue
            if MIN_SHARP_VENDORS > 0 and sharp_used < MIN_SHARP_VENDORS:
                continue

            offer = best_offer_near_consensus(rows, cons)
            if not offer:
                continue

            line = float(cons)
            # compute threes probability
            proj, prob_over, sigma_like, att_mean = threes_prob_over_beta_binom(games, line)

            # injury boost to threes is mostly minutes-driven (stable)
            min_s, min_l, rate_s, rate_l = _role_trend([(d, m, mn) for (d, m, a, mn) in games])
            min_delta = min_s - min_l
            rate_delta = rate_s - rate_l

            absorption = 0.0
            if l10_min >= 28:
                absorption += 0.25
            if l10_min >= 34:
                absorption += 0.10
            if min_delta >= 2.0:
                absorption += 0.15
            absorption = min(0.55, absorption)

            # minutes boost translates to extra attempts
            injury_boost_min = min(BOOST_CAP_MIN, vac_min * absorption * 0.18)
            # project attempts via minutes (rough)
            att_per_min = att_mean / max(l10_min, 1e-6)
            n2 = int(max(1, round(att_mean + injury_boost_min * att_per_min)))
            # recompute prob with boosted attempts using same posterior for p
            # posterior from l10
            sum_m = sum(float(x[1]) for x in l10)
            sum_a = sum(float(x[2]) for x in l10)
            a = THREES_ALPHA0 + max(0.0, sum_m)
            b = THREES_BETA0 + max(0.0, (sum_a - sum_m))
            prob_over2 = beta_binom_tail_prob_over(line=line, n=n2, a=a, b=b)
            proj2 = (a/(a+b)) * n2

            edge = proj2 - line
            if edge < MIN_EDGE or prob_over2 < MIN_PROB:
                continue

            p_over = american_to_prob(float(offer["over_odds"]))
            p_under = american_to_prob(float(offer["under_odds"]))
            p_market = p_over / max(p_over + p_under, 1e-9)

            value_edge = prob_over2 - p_market
            if value_edge < VALUE_EDGE_MIN:
                continue

            ev = ev_per_dollar(prob_over2, float(offer["over_odds"]))
            if ev < EV_MIN:
                continue

            steam = 0.0
            if ENABLE_STEAM:
                prev = get_prev_market(state, prop_type, pid, now_ts)
                if prev:
                    cur = {"line": line, "over_odds": offer["over_odds"], "under_odds": offer["under_odds"], "ts": now_ts}
                    steam = steam_score(prev, cur)
                    if steam < STEAM_MIN_SCORE:
                        steam = 0.0

            why = (
                f"TriggerStrength {trigger_strength:.0f} | Absorb {absorption:.2f}. "
                f"{injured_name} {injured_status.upper()} vacates ~{vac_stat:.1f} pts / {vac_min:.1f} min. "
                f"{nm} L10 3PA≈{l10_3pa:.1f}, mins≈{l10_min:.1f}. "
                f"BetaBinom n≈{n2} att | Proj {proj2:.2f} vs CONS {line:.1f} (n={n_cons}, sharp={sharp_used}) "
                f"| offer {offer['vendor']} {offer['line']:.1f} ({int(offer['over_odds']):+d}) "
                f"| edge +{edge:.2f} | P≈{prob_over2*100:.0f}% (mkt≈{p_market*100:.0f}%, val_edge≈{value_edge:+.2f}) "
                f"| EV≈{ev:+.2f}/$1 | steam={steam:.1f}."
            )

            ideas.append({
                "section": "injury",
                "prop_type": prop_type,
                "player_name": nm,
                "player_id": int(pid),
                "gid": int(offer.get("gid") or 0),
                "cons_line": float(line),
                "line": float(offer["line"]),
                "proj": float(proj2),
                "edge": float(edge),
                "prob_over": float(prob_over2),
                "market_prob": float(p_market),
                "value_edge": float(value_edge),
                "ev": float(ev),
                "vendor": str(offer["vendor"]),
                "over_odds": float(offer["over_odds"]),
                "under_odds": float(offer["under_odds"]),
                "n_cons": int(n_cons),
                "steam": float(steam),
                "trigger_strength": float(trigger_strength),
                "trigger": f"{injured_name} ({team_short}) {injured_status}",
                "why": why,
            })

            remember_market(state, prop_type, pid, offer, line, n_cons, now_ts)

    else:
        # points (and non-beta threes fallback): normal Gaussian model using stat key
        stat_key = "pts" if prop_type == "points" else "fg3m"
        stats_all = {}
        for chunk_ids in _chunk(cand_ids, STAT_BATCH_SIZE):
            if deadline_exceeded():
                break
            stats_all.update(bdl_last_n_games_stats(chunk_ids, season, BASELINE_GAMES, stat_key))

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
            cons, n_cons, sharp_used = consensus_line(rows)
            if cons is None or n_cons < MIN_VENDORS_FOR_CONSENSUS:
                continue
            if MIN_SHARP_VENDORS > 0 and sharp_used < MIN_SHARP_VENDORS:
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

            proj, edge, prob_over, aux = compute_projection_and_prob(games_all=games, line=line)
            base_avg, l10_avg, l3_avg, l10_min, sigma = aux

            rate = l10_avg / max(l10_min, 1e-6)
            proj = proj + injury_boost_stat + (injury_boost_min * rate * BOOST_CAP_RATE)
            edge = proj - line
            z = (proj - line) / max(sigma, 1e-6)
            prob_over = _norm_cdf(z)

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
                    if steam < STEAM_MIN_SCORE:
                        steam = 0.0

            why = (
                f"TriggerStrength {trigger_strength:.0f} | Absorb {absorption:.2f}. "
                f"{injured_name} {injured_status.upper()} vacates ~{vac_stat:.1f} pts / {vac_min:.1f} min. "
                f"{nm} base(L{BASELINE_GAMES}) {base_avg:.1f}, L10 {l10_avg:.1f}, L3 {l3_avg:.1f} "
                f"(mins L10 {l10_min:.1f}). Role Δmin={min_delta:+.1f}, Δrate={rate_delta:+.2f}. "
                f"Proj {proj:.1f} vs CONS {line:.1f} (n={n_cons}, sharp={sharp_used}) | offer {offer['vendor']} {offer['line']:.1f} ({int(offer['over_odds']):+d}) "
                f"| edge +{edge:.1f} | P≈{prob_over*100:.0f}% (mkt≈{p_market*100:.0f}%, val_edge≈{value_edge:+.2f}) "
                f"| EV≈{ev:+.2f}/$1 | steam={steam:.1f}."
            )

            ideas.append({
                "section": "injury",
                "prop_type": prop_type,
                "player_name": nm,
                "player_id": int(pid),
                "gid": int(offer.get("gid") or 0),
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
                "steam": float(steam),
                "trigger_strength": float(trigger_strength),
                "trigger": f"{injured_name} ({team_short}) {injured_status}",
                "why": why,
            })

            remember_market(state, prop_type, pid, offer, line, n_cons, now_ts)

    ideas.sort(key=lambda x: (x["trigger_strength"], x["ev"], x["value_edge"], x["edge"], x["prob_over"]), reverse=True)
    return ideas

def slate_scan_edges(now_et, prop_type, lines_map_for_prop, state, now_ts):
    if not ENABLE_SLATE_SCAN or deadline_exceeded():
        return []

    season = _season_year(now_et)

    pids = list((lines_map_for_prop or {}).keys())
    if not pids:
        return []

    ideas = []

    if prop_type == "threes" and THREES_BETA_BINOM:
        stats_all = {}
        for chunk_ids in _chunk(pids, STAT_BATCH_SIZE):
            if deadline_exceeded():
                break
            stats_all.update(bdl_last_n_games_threes(chunk_ids, season, BASELINE_GAMES))

        for pid in pids:
            if deadline_exceeded():
                break
            games = stats_all.get(int(pid), [])
            if len(games) < 8:
                continue

            l10 = games[-min(len(games), LOOKBACK_GAMES):]
            l10_3pa = sum(float(x[2]) for x in l10) / max(1, len(l10))
            l10_min = sum(float(x[3]) for x in l10) / max(1, len(l10))
            if l10_min < MIN_L10_MIN or l10_3pa < THREES_MIN_L10_3PA:
                continue

            rows = (lines_map_for_prop or {}).get(int(pid), [])
            cons, n_cons, sharp_used = consensus_line(rows)
            if cons is None or n_cons < MIN_VENDORS_FOR_CONSENSUS:
                continue
            if MIN_SHARP_VENDORS > 0 and sharp_used < MIN_SHARP_VENDORS:
                continue

            offer = best_offer_near_consensus(rows, cons)
            if not offer:
                continue

            line = float(cons)
            proj, prob_over, sigma_like, att_mean = threes_prob_over_beta_binom(games, line)
            edge = proj - line

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
                prev = get_prev_market(state, prop_type, int(pid), now_ts)
                if prev:
                    cur = {"line": line, "over_odds": offer["over_odds"], "under_odds": offer["under_odds"], "ts": now_ts}
                    steam = steam_score(prev, cur)
                    if steam < STEAM_MIN_SCORE:
                        steam = 0.0

            name = PLAYER_NAME_CACHE.get(int(pid), f"Player {pid}")

            why = (
                f"SlateScan. BetaBinom L10 3PA≈{l10_3pa:.1f}, mins≈{l10_min:.1f}. "
                f"Proj {proj:.2f} vs CONS {line:.1f} (n={n_cons}, sharp={sharp_used}) "
                f"| offer {offer['vendor']} {offer['line']:.1f} ({int(offer['over_odds']):+d}) "
                f"| edge +{edge:.2f} | P≈{prob_over*100:.0f}% (mkt≈{p_market*100:.0f}%, val_edge≈{value_edge:+.2f}) "
                f"| EV≈{ev:+.2f}/$1 | steam={steam:.1f}."
            )

            ideas.append({
                "section": "slate",
                "prop_type": prop_type,
                "player_name": name,
                "player_id": int(pid),
                "gid": int(offer.get("gid") or 0),
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
                "steam": float(steam),
                "trigger_strength": 0.0,
                "trigger": "No injury trigger (league-wide scan)",
                "why": why,
            })

            remember_market(state, prop_type, int(pid), offer, line, n_cons, now_ts)

    else:
        stat_key = "pts" if prop_type == "points" else "fg3m"
        stats_all = {}
        for chunk_ids in _chunk(pids, STAT_BATCH_SIZE):
            if deadline_exceeded():
                break
            stats_all.update(bdl_last_n_games_stats(chunk_ids, season, BASELINE_GAMES, stat_key))

        for pid in pids:
            if deadline_exceeded():
                break

            games = stats_all.get(int(pid), [])
            if len(games) < 8:
                continue

            rows = (lines_map_for_prop or {}).get(int(pid), [])
            cons, n_cons, sharp_used = consensus_line(rows)
            if cons is None or n_cons < MIN_VENDORS_FOR_CONSENSUS:
                continue
            if MIN_SHARP_VENDORS > 0 and sharp_used < MIN_SHARP_VENDORS:
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

            proj, edge, prob_over, aux = compute_projection_and_prob(games_all=games, line=line)
            base_avg, l10_avg, l3_avg, l10_min, sigma = aux

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
                prev = get_prev_market(state, prop_type, int(pid), now_ts)
                if prev:
                    cur = {"line": line, "over_odds": offer["over_odds"], "under_odds": offer["under_odds"], "ts": now_ts}
                    steam = steam_score(prev, cur)
                    if steam < STEAM_MIN_SCORE:
                        steam = 0.0

            name = PLAYER_NAME_CACHE.get(int(pid), f"Player {pid}")

            why = (
                f"SlateScan. base(L{BASELINE_GAMES}) {base_avg:.1f}, L10 {l10_avg:.1f}, L3 {l3_avg:.1f} "
                f"(mins L10 {l10_min:.1f}). Role Δmin={min_delta:+.1f}, Δrate={rate_delta:+.2f}. "
                f"Proj {proj:.1f} vs CONS {line:.1f} (n={n_cons}, sharp={sharp_used}) | offer {offer['vendor']} {offer['line']:.1f} ({int(offer['over_odds']):+d}) "
                f"| edge +{edge:.1f} | P≈{prob_over*100:.0f}% (mkt≈{p_market*100:.0f}%, val_edge≈{value_edge:+.2f}) "
                f"| EV≈{ev:+.2f}/$1 | steam={steam:.1f}."
            )

            ideas.append({
                "section": "slate",
                "prop_type": prop_type,
                "player_name": name,
                "player_id": int(pid),
                "gid": int(offer.get("gid") or 0),
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
                "steam": float(steam),
                "trigger_strength": 0.0,
                "trigger": "No injury trigger (league-wide scan)",
                "why": why,
            })

            remember_market(state, prop_type, int(pid), offer, line, n_cons, now_ts)

    ideas.sort(key=lambda x: (x["ev"], x["value_edge"], x["edge"], x["prob_over"]), reverse=True)
    return ideas

# -------------------- SYNTH LADDER (MODEL ONLY) --------------------
def synth_ladder_from_model(points_ideas, stats_cache_by_pid):
    if not ENABLE_SYNTH_LADDER:
        return []
    if not points_ideas:
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

# -------------------- EXPOSURE CAPS --------------------
def apply_exposure_caps(now_et, ideas):
    """
    Applies caps on final_out only:
      - MAX_PLAYS_PER_GAME
      - MAX_PLAYS_PER_TEAM (uses BDL player team lookup; small overhead for shortlisted ideas)
    """
    if not ideas:
        return ideas

    games_today = bdl_games_today(now_et)  # for per-game counting
    game_counts = {}
    team_counts = {}

    kept = []
    for it in ideas:
        gid = int(it.get("gid") or 0)
        if gid and MAX_PLAYS_PER_GAME > 0:
            if game_counts.get(gid, 0) >= MAX_PLAYS_PER_GAME:
                continue

        pid = int(it.get("player_id"))
        tname = ""
        if MAX_PLAYS_PER_TEAM > 0:
            pinfo = bdl_player_by_id(pid)
            tname = (pinfo.get("team_name") or "").strip()
            if tname:
                if team_counts.get(tname, 0) >= MAX_PLAYS_PER_TEAM:
                    continue

        kept.append(it)
        if gid and MAX_PLAYS_PER_GAME > 0:
            game_counts[gid] = game_counts.get(gid, 0) + 1
        if tname and MAX_PLAYS_PER_TEAM > 0:
            team_counts[tname] = team_counts.get(tname, 0) + 1

    return kept

# -------------------- LINEUPEXPERTS NEWS ADJUST --------------------
def apply_news_overlay(state, now_ts: int, ideas):
    """
    Pulls LineupExperts news for the idea players and adjusts 'news_adj' + attaches 'news_blurb'.
    This does NOT change your math model; it changes ranking and can optionally drop strong negative news.
    """
    if not ENABLE_LINEUPEXPERTS or not LINEUPEXPERTS_API_KEY or not ideas:
        return ideas

    try:
        name_to_le = le_players_map_cached(state, now_ts)
    except Exception as e:
        print(f"[WARN] LineupExperts players map failed: {e}")
        return ideas

    # Map BDL player names -> LE IDs
    bdl_names = {}
    for it in ideas:
        pid = int(it.get("player_id"))
        nm = it.get("player_name") or PLAYER_NAME_CACHE.get(pid, "")
        if nm:
            bdl_names[pid] = nm

    le_ids = []
    pid_to_le = {}
    for pid, nm in bdl_names.items():
        le_id = name_to_le.get(_clean_name(nm))
        if le_id:
            pid_to_le[pid] = le_id
            le_ids.append(le_id)

    if not le_ids:
        return ideas

    news_map = le_news_by_players(le_ids)

    max_age_sec = int(NEWS_MAX_AGE_HOURS * 3600)

    out = []
    for it in ideas:
        pid = int(it.get("player_id"))
        le_id = pid_to_le.get(pid)
        news_adj = 0.0
        news_blurb = ""

        if le_id and le_id in news_map:
            story = news_map[le_id]
            dt = story.get("_dt")
            age_ok = True
            if isinstance(dt, datetime):
                age_sec = abs(int((datetime.now(dt.tzinfo) - dt).total_seconds()))
                age_ok = age_sec <= max_age_sec

            title = str(story.get("Title", "") or "")
            content = str(story.get("Content", "") or "")
            blob = (title + " " + content).strip()

            if age_ok and blob:
                s, label = score_news(blob)
                if s <= -1.0:
                    news_adj = -NEWS_PENALTY
                elif s >= +1.0:
                    news_adj = +NEWS_BOOST
                elif s < 0:
                    news_adj = -NEWS_PENALTY * 0.5
                elif s > 0:
                    news_adj = +NEWS_BOOST * 0.5

                news_blurb = title[:120].strip()

                if NEWS_REQUIRE_OK and s <= -1.0:
                    # hard drop on strong negative news
                    continue

        it2 = dict(it)
        it2["news_adj"] = float(news_adj)
        it2["news_blurb"] = news_blurb
        # composite score used for ranking (EV-first but news can break ties / prevent bad plays)
        it2["_score"] = float(it2.get("ev", 0.0)) + float(it2.get("value_edge", 0.0)) + float(it2.get("news_adj", 0.0))
        out.append(it2)

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
        f"MIN_VENDORS_FOR_CONSENSUS={MIN_VENDORS_FOR_CONSENSUS} MIN_SHARP_VENDORS={MIN_SHARP_VENDORS} "
        f"ENABLE_STEAM={int(ENABLE_STEAM)} MAX_PLAYS_PER_TEAM={MAX_PLAYS_PER_TEAM} MAX_PLAYS_PER_GAME={MAX_PLAYS_PER_GAME} "
        f"THREES_BETA_BINOM={int(THREES_BETA_BINOM)} LINEUPEXPERTS={int(ENABLE_LINEUPEXPERTS)}"
    )

    if TEST_MODE:
        send_one(f"✅ NBA betting agent test OK ({ts_et})")
        return

    state = load_state()
    old_players = state.get("players", {})

    # props
    lines_map = build_today_props(now_et)

    # injuries
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

        exclude_names_lower = {_clean_name(v.get("name", "")) for v in new_players.values() if v.get("name")}

        for pid, cur in new_players.items():
            if deadline_exceeded():
                break

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
                    now_ts=now_ts
                )
                if ideas:
                    got_any = True
                    injury_ideas_all.extend(ideas)

            if got_any:
                triggers.append(f"{injured_name} ({team_short}) {injured_status}")

    # slate scan
    slate_ideas_all = []
    if ENABLE_SLATE_SCAN and (not deadline_exceeded()):
        for pt in PROP_TYPES:
            if deadline_exceeded():
                break
            slate_ideas_all.extend(
                slate_scan_edges(now_et, pt, lines_map.get(pt, {}), state=state, now_ts=now_ts)
            )

    combined = injury_ideas_all + slate_ideas_all
    combined = apply_cooldown(state, combined, now_ts)

    # Apply LineupExperts overlay BEFORE final selection (ranking + optional drops)
    combined = apply_news_overlay(state, now_ts, combined)

    # Dedup best per player+prop using composite score
    best = {}
    for i in combined:
        k = (i["prop_type"], int(i["player_id"]))
        score = float(i.get("_score", float(i.get("ev", 0.0)) + float(i.get("value_edge", 0.0))))
        if (k not in best) or (score > best[k][0]):
            best[k] = (score, i)

    combined = [v[1] for v in best.values()]

    # Per market limits
    out_by_market = {}
    for pt in PROP_TYPES:
        inj = [x for x in combined if x["prop_type"] == pt and x["section"] == "injury"]
        slt = [x for x in combined if x["prop_type"] == pt and x["section"] == "slate"]

        inj.sort(key=lambda x: (x.get("trigger_strength", 0.0), x.get("_score", 0.0), x.get("ev", 0.0)), reverse=True)
        slt.sort(key=lambda x: (x.get("_score", 0.0), x.get("ev", 0.0)), reverse=True)

        picks = inj + slt
        if MIN_PER_MARKET > 0:
            picks = picks[:max(MIN_PER_MARKET, MAX_PER_MARKET)]
        picks = picks[:MAX_PER_MARKET]
        out_by_market[pt] = picks

    final_out = []
    for pt in PROP_TYPES:
        final_out.extend(out_by_market.get(pt, []))
    final_out = final_out[:MAX_TOTAL_PLAYS]

    # Exposure caps (this is your “stop the 3 Bucks” guardrail)
    final_out = apply_exposure_caps(now_et, final_out)

    # Plus odds bucket
    plus_bucket = []
    for x in final_out:
        try:
            if float(x.get("over_odds", -999)) >= PLUS_ODDS_MIN:
                plus_bucket.append(x)
        except Exception:
            pass
    plus_bucket.sort(key=lambda x: (x.get("_score", 0.0), x.get("ev", 0.0)), reverse=True)
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

        for pt in PROP_TYPES:
            picks = out_by_market.get(pt, [])
            picks = [x for x in picks if x in final_out]  # after exposure caps
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
                    news = f" 📰 {i['news_blurb']}" if i.get("news_blurb") else ""
                    msg.append(
                        f"• {i['player_name']} OVER {i['cons_line']:.1f}  "
                        f"(edge +{i['edge']:.1f}, P≈{i['prob_over']*100:.0f}%, EV≈{i['ev']:+.2f}/$1){fire}"
                    )
                    msg.append(f"  Offer: {i['vendor']} {i['line']:.1f} ({int(i['over_odds']):+d}) | CONS n={i.get('n_cons',0)} | news_adj={i.get('news_adj',0.0):+.2f}{news}")
                    msg.append(f"  Trigger: {i['trigger']}")
                    msg.append(f"  Why: {i['why']}")
                    msg.append("")

            if slt:
                msg.append("🌎 League-Wide Slate Scan (no injury required):")
                msg.append("")
                for i in slt:
                    fire = " 🔥" if i.get("ev", 0) >= 0.25 else ""
                    news = f" 📰 {i['news_blurb']}" if i.get("news_blurb") else ""
                    msg.append(
                        f"• {i['player_name']} OVER {i['cons_line']:.1f}  "
                        f"(edge +{i['edge']:.1f}, P≈{i['prob_over']*100:.0f}%, EV≈{i['ev']:+.2f}/$1){fire}"
                    )
                    msg.append(f"  Offer: {i['vendor']} {i['line']:.1f} ({int(i['over_odds']):+d}) | CONS n={i.get('n_cons',0)} | news_adj={i.get('news_adj',0.0):+.2f}{news}")
                    msg.append(f"  Why: {i['why']}")
                    msg.append("")

            msg.append("")

        if plus_bucket:
            msg.append("💎 Plus-odds value bucket:")
            msg.append("")
            for i in plus_bucket:
                msg.append(
                    f"• {i['player_name']} OVER {i['cons_line']:.1f}  "
                    f"(offer {i['vendor']} {int(i['over_odds']):+d}, P≈{i['prob_over']*100:.0f}%, EV≈{i['ev']:+.2f}/$1, news_adj={i.get('news_adj',0.0):+.2f})"
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
