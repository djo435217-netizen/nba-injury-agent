import os
import json
import re
import time
import math
from datetime import datetime, timedelta
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


def _strip_quotes(s: str) -> str:
    """
    Fixes env var issues like:
      LINEUPEXPERTS_BASE_URL="https://api.lineupexperts.com/v1"
    which becomes a literal string containing quotes.
    """
    s = (s or "").strip()
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        s = s[1:-1].strip()
    return s


BOOK_VENDOR_RAW = _strip_quotes(os.environ.get("BOOK_VENDORS", os.environ.get("BOOK_VENDOR", "fanduel"))).strip().lower()
BOOK_VENDORS = [v.strip() for v in BOOK_VENDOR_RAW.split(",") if v.strip()]

PROP_TYPES_RAW = _strip_quotes(os.environ.get("PROP_TYPES", "points")).strip().lower()
PROP_TYPES = [p.strip() for p in PROP_TYPES_RAW.split(",") if p.strip()]

ENABLE_INJURY_TRIGGERS = os.environ.get("ENABLE_INJURY_TRIGGERS", "1") == "1"
ENABLE_SLATE_SCAN = os.environ.get("ENABLE_SLATE_SCAN", "1") == "1"

# Exposure caps (quick win)
MAX_PLAYS_PER_TEAM = int(os.environ.get("MAX_PLAYS_PER_TEAM", "2"))
MAX_PLAYS_PER_GAME = int(os.environ.get("MAX_PLAYS_PER_GAME", "2"))

# Consensus + Steam + EV + Plus odds + "market respect"
MIN_VENDORS_FOR_CONSENSUS = int(os.environ.get("MIN_VENDORS_FOR_CONSENSUS", "2"))
MIN_SHARP_VENDORS = int(os.environ.get("MIN_SHARP_VENDORS", "1"))
SHARP_VENDORS_RAW = _strip_quotes(os.environ.get("SHARP_VENDORS", "draftkings,caesars,betmgm,bet365,pointsbet,hardrock")).strip().lower()
SHARP_VENDORS = {x.strip() for x in SHARP_VENDORS_RAW.split(",") if x.strip()}

ENABLE_STEAM = os.environ.get("ENABLE_STEAM", "0") == "1"
STEAM_MIN_SCORE = float(os.environ.get("STEAM_MIN_SCORE", "1.0"))
STEAM_MAX_AGE_MIN = int(os.environ.get("STEAM_MAX_AGE_MIN", "240"))

# Output sizing
MIN_PER_MARKET = int(os.environ.get("MIN_PER_MARKET", "0"))
MAX_PER_MARKET = int(os.environ.get("MAX_PER_MARKET", "6"))
MAX_TOTAL_PLAYS = int(os.environ.get("MAX_TOTAL_PLAYS", "10"))

# Windows
BASELINE_GAMES = int(os.environ.get("BASELINE_GAMES", "30"))
LOOKBACK_GAMES = int(os.environ.get("LOOKBACK_GAMES", "10"))
SHORT_GAMES = int(os.environ.get("SHORT_GAMES", "3"))

# Model thresholds (points engine)
MIN_EDGE = float(os.environ.get("MIN_EDGE", "2.5"))
MIN_PROB = float(os.environ.get("MIN_PROB", "0.62"))
STD_FLOOR = float(os.environ.get("STD_FLOOR", "5.0"))

# EV filter
EV_MIN = float(os.environ.get("EV_MIN", "0.00"))  # keep >= 0 EV by default
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
STAT_BATCH_SIZE = int(os.environ.get("STAT_BATCH_SIZE", "90"))

DEBUG_PROP_SAMPLE_TYPES = os.environ.get("DEBUG_PROP_SAMPLE_TYPES", "0").strip().lower()

# Plus-odds bucket
PLUS_ODDS_MIN = float(os.environ.get("PLUS_ODDS_MIN", "100"))
PLUS_ODDS_TOPN = int(os.environ.get("PLUS_ODDS_TOPN", "3"))

# Threes: beta-binomial
THREES_BETA_BINOM = os.environ.get("THREES_BETA_BINOM", "1") == "1"
THREES_MIN_ATT_GAMES = int(os.environ.get("THREES_MIN_ATT_GAMES", "8"))

# LineupExperts news integration (NBA Core endpoint)
LINEUPEXPERTS = os.environ.get("LINEUPEXPERTS", "0") == "1"
LINEUPEXPERTS_KEY = _strip_quotes(os.environ.get("LINEUPEXPERTS_KEY", os.environ.get("LINEUPEXPERTS_API_KEY", ""))).strip()
LINEUPEXPERTS_BASE_URL = _strip_quotes(os.environ.get("LINEUPEXPERTS_BASE_URL", "https://api.lineupexperts.com/v1")).strip().rstrip("/")
LINEUPEXPERTS_TIMEOUT = int(os.environ.get("LINEUPEXPERTS_TIMEOUT", "12"))
LINEUPEXPERTS_MAX_ITEMS = int(os.environ.get("LINEUPEXPERTS_MAX_ITEMS", "200"))
NEWS_LOOKBACK_HOURS = int(os.environ.get("NEWS_LOOKBACK_HOURS", "36"))
NEWS_BOOST_OUT = float(os.environ.get("NEWS_BOOST_OUT", "0.18"))
NEWS_BOOST_QUESTIONABLE = float(os.environ.get("NEWS_BOOST_QUESTIONABLE", "0.08"))
NEWS_BOOST_MINUTES = float(os.environ.get("NEWS_BOOST_MINUTES", "0.05"))
NEWS_MIN_CONFIDENCE = float(os.environ.get("NEWS_MIN_CONFIDENCE", "0.25"))

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
PLAYER_NAME_CACHE = {}   # pid -> "First Last"
PLAYER_TEAM_CACHE = {}   # pid -> team name (best-effort)
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

def bdl_games_today(now_et: datetime):
    today = now_et.strftime("%Y-%m-%d")
    resp = _bdl_get("/v1/games", params={"dates[]": [today], "per_page": 100})
    games = resp.get("data") or []
    out = {}
    for g in games:
        try:
            gid = int(g["id"])
        except Exception:
            continue
        home = ((g.get("home_team") or {}).get("name") or "").strip()
        away = ((g.get("visitor_team") or {}).get("name") or "").strip()
        out[gid] = {"home": home, "away": away}
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

            team = row.get("team") or {}
            tname = (team.get("name") or "").strip()
            if tname:
                PLAYER_TEAM_CACHE[pid] = tname

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
            tname = (team.get("name") or "").strip()
            if tname:
                PLAYER_TEAM_CACHE[pid] = tname

            game = row.get("game") or {}
            date = game.get("date")
            fg3m = float(row.get("fg3m", 0) or 0)
            fg3a = float(row.get("fg3a", 0) or 0)
            mins = _parse_minutes(row.get("min"))
            if date:
                out[pid].append((date, fg3m, fg3a, mins))

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
    games_map = bdl_games_today(now_et)
    game_ids = list(games_map.keys())
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
                    "vendor": str((pp.get("vendor") or (v or "no_vendor"))).strip().lower(),
                    "prop_type": (pp.get("prop_type") or pt),
                    "line": float(line),
                    "over_odds": float(over_odds),
                    "under_odds": float(under_odds),
                    "updated_at": pp.get("updated_at"),
                }
                lines_map.setdefault(pt, {}).setdefault(pid, []).append(row)

    return lines_map, games_map

# -------------------- CONSENSUS + OFFER PICKING --------------------
def _round_to_half(x: float) -> float:
    return round(float(x) * 2.0) / 2.0

def consensus_line(rows):
    if not rows:
        return None, 0, 0

    by_vendor = {}
    sharp_count = 0

    for r in rows:
        v = str(r.get("vendor") or "").strip().lower()
        if not v:
            continue
        try:
            line = float(r["line"])
        except Exception:
            continue

        if v not in by_vendor:
            by_vendor[v] = _round_to_half(line)

    for v in by_vendor.keys():
        if v in SHARP_VENDORS:
            sharp_count += 1

    lines = sorted(by_vendor.values())
    n = len(lines)
    if n == 0:
        return None, 0, sharp_count

    mid = n // 2
    if n % 2 == 1:
        return float(lines[mid]), n, sharp_count
    return float(0.5 * (lines[mid - 1] + lines[mid])), n, sharp_count

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
def compute_projection_and_prob_points(games_all, line, w_base=0.45, w_l10=0.35, w_l3=0.10, w_line=0.10):
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

# -------------------- THREES (BETA-BINOMIAL) --------------------
def _betaln(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

def _log_choose(n: int, k: int) -> float:
    # guard invalid combinations
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

def beta_binom_pmf(k: int, n: int, a: float, b: float) -> float:
    # Guard to avoid lgamma(0)/invalid combs
    if n < 0 or k < 0 or k > n:
        return 0.0
    if n == 0:
        # Only k=0 is possible
        return 1.0 if k == 0 else 0.0
    lc = _log_choose(n, k)
    if not math.isfinite(lc):
        return 0.0
    return math.exp(lc + _betaln(k + a, (n - k) + b) - _betaln(a, b))

def beta_binom_cdf(k: int, n: int, a: float, b: float) -> float:
    # Clamp k into [0,n] so we never sum pmf(i) beyond n
    if n <= 0:
        return 1.0 if k >= 0 else 0.0
    if k < 0:
        return 0.0
    k = min(int(k), int(n))
    s = 0.0
    for i in range(0, k + 1):
        s += beta_binom_pmf(i, n, a, b)
    return min(1.0, max(0.0, s))

def threes_prob_over_beta_binom(threes_games, line: float):
    if not threes_games:
        return None

    base = _slice_last(threes_games, BASELINE_GAMES)
    l10 = _slice_last(threes_games, LOOKBACK_GAMES)

    makes = sum(float(x[1]) for x in base)
    atts = sum(float(x[2]) for x in base)
    if atts <= 0:
        return None

    a = 1.0 + makes
    b = 1.0 + max(0.0, atts - makes)

    att_list = [int(round(float(x[2]))) for x in l10 if float(x[2]) > 0]
    if len(att_list) < max(3, min(THREES_MIN_ATT_GAMES, LOOKBACK_GAMES // 2)):
        att_list = [int(round(float(x[2]))) for x in base if float(x[2]) > 0]

    if not att_list:
        return None

    k = int(math.floor(float(line)))  # over line means >= k+1
    probs = []
    for n_att in att_list:
        n_att = int(n_att)
        if n_att <= 0:
            probs.append(0.0)
            continue
        p_le_k = beta_binom_cdf(k, n_att, a, b)
        probs.append(1.0 - p_le_k)

    return sum(probs) / len(probs)

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
def _try_parse_dt(s: str):
    if not s:
        return None
    s = str(s).strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z",
                "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None

def fetch_lineupexperts_news(now_et: datetime):
    if not LINEUPEXPERTS or not LINEUPEXPERTS_KEY:
        return []

    url = f"{LINEUPEXPERTS_BASE_URL}/nba-NewsBySport"
    try:
        r = requests.get(url, params={"key": LINEUPEXPERTS_KEY}, timeout=LINEUPEXPERTS_TIMEOUT)
    except Exception as e:
        print(f"[WARN] LineupExperts request failed: {type(e).__name__}: {e}")
        return []

    if r.status_code != 200:
        print(f"[WARN] LineupExperts HTTP {r.status_code} for {r.url}: {r.text[:300]}")
        return []

    try:
        data = r.json()
    except Exception:
        print(f"[WARN] LineupExperts non-JSON response: {r.text[:200]}")
        return []

    if isinstance(data, dict):
        items = data.get("data") or data.get("results") or data.get("news") or data.get("items") or []
    elif isinstance(data, list):
        items = data
    else:
        items = []

    if not isinstance(items, list):
        return []

    items = items[:max(1, LINEUPEXPERTS_MAX_ITEMS)]

    cutoff = now_et - timedelta(hours=NEWS_LOOKBACK_HOURS)
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue

        dt = None
        for k in ("date", "updated", "updated_at", "created", "created_at", "publishDate", "published", "time"):
            if k in it:
                dt = _try_parse_dt(it.get(k))
                break

        if dt is not None and dt.tzinfo is None:
            dt = dt.replace(tzinfo=ET)

        if dt is not None and dt < cutoff:
            continue

        out.append(it)

    return out

def build_news_boost_map(news_items):
    boosts = {}

    def push(player_name: str, boost: float, confidence: float, why: str):
        if not player_name:
            return
        k = _clean_name(player_name)
        if not k:
            return
        if confidence < NEWS_MIN_CONFIDENCE:
            return
        cur = boosts.get(k)
        score = boost * confidence
        if (cur is None) or (score > (cur["boost"] * cur["confidence"])):
            boosts[k] = {"boost": float(boost), "confidence": float(confidence), "why": why[:220]}

    out_pat = re.compile(r"\b(ruled out|will miss|out for|out vs|out tonight|inactive)\b", re.I)
    q_pat = re.compile(r"\b(questionable|probable|doubtful|game-time decision|gtd)\b", re.I)
    min_up_pat = re.compile(r"\b(minutes restriction lifted|minutes limit lifted|expected to start|will start|increase in minutes|bigger role)\b", re.I)

    for it in news_items:
        try:
            title = str(it.get("title") or it.get("headline") or "").strip()
            body = str(it.get("news") or it.get("description") or it.get("content") or it.get("analysis") or "").strip()
            player = str(it.get("player") or it.get("playerName") or it.get("full_name") or "").strip()
        except Exception:
            continue

        text = f"{title}\n{body}".strip()
        if not text:
            continue

        conf = None
        for ck in ("confidence", "impact", "weight", "rating", "score"):
            if ck in it:
                try:
                    conf = float(it.get(ck))
                    break
                except Exception:
                    pass
        if conf is None:
            conf = 0.35
            if out_pat.search(text) or q_pat.search(text):
                conf = 0.55

        if not player:
            continue

        boost = 0.0
        why_bits = []
        if out_pat.search(text):
            boost += NEWS_BOOST_OUT
            why_bits.append("out-news")
        if q_pat.search(text):
            boost += NEWS_BOOST_QUESTIONABLE
            why_bits.append("status-news")
        if min_up_pat.search(text):
            boost += NEWS_BOOST_MINUTES
            why_bits.append("minutes-news")

        if boost <= 0:
            continue

        push(player, boost, conf, f"{'|'.join(why_bits)}: {title or body}")

    return boosts

def apply_news_to_projection(proj: float, boost_rec: dict | None, cap: float = 0.30):
    if not boost_rec:
        return proj, 0.0, None
    b = float(boost_rec.get("boost", 0.0))
    c = float(boost_rec.get("confidence", 0.0))
    eff = max(0.0, min(cap, b * c))
    if eff <= 0:
        return proj, 0.0, None
    why = boost_rec.get("why") or ""
    return proj * (1.0 + eff), eff, why

# -------------------- ENGINES --------------------
def build_injury_edges(team_short, injured_name, injured_status, exclude_names_lower, now_et, prop_type, lines_map_for_prop, state, now_ts, news_boosts):
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

    if prop_type == "threes":
        inj_games = bdl_last_n_games_threes([injured_pid], season, BASELINE_GAMES).get(injured_pid, [])
        ip10 = sum(float(x[1]) for x in _slice_last(inj_games, LOOKBACK_GAMES)) / max(1, len(_slice_last(inj_games, LOOKBACK_GAMES)))
        im10 = sum(float(x[3]) for x in _slice_last(inj_games, LOOKBACK_GAMES)) / max(1, len(_slice_last(inj_games, LOOKBACK_GAMES)))
    else:
        inj_games = bdl_last_n_games_stats([injured_pid], season, BASELINE_GAMES, "pts").get(injured_pid, [])
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
    if prop_type == "threes":
        for chunk_ids in _chunk(cand_ids, STAT_BATCH_SIZE):
            if deadline_exceeded():
                break
            stats_all.update(bdl_last_n_games_threes(chunk_ids, season, BASELINE_GAMES))
    else:
        for chunk_ids in _chunk(cand_ids, STAT_BATCH_SIZE):
            if deadline_exceeded():
                break
            stats_all.update(bdl_last_n_games_stats(chunk_ids, season, BASELINE_GAMES, "pts"))

    ideas = []
    for pid, nm in roster_tuples:
        if deadline_exceeded():
            break

        games = stats_all.get(pid, [])
        if len(games) < 8:
            continue

        if prop_type == "threes":
            m10 = sum(float(x[3]) for x in _slice_last(games, LOOKBACK_GAMES)) / max(1, len(_slice_last(games, LOOKBACK_GAMES)))
            v10 = sum(float(x[1]) for x in _slice_last(games, LOOKBACK_GAMES)) / max(1, len(_slice_last(games, LOOKBACK_GAMES)))
        else:
            v10, m10, _ = avg_stat_min_std(_slice_last(games, LOOKBACK_GAMES))

        if m10 < MIN_L10_MIN:
            continue

        rows = (lines_map_for_prop or {}).get(pid, [])
        cons, n_cons, n_sharp = consensus_line(rows)
        if cons is None or n_cons < MIN_VENDORS_FOR_CONSENSUS or n_sharp < MIN_SHARP_VENDORS:
            continue

        offer = best_offer_near_consensus(rows, cons)
        if not offer:
            continue

        line = float(cons)

        if (v10 - line) > LINE_MIN_GAP:
            continue

        if prop_type == "threes":
            long_slice = _slice_last(games, LOOKBACK_GAMES)
            short_slice = _slice_last(games, SHORT_GAMES)
            v_l = sum(float(x[1]) for x in long_slice) / max(1, len(long_slice))
            m_l = sum(float(x[3]) for x in long_slice) / max(1, len(long_slice))
            v_s = sum(float(x[1]) for x in short_slice) / max(1, len(short_slice))
            m_s = sum(float(x[3]) for x in short_slice) / max(1, len(short_slice))
            rate_l = v_l / max(m_l, 1e-6)
            rate_s = v_s / max(m_s, 1e-6)
            min_delta = m_s - m_l
            rate_delta = rate_s - rate_l
            l10_min = m_l
        else:
            min_s, min_l, rate_s, rate_l = _role_trend(games)
            min_delta = min_s - min_l
            rate_delta = rate_s - rate_l
            l10_min = m10

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

        if prop_type == "threes" and THREES_BETA_BINOM:
            base_line = float(line)
            base = _slice_last(games, BASELINE_GAMES)
            l10 = _slice_last(games, LOOKBACK_GAMES)
            l3 = _slice_last(games, SHORT_GAMES)
            base_avg = sum(float(x[1]) for x in base) / max(1, len(base))
            l10_avg = sum(float(x[1]) for x in l10) / max(1, len(l10))
            l3_avg = sum(float(x[1]) for x in l3) / max(1, len(l3))
            proj = 0.45 * base_avg + 0.35 * l10_avg + 0.10 * l3_avg + 0.10 * base_line

            proj = proj + min(0.6, injury_boost_min * 0.03) + min(0.4, injury_boost_stat * 0.05)

            boost_rec = news_boosts.get(_clean_name(nm))
            proj, news_eff, news_why = apply_news_to_projection(proj, boost_rec)

            prob_over = threes_prob_over_beta_binom(games, base_line)
            if prob_over is None:
                continue

            edge = proj - base_line
            aux = (base_avg, l10_avg, l3_avg, l10_min, None)
        else:
            proj, edge, prob_over, aux = compute_projection_and_prob_points(games_all=games, line=line)
            base_avg, l10_avg, l3_avg, l10_min, sigma = aux

            rate = l10_avg / max(l10_min, 1e-6)
            proj = proj + injury_boost_stat + (injury_boost_min * rate * BOOST_CAP_RATE)

            boost_rec = news_boosts.get(_clean_name(nm))
            proj, news_eff, news_why = apply_news_to_projection(proj, boost_rec)

            edge = proj - line
            z = (proj - line) / max(sigma, 1e-6)
            prob_over = _norm_cdf(z)

            aux = (base_avg, l10_avg, l3_avg, l10_min, sigma)

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

        gid = int(offer.get("gid") or offer.get("game_id") or offer.get("gid", 0) or 0)
        team_name = PLAYER_TEAM_CACHE.get(pid) or team_short or ""

        base_avg, l10_avg, l3_avg, l10_min, sigma = aux
        news_note = f" | news_boost≈{news_eff:+.2f}" if 'news_eff' in locals() and news_eff else ""
        news_note2 = f" ({news_why})" if 'news_why' in locals() and news_why else ""

        why = (
            f"TriggerStrength {trigger_strength:.0f} | Absorb {absorption:.2f}. "
            f"{injured_name} {injured_status.upper()} vacates ~{vac_stat:.1f} {prop_type.title()} / {vac_min:.1f} min. "
            f"{nm} base(L{BASELINE_GAMES}) {base_avg:.1f}, L10 {l10_avg:.1f}, L3 {l3_avg:.1f} "
            f"(mins L10 {l10_min:.1f}). Role Δmin={min_delta:+.1f}, Δrate={rate_delta:+.2f}. "
            f"Proj {proj:.1f} vs CONS {line:.1f} (n={n_cons}, sharp={n_sharp}) | "
            f"offer {offer['vendor']} {offer['line']:.1f} ({int(offer['over_odds']):+d}) "
            f"| edge +{edge:.1f} | P≈{prob_over*100:.0f}% (mkt≈{p_market*100:.0f}%, val_edge≈{value_edge:+.2f}) "
            f"| EV≈{ev:+.2f}/$1 | steam={steam:.1f}{news_note}{news_note2}."
        )

        ideas.append({
            "section": "injury",
            "prop_type": prop_type,
            "player_name": nm,
            "player_id": int(pid),
            "team": team_name,
            "gid": gid,
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
            "n_sharp": int(n_sharp),
            "steam": float(steam),
            "trigger_strength": float(trigger_strength),
            "trigger": f"{injured_name} ({team_short}) {injured_status}",
            "why": why,
        })

        remember_market(state, prop_type, pid, offer, line, n_cons, now_ts)

    ideas.sort(key=lambda x: (x["trigger_strength"], x["ev"], x["value_edge"], x["edge"], x["prob_over"]), reverse=True)
    return ideas

def slate_scan_edges(now_et, prop_type, lines_map_for_prop, state, now_ts, news_boosts):
    if not ENABLE_SLATE_SCAN or deadline_exceeded():
        return []

    season = _season_year(now_et)

    pids = list((lines_map_for_prop or {}).keys())
    if not pids:
        return []

    stats_all = {}
    if prop_type == "threes" and THREES_BETA_BINOM:
        for chunk_ids in _chunk(pids, STAT_BATCH_SIZE):
            if deadline_exceeded():
                break
            stats_all.update(bdl_last_n_games_threes(chunk_ids, season, BASELINE_GAMES))
    else:
        for chunk_ids in _chunk(pids, STAT_BATCH_SIZE):
            if deadline_exceeded():
                break
            stats_all.update(bdl_last_n_games_stats(chunk_ids, season, BASELINE_GAMES, "pts"))

    ideas = []
    for pid in pids:
        if deadline_exceeded():
            break

        games = stats_all.get(int(pid), [])
        if len(games) < 8:
            continue

        rows = (lines_map_for_prop or {}).get(int(pid), [])
        cons, n_cons, n_sharp = consensus_line(rows)
        if cons is None or n_cons < MIN_VENDORS_FOR_CONSENSUS or n_sharp < MIN_SHARP_VENDORS:
            continue

        offer = best_offer_near_consensus(rows, cons)
        if not offer:
            continue

        if prop_type == "threes":
            v10 = sum(float(x[1]) for x in _slice_last(games, LOOKBACK_GAMES)) / max(1, len(_slice_last(games, LOOKBACK_GAMES)))
            m10 = sum(float(x[3]) for x in _slice_last(games, LOOKBACK_GAMES)) / max(1, len(_slice_last(games, LOOKBACK_GAMES)))
        else:
            v10, m10, _ = avg_stat_min_std(_slice_last(games, LOOKBACK_GAMES))

        if m10 < MIN_L10_MIN:
            continue

        line = float(cons)
        if (v10 - line) > LINE_MIN_GAP:
            continue

        if prop_type == "threes":
            long_slice = _slice_last(games, LOOKBACK_GAMES)
            short_slice = _slice_last(games, SHORT_GAMES)
            v_l = sum(float(x[1]) for x in long_slice) / max(1, len(long_slice))
            m_l = sum(float(x[3]) for x in long_slice) / max(1, len(long_slice))
            v_s = sum(float(x[1]) for x in short_slice) / max(1, len(short_slice))
            m_s = sum(float(x[3]) for x in short_slice) / max(1, len(short_slice))
            rate_l = v_l / max(m_l, 1e-6)
            rate_s = v_s / max(m_s, 1e-6)
            min_delta = m_s - m_l
            rate_delta = rate_s - rate_l
            l10_min = m10
        else:
            min_s, min_l, rate_s, rate_l = _role_trend(games)
            min_delta = min_s - min_l
            rate_delta = rate_s - rate_l
            l10_min = m10

        if prop_type == "threes" and THREES_BETA_BINOM:
            base = _slice_last(games, BASELINE_GAMES)
            l10 = _slice_last(games, LOOKBACK_GAMES)
            l3 = _slice_last(games, SHORT_GAMES)
            base_avg = sum(float(x[1]) for x in base) / max(1, len(base))
            l10_avg = sum(float(x[1]) for x in l10) / max(1, len(l10))
            l3_avg = sum(float(x[1]) for x in l3) / max(1, len(l3))
            proj = 0.45 * base_avg + 0.35 * l10_avg + 0.10 * l3_avg + 0.10 * float(line)

            name = PLAYER_NAME_CACHE.get(int(pid), f"Player {pid}")
            boost_rec = news_boosts.get(_clean_name(name))
            proj, news_eff, news_why = apply_news_to_projection(proj, boost_rec)

            prob_over = threes_prob_over_beta_binom(games, float(line))
            if prob_over is None:
                continue

            edge = proj - float(line)
            sigma = None
        else:
            proj, edge, prob_over, aux = compute_projection_and_prob_points(games_all=games, line=line)
            base_avg, l10_avg, l3_avg, l10_min, sigma = aux

            name = PLAYER_NAME_CACHE.get(int(pid), f"Player {pid}")
            boost_rec = news_boosts.get(_clean_name(name))
            proj, news_eff, news_why = apply_news_to_projection(proj, boost_rec)

            edge = proj - line
            z = (proj - line) / max(sigma, 1e-6)
            prob_over = _norm_cdf(z)

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
            prev = get_prev_market(state, prop_type, pid, now_ts)
            if prev:
                cur = {"line": line, "over_odds": offer["over_odds"], "under_odds": offer["under_odds"], "ts": now_ts}
                steam = steam_score(prev, cur)

        name = PLAYER_NAME_CACHE.get(int(pid), f"Player {pid}")
        team_name = PLAYER_TEAM_CACHE.get(int(pid), "")
        gid = int(offer.get("gid") or offer.get("game_id") or 0)

        news_note = f" | news_boost≈{news_eff:+.2f}" if 'news_eff' in locals() and news_eff else ""
        news_note2 = f" ({news_why})" if 'news_why' in locals() and news_why else ""

        if prop_type == "threes":
            base = _slice_last(games, BASELINE_GAMES)
            l10 = _slice_last(games, LOOKBACK_GAMES)
            l3 = _slice_last(games, SHORT_GAMES)
            base_avg = sum(float(x[1]) for x in base) / max(1, len(base))
            l10_avg = sum(float(x[1]) for x in l10) / max(1, len(l10))
            l3_avg = sum(float(x[1]) for x in l3) / max(1, len(l3))

        why = (
            f"SlateScan. base(L{BASELINE_GAMES}) {base_avg:.1f}, L10 {l10_avg:.1f}, L3 {l3_avg:.1f} "
            f"(mins L10 {l10_min:.1f}). Role Δmin={min_delta:+.1f}, Δrate={rate_delta:+.2f}. "
            f"Proj {proj:.1f} vs CONS {line:.1f} (n={n_cons}, sharp={n_sharp}) | "
            f"offer {offer['vendor']} {offer['line']:.1f} ({int(offer['over_odds']):+d}) "
            f"| edge +{edge:.1f} | P≈{prob_over*100:.0f}% (mkt≈{p_market*100:.0f}%, val_edge≈{value_edge:+.2f}) "
            f"| EV≈{ev:+.2f}/$1 | steam={steam:.1f}{news_note}{news_note2}."
        )

        ideas.append({
            "section": "slate",
            "prop_type": prop_type,
            "player_name": name,
            "player_id": int(pid),
            "team": team_name,
            "gid": gid,
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
            "n_sharp": int(n_sharp),
            "steam": float(steam),
            "trigger_strength": 0.0,
            "trigger": "No injury trigger (league-wide scan)",
            "why": why,
        })

        remember_market(state, prop_type, pid, offer, line, n_cons, now_ts)

    ideas.sort(key=lambda x: (x["ev"], x["value_edge"], x["edge"], x["prob_over"]), reverse=True)
    return ideas

# -------------------- COOLDOWN FILTER --------------------
def apply_cooldown(state, ideas, now_ts: int):
    sent = state.get("sent_bets", {}) or {}
    cooldown_sec = BET_COOLDOWN_MIN * 60

    kept = []
    for i in ideas:
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
        key = f"{i['prop_type']}|{i['section']}|{int(i['player_id'])}|{float(i.get('cons_line', i.get('line', 0.0))):.1f}"
        sent[key] = {"ts": now_ts, "edge": float(i.get("edge", 0.0))}
    state["sent_bets"] = sent

# -------------------- EXPOSURE CAPS --------------------
def apply_exposure_caps(ideas):
    if not ideas:
        return ideas

    team_ct = {}
    game_ct = {}
    out = []
    for it in ideas:
        team = (it.get("team") or "").strip().lower()
        gid = int(it.get("gid") or 0)

        if team and team_ct.get(team, 0) >= MAX_PLAYS_PER_TEAM:
            continue
        if gid and game_ct.get(gid, 0) >= MAX_PLAYS_PER_GAME:
            continue

        out.append(it)

        if team:
            team_ct[team] = team_ct.get(team, 0) + 1
        if gid:
            game_ct[gid] = game_ct.get(gid, 0) + 1

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
        f"ENABLE_STEAM={int(ENABLE_STEAM)} "
        f"MAX_PLAYS_PER_TEAM={MAX_PLAYS_PER_TEAM} MAX_PLAYS_PER_GAME={MAX_PLAYS_PER_GAME} "
        f"THREES_BETA_BINOM={int(THREES_BETA_BINOM)} "
        f"LINEUPEXPERTS={int(LINEUPEXPERTS)} LINEUPEXPERTS_BASE_URL={LINEUPEXPERTS_BASE_URL}"
    )

    if TEST_MODE:
        send_one(f"✅ NBA betting agent test OK ({ts_et})")
        return

    state = load_state()
    old_players = state.get("players", {})

    lines_map, games_map = build_today_props(now_et)

    news_items = fetch_lineupexperts_news(now_et) if LINEUPEXPERTS else []
    news_boosts = build_news_boost_map(news_items) if news_items else {}

    if LINEUPEXPERTS:
        print(f"[INFO] LineupExperts news_items={len(news_items)} boosts={len(news_boosts)} lookback={NEWS_LOOKBACK_HOURS}h base_url={LINEUPEXPERTS_BASE_URL}")

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
                    now_ts=now_ts,
                    news_boosts=news_boosts
                )
                if ideas:
                    got_any = True
                    injury_ideas_all.extend(ideas)

            if got_any:
                triggers.append(f"{injured_name} ({team_short}) {injured_status}")

    slate_ideas_all = []
    if ENABLE_SLATE_SCAN and (not deadline_exceeded()):
        for pt in PROP_TYPES:
            if deadline_exceeded():
                break
            slate_ideas_all.extend(
                slate_scan_edges(now_et, pt, lines_map.get(pt, {}), state=state, now_ts=now_ts, news_boosts=news_boosts)
            )

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

    final_out = []
    for pt in PROP_TYPES:
        final_out.extend(out_by_market.get(pt, []))
    final_out = final_out[:MAX_TOTAL_PLAYS]

    # Apply exposure caps after ranking, before messaging
    final_out = apply_exposure_caps(final_out)

    capped_by_market = {pt: [] for pt in PROP_TYPES}
    for it in final_out:
        capped_by_market[it["prop_type"]].append(it)
    out_by_market = capped_by_market

    plus_bucket = []
    for x in final_out:
        try:
            if float(x.get("over_odds", -999)) >= PLUS_ODDS_MIN:
                plus_bucket.append(x)
        except Exception:
            pass
    plus_bucket.sort(key=lambda x: (x["ev"], x["value_edge"], x["prob_over"]), reverse=True)
    plus_bucket = plus_bucket[:PLUS_ODDS_TOPN]

    if final_out:
        msg = [f"💰 FanDuel Props ({ts_et})", ""]

        if LINEUPEXPERTS:
            msg.append(f"📰 News signals: items={len(news_items)}, boosted_players={len(news_boosts)} (lookback {NEWS_LOOKBACK_HOURS}h)")
            msg.append("")

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
                    team_tag = f"  [{i.get('team','')}]".rstrip() if i.get("team") else ""
                    msg.append(
                        f"• {i['player_name']} OVER {i['cons_line']:.1f}  "
                        f"(edge +{i['edge']:.1f}, P≈{i['prob_over']*100:.0f}%, EV≈{i['ev']:+.2f}/$1){fire}{team_tag}"
                    )
                    msg.append(f"  Trigger: {i['trigger']}")
                    msg.append(f"  Why: {i['why']}")
                    msg.append("")

            if slt:
                msg.append("🌎 League-Wide Slate Scan (no injury required):")
                msg.append("")
                for i in slt:
                    fire = " 🔥" if i.get("ev", 0) >= 0.25 else ""
                    team_tag = f"  [{i.get('team','')}]".rstrip() if i.get("team") else ""
                    msg.append(
                        f"• {i['player_name']} OVER {i['cons_line']:.1f}  "
                        f"(edge +{i['edge']:.1f}, P≈{i['prob_over']*100:.0f}%, EV≈{i['ev']:+.2f}/$1){fire}{team_tag}"
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

        msg.append(f"🧢 Exposure caps applied: team≤{MAX_PLAYS_PER_TEAM}, game≤{MAX_PLAYS_PER_GAME}")
        send_chunked("\n".join(msg).strip())

        record_sent(state, final_out, now_ts)

    else:
        print("[INFO] No plays cleared thresholds this run.")

    state["players"] = new_players
    save_state(state)

if __name__ == "__main__":
    run()
