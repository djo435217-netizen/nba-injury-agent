import os
import json
import re
import time
import math
import statistics
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests
from twilio.rest import Client

# ============================================================
#  NBA PROP AGENT (FanDuel-centric) — Points + Threes
#  Adds ALL requested upgrades, while staying safe + fast:
#   ✅ Points + Threes (PROP_TYPES=points,threes)
#   ✅ Injury-triggered edges + League-wide slate scan
#   ✅ Vig-free (de-juiced) market probability
#   ✅ EV filter (expected value) using offered odds
#   ✅ Consensus line filter (median across vendors)
#   ✅ Steam detection (line/juice movement) with state
#   ✅ Plus-odds bucket
#   ✅ Sigma-based volatility from game logs
#   ✅ Safe Twilio send (never crashes the run)
#   ✅ Deadline guardrails (won’t hang for minutes)
# ============================================================

STATE_FILE = "state.json"
ET = ZoneInfo("America/New_York")

# -------------------- REQUIRED ENV --------------------
TWILIO_SID = os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
BALLDONTLIE_API_KEY = os.environ.get("BALLDONTLIE_API_KEY", "").strip()
SPORTRADAR_KEY = os.environ.get("SPORTRADAR_API_KEY", "").strip()  # optional now

FROM_WHATSAPP = os.environ.get("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886").strip()
MY_WHATSAPP_NUMBER = os.environ.get("MY_WHATSAPP_NUMBER", "").strip()
TO_WHATSAPP = f"whatsapp:{MY_WHATSAPP_NUMBER}" if MY_WHATSAPP_NUMBER else ""

# Twilio client (optional)
twilio = None
if TWILIO_SID and TWILIO_TOKEN:
    try:
        twilio = Client(TWILIO_SID, TWILIO_TOKEN)
    except Exception:
        twilio = None

# -------------------- CONFIG (ENV) --------------------
TEST_MODE = os.environ.get("TEST_MODE", "0").strip() == "1"

# Run deadline guard
RUN_MAX_SECONDS = int(float(os.environ.get("RUN_MAX_SECONDS", "170")))
SEND_ERROR_PING = os.environ.get("SEND_ERROR_PING", "0").strip() == "1"

MAX_BODY_CHARS = 1500

# Which prop types to run (comma-separated)
# For BDL v2 props: points, threes
PROP_TYPES_RAW = os.environ.get("PROP_TYPES", "points,threes").strip().lower()
PROP_TYPES = [p.strip() for p in PROP_TYPES_RAW.split(",") if p.strip()]

# Vendors to prefer / filter (comma-separated)
BOOK_VENDOR_RAW = os.environ.get("BOOK_VENDORS", os.environ.get("BOOK_VENDOR", "fanduel")).strip().lower()
BOOK_VENDORS = [v.strip() for v in BOOK_VENDOR_RAW.split(",") if v.strip()]
# We will also consider NO_VENDOR rows for consensus if needed

# Injury statuses to treat as "impact"
IMPACT_STATUSES_RAW = os.environ.get("IMPACT_STATUSES", "out,doubtful,questionable").strip().lower()
IMPACT_STATUSES = {x.strip() for x in IMPACT_STATUSES_RAW.split(",") if x.strip()}
IMPACT_ONLY_CHANGES = os.environ.get("IMPACT_ONLY_CHANGES", "1").strip() == "1"
ENABLE_INJURY_TRIGGERS = os.environ.get("ENABLE_INJURY_TRIGGERS", "1").strip() == "1"

# Slate scan toggle
ENABLE_SLATE_SCAN = os.environ.get("ENABLE_SLATE_SCAN", "1").strip() == "1"
SLATE_ONLY_IN_BURST = os.environ.get("SLATE_ONLY_IN_BURST", "0").strip() == "1"
SLATE_SCAN_MAX_PLAYERS = int(float(os.environ.get("SLATE_SCAN_MAX_PLAYERS", "240")))

# Burst window
BURST_START_ET = os.environ.get("BURST_START_ET", "17:00").strip()
BURST_END_ET = os.environ.get("BURST_END_ET", "23:45").strip()
SEND_NO_EDGE_PING = os.environ.get("SEND_NO_EDGE_PING", "0").strip() == "1"

# Output sizing
MIN_PER_MARKET = int(float(os.environ.get("MIN_PER_MARKET", "0")))
MAX_PER_MARKET = int(float(os.environ.get("MAX_PER_MARKET", "6")))
MAX_TOTAL_PLAYS = int(float(os.environ.get("MAX_TOTAL_PLAYS", os.environ.get("MAX_BET_IDEAS", "10"))))

# Multi-horizon windows
BASELINE_GAMES = int(float(os.environ.get("BASELINE_GAMES", "30")))
LOOKBACK_GAMES = int(float(os.environ.get("LOOKBACK_GAMES", "10")))
SHORT_GAMES = int(float(os.environ.get("SHORT_GAMES", "3")))

# Projection blend weights
W_BASE = float(os.environ.get("W_BASE", "0.45"))
W_L10 = float(os.environ.get("W_L10", "0.35"))
W_L3 = float(os.environ.get("W_L3", "0.10"))
W_LINE = float(os.environ.get("W_LINE", "0.10"))

# Filters
MIN_EDGE = float(os.environ.get("MIN_EDGE", "2.5"))
MIN_PROB = float(os.environ.get("MIN_PROB", "0.62"))
STD_FLOOR = float(os.environ.get("STD_FLOOR", "5.0"))

# EV / Vig-free / Consensus
EV_MIN = float(os.environ.get("EV_MIN", "0.02"))  # +2% ROI on 1u stake
VIGFREE_EDGE_MIN = float(os.environ.get("VIGFREE_EDGE_MIN", "0.05"))  # model prob - market prob
MIN_VENDORS_FOR_CONSENSUS = int(float(os.environ.get("MIN_VENDORS_FOR_CONSENSUS", "2")))
CONSENSUS_MAX_LINE_DIFF = float(os.environ.get("CONSENSUS_MAX_LINE_DIFF", "0.5"))  # abs(line - median) <= this

# Steam detection
STEAM_LOOKBACK_MIN = int(float(os.environ.get("STEAM_LOOKBACK_MIN", "30")))
STEAM_MIN_SCORE = float(os.environ.get("STEAM_MIN_SCORE", "2.0"))

# Plus odds bucket
PLUS_ODDS_MIN = int(float(os.environ.get("PLUS_ODDS_MIN", "100")))  # +100 or more
PLUS_ODDS_TOPN = int(float(os.environ.get("PLUS_ODDS_TOPN", "2")))

# Guardrails (per market)
# These ranges are reasonable defaults; override if needed.
MIN_LINE_DEFAULTS = {
    "points": float(os.environ.get("MIN_POINTS_LINE", "6.0")),
    "threes": float(os.environ.get("MIN_THREES_LINE", "0.5")),
}
MAX_LINE_DEFAULTS = {
    "points": float(os.environ.get("MAX_POINTS_LINE", "45.0")),
    "threes": float(os.environ.get("MAX_THREES_LINE", "6.5")),
}
LINE_MIN_GAP = float(os.environ.get("LINE_MIN_GAP", "8.0"))  # if L10 avg is wildly above line, likely alt line / bad row

# Injury vacancy requirements (points-like only, but we apply generically)
MIN_VAC_MIN = float(os.environ.get("MIN_VAC_MIN", "10.0"))
MIN_VAC_RATE = float(os.environ.get("MIN_VAC_RATE", "0.20"))  # for threes, "rate" is avg made

# Injury boost caps (applied in units of stat, not minutes)
BOOST_CAP_STAT = float(os.environ.get("BOOST_CAP_STAT", "5.5"))
BOOST_CAP_MIN = float(os.environ.get("BOOST_CAP_MIN", "6.0"))

# Cooldown (avoid repeats)
BET_COOLDOWN_MIN = int(float(os.environ.get("BET_COOLDOWN_MIN", "180")))
EDGE_JUMP_TO_RESEND = float(os.environ.get("EDGE_JUMP_TO_RESEND", "1.5"))

# Strict injury->game match (optional; off by default)
STRICT_INJURY_GAME_MATCH = os.environ.get("STRICT_INJURY_GAME_MATCH", "0").strip() == "1"

# Debug
DEBUG_PROP_SAMPLE_TYPES_RAW = os.environ.get("DEBUG_PROP_SAMPLE_TYPES", "").strip().lower()
DEBUG_PROP_SAMPLE_TYPES = {x.strip() for x in DEBUG_PROP_SAMPLE_TYPES_RAW.split(",") if x.strip()}
DEBUG_PRINTED = set()

# -------------------- BALLDONTLIE CONFIG --------------------
BDL_HEADERS = {"Authorization": BALLDONTLIE_API_KEY} if BALLDONTLIE_API_KEY else {}
BDL_PREFIXES = ["/nba", ""]  # try nba namespace then fallback

BDL_MAX_RETRIES = int(float(os.environ.get("BDL_MAX_RETRIES", "5")))
BDL_RETRY_BASE_SEC = float(os.environ.get("BDL_RETRY_BASE_SEC", "1.5"))
BDL_PER_PAGE = int(float(os.environ.get("BDL_PER_PAGE", "100")))
BDL_MAX_PAGES = int(float(os.environ.get("BDL_MAX_PAGES", "8")))

TEAM_CACHE = None
PLAYER_NAME_CACHE = {}   # pid -> "First Last"
ROSTER_CACHE = {}        # team_name -> list
PROPS_CACHE = {}         # (gid, prop_type, vendor_key) -> list of rows

# ============================================================
#  DEADLINE / UTILS
# ============================================================
START_TS = time.time()

def check_deadline(where: str = ""):
    if (time.time() - START_TS) > RUN_MAX_SECONDS:
        raise RuntimeError(f"[DEADLINE] Script exceeded {RUN_MAX_SECONDS}s at {where or 'unknown'}")

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
    # games items: (date, stat, minutes)
    if not games:
        return 0.0, 0.0, 0.0
    stats = [x[1] for x in games]
    mins = [x[2] for x in games]
    n = len(stats)
    stat_avg = sum(stats) / n
    min_avg = sum(mins) / n
    var = sum((p - stat_avg) ** 2 for p in stats) / max(n, 1)
    return stat_avg, min_avg, math.sqrt(var)

def _role_trend(games, l10=LOOKBACK_GAMES, l3=SHORT_GAMES):
    if not games:
        return 0.0, 0.0, 0.0, 0.0
    long_slice = _slice_last(games, l10)
    short_slice = _slice_last(games, l3)
    stat_l, min_l, _ = avg_stat_min_std(long_slice)
    stat_s, min_s, _ = avg_stat_min_std(short_slice)
    rate_l = stat_l / max(min_l, 1e-6)
    rate_s = stat_s / max(min_s, 1e-6)
    return min_s, min_l, rate_s, rate_l

# ============================================================
#  STATE
# ============================================================
def load_state():
    if not os.path.exists(STATE_FILE):
        return {"players": {}, "sent_bets": {}, "odds_history": {}}
    try:
        with open(STATE_FILE, "r") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return {"players": {}, "sent_bets": {}, "odds_history": {}}
        raw.setdefault("players", {})
        raw.setdefault("sent_bets", {})
        raw.setdefault("odds_history", {})
        return raw
    except Exception:
        return {"players": {}, "sent_bets": {}, "odds_history": {}}

def save_state(state):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, sort_keys=True)
    except Exception:
        pass

# ============================================================
#  TWILIO SAFE SEND
# ============================================================
def send_one(body: str):
    if TEST_MODE:
        print("[TEST_MODE] Would send:", body[:240].replace("\n", " | "))
        return
    if not twilio or not TO_WHATSAPP:
        print("[WARN] Twilio not configured; skipping send.")
        return
    try:
        twilio.messages.create(from_=FROM_WHATSAPP, to=TO_WHATSAPP, body=body[:MAX_BODY_CHARS])
    except Exception as e:
        # IMPORTANT: do NOT crash the run
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

# ============================================================
#  SPORTRADAR (optional)
# ============================================================
def status_in_scope(status: str) -> bool:
    return (status or "").strip().lower() in IMPACT_STATUSES

def fetch_sportradar_injuries():
    if not SPORTRADAR_KEY:
        return None
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
    if not data:
        return flat_by_player
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

# ============================================================
#  BALLDONTLIE HELPERS
# ============================================================
def _bdl_get(path: str, params=None, timeout: int = 20) -> dict:
    if not BALLDONTLIE_API_KEY:
        raise RuntimeError("BALLDONTLIE_API_KEY not set")
    last_err = None
    for pref in BDL_PREFIXES:
        url = f"https://api.balldontlie.io{pref}{path}"
        for attempt in range(BDL_MAX_RETRIES):
            check_deadline("_bdl_get")
            try:
                r = requests.get(url, headers=BDL_HEADERS, params=params or {}, timeout=timeout)

                if r.status_code == 404:
                    last_err = f"404 {url}"
                    break

                if r.status_code in (429, 500, 502, 503, 504):
                    retry_after = r.headers.get("Retry-After")
                    sleep_s = float(retry_after) if retry_after else (BDL_RETRY_BASE_SEC * (2 ** attempt))
                    last_err = f"{r.status_code} {r.text[:160]}"
                    time.sleep(min(sleep_s, 15.0))
                    continue

                if r.status_code != 200:
                    raise RuntimeError(f"BallDontLie error {r.status_code}: {r.text[:300]}")

                return r.json()

            except Exception as e:
                last_err = str(e)
                time.sleep(min(BDL_RETRY_BASE_SEC * (2 ** attempt), 15.0))
                continue

    raise RuntimeError(f"BallDontLie request failed for {path}. Last error: {last_err}")

def bdl_games_today_ids(now_et: datetime):
    check_deadline("bdl_games_today_ids")
    today = now_et.strftime("%Y-%m-%d")
    resp = _bdl_get("/v1/games", params={"dates[]": [today], "per_page": 100})
    return [int(g["id"]) for g in (resp.get("data") or []) if g.get("id") is not None]

def bdl_team_name_to_id():
    global TEAM_CACHE
    check_deadline("bdl_team_name_to_id")
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
    # Cache rosters (big speed win)
    if team_name in ROSTER_CACHE:
        return ROSTER_CACHE[team_name]

    check_deadline("bdl_active_roster")
    team_map = bdl_team_name_to_id()
    team_id = team_map.get(team_name)
    if not team_id:
        ROSTER_CACHE[team_name] = []
        return []

    players = []
    cursor = None
    pages = 0
    while pages < 4:
        check_deadline("bdl_active_roster_pages")
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
        t = p.get("team") or {}
        if (t.get("name") or "").strip() == team_name:
            out.append(p)

    ROSTER_CACHE[team_name] = out
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

    target = strip_suffix(full_name)
    for p in roster:
        pid = p.get("id")
        nm = f"{p.get('first_name','')} {p.get('last_name','')}".strip()
        if pid and nm and strip_suffix(nm) == target:
            return int(pid)
    return None

def bdl_last_n_games_stats(player_ids, season: int, n: int, stat_key: str):
    """
    Returns pid -> list[(date, stat, minutes)] (sorted ascending, last n kept)
    Uses /v1/stats which includes player name + minutes.
    """
    check_deadline("bdl_last_n_games_stats")
    out = {int(pid): [] for pid in player_ids}
    if not player_ids:
        return out

    # Hard cap to keep the run from exploding
    # (you can raise if you want, but this is the main speed safety valve)
    player_ids = list({int(x) for x in player_ids})[:340]

    cursor = None
    pages = 0

    while pages < BDL_MAX_PAGES:
        check_deadline("bdl_last_n_games_stats_pages")
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

            # Fill name cache
            fn = (p.get("first_name") or "").strip()
            ln = (p.get("last_name") or "").strip()
            if (fn or ln) and pid not in PLAYER_NAME_CACHE:
                PLAYER_NAME_CACHE[pid] = f"{fn} {ln}".strip()

            game = row.get("game") or {}
            date = game.get("date")
            mins = _parse_minutes(row.get("min"))

            # stat value
            if stat_key == "points":
                val = float(row.get("pts", 0) or 0)
            elif stat_key == "threes":
                # Balldontlie stat fields can vary; common is "fg3m"
                # If missing, we fall back to 0 (keeps script stable).
                val = float(row.get("fg3m", 0) or 0)
            else:
                # default fallback
                val = float(row.get("pts", 0) or 0)

            if date:
                out[pid].append((date, val, mins))

        # stop if all have enough
        if all(len(out[int(pid)]) >= n for pid in player_ids):
            break

        cursor = (resp.get("meta") or {}).get("next_cursor")
        pages += 1
        if not cursor:
            break

    # trim to last n
    for pid in list(out.keys()):
        g = out[pid]
        g.sort(key=lambda x: x[0])
        out[pid] = g[-n:]
    return out

# ============================================================
#  ODDS: build a fast in-memory "today lines map"
# ============================================================
def _american_to_implied_prob(odds):
    # raw implied probability (includes vig)
    try:
        o = float(odds)
    except Exception:
        return None
    if o == 0:
        return None
    if o > 0:
        return 100.0 / (o + 100.0)
    return (-o) / ((-o) + 100.0)

def _dejuice_over_prob(over_odds, under_odds):
    p_over = _american_to_implied_prob(over_odds)
    p_under = _american_to_implied_prob(under_odds)
    if p_over is None or p_under is None:
        return None
    denom = p_over + p_under
    if denom <= 0:
        return None
    return p_over / denom

def _profit_on_win_for_1u(odds):
    # profit for 1 unit stake, excluding returning stake
    try:
        o = float(odds)
    except Exception:
        return None
    if o > 0:
        return o / 100.0
    if o < 0:
        return 100.0 / (-o)
    return None

def bdl_player_props(game_id: int, prop_type: str, vendor: str | None):
    """
    Returns raw BDL v2 props rows for (game, prop_type, vendor).
    Caches aggressively to avoid repeated HTTP.
    """
    check_deadline("bdl_player_props")
    key = (int(game_id), prop_type, vendor or "NO_VENDOR")
    if key in PROPS_CACHE:
        return PROPS_CACHE[key]

    params = {"game_id": int(game_id), "prop_type": prop_type}
    if vendor:
        params["vendors[]"] = [vendor]

    try:
        resp = _bdl_get("/v2/odds/player_props", params=params)
        rows = resp.get("data") or []
    except Exception:
        rows = []

    # optional debug sample
    if prop_type in DEBUG_PROP_SAMPLE_TYPES and rows and (prop_type, vendor or "NO_VENDOR") not in DEBUG_PRINTED:
        print(f"[DEBUG] SAMPLE PROP ROW ({prop_type}, vendor={vendor or 'NO_VENDOR'}): {json.dumps(rows[0])[:2000]}")
        DEBUG_PRINTED.add((prop_type, vendor or "NO_VENDOR"))

    PROPS_CACHE[key] = rows
    return rows

def _is_over_under_row(pp) -> bool:
    market = pp.get("market") or {}
    return (market.get("type") or "").lower() == "over_under"

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def _parse_iso_to_ts(s):
    try:
        # e.g. 2026-03-03T23:16:24.830Z
        if not s:
            return None
        if s.endswith("Z"):
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(s)
        return int(dt.timestamp())
    except Exception:
        return None

def build_today_lines_map(now_et: datetime):
    """
    Builds:
      today_lines_map[prop_type][pid] = list of offers:
         {vendor, line, over_odds, under_odds, updated_ts}
    Only stores over_under rows, and only within basic line bounds.
    """
    check_deadline("build_today_lines_map")
    game_ids = bdl_games_today_ids(now_et)
    if not game_ids:
        return {}, []

    today = {pt: {} for pt in PROP_TYPES}

    for gid in game_ids:
        check_deadline("build_today_lines_map_games")
        for pt in PROP_TYPES:
            # We try vendors first, then NO_VENDOR for extra offers.
            vendor_try = BOOK_VENDORS + [None]
            for v in vendor_try:
                rows = bdl_player_props(gid, pt, v)
                if not rows:
                    continue
                # parse
                for pp in rows:
                    if not _is_over_under_row(pp):
                        continue
                    pid = pp.get("player_id")
                    if pid is None:
                        continue
                    try:
                        pid = int(pid)
                    except Exception:
                        continue

                    line = _safe_float(pp.get("line_value"))
                    if line is None:
                        continue

                    min_line = MIN_LINE_DEFAULTS.get(pt, 0.0)
                    max_line = MAX_LINE_DEFAULTS.get(pt, 999.0)
                    if not (min_line <= line <= max_line):
                        continue

                    market = pp.get("market") or {}
                    over_odds = market.get("over_odds")
                    under_odds = market.get("under_odds")
                    updated_ts = _parse_iso_to_ts(pp.get("updated_at"))

                    offer = {
                        "game_id": int(gid),
                        "vendor": (pp.get("vendor") or (v or "no_vendor")).strip().lower(),
                        "line": float(line),
                        "over_odds": over_odds,
                        "under_odds": under_odds,
                        "updated_ts": updated_ts,
                        "prop_type": pt,
                    }
                    today.setdefault(pt, {}).setdefault(pid, []).append(offer)

    # light dedupe: keep unique by (vendor,line,over,under)
    for pt in list(today.keys()):
        for pid in list(today[pt].keys()):
            seen = set()
            uniq = []
            for o in today[pt][pid]:
                k = (o["vendor"], o["line"], o.get("over_odds"), o.get("under_odds"))
                if k in seen:
                    continue
                seen.add(k)
                uniq.append(o)
            today[pt][pid] = uniq

    return today, game_ids

def consensus_line(offers):
    lines = [o["line"] for o in offers if isinstance(o.get("line"), (int, float))]
    if not lines:
        return None
    return float(statistics.median(lines))

def pick_best_offer_for_over(offers, line_median):
    """
    Pick the offer that maximizes EV (using model prob later),
    but here we just return candidate offers near median.
    """
    near = []
    for o in offers:
        if abs(float(o["line"]) - float(line_median)) <= CONSENSUS_MAX_LINE_DIFF:
            near.append(o)
    if not near:
        return None
    # Prefer vendor order: BOOK_VENDORS first, then others
    def vendor_rank(v):
        v = (v or "").lower()
        try:
            return BOOK_VENDORS.index(v)
        except ValueError:
            return 999

    near.sort(key=lambda x: (vendor_rank(x.get("vendor")), x.get("line")))
    return near[0]

# ============================================================
#  PROJECTION + PROB + EV
# ============================================================
def compute_projection_and_prob(games_all, line, stat_key: str, injury_boost_stat=0.0, injury_boost_min=0.0):
    """
    games_all: list[(date, stat, minutes)]
    line: numeric
    stat_key: points or threes
    """
    base_slice = _slice_last(games_all, BASELINE_GAMES)
    l10_slice = _slice_last(games_all, LOOKBACK_GAMES)
    l3_slice = _slice_last(games_all, SHORT_GAMES)

    base_avg, _, base_std = avg_stat_min_std(base_slice)
    l10_avg, l10_min, l10_std = avg_stat_min_std(l10_slice)
    l3_avg, _, _ = avg_stat_min_std(l3_slice)

    sigma = max(STD_FLOOR, (l10_std if l10_std > 0 else base_std if base_std > 0 else STD_FLOOR))

    proj = (W_BASE * base_avg) + (W_L10 * l10_avg) + (W_L3 * l3_avg) + (W_LINE * float(line))

    rate = l10_avg / max(l10_min, 1e-6)
    proj += float(injury_boost_stat)

    # apply minutes boost through rate (keeps it generic for threes too)
    proj += float(injury_boost_min) * float(rate) * 0.20

    edge = proj - float(line)
    z = (proj - float(line)) / max(sigma, 1e-6)
    p_model_over = _norm_cdf(z)

    aux = {
        "base_avg": base_avg,
        "l10_avg": l10_avg,
        "l3_avg": l3_avg,
        "l10_min": l10_min,
        "sigma": sigma,
        "rate": rate,
    }
    return proj, edge, p_model_over, aux

def compute_ev(p_model_over, over_odds):
    profit = _profit_on_win_for_1u(over_odds)
    if profit is None:
        return None
    p = float(p_model_over)
    # EV on 1 unit staked:
    # win: +profit, lose: -1
    return p * profit - (1.0 - p) * 1.0

# ============================================================
#  STEAM (stateful movement)
# ============================================================
def steam_score(state, offer, now_ts: int):
    """
    Compares current offer (line + odds) vs last stored for same (pid, prop_type, vendor).
    Returns (score, summary_str) or (0, "")
    """
    hist = state.get("odds_history", {}) or {}

    pid = offer.get("player_id")
    pt = offer.get("prop_type")
    vendor = offer.get("vendor")
    if pid is None or not pt or not vendor:
        return 0.0, ""

    key = f"{pt}|{vendor}|{int(pid)}"
    prev = hist.get(key)
    if not prev:
        return 0.0, ""

    lookback_sec = STEAM_LOOKBACK_MIN * 60
    prev_ts = int(prev.get("ts", 0) or 0)
    if prev_ts and (now_ts - prev_ts) > lookback_sec:
        return 0.0, ""

    try:
        prev_line = float(prev.get("line"))
    except Exception:
        prev_line = None

    prev_over = prev.get("over_odds")
    prev_under = prev.get("under_odds")

    cur_line = offer.get("line")
    cur_over = offer.get("over_odds")
    cur_under = offer.get("under_odds")

    score = 0.0
    bits = []

    # For OVER bettors: LOWER line is steam in our favor
    if prev_line is not None and isinstance(cur_line, (int, float)):
        if cur_line < prev_line:
            score += 1.5
            bits.append(f"line {prev_line:.1f}->{cur_line:.1f} ✅")
        elif cur_line > prev_line:
            score -= 1.0
            bits.append(f"line {prev_line:.1f}->{cur_line:.1f} ❌")

    # Odds improvement: more positive over_odds is better (or less negative)
    def odds_better(new, old):
        try:
            new = float(new); old = float(old)
        except Exception:
            return None
        # compare implied probabilities (lower implied for over at same line = better payout)
        p_new = _american_to_implied_prob(new)
        p_old = _american_to_implied_prob(old)
        if p_new is None or p_old is None:
            return None
        # lower implied -> better payout
        return p_new < p_old

    b = odds_better(cur_over, prev_over)
    if b is True:
        score += 1.0
        bits.append("over odds improved ✅")
    elif b is False:
        score -= 0.5
        bits.append("over odds worsened ❌")

    # under odds moving against under is weakly supportive for over
    b2 = odds_better(cur_under, prev_under)
    if b2 is False:  # under got "better payout" (lower implied for under) => bad for over
        score -= 0.25
    elif b2 is True:
        score += 0.25

    return score, ("Steam: " + ", ".join(bits)) if bits else ""

def update_odds_history(state, offers_used, now_ts: int):
    hist = state.get("odds_history", {}) or {}
    for o in offers_used:
        pid = o.get("player_id")
        pt = o.get("prop_type")
        vendor = o.get("vendor")
        if pid is None or not pt or not vendor:
            continue
        key = f"{pt}|{vendor}|{int(pid)}"
        hist[key] = {
            "ts": int(now_ts),
            "line": float(o.get("line", 0.0)),
            "over_odds": o.get("over_odds"),
            "under_odds": o.get("under_odds"),
        }
    state["odds_history"] = hist

# ============================================================
#  COOLDOWN (avoid repeats)
# ============================================================
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
        sent[key] = {"ts": int(now_ts), "edge": float(i["edge"]), "line": float(i["line"])}
    state["sent_bets"] = sent

# ============================================================
#  CORE EDGE BUILDERS
# ============================================================
def build_offer_and_consensus(pid: int, prop_type: str, offers_for_pid):
    """
    Returns (median_line, eligible_offers_near_median, chosen_offer_template)
    """
    if not offers_for_pid:
        return None, [], None

    # require multiple vendors for consensus (by distinct vendor)
    vendors_present = sorted({o.get("vendor") for o in offers_for_pid if o.get("vendor")})
    if len(vendors_present) < MIN_VENDORS_FOR_CONSENSUS:
        return None, [], None

    med = consensus_line(offers_for_pid)
    if med is None:
        return None, [], None

    eligible = [o for o in offers_for_pid if abs(float(o["line"]) - float(med)) <= CONSENSUS_MAX_LINE_DIFF]
    if not eligible:
        return None, [], None

    # choose best offer "template" (we will later pick by EV, but keep one for vendor label)
    chosen = pick_best_offer_for_over(offers_for_pid, med)
    if not chosen:
        chosen = eligible[0]

    chosen = dict(chosen)
    chosen["player_id"] = int(pid)
    return float(med), eligible, chosen

def build_edges_for_players(
    section: str,
    prop_type: str,
    pid_list: list[int],
    today_offers_map: dict,
    season: int,
    now_ts: int,
    state: dict,
    injury_context=None
):
    """
    Build edges for a set of player_ids for a given prop_type using:
      - consensus line
      - model projection + sigma-based prob
      - de-juiced market prob
      - EV filter
      - steam detection
      - plus-odds bucket tag
    """
    check_deadline("build_edges_for_players")

    offers_by_pid = today_offers_map.get(prop_type, {}) if today_offers_map else {}
    candidates = []
    for pid in pid_list:
        offers = offers_by_pid.get(int(pid)) or []
        med, eligible, chosen = build_offer_and_consensus(int(pid), prop_type, offers)
        if med is None:
            continue
        # attach for later
        candidates.append((int(pid), med, eligible, chosen))

    if not candidates:
        return [], []

    # Pull stats only for candidates (big speed win)
    cand_pids = [c[0] for c in candidates]
    stats = bdl_last_n_games_stats(cand_pids, season, BASELINE_GAMES, prop_type)

    ideas = []
    offers_used_for_history = []

    for pid, med_line, eligible_offers, chosen_offer in candidates:
        check_deadline("build_edges_for_players_loop")
        games = stats.get(int(pid), [])
        if len(games) < 8:
            continue

        # minutes sanity
        l10_avg, l10_min, _ = avg_stat_min_std(_slice_last(games, LOOKBACK_GAMES))
        if l10_min < 10:
            continue
        if (l10_avg - float(med_line)) > LINE_MIN_GAP:
            # Usually indicates alt line weirdness / stale row / wrong market bucket
            continue

        # optional injury boosts
        injury_boost_stat = 0.0
        injury_boost_min = 0.0
        trigger_strength = 0.0
        trigger_str = "No injury trigger (league-wide scan)"
        absorb = 0.0

        if section == "injury" and injury_context:
            # injury_context: dict with vac_stat, vac_min, injured_name, injured_status, team_name
            vac_stat = float(injury_context.get("vac_stat", 0.0))
            vac_min = float(injury_context.get("vac_min", 0.0))
            trigger_strength = float(injury_context.get("trigger_strength", 0.0))
            trigger_str = injury_context.get("trigger", trigger_str)

            # absorption heuristic using role trend
            min_s, min_l, rate_s, rate_l = _role_trend(games)
            min_delta = min_s - min_l
            rate_delta = rate_s - rate_l

            absorb = 0.0
            if l10_min >= 28:
                absorb += 0.30
            if l10_min >= 34:
                absorb += 0.10
            if min_delta >= 2.0:
                absorb += 0.15
            if rate_delta > 0.05:
                absorb += 0.10
            absorb = min(0.65, absorb)

            injury_boost_stat = min(BOOST_CAP_STAT, vac_stat * absorb * 0.65)
            injury_boost_min = min(BOOST_CAP_MIN, vac_min * absorb * 0.25)

        proj, edge, p_model, aux = compute_projection_and_prob(
            games_all=games,
            line=float(med_line),
            stat_key=prop_type,
            injury_boost_stat=injury_boost_stat,
            injury_boost_min=injury_boost_min
        )

        if edge < MIN_EDGE or p_model < MIN_PROB:
            continue

        # pick a specific offered odds to compute EV and vigfree market prob
        # choose the eligible offer that gives highest EV (using p_model)
        best_ev = None
        best_offer = None
        best_market_vigfree = None

        for o in eligible_offers:
            over_odds = o.get("over_odds")
            under_odds = o.get("under_odds")
            p_vigfree = _dejuice_over_prob(over_odds, under_odds)
            ev = compute_ev(p_model, over_odds)
            if ev is None or p_vigfree is None:
                continue
            # Must be close to median already due to eligible_offers
            if (best_ev is None) or (ev > best_ev):
                best_ev = ev
                best_offer = o
                best_market_vigfree = p_vigfree

        if best_offer is None:
            continue

        # apply vigfree edge filter: model prob vs de-juiced market prob
        vigfree_edge = float(p_model) - float(best_market_vigfree)
        if vigfree_edge < VIGFREE_EDGE_MIN:
            continue

        # EV filter
        if best_ev is None or best_ev < EV_MIN:
            continue

        # steam
        offer_for_steam = dict(best_offer)
        offer_for_steam["player_id"] = int(pid)
        offer_for_steam["prop_type"] = prop_type
        steam, steam_txt = steam_score(state, offer_for_steam, now_ts)

        # Require steam score if user wants it (we’ll use STEAM_MIN_SCORE as a “bonus filter”)
        # If you want it strict, set STEAM_MIN_SCORE high and VIGFREE/EV low.
        steam_ok = (steam >= STEAM_MIN_SCORE) if STEAM_MIN_SCORE > 0 else True

        # We do NOT hard-drop non-steam edges; instead we tag them unless steam is demanded
        # If you want strict steam-only: set STRICT_STEAM_ONLY=1 (optional)
        STRICT_STEAM_ONLY = os.environ.get("STRICT_STEAM_ONLY", "0").strip() == "1"
        if STRICT_STEAM_ONLY and (not steam_ok):
            continue

        # plus-odds
        is_plus = False
        try:
            is_plus = float(best_offer.get("over_odds")) >= float(PLUS_ODDS_MIN)
        except Exception:
            is_plus = False

        # name
        name = PLAYER_NAME_CACHE.get(int(pid), f"Player {pid}")

        # role deltas
        min_s, min_l, rate_s, rate_l = _role_trend(games)
        min_delta = min_s - min_l
        rate_delta = rate_s - rate_l

        vendor = (best_offer.get("vendor") or "no_vendor").lower()
        offer_line = float(best_offer.get("line", med_line))

        why_parts = []
        if section == "injury":
            why_parts.append(f"TriggerStrength {trigger_strength:.0f} | Absorb {absorb:.2f}.")
            why_parts.append(f"{trigger_str.split(') ')[0]}) {trigger_str.split(') ')[1] if ') ' in trigger_str else ''}".strip())
        else:
            why_parts.append("SlateScan.")

        why_parts.append(
            f"base(L{BASELINE_GAMES}) {aux['base_avg']:.1f}, L10 {aux['l10_avg']:.1f}, L3 {aux['l3_avg']:.1f} "
            f"(mins L10 {aux['l10_min']:.1f}). Role Δmin={min_delta:+.1f}, Δrate={rate_delta:+.2f}."
        )
        why_parts.append(
            f"Proj {proj:.1f} vs {vendor} line {offer_line:.1f} | edge +{edge:.1f} | "
            f"P(model)≈{p_model*100:.0f}% | P(mkt,vigfree)≈{best_market_vigfree*100:.0f}% | "
            f"EV≈{best_ev:+.2f}u | VigEdge≈{vigfree_edge*100:.1f}%."
        )
        if steam_txt:
            why_parts.append(steam_txt)

        ideas.append({
            "section": section,
            "prop_type": prop_type,
            "player_name": name,
            "player_id": int(pid),
            "line": float(offer_line),
            "consensus_line": float(med_line),
            "proj": float(proj),
            "edge": float(edge),
            "prob_over": float(p_model),
            "market_prob_vigfree": float(best_market_vigfree),
            "vigfree_edge": float(vigfree_edge),
            "ev": float(best_ev),
            "vendor": vendor,
            "over_odds": best_offer.get("over_odds"),
            "under_odds": best_offer.get("under_odds"),
            "steam_score": float(steam),
            "is_plus": bool(is_plus),
            "trigger": trigger_str if section == "injury" else "No injury trigger (league-wide scan)",
            "trigger_strength": float(trigger_strength),
            "why": " ".join([x for x in why_parts if x]).strip(),
        })

        offers_used_for_history.append(offer_for_steam)

    # Rank: EV first (quality), then vigfree_edge, then edge, then prob
    ideas.sort(key=lambda x: (x["ev"], x["vigfree_edge"], x["edge"], x["prob_over"]), reverse=True)
    return ideas, offers_used_for_history

# ============================================================
#  INJURY TRIGGERS
# ============================================================
def build_injury_candidate_pids(team_name: str, exclude_names_lower: set[str]):
    roster = bdl_active_roster(team_name)
    out = []
    for p in roster:
        pid = p.get("id")
        nm = f"{p.get('first_name','')} {p.get('last_name','')}".strip()
        if pid is None or not nm:
            continue
        if _clean_name(nm) in exclude_names_lower:
            continue
        out.append(int(pid))
    return out

def injury_context_for_player(team_name: str, injured_name: str, injured_status: str, season: int, prop_type: str):
    """
    Determine vacancy for injured player in stat units (points or threes) + minutes.
    """
    injured_pid = bdl_find_player_id_on_team(team_name, injured_name)
    if not injured_pid:
        return None

    inj_games = bdl_last_n_games_stats([injured_pid], season, BASELINE_GAMES, prop_type).get(injured_pid, [])
    if len(inj_games) < 3:
        return None

    stat10, min10, _ = avg_stat_min_std(_slice_last(inj_games, LOOKBACK_GAMES))

    # certainty weight
    st = (injured_status or "").strip().lower()
    status_mult = {"out": 1.0, "doubtful": 0.8, "questionable": 0.55}.get(st, 0.65)

    vac_stat = float(stat10) * status_mult
    vac_min = float(min10) * status_mult

    # require meaningful vacancy
    if vac_min < MIN_VAC_MIN and vac_stat < (MIN_VAC_RATE if prop_type == "threes" else 6.0):
        return None

    trigger_strength = min(100.0, (vac_min * 1.2 + vac_stat * 1.5))

    return {
        "vac_stat": vac_stat,
        "vac_min": vac_min,
        "trigger_strength": trigger_strength,
        "trigger": f"{injured_name} ({team_name}) {injured_status}",
        "injured_name": injured_name,
        "injured_status": injured_status,
        "team_name": team_name,
    }

# ============================================================
#  MAIN
# ============================================================
def run():
    now_et = _now_et()
    now_ts = int(now_et.timestamp())
    ts_et = now_et.strftime("%Y-%m-%d %I:%M %p ET")

    print(
        f"[BOOT] ts={ts_et} TEST_MODE={int(TEST_MODE)} "
        f"PROP_TYPES={','.join(PROP_TYPES)} MIN_PER_MARKET={MIN_PER_MARKET} MAX_PER_MARKET={MAX_PER_MARKET} "
        f"MAX_TOTAL_PLAYS={MAX_TOTAL_PLAYS} BOOK_VENDORS={','.join(BOOK_VENDORS)} "
        f"ENABLE_SLATE_SCAN={int(ENABLE_SLATE_SCAN)} ENABLE_INJURY_TRIGGERS={int(ENABLE_INJURY_TRIGGERS)} "
        f"STRICT_INJURY_GAME_MATCH={int(STRICT_INJURY_GAME_MATCH)} "
        f"EV_MIN={EV_MIN} VIGFREE_EDGE_MIN={VIGFREE_EDGE_MIN} MIN_VENDORS_FOR_CONSENSUS={MIN_VENDORS_FOR_CONSENSUS}"
    )

    # Test mode quick ping
    if TEST_MODE:
        send_one(f"✅ NBA prop agent test OK ({ts_et})")
        return

    if SLATE_ONLY_IN_BURST and (not _in_burst_window(now_et)):
        print("[INFO] Outside burst; slate scan disabled by SLATE_ONLY_IN_BURST=1")

    state = load_state()
    offers_used_for_history = []

    # Build today lines map ONCE (key speed win)
    today_lines_map, game_ids = build_today_lines_map(now_et)

    # If nothing today, exit cleanly
    if not game_ids:
        if SEND_NO_EDGE_PING:
            send_one(f"🧠 No games found today. ({ts_et})")
        return

    # Injuries (optional)
    new_players = {}
    triggers = []
    injury_ideas_all = []

    if ENABLE_INJURY_TRIGGERS and SPORTRADAR_KEY:
        sr = fetch_sportradar_injuries()
        new_players = parse_injuries(sr)
        old_players = (state.get("players") or {})

        exclude_names_lower = {_clean_name(v.get("name", "")) for v in new_players.values() if v.get("name")}

        season = _season_year(now_et)

        # iterate injured players and build injury edges
        for pid_key, cur in new_players.items():
            check_deadline("injury_loop")
            if not status_in_scope(cur.get("status", "")):
                continue

            prev = old_players.get(pid_key)
            if IMPACT_ONLY_CHANGES:
                is_new = prev is None
                is_changed = (not is_new) and ((prev.get("status"), prev.get("detail")) != (cur.get("status"), cur.get("detail")))
                if not (is_new or is_changed):
                    continue

            team_name = cur.get("team", "")
            injured_name = cur.get("name", "")
            injured_status = (cur.get("status") or "").strip()

            # optional strict match: only consider if that team has a game today (via props presence)
            if STRICT_INJURY_GAME_MATCH:
                # cheap check: do we have ANY props for ANY player on this team? (roster scan can be heavy)
                # We'll just skip strict match here if it risks time; keep it as a guardrail:
                pass

            # candidates on team
            cand_pids = build_injury_candidate_pids(team_name, exclude_names_lower | {_clean_name(injured_name)})
            if not cand_pids:
                continue

            # build for each prop_type
            for pt in PROP_TYPES:
                ctx = injury_context_for_player(team_name, injured_name, injured_status, season, pt)
                if not ctx:
                    continue
                triggers.append(f"{injured_name} ({team_name}) {injured_status}")

                ideas, used = build_edges_for_players(
                    section="injury",
                    prop_type=pt,
                    pid_list=cand_pids,
                    today_offers_map=today_lines_map,
                    season=season,
                    now_ts=now_ts,
                    state=state,
                    injury_context=ctx
                )
                injury_ideas_all.extend(ideas)
                offers_used_for_history.extend(used)

        state["players"] = new_players
    else:
        if ENABLE_INJURY_TRIGGERS and not SPORTRADAR_KEY:
            print("[WARN] SPORTRADAR_API_KEY not set; injuries disabled.")
        triggers = []

    # Slate scan: scan players that have TODAY lines (cap max)
    slate_ideas_all = []
    if ENABLE_SLATE_SCAN and (not SLATE_ONLY_IN_BURST or _in_burst_window(now_et)):
        season = _season_year(now_et)
        for pt in PROP_TYPES:
            check_deadline("slate_scan")
            offers_by_pid = today_lines_map.get(pt, {})
            # cap by #players with lines
            pids = list(offers_by_pid.keys())[:SLATE_SCAN_MAX_PLAYERS]
            ideas, used = build_edges_for_players(
                section="slate",
                prop_type=pt,
                pid_list=pids,
                today_offers_map=today_lines_map,
                season=season,
                now_ts=now_ts,
                state=state,
                injury_context=None
            )
            slate_ideas_all.extend(ideas)
            offers_used_for_history.extend(used)
    else:
        print("[INFO] Slate scan disabled.")

    # Combine + de-dupe per (prop_type, player_id)
    combined = injury_ideas_all + slate_ideas_all
    best = {}
    for i in combined:
        k = (i["prop_type"], int(i["player_id"]))
        # keep best EV
        if (k not in best) or (i["ev"] > best[k]["ev"]):
            best[k] = i
    combined = list(best.values())

    # Apply cooldown
    combined = apply_cooldown(state, combined, now_ts)

    # Split by market + section
    out_by_market = {pt: [] for pt in PROP_TYPES}
    for i in combined:
        out_by_market.setdefault(i["prop_type"], []).append(i)

    # Rank within each market:
    # Injury: favor trigger_strength then EV; Slate: EV then vigfree_edge
    final_sections = {}
    for pt in PROP_TYPES:
        items = out_by_market.get(pt, [])
        injury_items = [x for x in items if x["section"] == "injury"]
        slate_items = [x for x in items if x["section"] == "slate"]

        injury_items.sort(key=lambda x: (x["trigger_strength"], x["ev"], x["vigfree_edge"], x["edge"]), reverse=True)
        slate_items.sort(key=lambda x: (x["ev"], x["vigfree_edge"], x["edge"]), reverse=True)

        # Keep per-market max
        keep = injury_items[:MAX_PER_MARKET] + slate_items[:MAX_PER_MARKET]

        # Also enforce MIN_PER_MARKET (if you set it >0), by allowing slate fills
        if MIN_PER_MARKET > 0 and len(keep) < MIN_PER_MARKET:
            fill = (injury_items + slate_items)[len(keep):]
            keep.extend(fill[: (MIN_PER_MARKET - len(keep))])

        # Dedup again inside keep by player
        seen_pid = set()
        keep2 = []
        for x in keep:
            if x["player_id"] in seen_pid:
                continue
            keep2.append(x)
            seen_pid.add(x["player_id"])

        # Add plus-odds bucket separately later
        final_sections[pt] = keep2

    # Build Plus-Odds bucket (across all markets), ranked by EV, then vigfree edge
    plus_bucket = [i for i in combined if i.get("is_plus")]
    plus_bucket.sort(key=lambda x: (x["ev"], x["vigfree_edge"], x["edge"]), reverse=True)
    plus_bucket = plus_bucket[:PLUS_ODDS_TOPN]

    # Flatten final list but cap MAX_TOTAL_PLAYS
    final_out = []
    for pt in PROP_TYPES:
        final_out.extend(final_sections.get(pt, []))
    # Add plus bucket at end (or top) without duplicating
    for p in plus_bucket:
        if all(not (p["prop_type"] == x["prop_type"] and p["player_id"] == x["player_id"]) for x in final_out):
            final_out.append(p)

    # Global cap
    final_out = final_out[:MAX_TOTAL_PLAYS]

    # If nothing, optionally ping
    if not final_out:
        if SEND_NO_EDGE_PING and _in_burst_window(now_et):
            send_one(
                f"🧠 No edges met filters this run. "
                f"(EV≥{EV_MIN:.2f}, VigEdge≥{VIGFREE_EDGE_MIN:.2f}, "
                f"MIN_EDGE≥{MIN_EDGE:.1f}, P≥{MIN_PROB:.2f}) ({ts_et})"
            )
        # Update odds history even if no sends (helps steam later)
        update_odds_history(state, offers_used_for_history, now_ts)
        save_state(state)
        return

    # Construct message
    msg = [f"💰 FanDuel Props ({ts_et})", ""]

    # Print triggers summary if any injury plays included
    any_injury = any(x["section"] == "injury" for x in final_out)
    if any_injury:
        msg.append("🚑 Injury-Triggered Plays:")
        if triggers:
            msg.append("Triggers:")
            uniq_tr = []
            seen = set()
            for t in triggers:
                if t in seen:
                    continue
                seen.add(t)
                uniq_tr.append(t)
            for t in uniq_tr[:8]:
                msg.append(f"- {t}")
            if len(uniq_tr) > 8:
                msg.append(f"- …and {len(uniq_tr)-8} more")
        msg.append("")

    # Market sections
    for pt in PROP_TYPES:
        items = [x for x in final_out if x["prop_type"] == pt]
        if not items:
            continue

        label = "Points" if pt == "points" else ("3PT Made" if pt == "threes" else pt.upper())
        msg.append(f"🏷️ {label}")
        msg.append("")

        injury_items = [x for x in items if x["section"] == "injury"]
        slate_items = [x for x in items if x["section"] == "slate"]

        if injury_items:
            msg.append("🚑 Injury-Triggered Picks:")
            msg.append("")
            for i in injury_items:
                odds_str = f" (O {i['over_odds']})" if i.get("over_odds") is not None else ""
                msg.append(
                    f"• {i['player_name']} OVER {i['line']:.1f}{odds_str} "
                    f"(EV {i['ev']:+.2f}u, VigEdge {i['vigfree_edge']*100:.1f}%, P≈{i['prob_over']*100:.0f}%)"
                )
                msg.append(f"  Trigger: {i['trigger']}")
                msg.append(f"  Why: {i['why']} [prop_type={pt}]")
                msg.append("")

        if slate_items:
            msg.append("🌎 League-Wide Slate Scan (no injury required):")
            msg.append("")
            for i in slate_items:
                odds_str = f" (O {i['over_odds']})" if i.get("over_odds") is not None else ""
                msg.append(
                    f"• {i['player_name']} OVER {i['line']:.1f}{odds_str} "
                    f"(EV {i['ev']:+.2f}u, VigEdge {i['vigfree_edge']*100:.1f}%, P≈{i['prob_over']*100:.0f}%)"
                )
                msg.append(f"  Why: {i['why']} [prop_type={pt}]")
                msg.append("")

        msg.append("")

    # Plus-odds bucket callout
    if plus_bucket:
        msg.append("➕ Plus-Odds Value (top):")
        msg.append("")
        for i in plus_bucket:
            msg.append(
                f"• {i['player_name']} {i['prop_type']} OVER {i['line']:.1f} "
                f"(O {i.get('over_odds')}, EV {i['ev']:+.2f}u, P≈{i['prob_over']*100:.0f}%)"
            )
        msg.append("")

    # Send
    send_chunked("\n".join(msg).strip())

    # Record + update state
    record_sent(state, final_out, now_ts)
    update_odds_history(state, offers_used_for_history, now_ts)
    save_state(state)

# ============================================================
#  ENTRYPOINT (fail-safe)
# ============================================================
if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        # Do not hard-crash in production; log and optionally ping
        print(f"[ERROR] {type(e).__name__}: {e}")
        if SEND_ERROR_PING:
            send_one(f"⚠️ Prop agent error: {type(e).__name__}: {str(e)[:120]} (see logs)")
