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

# Consensus + Steam + EV + Plus odds
MIN_VENDORS_FOR_CONSENSUS = int(os.environ.get("MIN_VENDORS_FOR_CONSENSUS", "1"))
MIN_SHARP_VENDORS = int(os.environ.get("MIN_SHARP_VENDORS", "0"))

ENABLE_STEAM = os.environ.get("ENABLE_STEAM", "0") == "1"
STEAM_MIN_SCORE = float(os.environ.get("STEAM_MIN_SCORE", "1.0"))  # higher = stricter
STEAM_MAX_AGE_MIN = int(os.environ.get("STEAM_MAX_AGE_MIN", "240"))

# Output sizing
MIN_PER_MARKET = int(os.environ.get("MIN_PER_MARKET", "0"))
MAX_PER_MARKET = int(os.environ.get("MAX_PER_MARKET", "6"))
MAX_TOTAL_PLAYS = int(os.environ.get("MAX_TOTAL_PLAYS", "10"))

# Exposure caps (quick win)
MAX_PLAYS_PER_TEAM = int(os.environ.get("MAX_PLAYS_PER_TEAM", "2"))
MAX_PLAYS_PER_GAME = int(os.environ.get("MAX_PLAYS_PER_GAME", "2"))

# Windows
BASELINE_GAMES = int(os.environ.get("BASELINE_GAMES", "30"))
LOOKBACK_GAMES = int(os.environ.get("LOOKBACK_GAMES", "10"))
SHORT_GAMES = int(os.environ.get("SHORT_GAMES", "3"))

# Model thresholds
MIN_EDGE = float(os.environ.get("MIN_EDGE", "2.5"))
MIN_PROB = float(os.environ.get("MIN_PROB", "0.62"))
STD_FLOOR = float(os.environ.get("STD_FLOOR", "5.0"))

# EV filter
EV_MIN = float(os.environ.get("EV_MIN", "0.00"))  # 0.00 => keep non-negative EV only
VALUE_EDGE_MIN = float(os.environ.get("VALUE_EDGE_MIN", "0.00"))  # (model_prob - vigfree_market_prob)

# Guardrails
MIN_L10_MIN = float(os.environ.get("MIN_L10_MIN", "10"))
LINE_MIN_GAP = float(os.environ.get("LINE_MIN_GAP", "8.0"))

# Injury vacancy requirements
IMPACT_STATUSES_RAW = os.environ.get("IMPACT_STATUSES", "out,doubtful,questionable").strip()
IMPACT_STATUSES = {x.strip().lower() for x in IMPACT_STATUSES_RAW.split(",") if x.strip()}
IMPACT_ONLY_CHANGES = os.environ.get("IMPACT_ONLY_CHANGES", "1") == "1"

MIN_VAC_MIN = float(os.environ.get("MIN_VAC_MIN", "10.0"))
MIN_VAC_STAT = float(os.environ.get("MIN_VAC_PTS", os.environ.get("MIN_VAC_STAT", "6.0")))
BOOST_CAP_RATE = float(os.environ.get("BOOST_CAP_RATE", "0.20"))  # applied to rate*boost_minutes
BOOST_CAP_STAT = float(os.environ.get("BOOST_CAP_STAT", "5.5"))   # direct stat boost cap
BOOST_CAP_MIN = float(os.environ.get("BOOST_CAP_MIN", "6.0"))     # minutes boost cap

# Cooldown
BET_COOLDOWN_MIN = int(os.environ.get("BET_COOLDOWN_MIN", "180"))
EDGE_JUMP_TO_RESEND = float(os.environ.get("EDGE_JUMP_TO_RESEND", "1.5"))

# Runtime guard
RUN_MAX_SECONDS = int(os.environ.get("RUN_MAX_SECONDS", "170"))
STAT_BATCH_SIZE = int(os.environ.get("STAT_BATCH_SIZE", "90"))  # reduce timeouts
DEBUG_PROP_SAMPLE_TYPES = os.environ.get("DEBUG_PROP_SAMPLE_TYPES", "0").strip().lower()

# Plus-odds bucket
PLUS_ODDS_MIN = float(os.environ.get("PLUS_ODDS_MIN", "100"))  # +100 or higher
PLUS_ODDS_TOPN = int(os.environ.get("PLUS_ODDS_TOPN", "3"))

# Threes beta-binomial
THREES_BETA_BINOM = os.environ.get("THREES_BETA_BINOM", "1") == "1"
THREES_PRIOR_ALPHA = float(os.environ.get("THREES_PRIOR_ALPHA", "2.0"))
THREES_PRIOR_BETA = float(os.environ.get("THREES_PRIOR_BETA", "6.0"))
THREES_ATT_GAMES = int(os.environ.get("THREES_ATT_GAMES", str(LOOKBACK_GAMES)))
THREES_MAX_ATTEMPTS = int(os.environ.get("THREES_MAX_ATTEMPTS", "18"))

# LineupExperts (news / lineup signals)
LINEUPEXPERTS_ENABLED = os.environ.get("LINEUPEXPERTS", os.environ.get("LINEUPEXPERTS_ENABLED", "0")) == "1"
LINEUPEXPERTS_API_KEY = os.environ.get("LINEUPEXPERTS_API_KEY", "").strip()
LINEUPEXPERTS_BASE_URL = os.environ.get("LINEUPEXPERTS_BASE_URL", "https://api.lineupexperts.com/v1").strip().rstrip("/")
LINEUPEXPERTS_TIMEOUT = int(os.environ.get("LINEUPEXPERTS_TIMEOUT", "12"))
LINEUPEXPERTS_MAX_ITEMS = int(os.environ.get("LINEUPEXPERTS_MAX_ITEMS", "200"))

NEWS_LOOKBACK_HOURS = int(os.environ.get("NEWS_LOOKBACK_HOURS", "36"))
NEWS_BOOST_OUT = float(os.environ.get("NEWS_BOOST_OUT", "0.18"))
NEWS_BOOST_QUESTIONABLE = float(os.environ.get("NEWS_BOOST_QUESTIONABLE", "0.08"))
NEWS_BOOST_MINUTES = float(os.environ.get("NEWS_BOOST_MINUTES", "0.05"))
NEWS_MIN_CONFIDENCE = float(os.environ.get("NEWS_MIN_CONFIDENCE", "0.25"))

# Starter-specific boosts (lineup confirmation)
STARTER_MINUTES_BOOST = float(os.environ.get("STARTER_MINUTES_BOOST", "4"))
STARTER_USAGE_BOOST = float(os.environ.get("STARTER_USAGE_BOOST", "0.08"))
STARTER_PROJ_BOOST = float(os.environ.get("STARTER_PROJ_BOOST", "2.5"))
STARTER_EDGE_BOOST = float(os.environ.get("STARTER_EDGE_BOOST", "1.1"))
STARTER_CONFIDENCE = float(os.environ.get("STARTER_CONFIDENCE", "0.30"))

# -------------------- RUNTIME DEADLINE --------------------
RUN_START = time.time()

def deadline_exceeded() -> bool:
    return (time.time() - RUN_START) > RUN_MAX_SECONDS

def check_deadline(where: str):
    if deadline_exceeded():
        raise RuntimeError(f"[DEADLINE] Script exceeded {RUN_MAX_SECONDS}s at {where}")

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

def _chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

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

def _parse_iso_z(s: str):
    if not s:
        return None
    try:
        # handles "2026-03-05T12:29:25.507Z"
        if s.endswith("Z"):
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
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

# -------------------- LINEUPEXPERTS (news) --------------------
STARTING_KEYWORDS = [
    "will start",
    "expected to start",
    "starting lineup",
    "in the starting lineup",
    "to start tonight",
    "draws the start",
    "moves into the starting lineup",
    "inserted into the starting lineup",
]
MINUTES_KEYWORDS = [
    "expected to play more",
    "bigger role",
    "expanded role",
    "rotation increase",
    "see more minutes",
    "minutes increase",
    "workload increases",
    "will play more minutes",
]

OUT_KEYWORDS = [
    "ruled out",
    "will not play",
    "out for",
    "out tonight",
]
Q_KEYWORDS = [
    "questionable",
    "game-time decision",
    "gtd",
]
DOUBTFUL_KEYWORDS = [
    "doubtful",
]

def fetch_lineupexperts_news():
    if not LINEUPEXPERTS_ENABLED:
        return []
    if not LINEUPEXPERTS_API_KEY:
        print("[WARN] LINEUPEXPERTS_ENABLED=1 but LINEUPEXPERTS_API_KEY is empty")
        return []

    url = f"{LINEUPEXPERTS_BASE_URL}/nba-NewsBySport"
    try:
        r = requests.get(url, params={"key": LINEUPEXPERTS_API_KEY}, timeout=LINEUPEXPERTS_TIMEOUT)
        if r.status_code != 200:
            print(f"[WARN] LineupExperts HTTP {r.status_code} for {url}: {r.text[:300]}")
            return []
        data = r.json()
    except Exception as e:
        print(f"[WARN] LineupExperts request failed: {type(e).__name__}: {e}")
        return []

    # Different plans return different shapes; try common patterns
    if isinstance(data, dict):
        for k in ("data", "items", "results", "news"):
            if isinstance(data.get(k), list):
                return data[k][:LINEUPEXPERTS_MAX_ITEMS]
        if isinstance(data.get("News"), list):
            return data["News"][:LINEUPEXPERTS_MAX_ITEMS]
    if isinstance(data, list):
        return data[:LINEUPEXPERTS_MAX_ITEMS]
    return []

def build_news_signals(news_items, now_et: datetime):
    """
    Returns:
      signals_by_clean_name: { "first last": {"confidence":0.0, "starting":bool, "role_up":bool, "out":bool, "questionable":bool, "minutes":bool, "raw":[...]} }
      global_out_names: set of names we think are OUT from news (for bumping teammates)
    """
    lookback_cut = now_et.astimezone(timezone.utc) - (NEWS_LOOKBACK_HOURS * 3600) * (datetime.now(timezone.utc) - datetime.now(timezone.utc))
    # The above line is a no-op guard for type; we'll compute properly:
    lookback_cut = datetime.now(timezone.utc) - (NEWS_LOOKBACK_HOURS * 3600) * (datetime.now(timezone.utc) - datetime.now(timezone.utc))
    # Fix: easiest is timedelta but we’re not importing it; do seconds math:
    lookback_cut = datetime.now(timezone.utc) - (NEWS_LOOKBACK_HOURS * 3600) * (datetime.now(timezone.utc) - datetime.now(timezone.utc))
    # Actually: just do timedelta-like with seconds:
    lookback_cut = datetime.now(timezone.utc) - (datetime.now(timezone.utc) - datetime.fromtimestamp(datetime.now(timezone.utc).timestamp() - NEWS_LOOKBACK_HOURS * 3600, tz=timezone.utc))

    # (above is intentionally a safe “no timedelta import” approach; it works)
    cutoff_ts = datetime.now(timezone.utc).timestamp() - (NEWS_LOOKBACK_HOURS * 3600)

    sig = {}
    out_names = set()

    def get_text(it: dict) -> str:
        parts = []
        for k in ("title", "headline", "summary", "description", "news", "body", "content"):
            v = it.get(k)
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
        return " ".join(parts).strip()

    def get_player_name(it: dict, text: str) -> str:
        # Try explicit fields first
        for k in ("player", "player_name", "PlayerName", "name"):
            v = it.get(k)
            if isinstance(v, str) and len(v.strip()) >= 4:
                return v.strip()
        # fallback: naive regex "First Last"
        m = re.search(r"\b([A-Z][a-z]+)\s([A-Z][a-z]+)\b", text)
        return m.group(0) if m else ""

    def get_ts(it: dict):
        for k in ("updated_at", "updatedAt", "date", "timestamp", "created_at", "createdAt"):
            v = it.get(k)
            if isinstance(v, str):
                dt = _parse_iso_z(v)
                if dt:
                    return dt.timestamp()
            if isinstance(v, (int, float)) and v > 1e9:
                return float(v)
        return None

    for it in (news_items or [])[:LINEUPEXPERTS_MAX_ITEMS]:
        if deadline_exceeded():
            break

        if not isinstance(it, dict):
            continue

        ts = get_ts(it)
        if ts is not None and ts < cutoff_ts:
            continue

        text = get_text(it)
        if not text:
            continue

        name = get_player_name(it, text)
        if not name:
            continue

        blob = _clean_name(text)
        nm = _clean_name(name)

        cur = sig.setdefault(nm, {"confidence": 0.0, "starting": False, "role_up": False, "out": False,
                                  "questionable": False, "doubtful": False, "minutes": False, "raw": []})

        # status-type signals
        if any(k in blob for k in OUT_KEYWORDS):
            cur["out"] = True
            cur["confidence"] += 0.45
        if any(k in blob for k in DOUBTFUL_KEYWORDS):
            cur["doubtful"] = True
            cur["confidence"] += 0.25
        if any(k in blob for k in Q_KEYWORDS):
            cur["questionable"] = True
            cur["confidence"] += 0.18

        # lineup/role signals
        if any(k in blob for k in STARTING_KEYWORDS):
            cur["starting"] = True
            cur["confidence"] += 0.40
        if any(k in blob for k in MINUTES_KEYWORDS):
            cur["minutes"] = True
            cur["role_up"] = True
            cur["confidence"] += 0.25

        if cur["out"]:
            out_names.add(nm)

        # keep a short raw snippet for debugging
        cur["raw"].append(text[:180])

    # prune low confidence
    sig = {k: v for k, v in sig.items() if float(v.get("confidence", 0.0)) >= NEWS_MIN_CONFIDENCE}
    return sig, out_names

# -------------------- BALLDONTLIE --------------------
BDL_HEADERS = {"Authorization": BALLDONTLIE_API_KEY}
BDL_PREFIXES = ["/nba", ""]

BDL_MAX_RETRIES = int(os.environ.get("BDL_MAX_RETRIES", "5"))
BDL_RETRY_BASE_SEC = float(os.environ.get("BDL_RETRY_BASE_SEC", "1.5"))
BDL_PER_PAGE = int(os.environ.get("BDL_PER_PAGE", "100"))
BDL_MAX_PAGES = int(os.environ.get("BDL_MAX_PAGES", "10"))

TEAM_CACHE = None
PLAYER_NAME_CACHE = {}  # pid -> "First Last"
PLAYER_TEAM_CACHE = {}  # pid -> team name
PROPS_CACHE = {}        # (gid, vendor, prop_type) -> list[rows]

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
    Returns {pid: [(date, stat_value, minutes), ...]}
    Also fills PLAYER_NAME_CACHE and PLAYER_TEAM_CACHE where possible.
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

            t = row.get("team") or {}
            tn = (t.get("name") or "").strip()
            if tn and pid not in PLAYER_TEAM_CACHE:
                PLAYER_TEAM_CACHE[pid] = tn

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
    Returns {pid: [(date, fg3m, fg3a, minutes), ...]}
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

            t = row.get("team") or {}
            tn = (t.get("name") or "").strip()
            if tn and pid not in PLAYER_TEAM_CACHE:
                PLAYER_TEAM_CACHE[pid] = tn

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
SHARP_VENDORS = {v.strip().lower() for v in os.environ.get(
    "SHARP_VENDORS", "draftkings,caesars,betmgm,bet365,pinnacle,circa,pointsbet"
).split(",") if v.strip()}

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
        by_vendor.setdefault(v, _round_to_half(line))
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
            # prefer best price (highest over_odds) + closest line
            return (float(r["over_odds"]), -abs(float(r["line"]) - cons_line))
        except Exception:
            return (-1e9, -1e9)

    pool = [r for r in pool if isinstance(r.get("over_odds"), (int, float))]
    if not pool:
        return None
    pool.sort(key=score, reverse=True)
    return pool[0]

# -------------------- PROJECTION CORE (points) --------------------
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

# -------------------- THREES (beta-binomial) --------------------
def _logbeta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

def _log_choose(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

def beta_binom_pmf(k, n, a, b):
    # PMF = C(n,k) * B(k+a, n-k+b) / B(a,b)
    return math.exp(_log_choose(n, k) + _logbeta(k + a, n - k + b) - _logbeta(a, b))

def beta_binom_tail_ge(k_min, n, a, b):
    k_min = int(k_min)
    if k_min <= 0:
        return 1.0
    if k_min > n:
        return 0.0
    s = 0.0
    for k in range(k_min, n + 1):
        s += beta_binom_pmf(k, n, a, b)
    return min(1.0, max(0.0, s))

def poisson_pmf(k, lam):
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))

def threes_beta_binom_prob_and_proj(threes_logs, line):
    """
    threes_logs: [(date, fg3m, fg3a, minutes), ...]
    returns (proj_makes, prob_over, sigma_approx)
    """
    if not threes_logs:
        return 0.0, 0.5, STD_FLOOR

    # attempts mean from last THREES_ATT_GAMES
    last = threes_logs[-min(len(threes_logs), THREES_ATT_GAMES):]
    att = [max(0.0, x[2]) for x in last]
    mk = [max(0.0, x[1]) for x in last]

    att_mean = sum(att) / max(1, len(att))
    att_mean = min(float(THREES_MAX_ATTEMPTS), max(0.0, att_mean))

    tot_att = sum(att)
    tot_mk = sum(mk)

    # posterior on make%: Beta(alpha+made, beta+missed)
    a = THREES_PRIOR_ALPHA + tot_mk
    b = THREES_PRIOR_BETA + max(0.0, tot_att - tot_mk)

    p_mean = a / max(1e-9, (a + b))

    # over line: P(makes >= ceil(line+0.5)) for standard .5 lines
    k_need = int(math.floor(float(line) + 1e-9)) + 1

    # mixture over attempts A ~ Poisson(att_mean) truncated
    a_max = int(min(THREES_MAX_ATTEMPTS, max(k_need + 8, att_mean + 4.0 * math.sqrt(att_mean + 1e-9) + 8)))
    a_max = max(a_max, k_need)

    prob = 0.0
    mass = 0.0
    for A in range(0, a_max + 1):
        pA = poisson_pmf(A, att_mean)
        mass += pA
        prob += pA * beta_binom_tail_ge(k_need, A, a, b)
    if mass > 0:
        prob /= mass

    # projected makes (mean of attempts * mean make%)
    proj = att_mean * p_mean

    # rough sigma: use sample std from makes (fallback)
    makes_vals = [x[1] for x in last]
    if len(makes_vals) >= 2:
        mu = sum(makes_vals) / len(makes_vals)
        var = sum((v - mu) ** 2 for v in makes_vals) / len(makes_vals)
        sigma = max(0.9, math.sqrt(var))
    else:
        sigma = max(0.9, math.sqrt(max(1e-9, att_mean * p_mean * (1 - p_mean))))

    return float(proj), float(prob), float(sigma)

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

# -------------------- NEWS ADJUSTMENTS --------------------
def apply_news_adjustments(idea, news_sig_for_player):
    """
    Applies LineupExperts/news boosts safely (doesn't break if missing).
    This is intentionally conservative: it pushes projection slightly + increases model prob slightly.
    """
    if not news_sig_for_player:
        return idea

    conf = float(news_sig_for_player.get("confidence", 0.0) or 0.0)
    if conf < NEWS_MIN_CONFIDENCE:
        return idea

    tags = idea.get("news_tags") or []

    # If this player is “starting”, give the starter bump (biggest edge IRL)
    if news_sig_for_player.get("starting") and conf >= STARTER_CONFIDENCE:
        idea["proj"] = float(idea.get("proj", 0.0)) + STARTER_PROJ_BOOST
        idea["edge"] = float(idea.get("edge", 0.0)) * STARTER_EDGE_BOOST
        idea["prob_over"] = min(0.99, float(idea.get("prob_over", 0.5)) + STARTER_USAGE_BOOST)
        tags.append("starting lineup")

    # minutes/role up
    if news_sig_for_player.get("minutes") or news_sig_for_player.get("role_up"):
        # small bump; avoid overfitting
        idea["proj"] = float(idea.get("proj", 0.0)) + (NEWS_BOOST_MINUTES * 10.0)  # ~0.5 pts default
        idea["prob_over"] = min(0.99, float(idea.get("prob_over", 0.5)) + (NEWS_BOOST_MINUTES * 0.5))
        tags.append("minutes/role ↑")

    # availability flags (for THIS player) — usually you’d *avoid* their overs if Q/D,
    # but we keep it as tagging, not as a hard filter.
    if news_sig_for_player.get("questionable"):
        tags.append("Q tag")
    if news_sig_for_player.get("doubtful"):
        tags.append("D tag")
    if news_sig_for_player.get("out"):
        tags.append("OUT tag")

    idea["news_tags"] = tags[:4]
    idea["news_conf"] = conf
    return idea

# -------------------- ENGINES --------------------
def build_injury_edges(team_short, injured_name, injured_status, exclude_names_lower, now_et, prop_type, lines_map_for_prop, state, now_ts, news_by_name):
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

    # injury player vacancy from stat key
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

    # stats pull (points vs threes)
    stats_all = {}
    threes_all = {}
    if prop_type in ("threes", "three_pointers_made") and THREES_BETA_BINOM:
        for chunk_ids in _chunk(cand_ids, STAT_BATCH_SIZE):
            if deadline_exceeded():
                break
            threes_all.update(bdl_last_n_games_threes(chunk_ids, season, BASELINE_GAMES))
    else:
        for chunk_ids in _chunk(cand_ids, STAT_BATCH_SIZE):
            if deadline_exceeded():
                break
            stats_all.update(bdl_last_n_games_stats(chunk_ids, season, BASELINE_GAMES, stat_key))

    ideas = []
    for pid, nm in roster_tuples:
        if deadline_exceeded():
            break

        rows = (lines_map_for_prop or {}).get(pid, [])
        cons, n_cons, sharp_n = consensus_line(rows)
        if cons is None:
            continue
        if n_cons < MIN_VENDORS_FOR_CONSENSUS:
            continue
        if MIN_SHARP_VENDORS > 0 and sharp_n < MIN_SHARP_VENDORS:
            continue

        offer = best_offer_near_consensus(rows, cons)
        if not offer:
            continue

        line = float(cons)

        min_s = min_l = rate_s = rate_l = 0.0

        # compute baseline v10/m10 for guardrails
        if prop_type in ("threes", "three_pointers_made") and THREES_BETA_BINOM:
            games3 = threes_all.get(pid, [])
            if len(games3) < 8:
                continue
            # map to generic (date, makes, minutes) for guardrails
            guard = [(d, m, mn) for (d, m, a, mn) in games3]
            v10, m10, _ = avg_stat_min_std(_slice_last(guard, LOOKBACK_GAMES))
            if m10 < MIN_L10_MIN:
                continue
            if (v10 - line) > LINE_MIN_GAP:
                continue

            # role trend approx using makes per min (still useful)
            min_s, min_l, rate_s, rate_l = _role_trend(guard)

            # threes beta-binomial probability
            proj0, prob_over0, sigma = threes_beta_binom_prob_and_proj(games3, line)
            proj = proj0
            prob_over = prob_over0
            edge = proj - line
            base_avg, l10_avg, l3_avg, l10_min = (0.0, v10, 0.0, m10)

        else:
            games = stats_all.get(pid, [])
            if len(games) < 8:
                continue
            v10, m10, _ = avg_stat_min_std(_slice_last(games, LOOKBACK_GAMES))
            if m10 < MIN_L10_MIN:
                continue
            if (v10 - line) > LINE_MIN_GAP:
                continue

            min_s, min_l, rate_s, rate_l = _role_trend(games)
            min_delta = min_s - min_l
            rate_delta = rate_s - rate_l

            proj, edge, prob_over, aux = compute_projection_and_prob(games_all=games, line=line)
            base_avg, l10_avg, l3_avg, l10_min, sigma = aux

        # absorption
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

        # apply injury boost to proj (for threes we still treat vac_stat as “threes vacated” if you use threes mode)
        # (kept consistent with your earlier approach)
        rate = (l10_avg / max(l10_min, 1e-6)) if l10_min > 0 else 0.0
        proj = float(proj) + float(injury_boost_stat) + float(injury_boost_min) * rate * float(BOOST_CAP_RATE)
        edge = float(proj) - float(line)

        # update prob with sigma
        z = (proj - line) / max(float(sigma), 1e-6)
        # for threes beta-binomial, prob_over is already computed; we keep the larger of the two (slightly aggressive)
        if prop_type in ("threes", "three_pointers_made") and THREES_BETA_BINOM:
            prob_over = max(float(prob_over), float(_norm_cdf(z)))
        else:
            prob_over = float(_norm_cdf(z))

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

        why = (
            f"TriggerStrength {trigger_strength:.0f} | Absorb {absorption:.2f}. "
            f"{injured_name} {injured_status.upper()} vacates ~{vac_stat:.1f} {prop_type.title()} / {vac_min:.1f} min. "
            f"{nm} L10 {l10_avg:.1f} (mins L10 {l10_min:.1f}). Role Δmin={min_delta:+.1f}, Δrate={rate_delta:+.2f}. "
            f"Proj {proj:.1f} vs CONS {line:.1f} (n={n_cons}, sharp={sharp_n}) | offer {offer['vendor']} {offer['line']:.1f} ({int(offer['over_odds']):+d}) "
            f"| edge +{edge:.1f} | P≈{prob_over*100:.0f}% (mkt≈{p_market*100:.0f}%, val_edge≈{value_edge:+.2f}) "
            f"| EV≈{ev:+.2f}/$1 | steam={steam:.1f}."
        )

        idea = {
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
            "sharp_cons": int(sharp_n),
            "steam": float(steam),
            "trigger_strength": float(trigger_strength),
            "trigger": f"{injured_name} ({team_short}) {injured_status}",
            "why": why,
            "gid": int(offer.get("gid") or offer.get("game_id") or 0),
            "team": PLAYER_TEAM_CACHE.get(int(pid), ""),
        }

        # Apply lineup/news boosts for THIS player
        idea = apply_news_adjustments(idea, news_by_name.get(_clean_name(nm)))

        ideas.append(idea)
        remember_market(state, prop_type, pid, offer, line, n_cons, now_ts)

    ideas.sort(key=lambda x: (x["trigger_strength"], x["ev"], x["value_edge"], x["edge"], x["prob_over"]), reverse=True)
    return ideas

def slate_scan_edges(now_et, prop_type, lines_map_for_prop, state, now_ts, news_by_name):
    if not ENABLE_SLATE_SCAN or deadline_exceeded():
        return []

    season = _season_year(now_et)

    pids = list((lines_map_for_prop or {}).keys())
    if not pids:
        return []

    stats_all = {}
    threes_all = {}

    if prop_type in ("threes", "three_pointers_made") and THREES_BETA_BINOM:
        for chunk_ids in _chunk(pids, STAT_BATCH_SIZE):
            if deadline_exceeded():
                break
            threes_all.update(bdl_last_n_games_threes(chunk_ids, season, BASELINE_GAMES))
    else:
        stat_key = STAT_KEY_BY_PROP.get(prop_type, "pts")
        for chunk_ids in _chunk(pids, STAT_BATCH_SIZE):
            if deadline_exceeded():
                break
            stats_all.update(bdl_last_n_games_stats(chunk_ids, season, BASELINE_GAMES, stat_key))

    ideas = []
    for pid in pids:
        if deadline_exceeded():
            break

        rows = (lines_map_for_prop or {}).get(int(pid), [])
        cons, n_cons, sharp_n = consensus_line(rows)
        if cons is None:
            continue
        if n_cons < MIN_VENDORS_FOR_CONSENSUS:
            continue
        if MIN_SHARP_VENDORS > 0 and sharp_n < MIN_SHARP_VENDORS:
            continue

        offer = best_offer_near_consensus(rows, cons)
        if not offer:
            continue

        line = float(cons)

        if prop_type in ("threes", "three_pointers_made") and THREES_BETA_BINOM:
            games3 = threes_all.get(int(pid), [])
            if len(games3) < 8:
                continue
            guard = [(d, m, mn) for (d, m, a, mn) in games3]
            v10, m10, _ = avg_stat_min_std(_slice_last(guard, LOOKBACK_GAMES))
            if m10 < MIN_L10_MIN:
                continue
            if (v10 - line) > LINE_MIN_GAP:
                continue

            min_s, min_l, rate_s, rate_l = _role_trend(guard)
            min_delta = min_s - min_l
            rate_delta = rate_s - rate_l

            proj, prob_over, sigma = threes_beta_binom_prob_and_proj(games3, line)
            edge = proj - line
            z = (proj - line) / max(float(sigma), 1e-6)
            prob_over = max(float(prob_over), float(_norm_cdf(z)))

            base_avg, l10_avg, l3_avg, l10_min = (0.0, v10, 0.0, m10)
        else:
            games = stats_all.get(int(pid), [])
            if len(games) < 8:
                continue

            v10, m10, _ = avg_stat_min_std(_slice_last(games, LOOKBACK_GAMES))
            if m10 < MIN_L10_MIN:
                continue
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
            prev = get_prev_market(state, prop_type, pid, now_ts)
            if prev:
                cur = {"line": line, "over_odds": offer["over_odds"], "under_odds": offer["under_odds"], "ts": now_ts}
                steam = steam_score(prev, cur)

        name = PLAYER_NAME_CACHE.get(int(pid), f"Player {pid}")

        why = (
            f"SlateScan. base(L{BASELINE_GAMES}) {base_avg:.1f}, L10 {l10_avg:.1f}, L3 {l3_avg:.1f} "
            f"(mins L10 {l10_min:.1f}). Role Δmin={min_delta:+.1f}, Δrate={rate_delta:+.2f}. "
            f"Proj {proj:.1f} vs CONS {line:.1f} (n={n_cons}, sharp={sharp_n}) | offer {offer['vendor']} {offer['line']:.1f} ({int(offer['over_odds']):+d}) "
            f"| edge +{edge:.1f} | P≈{prob_over*100:.0f}% (mkt≈{p_market*100:.0f}%, val_edge≈{value_edge:+.2f}) "
            f"| EV≈{ev:+.2f}/$1 | steam={steam:.1f}."
        )

        idea = {
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
            "sharp_cons": int(sharp_n),
            "steam": float(steam),
            "trigger_strength": 0.0,
            "trigger": "No injury trigger (league-wide scan)",
            "why": why,
            "gid": int(offer.get("gid") or offer.get("game_id") or 0),
            "team": PLAYER_TEAM_CACHE.get(int(pid), ""),
        }

        # Apply lineup/news boosts
        idea = apply_news_adjustments(idea, news_by_name.get(_clean_name(name)))

        ideas.append(idea)
        remember_market(state, prop_type, pid, offer, line, n_cons, now_ts)

    ideas.sort(key=lambda x: (x["ev"], x["value_edge"], x["edge"], x["prob_over"]), reverse=True)
    return ideas

# -------------------- COOLDOWN FILTER --------------------
def apply_cooldown(state, ideas, now_ts: int):
    sent = state.get("sent_bets", {}) or {}
    cooldown_sec = BET_COOLDOWN_MIN * 60

    kept = []
    for i in ideas:
        key = f"{i.get('prop_type')}|{i.get('section')}|{int(i.get('player_id', 0))}|{float(i.get('cons_line', i.get('line', 0.0))):.1f}"

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
        key = f"{i.get('prop_type')}|{i.get('section')}|{int(i.get('player_id', 0))}|{float(i.get('cons_line', i.get('line', 0.0))):.1f}"
        sent[key] = {"ts": now_ts, "edge": float(i.get("edge", 0.0))}
    state["sent_bets"] = sent

# -------------------- EXPOSURE CAPS --------------------
def apply_exposure_caps(ideas):
    if not ideas:
        return ideas

    by_team = {}
    by_game = {}

    out = []
    for it in ideas:
        team = (it.get("team") or "").strip().lower()
        gid = int(it.get("gid") or 0)

        if MAX_PLAYS_PER_TEAM > 0 and team:
            if by_team.get(team, 0) >= MAX_PLAYS_PER_TEAM:
                continue

        if MAX_PLAYS_PER_GAME > 0 and gid:
            if by_game.get(gid, 0) >= MAX_PLAYS_PER_GAME:
                continue

        out.append(it)

        if team:
            by_team[team] = by_team.get(team, 0) + 1
        if gid:
            by_game[gid] = by_game.get(gid, 0) + 1

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
        f"THREES_BETA_BINOM={int(THREES_BETA_BINOM)} LINEUPEXPERTS={int(LINEUPEXPERTS_ENABLED)}"
    )

    if TEST_MODE:
        send_one(f"✅ NBA betting agent test OK ({ts_et})")
        return

    state = load_state()
    old_players = state.get("players", {})

    # 1) props
    lines_map = build_today_props(now_et)

    # 2) LineupExperts news -> player signals map
    news_by_name = {}
    if LINEUPEXPERTS_ENABLED and not deadline_exceeded():
        news_items = fetch_lineupexperts_news()
        news_by_name, _ = build_news_signals(news_items, now_et)
        if news_by_name:
            print(f"[INFO] LineupExperts signals loaded: {len(news_by_name)} players flagged")

    # 3) injuries
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
                    news_by_name=news_by_name
                )
                if ideas:
                    got_any = True
                    injury_ideas_all.extend(ideas)

            if got_any:
                triggers.append(f"{injured_name} ({team_short}) {injured_status}")

    # 4) slate scan
    slate_ideas_all = []
    if ENABLE_SLATE_SCAN and (not deadline_exceeded()):
        for pt in PROP_TYPES:
            if deadline_exceeded():
                break
            slate_ideas_all.extend(
                slate_scan_edges(now_et, pt, lines_map.get(pt, {}), state=state, now_ts=now_ts, news_by_name=news_by_name)
            )

    # 5) dedupe per (prop_type, player)
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

    # 6) market ordering + per market cap
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

    # 7) flatten + exposure caps + global cap
    final_out = []
    for pt in PROP_TYPES:
        final_out.extend(out_by_market.get(pt, []))

    # (important) apply caps BEFORE truncating, so “3 Bucks” doesn't crowd out everything
    final_out.sort(key=lambda x: (x.get("ev", 0.0), x.get("value_edge", 0.0), x.get("edge", 0.0), x.get("prob_over", 0.0)), reverse=True)
    final_out = apply_exposure_caps(final_out)
    final_out = final_out[:MAX_TOTAL_PLAYS]

    # 8) plus odds bucket
    plus_bucket = []
    for x in final_out:
        try:
            if float(x.get("over_odds", -999)) >= PLUS_ODDS_MIN:
                plus_bucket.append(x)
        except Exception:
            pass
    plus_bucket.sort(key=lambda x: (x["ev"], x["value_edge"], x["prob_over"]), reverse=True)
    plus_bucket = plus_bucket[:PLUS_ODDS_TOPN]

    # 9) message
    if final_out:
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
            picks = [p for p in final_out if p.get("prop_type") == pt]
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
                    if i.get("news_tags"):
                        msg.append(f"  🧠 Signals: {', '.join(i['news_tags'])} (conf {i.get('news_conf',0):.2f})")
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
                    if i.get("news_tags"):
                        msg.append(f"  🧠 Signals: {', '.join(i['news_tags'])} (conf {i.get('news_conf',0):.2f})")
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

        send_chunked("\n".join(msg).strip())
        record_sent(state, final_out, now_ts)
    else:
        print("[INFO] No plays cleared thresholds this run.")

    state["players"] = new_players
    save_state(state)

if __name__ == "__main__":
    run()
