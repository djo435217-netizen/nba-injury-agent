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

# ============================================================
# NBA PROPS AGENT (Points + 3PT Made) — Stable Cron Version
# - Run watchdog (prevents 10+ minute hangs)
# - Safer BDL retries/paging
# - Injury-triggered + League-wide slate scan
# - Multi-vendor (FanDuel + fallback) + main-line picker
# - Robust Twilio send (won't crash the run)
# ============================================================

STATE_FILE = "state.json"
ET = ZoneInfo("America/New_York")

# -------------------- RUN WATCHDOG --------------------
RUN_MAX_SECONDS = int(os.environ.get("RUN_MAX_SECONDS", "85"))
_RUN_START_TS = time.time()

def check_deadline(where: str = ""):
    if (time.time() - _RUN_START_TS) > RUN_MAX_SECONDS:
        raise TimeoutError(f"[DEADLINE] Script exceeded {RUN_MAX_SECONDS}s at {where}")

# -------------------- REQUIRED ENV --------------------
TWILIO_SID = os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
BALLDONTLIE_API_KEY = os.environ.get("BALLDONTLIE_API_KEY", "").strip()
SPORTRADAR_KEY = os.environ.get("SPORTRADAR_API_KEY", "").strip()

FROM_WHATSAPP = os.environ.get("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886").strip()
MY_WHATSAPP_NUMBER = os.environ.get("MY_WHATSAPP_NUMBER", "").strip()
TO_WHATSAPP = f"whatsapp:{MY_WHATSAPP_NUMBER}" if MY_WHATSAPP_NUMBER else ""

twilio = Client(TWILIO_SID, TWILIO_TOKEN) if (TWILIO_SID and TWILIO_TOKEN) else None

# -------------------- CONFIG (ENV) --------------------
TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
MAX_BODY_CHARS = 1500

# Injury feed filtering
IMPACT_STATUSES_RAW = os.environ.get("IMPACT_STATUSES", "out,doubtful,questionable").strip()
IMPACT_STATUSES = {x.strip().lower() for x in IMPACT_STATUSES_RAW.split(",") if x.strip()}
IMPACT_ONLY_CHANGES = os.environ.get("IMPACT_ONLY_CHANGES", "1") == "1"

# Enable modules
ENABLE_SLATE_SCAN = os.environ.get("ENABLE_SLATE_SCAN", "1") == "1"
ENABLE_INJURY_TRIGGERS = os.environ.get("ENABLE_INJURY_TRIGGERS", "1") == "1"
STRICT_INJURY_GAME_MATCH = os.environ.get("STRICT_INJURY_GAME_MATCH", "0") == "1"  # safer default OFF

# Vendors (comma-separated)
BOOK_VENDOR_RAW = os.environ.get("BOOK_VENDORS", os.environ.get("BOOK_VENDOR", "fanduel")).strip().lower()
BOOK_VENDORS = [v.strip() for v in BOOK_VENDOR_RAW.split(",") if v.strip()]

# Prop types requested (comma-separated)
# Allowed inputs: points, threes, three_pointers_made
PROP_TYPES_RAW = os.environ.get("PROP_TYPES", os.environ.get("PROP_TYPE", "points")).strip().lower()
PROP_TYPES = [p.strip() for p in PROP_TYPES_RAW.split(",") if p.strip()]

# Output sizing
MIN_PER_MARKET = int(os.environ.get("MIN_PER_MARKET", "2"))
MAX_PER_MARKET = int(os.environ.get("MAX_PER_MARKET", "6"))
MAX_TOTAL_PLAYS = int(os.environ.get("MAX_TOTAL_PLAYS", os.environ.get("MAX_BET_IDEAS", "10")))

# Multi-horizon windows
BASELINE_GAMES = int(os.environ.get("BASELINE_GAMES", "30"))
LOOKBACK_GAMES = int(os.environ.get("LOOKBACK_GAMES", "10"))
SHORT_GAMES = int(os.environ.get("SHORT_GAMES", "3"))

# Projection blend weights
W_BASE = float(os.environ.get("W_BASE", "0.45"))
W_L10  = float(os.environ.get("W_L10", "0.35"))
W_L3   = float(os.environ.get("W_L3", "0.10"))
W_LINE = float(os.environ.get("W_LINE", "0.10"))

# Thresholds
MIN_EDGE = float(os.environ.get("MIN_EDGE", "2.5"))
MIN_PROB = float(os.environ.get("MIN_PROB", "0.62"))
STD_FLOOR = float(os.environ.get("STD_FLOOR", "5.0"))

# Guardrails
MIN_LINE = float(os.environ.get("MIN_POINTS_LINE", os.environ.get("MIN_LINE", "0.5")))
MAX_LINE = float(os.environ.get("MAX_POINTS_LINE", os.environ.get("MAX_LINE", "60.0")))
LINE_MIN_GAP = float(os.environ.get("LINE_MIN_GAP", "8.0"))
MIN_DELTA_FLOOR = float(os.environ.get("MIN_DELTA_FLOOR", "-3.0"))

# Injury vacancy requirements
MIN_VAC_MIN = float(os.environ.get("MIN_VAC_MIN", "10.0"))
MIN_VAC_STAT = float(os.environ.get("MIN_VAC_PTS", os.environ.get("MIN_VAC_STAT", "6.0")))

# Injury boost caps
BOOST_CAP_STAT = float(os.environ.get("BOOST_CAP_PTS", os.environ.get("BOOST_CAP_STAT", "5.5")))
BOOST_CAP_MIN = float(os.environ.get("BOOST_CAP_MIN", "6.0"))

# Cooldown
BET_COOLDOWN_MIN = int(os.environ.get("BET_COOLDOWN_MIN", "180"))
EDGE_JUMP_TO_RESEND = float(os.environ.get("EDGE_JUMP_TO_RESEND", "1.5"))

# Debug sampling: comma-separated prop types to sample in logs (e.g. "threes" or "points,threes")
DEBUG_PROP_SAMPLE_TYPES_RAW = os.environ.get("DEBUG_PROP_SAMPLE_TYPES", "").strip().lower()
DEBUG_PROP_SAMPLE_TYPES = {x.strip() for x in DEBUG_PROP_SAMPLE_TYPES_RAW.split(",") if x.strip()}

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

def _slice_last(games, n: int):
    if not games:
        return []
    return games[-min(len(games), n):]

def avg_stat_min_std(games):
    # games: [(date, stat, min)]
    if not games:
        return 0.0, 0.0, 0.0
    vals = [x[1] for x in games]
    mins = [x[2] for x in games]
    n = len(vals)
    v_avg = sum(vals) / n
    m_avg = sum(mins) / n
    var = sum((v - v_avg) ** 2 for v in vals) / max(n, 1)
    return v_avg, m_avg, math.sqrt(var)

def _role_trend(games):
    # rate per minute trend
    if not games:
        return 0.0, 0.0, 0.0, 0.0
    long_slice = _slice_last(games, LOOKBACK_GAMES)
    short_slice = _slice_last(games, SHORT_GAMES)

    v_l, m_l, _ = avg_stat_min_std(long_slice)
    v_s, m_s, _ = avg_stat_min_std(short_slice)
    rpm_l = v_l / max(m_l, 1e-6)
    rpm_s = v_s / max(m_s, 1e-6)
    return m_s, m_l, rpm_s, rpm_l

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
    # Never crash the run because Twilio is flaky / sandbox out-of-sync
    if not twilio:
        print("[WARN] Twilio client not configured; skipping send.")
        return
    if not TO_WHATSAPP:
        print("[WARN] MY_WHATSAPP_NUMBER missing; skipping send.")
        return
    try:
        twilio.messages.create(from_=FROM_WHATSAPP, to=TO_WHATSAPP, body=body[:MAX_BODY_CHARS])
    except TwilioRestException as e:
        print(f"[TWILIO_ERR] {e.status} {e.msg}")
    except Exception as e:
        print(f"[TWILIO_ERR] {type(e).__name__}: {e}")

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

# -------------------- SPORTRADAR (optional) --------------------
def fetch_sportradar_injuries():
    if not SPORTRADAR_KEY:
        print("[WARN] SPORTRADAR_KEY not set; continuing WITHOUT injuries.")
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

# -------------------- BALLDONTLIE --------------------
BDL_HEADERS = {"Authorization": BALLDONTLIE_API_KEY} if BALLDONTLIE_API_KEY else {}
BDL_PREFIXES = ["/nba", ""]

# Safer defaults to prevent long hangs
BDL_MAX_RETRIES = int(os.environ.get("BDL_MAX_RETRIES", "2"))
BDL_RETRY_BASE_SEC = float(os.environ.get("BDL_RETRY_BASE_SEC", "1.2"))
BDL_PER_PAGE = int(os.environ.get("BDL_PER_PAGE", "100"))
BDL_MAX_PAGES = int(os.environ.get("BDL_MAX_PAGES", "3"))

TEAM_CACHE = None
PROPS_CACHE = {}         # key: (game_id, vendor, prop_type_api)
DEBUG_PRINTED = set()    # (prop_type_api, vendor) once
PLAYER_NAME_CACHE = {}   # pid -> name

def _bdl_get(path: str, params=None, timeout: int = 12) -> dict:
    check_deadline(f"_bdl_get {path}")
    if not BALLDONTLIE_API_KEY:
        raise RuntimeError("BALLDONTLIE_API_KEY not set")

    last_err = None
    for pref in BDL_PREFIXES:
        url = f"https://api.balldontlie.io{pref}{path}"
        for attempt in range(BDL_MAX_RETRIES):
            check_deadline(f"_bdl_get loop {path} attempt {attempt+1}")
            try:
                r = requests.get(url, headers=BDL_HEADERS, params=params or {}, timeout=timeout)

                if r.status_code == 404:
                    last_err = f"404 {url}"
                    break

                if r.status_code in (429, 500, 502, 503, 504):
                    retry_after = r.headers.get("Retry-After")
                    sleep_s = float(retry_after) if retry_after else (BDL_RETRY_BASE_SEC * (2 ** attempt))
                    last_err = f"{r.status_code} {r.text[:140]}"
                    time.sleep(min(sleep_s, 10.0))
                    continue

                if r.status_code != 200:
                    raise RuntimeError(f"BallDontLie error {r.status_code}: {r.text[:300]}")

                return r.json()

            except Exception as e:
                last_err = str(e)
                time.sleep(min(BDL_RETRY_BASE_SEC * (2 ** attempt), 10.0))
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
        check_deadline("bdl_active_roster")
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

def _stat_key_for_market(prop_type_api: str) -> str:
    # BDL stats endpoint fields
    # points -> "pts"
    # threes -> "fg3m" (3PT made)
    if prop_type_api == "points":
        return "pts"
    if prop_type_api == "threes":
        return "fg3m"
    return "pts"

def bdl_last_n_games_stats(player_ids, season: int, n: int, stat_key: str):
    # returns pid -> [(date, stat, min)]
    out = {int(pid): [] for pid in player_ids}
    if not player_ids:
        return out

    cursor = None
    pages = 0
    while pages < BDL_MAX_PAGES:
        check_deadline("bdl_last_n_games_stats")
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

            fn = (p.get("first_name") or "").strip()
            ln = (p.get("last_name") or "").strip()
            if (fn or ln) and pid not in PLAYER_NAME_CACHE:
                PLAYER_NAME_CACHE[pid] = f"{fn} {ln}".strip()

            game = row.get("game") or {}
            date = game.get("date")

            # stat field with fallback
            val = row.get(stat_key)
            if val is None and stat_key == "fg3m":
                val = row.get("fg3m")  # (redundant but safe)
            try:
                val = float(val or 0)
            except Exception:
                val = 0.0

            mins = _parse_minutes(row.get("min"))
            if date:
                out[pid].append((date, val, mins))

        # early stop if we have enough
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

# --- prop type normalization (ENV -> API) ---
def normalize_prop_type(p: str) -> str:
    p = (p or "").strip().lower()
    if p in ("three_pointers_made", "3pt", "3pm", "three", "threes", "3s"):
        return "threes"  # ✅ your logs show BDL uses "threes"
    if p in ("points", "pts"):
        return "points"
    # fallback: pass through
    return p

def bdl_player_props(game_id: int, vendor: str | None, prop_type_api: str):
    key = (int(game_id), (vendor or ""), prop_type_api)
    if key in PROPS_CACHE:
        return PROPS_CACHE[key]

    params = {"game_id": int(game_id), "prop_type": prop_type_api}
    if vendor:
        params["vendors[]"] = [vendor]

    try:
        resp = _bdl_get("/v2/odds/player_props", params=params)
        props = resp.get("data") or []
    except Exception:
        props = []

    # Optional debug sample per type/vendor
    if prop_type_api in DEBUG_PROP_SAMPLE_TYPES:
        tag = (prop_type_api, vendor or "NO_VENDOR")
        if tag not in DEBUG_PRINTED and props:
            print(f"[DEBUG] SAMPLE PROP ROW ({prop_type_api}, vendor={vendor or 'NO_VENDOR'}): {json.dumps(props[0])[:2000]}")
            DEBUG_PRINTED.add(tag)

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
            # closeness to -110/-110 implies "main" line
            dist = abs(abs(float(over)) - 110.0) + abs(abs(float(under)) - 110.0)
        else:
            dist = None
        candidates.append((dist, line))

    if not candidates:
        return None

    with_dist = [c for c in candidates if c[0] is not None]
    if with_dist:
        with_dist.sort(key=lambda x: x[0])
        return float(with_dist[0][1])

    lines = sorted([c[1] for c in candidates])
    mid = len(lines) // 2
    return float(lines[mid]) if len(lines) % 2 == 1 else float(0.5 * (lines[mid - 1] + lines[mid]))

def main_line_for_player(game_id: int, player_id: int, prop_type_api: str):
    for v in BOOK_VENDORS + [None]:
        props = bdl_player_props(game_id, v, prop_type_api)
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
            return float(line), (v or "NO_VENDOR")
    return None, None

def game_ids_with_team_props(game_ids, team_player_ids: set[int], prop_type_api: str):
    # used for STRICT_INJURY_GAME_MATCH: keep only games where team has props lines
    relevant = []
    for gid in game_ids:
        check_deadline("game_ids_with_team_props")
        found = False
        for v in BOOK_VENDORS + [None]:
            props = bdl_player_props(gid, v, prop_type_api)
            if not props:
                continue
            for pp in props:
                try:
                    pid = int(pp.get("player_id", -1))
                except Exception:
                    continue
                if pid in team_player_ids:
                    found = True
                    break
            if found:
                break
        if found:
            relevant.append(int(gid))
    return relevant

# -------------------- PROJECTION CORE --------------------
def compute_projection_and_prob(games_all, line: float, injury_boost_stat=0.0, injury_boost_min=0.0):
    base_slice = _slice_last(games_all, BASELINE_GAMES)
    l10_slice = _slice_last(games_all, LOOKBACK_GAMES)
    l3_slice = _slice_last(games_all, SHORT_GAMES)

    base_avg, _, base_std = avg_stat_min_std(base_slice)
    l10_avg, l10_min, l10_std = avg_stat_min_std(l10_slice)
    l3_avg, _, _ = avg_stat_min_std(l3_slice)

    sigma = max(STD_FLOOR, (l10_std if l10_std > 0 else base_std if base_std > 0 else STD_FLOOR))

    proj = (W_BASE * base_avg) + (W_L10 * l10_avg) + (W_L3 * l3_avg) + (W_LINE * line)

    rpm = l10_avg / max(l10_min, 1e-6)
    proj += injury_boost_stat
    proj += (injury_boost_min * rpm * 0.20)

    edge = proj - line
    z = (proj - line) / sigma
    prob_over = _norm_cdf(z)
    return proj, edge, prob_over, (base_avg, l10_avg, l3_avg, l10_min, sigma, rpm)

# -------------------- INJURY ENGINE --------------------
def build_injury_edges(team_short, injured_name, injured_status, exclude_names_lower, now_et, prop_type_api: str):
    if not ENABLE_INJURY_TRIGGERS:
        return []

    season = _season_year(now_et)
    stat_key = _stat_key_for_market(prop_type_api)

    roster = bdl_active_roster(team_short)
    if not roster:
        return []

    roster_tuples = []
    team_player_ids = set()
    for p in roster:
        pid = p.get("id")
        nm = f"{p.get('first_name','')} {p.get('last_name','')}".strip()
        if pid is None or not nm:
            continue
        if _clean_name(nm) in exclude_names_lower:
            continue
        pid_int = int(pid)
        roster_tuples.append((pid_int, nm))
        team_player_ids.add(pid_int)

    if not roster_tuples:
        return []

    injured_pid = bdl_find_player_id_on_team(team_short, injured_name)
    if not injured_pid:
        return []

    inj_games = bdl_last_n_games_stats([injured_pid], season, BASELINE_GAMES, stat_key).get(injured_pid, [])
    inj_l10 = _slice_last(inj_games, LOOKBACK_GAMES)
    vac_stat_l10, vac_min_l10, _ = avg_stat_min_std(inj_l10)
    if len(inj_games) < 3:
        return []

    status = (injured_status or "").lower()
    STATUS_MULT = {"out": 1.0, "doubtful": 0.8, "questionable": 0.55}.get(status, 0.65)

    vac_stat = vac_stat_l10 * STATUS_MULT
    vac_min = vac_min_l10 * STATUS_MULT

    if not ((vac_min >= MIN_VAC_MIN) or (vac_stat >= MIN_VAC_STAT)):
        return []

    trigger_strength = min(100.0, (vac_min * 1.2 + vac_stat * 1.5))

    cand_ids = [pid for pid, _ in roster_tuples]
    stats = bdl_last_n_games_stats(cand_ids, season, BASELINE_GAMES, stat_key)

    game_ids = bdl_games_today_ids(now_et)
    if not game_ids:
        return []

    # Optional strict game filter: only games where this team's players actually have props
    if STRICT_INJURY_GAME_MATCH:
        game_ids = game_ids_with_team_props(game_ids, team_player_ids, prop_type_api)
        if not game_ids:
            return []

    ideas = []
    for pid, nm in roster_tuples:
        check_deadline("injury edges loop")
        games = stats.get(pid, [])
        if len(games) < 6:
            continue

        min_s, min_l, rpm_s, rpm_l = _role_trend(games)
        min_delta = min_s - min_l
        rpm_delta = rpm_s - rpm_l

        l10_avg, l10_min, _ = avg_stat_min_std(_slice_last(games, LOOKBACK_GAMES))
        if l10_min < 10:
            continue

        absorption = 0.0
        if l10_min >= 28:
            absorption += 0.30
        if l10_min >= 34:
            absorption += 0.10
        if min_delta >= 2.0:
            absorption += 0.15
        if rpm_delta > 0.05:
            absorption += 0.10
        absorption = min(0.65, absorption)

        # Find line across today's games
        line = None
        use_gid = None
        use_vendor = None
        for gid in game_ids:
            check_deadline("injury line search")
            l, v = main_line_for_player(gid, pid, prop_type_api)
            if l is not None:
                line = float(l)
                use_gid = int(gid)
                use_vendor = v
                break
        if line is None:
            continue

        # guardrail: skip obvious mis-mapped lines
        if (l10_avg - line) > LINE_MIN_GAP:
            continue

        injury_boost_stat = min(BOOST_CAP_STAT, vac_stat * absorption * 0.65)
        injury_boost_min = min(BOOST_CAP_MIN, vac_min * absorption * 0.25)

        proj, edge, prob_over, aux = compute_projection_and_prob(
            games_all=games,
            line=line,
            injury_boost_stat=injury_boost_stat,
            injury_boost_min=injury_boost_min
        )

        base_avg, l10_avg2, l3_avg, l10_min2, _, _ = aux

        if edge < MIN_EDGE or prob_over < MIN_PROB:
            continue
        if min_delta < MIN_DELTA_FLOOR and edge < (MIN_EDGE + 1.5):
            continue

        why = (
            f"TriggerStrength {trigger_strength:.0f} | Absorb {absorption:.2f}. "
            f"{injured_name} {injured_status.upper()} vacates ~{vac_stat:.1f} {prop_type_api.upper()} / {vac_min:.1f} min. "
            f"{nm} base(L{BASELINE_GAMES}) {base_avg:.1f}, L10 {l10_avg2:.1f}, L3 {l3_avg:.1f} "
            f"(mins L10 {l10_min2:.1f}). Role Δmin={min_delta:+.1f}, Δrate={rpm_delta:+.2f}. "
            f"Proj {proj:.1f} vs {use_vendor.lower()} line {line:.1f} | edge +{edge:.1f} | P≈{prob_over*100:.0f}%."
            f" [prop_type={prop_type_api}]"
        )

        ideas.append({
            "section": "injury",
            "prop_type": prop_type_api,
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
            "vendor": use_vendor,
        })

    ideas.sort(key=lambda x: (x["trigger_strength"], x["edge"], x["prob_over"]), reverse=True)
    return ideas

# -------------------- SLATE SCAN ENGINE --------------------
def slate_scan_edges(now_et, prop_type_api: str):
    if not ENABLE_SLATE_SCAN:
        return []

    season = _season_year(now_et)
    stat_key = _stat_key_for_market(prop_type_api)

    game_ids = bdl_games_today_ids(now_et)
    if not game_ids:
        return []

    # Gather one main line per player for this market
    player_to_best_line = {}  # pid -> (line, gid, vendor)
    for gid in game_ids:
        check_deadline("slate scan games")
        # pull props for any vendor that returns rows
        props = []
        used_vendor = None
        for v in BOOK_VENDORS + [None]:
            props = bdl_player_props(gid, v, prop_type_api)
            if props:
                used_vendor = (v or "NO_VENDOR")
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
            player_to_best_line[pid] = (float(line), int(gid), used_vendor)

    if not player_to_best_line:
        return []

    pids = list(player_to_best_line.keys())
    stats = bdl_last_n_games_stats(pids, season, BASELINE_GAMES, stat_key)  # fills PLAYER_NAME_CACHE

    ideas = []
    for pid in pids:
        check_deadline("slate scan players")
        games = stats.get(pid, [])
        if len(games) < 8:
            continue

        line, gid, vendor = player_to_best_line[pid]

        l10_avg, l10_min, _ = avg_stat_min_std(_slice_last(games, LOOKBACK_GAMES))
        if l10_min < 10:
            continue
        if (l10_avg - line) > LINE_MIN_GAP:
            continue

        min_s, min_l, rpm_s, rpm_l = _role_trend(games)
        min_delta = min_s - min_l
        rpm_delta = rpm_s - rpm_l

        proj, edge, prob_over, aux = compute_projection_and_prob(games_all=games, line=line)
        base_avg, l10_avg2, l3_avg, l10_min2, _, _ = aux

        if edge < MIN_EDGE or prob_over < MIN_PROB:
            continue
        if min_delta < MIN_DELTA_FLOOR and edge < (MIN_EDGE + 2.0):
            continue

        name = PLAYER_NAME_CACHE.get(int(pid), f"Player {pid}")

        why = (
            f"SlateScan. base(L{BASELINE_GAMES}) {base_avg:.1f}, L10 {l10_avg2:.1f}, L3 {l3_avg:.1f} "
            f"(mins L10 {l10_min2:.1f}). Role Δmin={min_delta:+.1f}, Δrate={rpm_delta:+.2f}. "
            f"Proj {proj:.1f} vs {vendor.lower()} line {line:.1f} | edge +{edge:.1f} | P≈{prob_over*100:.0f}%."
            f" [prop_type={prop_type_api}]"
        )

        ideas.append({
            "section": "slate",
            "prop_type": prop_type_api,
            "player_name": name,
            "player_id": int(pid),
            "line": float(line),
            "proj": float(proj),
            "edge": float(edge),
            "prob_over": float(prob_over),
            "trigger_strength": 0.0,
            "trigger": "No injury trigger (league-wide scan)",
            "why": why,
            "game_id": int(gid),
            "vendor": vendor,
        })

    ideas.sort(key=lambda x: (x["edge"], x["prob_over"]), reverse=True)
    return ideas

# -------------------- COOLDOWN FILTER --------------------
def apply_cooldown(state, ideas, now_ts: int):
    sent = state.get("sent_bets", {}) or {}
    cooldown_sec = BET_COOLDOWN_MIN * 60

    kept = []
    for i in ideas:
        key = f"{i['prop_type']}|{i['section']}|{int(i['player_id'])}|{i['line']:.1f}"
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
        key = f"{i['prop_type']}|{i['section']}|{int(i['player_id'])}|{i['line']:.1f}"
        sent[key] = {"ts": now_ts, "edge": float(i["edge"]), "line": float(i["line"])}
    state["sent_bets"] = sent

# -------------------- MAIN --------------------
def run():
    check_deadline("boot")
    now_et = _now_et()
    ts_et = now_et.strftime("%Y-%m-%d %I:%M %p ET")
    now_ts = int(now_et.timestamp())

    # Normalize requested markets
    markets = []
    for p in PROP_TYPES:
        api = normalize_prop_type(p)
        if api not in markets:
            markets.append(api)

    print(
        f"[BOOT] ts={ts_et} TEST_MODE={int(TEST_MODE)} "
        f"PROP_TYPES={','.join(markets)} "
        f"MIN_PER_MARKET={MIN_PER_MARKET} MAX_PER_MARKET={MAX_PER_MARKET} "
        f"MAX_TOTAL_PLAYS={MAX_TOTAL_PLAYS} "
        f"BOOK_VENDORS={','.join(BOOK_VENDORS)} "
        f"ENABLE_SLATE_SCAN={int(ENABLE_SLATE_SCAN)} ENABLE_INJURY_TRIGGERS={int(ENABLE_INJURY_TRIGGERS)} "
        f"STRICT_INJURY_GAME_MATCH={int(STRICT_INJURY_GAME_MATCH)}"
    )

    if TEST_MODE:
        send_one(f"✅ Props agent test OK ({ts_et})")
        return

    state = load_state()
    old_players = state.get("players", {})

    sr = fetch_sportradar_injuries()
    new_players = parse_injuries(sr)
    exclude_names_lower = {_clean_name(v.get("name", "")) for v in new_players.values() if v.get("name")}

    # Collect per market
    out_by_market = {m: {"injury": [], "slate": [], "triggers": []} for m in markets}

    # Injury triggers (shared feed)
    if ENABLE_INJURY_TRIGGERS and new_players:
        for pid, cur in new_players.items():
            check_deadline(f"injury loop {pid}")
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

            for m in markets:
                ideas = build_injury_edges(
                    team_short=team_short,
                    injured_name=injured_name,
                    injured_status=injured_status,
                    exclude_names_lower=exclude_names_lower | {_clean_name(injured_name)},
                    now_et=now_et,
                    prop_type_api=m
                )
                if ideas:
                    out_by_market[m]["triggers"].append(f"{injured_name} ({team_short}) {injured_status}")
                    out_by_market[m]["injury"].extend(ideas)

    # Slate scan per market
    if ENABLE_SLATE_SCAN:
        for m in markets:
            check_deadline(f"slate scan {m}")
            out_by_market[m]["slate"] = slate_scan_edges(now_et, m)

    # Combine, cooldown, and final select
    combined_all = []
    for m in markets:
        combined = out_by_market[m]["injury"] + out_by_market[m]["slate"]

        # best per (market, section, player)
        best = {}
        for i in combined:
            k = (i["prop_type"], i["section"], int(i["player_id"]))
            if (k not in best) or ((i["edge"], i["prob_over"]) > (best[k]["edge"], best[k]["prob_over"])):
                best[k] = i
        combined = list(best.values())

        combined = apply_cooldown(state, combined, now_ts)

        # per market caps
        injury_sorted = sorted(
            [i for i in combined if i["section"] == "injury"],
            key=lambda x: (x["trigger_strength"], x["edge"], x["prob_over"]),
            reverse=True
        )
        slate_sorted = sorted(
            [i for i in combined if i["section"] == "slate"],
            key=lambda x: (x["edge"], x["prob_over"]),
            reverse=True
        )

        # Ensure at least MIN_PER_MARKET if available, but cap at MAX_PER_MARKET
        picks = []
        picks.extend(injury_sorted[:MAX_PER_MARKET])
        if len(picks) < MIN_PER_MARKET:
            need = MIN_PER_MARKET - len(picks)
            picks.extend(slate_sorted[:need])

        # fill remaining up to MAX_PER_MARKET using slate
        remaining = MAX_PER_MARKET - len(picks)
        if remaining > 0:
            already = {(p["section"], p["player_id"], p["line"]) for p in picks}
            for s in slate_sorted:
                key = (s["section"], s["player_id"], s["line"])
                if key in already:
                    continue
                picks.append(s)
                if len(picks) >= MAX_PER_MARKET:
                    break

        out_by_market[m]["final"] = picks
        combined_all.extend(picks)

    # Global cap
    combined_all.sort(key=lambda x: (x["edge"], x["prob_over"]), reverse=True)
    final_out = combined_all[:MAX_TOTAL_PLAYS]

    if not final_out:
        print("[INFO] No plays passed thresholds/cooldown.")
        state["players"] = new_players
        save_state(state)
        return

    # Build message
    msg = [f"💰 FanDuel Props ({ts_et})", ""]

    for m in markets:
        picks = out_by_market[m].get("final", [])
        if not picks:
            continue

        header = "Points" if m == "points" else ("3PT Made" if m == "threes" else m)
        msg.append(f"🏷️ {header}")
        msg.append("")

        injury_picks = [p for p in picks if p["section"] == "injury"]
        slate_picks = [p for p in picks if p["section"] == "slate"]

        if injury_picks:
            msg.append("🚑 Injury-Triggered Plays:")
            triggers = out_by_market[m]["triggers"]
            if triggers:
                msg.append("Triggers:")
                for t in triggers[:8]:
                    msg.append(f"- {t}")
                if len(triggers) > 8:
                    msg.append(f"- …and {len(triggers)-8} more")
            msg.append("")
            for i in injury_picks:
                msg.append(f"• {i['player_name']} OVER {i['line']:.1f}  (edge +{i['edge']:.1f}, P≈{i['prob_over']*100:.0f}%)")
                msg.append(f"  Trigger: {i['trigger']}")
                msg.append(f"  Why: {i['why']}")
                msg.append("")

        if slate_picks:
            msg.append("🌎 League-Wide Slate Scan (no injury required):")
            msg.append("")
            for i in slate_picks:
                msg.append(f"• {i['player_name']} OVER {i['line']:.1f}  (edge +{i['edge']:.1f}, P≈{i['prob_over']*100:.0f}%)")
                msg.append(f"  Why: {i['why']}")
                msg.append("")

        msg.append("")  # spacer between markets

    send_chunked("\n".join(msg).strip())

    # Record cooldown + save state
    record_sent(state, final_out, now_ts)
    state["players"] = new_players
    save_state(state)

if __name__ == "__main__":
    try:
        run()
    except TimeoutError as e:
        print(str(e))
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
