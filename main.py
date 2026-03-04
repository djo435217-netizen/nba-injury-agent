import os
import json
import re
import time
import math
import statistics
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
MAX_BODY_CHARS = 1500

BOOK_VENDOR_RAW = os.environ.get("BOOK_VENDOR", "fanduel").strip().lower()
BOOK_VENDORS = [v.strip() for v in BOOK_VENDOR_RAW.split(",") if v.strip()]

PROP_TYPES_RAW = os.environ.get("PROP_TYPES", "points").strip().lower()
PROP_TYPES = [p.strip() for p in PROP_TYPES_RAW.split(",") if p.strip()]

ENABLE_INJURY_TRIGGERS = os.environ.get("ENABLE_INJURY_TRIGGERS", "1") == "1"
ENABLE_SLATE_SCAN = os.environ.get("ENABLE_SLATE_SCAN", "1") == "1"
ENABLE_LADDER_SCAN = os.environ.get("ENABLE_LADDER_SCAN", "1") == "1"

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

# Guardrails
MIN_L10_MIN = float(os.environ.get("MIN_L10_MIN", "10"))
LINE_MIN_GAP = float(os.environ.get("LINE_MIN_GAP", "8.0"))

# Injury vacancy requirements
IMPACT_STATUSES_RAW = os.environ.get("IMPACT_STATUSES", "out,doubtful,questionable").strip()
IMPACT_STATUSES = {x.strip().lower() for x in IMPACT_STATUSES_RAW.split(",") if x.strip()}
IMPACT_ONLY_CHANGES = os.environ.get("IMPACT_ONLY_CHANGES", "1") == "1"

MIN_VAC_MIN = float(os.environ.get("MIN_VAC_MIN", "10.0"))
MIN_VAC_PTS = float(os.environ.get("MIN_VAC_PTS", "6.0"))
BOOST_CAP_RATE = float(os.environ.get("BOOST_CAP_RATE", "0.20"))  # applied to rate*boost_minutes
BOOST_CAP_STAT = float(os.environ.get("BOOST_CAP_STAT", "5.5"))   # direct stat boost cap

# Cooldown (prevents repeating identical props too often)
BET_COOLDOWN_MIN = int(os.environ.get("BET_COOLDOWN_MIN", "180"))
EDGE_JUMP_TO_RESEND = float(os.environ.get("EDGE_JUMP_TO_RESEND", "1.5"))

# Runtime guard
RUN_MAX_SECONDS = int(os.environ.get("RUN_MAX_SECONDS", "170"))
DEBUG_PROP_SAMPLE_TYPES = os.environ.get("DEBUG_PROP_SAMPLE_TYPES", "0").strip().lower()

# Ladder settings (Points milestone)
LADDER_MIN_ODDS = float(os.environ.get("LADDER_MIN_ODDS", "300"))
LADDER_MAX_ODDS = float(os.environ.get("LADDER_MAX_ODDS", "1500"))
LADDER_TOPN = int(os.environ.get("LADDER_TOPN", "6"))
LADDER_EV_MIN = float(os.environ.get("LADDER_EV_MIN", "0.03"))
LADDER_MIN_L10_MIN = float(os.environ.get("LADDER_MIN_L10_MIN", "18"))

# Slate scan cap (prevents timeouts)
SLATE_SCAN_MAX_PLAYERS = int(os.environ.get("SLATE_SCAN_MAX_PLAYERS", "220"))

# -------------------- NEW: CONSENSUS LINE FILTER --------------------
MIN_VENDORS_FOR_CONSENSUS = int(os.environ.get("MIN_VENDORS_FOR_CONSENSUS", "2"))
CONSENSUS_MAX_DEVIATION = float(os.environ.get("CONSENSUS_MAX_DEVIATION", "1.5"))  # skip if offer line too far from consensus
CONSENSUS_USE_MEDIAN = os.environ.get("CONSENSUS_USE_MEDIAN", "1") == "1"  # 1=median, 0=mean

# -------------------- NEW: STEAM DETECTION --------------------
ENABLE_STEAM = os.environ.get("ENABLE_STEAM", "1") == "1"
STEAM_MIN_SCORE = float(os.environ.get("STEAM_MIN_SCORE", "0.75"))  # threshold to label as steam
STEAM_LINE_W = float(os.environ.get("STEAM_LINE_W", "1.0"))         # weight per 1.0 line move
STEAM_ODDS_W = float(os.environ.get("STEAM_ODDS_W", "0.25"))        # weight per 100 odds move (american)
STEAM_REQUIRE_FAVORABLE = os.environ.get("STEAM_REQUIRE_FAVORABLE", "0") == "1"  # if 1, only count steam if favorable for OVER

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
        return {"players": {}, "sent_bets": {}, "last_quotes": {}}
    try:
        with open(STATE_FILE, "r") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return {"players": {}, "sent_bets": {}, "last_quotes": {}}
        raw.setdefault("players", {})
        raw.setdefault("sent_bets", {})
        raw.setdefault("last_quotes", {})
        return raw
    except Exception:
        return {"players": {}, "sent_bets": {}, "last_quotes": {}}

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

def _median(vals):
    vals = [float(x) for x in vals]
    if not vals:
        return None
    if CONSENSUS_USE_MEDIAN:
        return float(statistics.median(vals))
    return float(sum(vals) / len(vals))

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
PLAYER_NAME_CACHE = {}  # pid -> "First Last"
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

    if DEBUG_PROP_SAMPLE_TYPES:
        dbg_types = {x.strip() for x in DEBUG_PROP_SAMPLE_TYPES.split(",") if x.strip()}
        if prop_type in dbg_types and props:
            print(f"[DEBUG] SAMPLE PROP ROW ({prop_type}, vendor={vendor or 'NO_VENDOR'}): {json.dumps(props[0])[:2000]}")

    PROPS_CACHE[key] = props
    return props

# -------------------- PROP PARSING --------------------
def pick_main_line(rows):
    """
    Picks a "main" line: closest to -110/-110 by total distance.
    Expects rows to include over_odds + under_odds.
    """
    if not rows:
        return None
    best = None
    best_dist = None
    for r in rows:
        try:
            o = float(r["over_odds"])
            u = float(r["under_odds"])
            dist = abs(abs(o) - 110.0) + abs(abs(u) - 110.0)
            if best is None or dist < best_dist:
                best = r
                best_dist = dist
        except Exception:
            continue
    return best

def build_today_props(now_et: datetime):
    """
    NEW: Fetch per-vendor so we can compute CONSENSUS (median) lines.
    Returns:
      offers_map[prop_type][pid] = {
          "consensus_line": float,
          "vendor_count": int,
          "best_offer": row,          # chosen offer row (vendor/odds/line close to consensus)
          "vendor_main": {vendor: row}# each vendor's main line row
      }
      ladders_points[pid] -> list of dict rows for milestone points only (odds, line, vendor, gid)
    """
    game_ids = bdl_games_today_ids(now_et)
    offers_map = {pt: {} for pt in PROP_TYPES}
    ladders_points = {}  # pid -> list

    for gid in game_ids:
        if deadline_exceeded():
            break

        for pt in PROP_TYPES:
            if deadline_exceeded():
                break

            # collect over_under rows per vendor for this game/prop_type
            per_vendor_rows = {}
            for v in BOOK_VENDORS + [None]:
                if deadline_exceeded():
                    break
                props = bdl_fetch_props_for_game(gid, v, pt)
                if not props:
                    continue

                for pp in props:
                    try:
                        pid = int(pp.get("player_id"))
                    except Exception:
                        continue

                    market = pp.get("market") or {}
                    mtype = (market.get("type") or "").lower()

                    if mtype == "over_under":
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
                        per_vendor_rows.setdefault(row["vendor"], []).append(row)

                    elif ENABLE_LADDER_SCAN and pt == "points" and mtype == "milestone":
                        odds = market.get("odds")
                        if odds is None:
                            continue
                        try:
                            odds = float(odds)
                            line = float(pp.get("line_value"))
                        except Exception:
                            continue
                        if not (LADDER_MIN_ODDS <= odds <= LADDER_MAX_ODDS):
                            continue
                        ladders_points.setdefault(pid, []).append({
                            "pid": pid,
                            "gid": int(pp.get("game_id")) if pp.get("game_id") is not None else int(gid),
                            "vendor": (pp.get("vendor") or (v or "no_vendor")),
                            "line": float(line),
                            "odds": float(odds),
                            "updated_at": pp.get("updated_at"),
                        })

            # Reduce each vendor to its "main" line for each player; then compute consensus across vendors
            if not per_vendor_rows:
                continue

            # vendor_main_for_pid[pid][vendor] = main_row
            vendor_main_for_pid = {}
            for vendor_name, rows in per_vendor_rows.items():
                by_pid = {}
                for r in rows:
                    by_pid.setdefault(int(r["pid"]), []).append(r)

                for pid, pid_rows in by_pid.items():
                    main = pick_main_line(pid_rows)
                    if main:
                        vendor_main_for_pid.setdefault(pid, {})[vendor_name] = main

            for pid, vendor_main in vendor_main_for_pid.items():
                if deadline_exceeded():
                    break

                lines = [float(r["line"]) for r in vendor_main.values()]
                if len(lines) < MIN_VENDORS_FOR_CONSENSUS:
                    continue

                consensus_line = _median(lines)
                if consensus_line is None:
                    continue

                # choose best offer row: closest to consensus line, tie-break by best over odds (higher = better payout)
                offers = list(vendor_main.values())
                offers.sort(key=lambda r: (abs(float(r["line"]) - float(consensus_line)), -float(r["over_odds"])))
                best_offer = offers[0]

                # Skip if best offer is too far from consensus
                if abs(float(best_offer["line"]) - float(consensus_line)) > CONSENSUS_MAX_DEVIATION:
                    continue

                offers_map.setdefault(pt, {}).setdefault(int(pid), {
                    "consensus_line": float(consensus_line),
                    "vendor_count": int(len(lines)),
                    "best_offer": best_offer,
                    "vendor_main": vendor_main,
                })

    return offers_map, ladders_points

# -------------------- PROJECTION CORE --------------------
STAT_KEY_BY_PROP = {
    "points": "pts",
    "threes": "fg3m",   # BDL stats field for 3PT made
}

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
    z = (proj - float(line)) / sigma
    prob_over = _norm_cdf(z)
    return proj, edge, prob_over, (base_avg, l10_avg, l3_avg, l10_min, sigma)

# -------------------- NEW: STEAM --------------------
def steam_score_and_note(state, prop_type: str, pid: int, consensus_line: float, offer_row: dict, now_ts: int):
    """
    Compares current consensus_line + offer odds vs last run.
    Returns (steam_score, note, favorable_bool)
    """
    if not ENABLE_STEAM:
        return 0.0, "", False

    last_quotes = (state.get("last_quotes", {}) or {})
    key = f"{prop_type}|{int(pid)}"
    prev = last_quotes.get(key)

    # record current after computing
    cur_line = float(consensus_line)
    cur_over = float(offer_row.get("over_odds", 0.0))
    cur_vendor = str(offer_row.get("vendor", "no_vendor"))

    if not prev:
        return 0.0, "", False

    try:
        prev_line = float(prev.get("line"))
        prev_over = float(prev.get("over_odds"))
    except Exception:
        return 0.0, "", False

    line_move = prev_line - cur_line      # positive means line went DOWN (good for OVER)
    odds_move = cur_over - prev_over      # positive means odds got better (more +) for OVER

    favorable = (line_move > 0) or (odds_move > 0)

    score = (abs(line_move) * STEAM_LINE_W) + (abs(odds_move) / 100.0 * STEAM_ODDS_W)

    if STEAM_REQUIRE_FAVORABLE and not favorable:
        score = 0.0

    if score >= STEAM_MIN_SCORE:
        note = f"STEAM score {score:.2f} (Δline {line_move:+.1f}, Δodds {odds_move:+.0f}) vs last {prev.get('vendor','?')}"
        return score, note, favorable

    return score, "", favorable

def update_last_quote(state, prop_type: str, pid: int, consensus_line: float, offer_row: dict, now_ts: int):
    state.setdefault("last_quotes", {})
    key = f"{prop_type}|{int(pid)}"
    state["last_quotes"][key] = {
        "ts": int(now_ts),
        "line": float(consensus_line),
        "over_odds": float(offer_row.get("over_odds", 0.0)),
        "under_odds": float(offer_row.get("under_odds", 0.0)),
        "vendor": str(offer_row.get("vendor", "no_vendor")),
    }

# -------------------- INJURY ENGINE --------------------
def build_injury_edges(team_short, injured_name, injured_status, exclude_names_lower, now_et, prop_type, offers_for_prop, state, now_ts):
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
    if not ((vac_min >= MIN_VAC_MIN) or (vac_stat >= MIN_VAC_PTS)):
        return []

    trigger_strength = min(100.0, (vac_min * 1.2 + vac_stat * 1.5))

    cand_ids = [pid for pid, _ in roster_tuples]
    stats = bdl_last_n_games_stats(cand_ids, season, BASELINE_GAMES, stat_key)

    ideas = []
    for pid, nm in roster_tuples:
        if deadline_exceeded():
            break

        games = stats.get(pid, [])
        if len(games) < 8:
            continue

        v10, m10, _ = avg_stat_min_std(_slice_last(games, LOOKBACK_GAMES))
        if m10 < MIN_L10_MIN:
            continue

        offer_pack = (offers_for_prop or {}).get(int(pid))
        if not offer_pack:
            continue

        consensus_line = float(offer_pack["consensus_line"])
        vendor_count = int(offer_pack["vendor_count"])
        offer = dict(offer_pack["best_offer"])

        if (v10 - consensus_line) > LINE_MIN_GAP:
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
        injury_boost_min = min(6.0, vac_min * absorption * 0.25)

        proj, edge, prob_over, aux = compute_projection_and_prob(games_all=games, line=consensus_line)
        base_avg, l10_avg, l3_avg, l10_min, sigma = aux

        rate = l10_avg / max(l10_min, 1e-6)
        proj = proj + injury_boost_stat + (injury_boost_min * rate * BOOST_CAP_RATE)
        edge = proj - consensus_line
        z = (proj - consensus_line) / max(sigma, 1e-6)
        prob_over = _norm_cdf(z)

        if edge < MIN_EDGE or prob_over < MIN_PROB:
            continue

        # vig-free market probability from offer OU odds
        p_over = american_to_prob(offer["over_odds"])
        p_under = american_to_prob(offer["under_odds"])
        p_market = p_over / max(p_over + p_under, 1e-9)

        ev = ev_per_dollar(prob_over, float(offer["over_odds"]))

        steam_score, steam_note, _ = steam_score_and_note(state, prop_type, int(pid), consensus_line, offer, now_ts)

        why = (
            f"TriggerStrength {trigger_strength:.0f} | Absorb {absorption:.2f}. "
            f"{injured_name} {injured_status.upper()} vacates ~{vac_stat:.1f} {prop_type.title()} / {vac_min:.1f} min. "
            f"{nm} base(L{BASELINE_GAMES}) {base_avg:.1f}, L10 {l10_avg:.1f}, L3 {l3_avg:.1f} "
            f"(mins L10 {l10_min:.1f}). Role Δmin={min_delta:+.1f}, Δrate={rate_delta:+.2f}. "
            f"Proj {proj:.1f} vs CONS {consensus_line:.1f} (n={vendor_count}) | "
            f"offer {offer['vendor']} {offer['line']:.1f} ({offer['over_odds']:+.0f}) | "
            f"edge +{edge:.1f} | P≈{prob_over*100:.0f}% (mkt≈{p_market*100:.0f}%) | EV≈{ev:+.2f}/$1."
        )
        if steam_note:
            why += f" {steam_note}."

        ideas.append({
            "section": "injury",
            "prop_type": prop_type,
            "player_name": nm,
            "player_id": int(pid),
            "line": float(consensus_line),
            "offer_line": float(offer["line"]),
            "proj": float(proj),
            "edge": float(edge),
            "prob_over": float(prob_over),
            "market_prob": float(p_market),
            "ev": float(ev),
            "vendor": offer["vendor"],
            "over_odds": float(offer["over_odds"]),
            "under_odds": float(offer["under_odds"]),
            "vendor_count": int(vendor_count),
            "steam_score": float(steam_score),
            "trigger_strength": float(trigger_strength),
            "trigger": f"{injured_name} ({team_short}) {injured_status}",
            "why": why,
        })

        update_last_quote(state, prop_type, int(pid), consensus_line, offer, now_ts)

    ideas.sort(key=lambda x: (x["trigger_strength"], x["steam_score"], x["ev"], x["edge"], x["prob_over"]), reverse=True)
    return ideas

# -------------------- SLATE SCAN --------------------
def slate_scan_edges(now_et, prop_type, offers_for_prop, state, now_ts):
    if not ENABLE_SLATE_SCAN:
        return []
    if deadline_exceeded():
        return []

    season = _season_year(now_et)
    stat_key = STAT_KEY_BY_PROP.get(prop_type, "pts")

    pids = list((offers_for_prop or {}).keys())
    if not pids:
        return []

    # cap to prevent timeouts
    if len(pids) > SLATE_SCAN_MAX_PLAYERS:
        # prefer higher vendor_count first (more reliable consensus)
        pids.sort(key=lambda pid: int(offers_for_prop[pid].get("vendor_count", 0)), reverse=True)
        pids = pids[:SLATE_SCAN_MAX_PLAYERS]

    stats = bdl_last_n_games_stats(pids, season, BASELINE_GAMES, stat_key)

    ideas = []
    for pid in pids:
        if deadline_exceeded():
            break

        games = stats.get(pid, [])
        if len(games) < 8:
            continue

        offer_pack = offers_for_prop.get(int(pid))
        if not offer_pack:
            continue

        consensus_line = float(offer_pack["consensus_line"])
        vendor_count = int(offer_pack["vendor_count"])
        offer = dict(offer_pack["best_offer"])

        v10, m10, _ = avg_stat_min_std(_slice_last(games, LOOKBACK_GAMES))
        if m10 < MIN_L10_MIN:
            continue

        if (v10 - consensus_line) > LINE_MIN_GAP:
            continue

        min_s, min_l, rate_s, rate_l = _role_trend(games)
        min_delta = min_s - min_l
        rate_delta = rate_s - rate_l

        proj, edge, prob_over, aux = compute_projection_and_prob(games_all=games, line=consensus_line)
        base_avg, l10_avg, l3_avg, l10_min, sigma = aux

        if edge < MIN_EDGE or prob_over < MIN_PROB:
            continue

        p_over = american_to_prob(offer["over_odds"])
        p_under = american_to_prob(offer["under_odds"])
        p_market = p_over / max(p_over + p_under, 1e-9)

        ev = ev_per_dollar(prob_over, float(offer["over_odds"]))

        steam_score, steam_note, _ = steam_score_and_note(state, prop_type, int(pid), consensus_line, offer, now_ts)

        name = PLAYER_NAME_CACHE.get(int(pid), f"Player {pid}")

        why = (
            f"SlateScan. base(L{BASELINE_GAMES}) {base_avg:.1f}, L10 {l10_avg:.1f}, L3 {l3_avg:.1f} "
            f"(mins L10 {l10_min:.1f}). Role Δmin={min_delta:+.1f}, Δrate={rate_delta:+.2f}. "
            f"Proj {proj:.1f} vs CONS {consensus_line:.1f} (n={vendor_count}) | "
            f"offer {offer['vendor']} {offer['line']:.1f} ({offer['over_odds']:+.0f}) | "
            f"edge +{edge:.1f} | P≈{prob_over*100:.0f}% (mkt≈{p_market*100:.0f}%) | EV≈{ev:+.2f}/$1."
        )
        if steam_note:
            why += f" {steam_note}."

        ideas.append({
            "section": "slate",
            "prop_type": prop_type,
            "player_name": name,
            "player_id": int(pid),
            "line": float(consensus_line),
            "offer_line": float(offer["line"]),
            "proj": float(proj),
            "edge": float(edge),
            "prob_over": float(prob_over),
            "market_prob": float(p_market),
            "ev": float(ev),
            "vendor": offer["vendor"],
            "over_odds": float(offer["over_odds"]),
            "under_odds": float(offer["under_odds"]),
            "vendor_count": int(vendor_count),
            "steam_score": float(steam_score),
            "trigger_strength": 0.0,
            "trigger": "No injury trigger (league-wide scan)",
            "why": why,
        })

        update_last_quote(state, prop_type, int(pid), consensus_line, offer, now_ts)

    ideas.sort(key=lambda x: (x["steam_score"], x["ev"], x["edge"], x["prob_over"]), reverse=True)
    return ideas

# -------------------- LADDER SCAN (POINTS ONLY) --------------------
def ladder_scan_points(now_et, ladders_points):
    if not ENABLE_LADDER_SCAN:
        return []
    if deadline_exceeded():
        return []
    if not ladders_points:
        return []

    season = _season_year(now_et)
    stat_key = "pts"

    pids = list(ladders_points.keys())
    stats = bdl_last_n_games_stats(pids, season, BASELINE_GAMES, stat_key)

    ideas = []
    for pid, rows in ladders_points.items():
        if deadline_exceeded():
            break

        games = stats.get(int(pid), [])
        if len(games) < 10:
            continue

        v10, m10, _ = avg_stat_min_std(_slice_last(games, LOOKBACK_GAMES))
        if m10 < LADDER_MIN_L10_MIN:
            continue

        name = PLAYER_NAME_CACHE.get(int(pid), f"Player {pid}")

        best_for_player = None
        for r in rows:
            try:
                line = float(r["line"])
                odds = float(r["odds"])
            except Exception:
                continue

            proj, edge, prob_over, aux = compute_projection_and_prob(games_all=games, line=line)
            base_avg, l10_avg, l3_avg, l10_min, sigma = aux

            p_model = prob_over
            p_imp = american_to_prob(odds)
            ev = ev_per_dollar(p_model, odds)

            if ev < LADDER_EV_MIN:
                continue

            item = {
                "section": "ladder",
                "prop_type": "points_ladder",
                "player_name": name,
                "player_id": int(pid),
                "line": float(line),
                "odds": float(odds),
                "prob_over": float(p_model),
                "implied_prob": float(p_imp),
                "ev": float(ev),
                "vendor": r.get("vendor", "no_vendor"),
                "why": (
                    f"Ladder (+odds). base(L{BASELINE_GAMES}) {base_avg:.1f}, L10 {l10_avg:.1f}, L3 {l3_avg:.1f} "
                    f"(mins L10 {l10_min:.1f}, sigma {sigma:.1f}). "
                    f"Model P≈{p_model*100:.0f}% vs implied≈{p_imp*100:.0f}% | EV≈{ev:+.2f}/$1."
                )
            }

            if (best_for_player is None) or ((item["ev"], item["prob_over"]) > (best_for_player["ev"], best_for_player["prob_over"])):
                best_for_player = item

        if best_for_player:
            ideas.append(best_for_player)

    ideas.sort(key=lambda x: (x["ev"], x["prob_over"]), reverse=True)
    return ideas[:LADDER_TOPN]

# -------------------- COOLDOWN FILTER --------------------
def apply_cooldown(state, ideas, now_ts: int):
    sent = state.get("sent_bets", {}) or {}
    cooldown_sec = BET_COOLDOWN_MIN * 60

    kept = []
    for i in ideas:
        if i["section"] == "ladder":
            key = f"ladder|{int(i['player_id'])}|{int(i['line'])}|{int(i['odds'])}"
        else:
            key = f"{i['prop_type']}|{i['section']}|{int(i['player_id'])}|{i['line']:.1f}"

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
        if i["section"] == "ladder":
            key = f"ladder|{int(i['player_id'])}|{int(i['line'])}|{int(i['odds'])}"
            sent[key] = {"ts": now_ts, "edge": 0.0}
        else:
            key = f"{i['prop_type']}|{i['section']}|{int(i['player_id'])}|{i['line']:.1f}"
            sent[key] = {"ts": now_ts, "edge": float(i.get("edge", 0.0))}
    state["sent_bets"] = sent

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
        f"ENABLE_LADDER_SCAN={int(ENABLE_LADDER_SCAN)} "
        f"MIN_VENDORS_FOR_CONSENSUS={MIN_VENDORS_FOR_CONSENSUS} ENABLE_STEAM={int(ENABLE_STEAM)}"
    )

    if TEST_MODE:
        send_one(f"✅ NBA betting agent test OK ({ts_et})")
        return

    state = load_state()
    old_players = state.get("players", {})

    # Build today's prop maps once (now includes consensus + best offer)
    offers_map, ladders_points = build_today_props(now_et)

    # Injuries
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
                    offers_for_prop=offers_map.get(pt, {}),
                    state=state,
                    now_ts=now_ts
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
            ideas = slate_scan_edges(now_et, pt, offers_map.get(pt, {}), state, now_ts)
            slate_ideas_all.extend(ideas)

    # Points ladders (only points)
    ladder_out = []
    if ENABLE_LADDER_SCAN and ("points" in PROP_TYPES) and (not deadline_exceeded()):
        ladder_out = ladder_scan_points(now_et, ladders_points)

    # Combine + dedupe per market/player keeping best EV
    combined = injury_ideas_all + slate_ideas_all
    best = {}
    for i in combined:
        k = (i["prop_type"], int(i["player_id"]))
        score = (
            float(i.get("steam_score", 0.0)),
            float(i.get("ev", 0.0)),
            float(i.get("edge", 0.0)),
            float(i.get("prob_over", 0.0))
        )
        if (k not in best) or (score > best[k][0]):
            best[k] = (score, i)

    combined = [v[1] for v in best.values()]
    combined = apply_cooldown(state, combined, now_ts)

    # Per market limits
    out_by_market = {}
    for pt in PROP_TYPES:
        inj = [x for x in combined if x["prop_type"] == pt and x["section"] == "injury"]
        slt = [x for x in combined if x["prop_type"] == pt and x["section"] == "slate"]

        inj.sort(key=lambda x: (x["trigger_strength"], x["steam_score"], x["ev"], x["edge"], x["prob_over"]), reverse=True)
        slt.sort(key=lambda x: (x["steam_score"], x["ev"], x["edge"], x["prob_over"]), reverse=True)

        picks = (inj + slt)
        if MIN_PER_MARKET > 0:
            picks = picks[:max(MIN_PER_MARKET, MAX_PER_MARKET)]
        picks = picks[:MAX_PER_MARKET]
        out_by_market[pt] = picks

    # Flatten with global cap
    final_out = []
    for pt in PROP_TYPES:
        final_out.extend(out_by_market.get(pt, []))
    final_out = final_out[:MAX_TOTAL_PLAYS]

    # Apply cooldown to ladders separately
    ladder_out = apply_cooldown(state, ladder_out, now_ts)

    # Message
    if final_out or ladder_out:
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
                    steam_tag = " 🔥" if float(i.get("steam_score", 0.0)) >= STEAM_MIN_SCORE else ""
                    msg.append(
                        f"• {i['player_name']} OVER {i['line']:.1f}  "
                        f"(edge +{i['edge']:.1f}, P≈{i['prob_over']*100:.0f}%, EV≈{i['ev']:+.2f}/$1){steam_tag}"
                    )
                    msg.append(f"  Trigger: {i['trigger']}")
                    msg.append(f"  Why: {i['why']}")
                    msg.append("")

            if slt:
                msg.append("🌎 League-Wide Slate Scan (no injury required):")
                msg.append("")
                for i in slt:
                    steam_tag = " 🔥" if float(i.get("steam_score", 0.0)) >= STEAM_MIN_SCORE else ""
                    msg.append(
                        f"• {i['player_name']} OVER {i['line']:.1f}  "
                        f"(edge +{i['edge']:.1f}, P≈{i['prob_over']*100:.0f}%, EV≈{i['ev']:+.2f}/$1){steam_tag}"
                    )
                    msg.append(f"  Why: {i['why']}")
                    msg.append("")

            msg.append("")

        if ladder_out:
            msg.append("🎯 Points Ladders (value longshots):")
            msg.append("")
            for i in ladder_out:
                msg.append(
                    f"• {i['player_name']} {int(i['line'])}+ Points  "
                    f"(odds {int(i['odds']):+d}, P≈{i['prob_over']*100:.0f}%, EV≈{i['ev']:+.2f}/$1)"
                )
                msg.append(f"  Vendor: {i.get('vendor','no_vendor')}")
                msg.append(f"  Why: {i['why']}")
                msg.append("")

        send_chunked("\n".join(msg).strip())

        record_sent(state, final_out, now_ts)
        record_sent(state, ladder_out, now_ts)

    else:
        print("[INFO] No plays cleared thresholds this run.")

    state["players"] = new_players
    save_state(state)

if __name__ == "__main__":
    run()
