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
MAX_BODY_CHARS = 1500

# statuses
IMPACT_STATUSES_RAW = os.environ.get("IMPACT_STATUSES", "out,doubtful,questionable").strip()
IMPACT_STATUSES = {x.strip().lower() for x in IMPACT_STATUSES_RAW.split(",") if x.strip()}
IMPACT_ONLY_CHANGES = os.environ.get("IMPACT_ONLY_CHANGES", "1") == "1"

# books/vendors
BOOK_VENDOR_RAW = os.environ.get("BOOK_VENDORS", os.environ.get("BOOK_VENDOR", "fanduel")).strip().lower()
BOOK_VENDORS = [v.strip() for v in BOOK_VENDOR_RAW.split(",") if v.strip()]

# prop types / markets
PROP_TYPES_RAW = os.environ.get("PROP_TYPES", "points,threes").strip().lower()
PROP_TYPES = [p.strip() for p in PROP_TYPES_RAW.split(",") if p.strip()]

# horizons
BASELINE_GAMES = int(os.environ.get("BASELINE_GAMES", "30"))
LOOKBACK_GAMES = int(os.environ.get("LOOKBACK_GAMES", "10"))
SHORT_GAMES = int(os.environ.get("SHORT_GAMES", "3"))

# projection weights
W_BASE = float(os.environ.get("W_BASE", "0.45"))
W_L10 = float(os.environ.get("W_L10", "0.35"))
W_L3 = float(os.environ.get("W_L3", "0.10"))
W_LINE = float(os.environ.get("W_LINE", "0.10"))

# thresholds
MIN_EDGE = float(os.environ.get("MIN_EDGE", "2.5"))
MIN_PROB = float(os.environ.get("MIN_PROB", "0.62"))
STD_FLOOR = float(os.environ.get("STD_FLOOR", "5.0"))

# per-market output sizes
MIN_PER_MARKET = int(os.environ.get("MIN_PER_MARKET", "0"))
MAX_PER_MARKET = int(os.environ.get("MAX_PER_MARKET", "6"))
MAX_TOTAL_PLAYS = int(os.environ.get("MAX_TOTAL_PLAYS", "10"))

# points line bounds
MIN_POINTS_LINE = float(os.environ.get("MIN_POINTS_LINE", "6.0"))
MAX_POINTS_LINE = float(os.environ.get("MAX_POINTS_LINE", "45.0"))

# threes line bounds (separate)
MIN_THREES_LINE = float(os.environ.get("MIN_THREES_LINE", "0.5"))
MAX_THREES_LINE = float(os.environ.get("MAX_THREES_LINE", "7.5"))

# guardrails
LINE_MIN_GAP = float(os.environ.get("LINE_MIN_GAP", "8.0"))
MIN_DELTA_FLOOR = float(os.environ.get("MIN_DELTA_FLOOR", "-3.0"))

# injury vacancy requirements + caps
MIN_VAC_MIN = float(os.environ.get("MIN_VAC_MIN", "10.0"))
MIN_VAC_PTS = float(os.environ.get("MIN_VAC_PTS", "6.0"))
BOOST_CAP_PTS = float(os.environ.get("BOOST_CAP_PTS", "5.5"))
BOOST_CAP_MIN = float(os.environ.get("BOOST_CAP_MIN", "6.0"))

# toggles
ENABLE_SLATE_SCAN = os.environ.get("ENABLE_SLATE_SCAN", "1") == "1"
ENABLE_INJURY_TRIGGERS = os.environ.get("ENABLE_INJURY_TRIGGERS", "1") == "1"
STRICT_INJURY_GAME_MATCH = os.environ.get("STRICT_INJURY_GAME_MATCH", "0") == "1"

# burst window
SEND_NO_EDGE_PING = os.environ.get("SEND_NO_EDGE_PING", "0") == "1"
BURST_START_ET = os.environ.get("BURST_START_ET", "17:00").strip()
BURST_END_ET = os.environ.get("BURST_END_ET", "23:45").strip()
SLATE_ONLY_IN_BURST = os.environ.get("SLATE_ONLY_IN_BURST", "0") == "1"

# slate scan cap (performance)
SLATE_SCAN_MAX_PLAYERS = int(os.environ.get("SLATE_SCAN_MAX_PLAYERS", "220"))

# cooldown
BET_COOLDOWN_MIN = int(os.environ.get("BET_COOLDOWN_MIN", "180"))
EDGE_JUMP_TO_RESEND = float(os.environ.get("EDGE_JUMP_TO_RESEND", "1.5"))

# value odds filter (optional)
VALUE_EDGE_MIN = float(os.environ.get("VALUE_EDGE_MIN", "0.05"))  # probability edge over implied prob

# debug
DEBUG_PROP_SAMPLE_TYPES_RAW = os.environ.get("DEBUG_PROP_SAMPLE_TYPES", "").strip().lower()
DEBUG_PROP_SAMPLE_TYPES = {x.strip() for x in DEBUG_PROP_SAMPLE_TYPES_RAW.split(",") if x.strip()}
DEBUG_PRINTED = set()

# runtime soft limit
RUN_MAX_SECONDS = int(os.environ.get("RUN_MAX_SECONDS", "170"))
RUN_T0 = time.time()

# ==================== UTILS ====================
def check_deadline(where: str):
    if (time.time() - RUN_T0) > RUN_MAX_SECONDS:
        # soft exit: do NOT crash cron; return signal via exception
        raise TimeoutError(f"[DEADLINE] exceeded {RUN_MAX_SECONDS}s at {where}")

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

def avg_stat_min_std(games):
    # games = [(date, stat, min)]
    if not games:
        return 0.0, 0.0, 0.0
    vals = [x[1] for x in games]
    mins = [x[2] for x in games]
    n = len(vals)
    vavg = sum(vals) / n
    mavg = sum(mins) / n
    var = sum((v - vavg) ** 2 for v in vals) / max(n, 1)
    return vavg, mavg, math.sqrt(var)

def _slice_last(games, n):
    if not games:
        return []
    return games[-min(len(games), n):]

def _role_trend(games, l10=10, l3=3):
    if not games:
        return 0.0, 0.0, 0.0, 0.0
    long_slice = _slice_last(games, l10)
    short_slice = _slice_last(games, l3)
    v_l, m_l, _ = avg_stat_min_std(long_slice)
    v_s, m_s, _ = avg_stat_min_std(short_slice)
    rate_l = v_l / max(m_l, 1e-6)
    rate_s = v_s / max(m_s, 1e-6)
    return m_s, m_l, rate_s, rate_l

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

def implied_prob_from_american(odds: float) -> float:
    # +120 => 100/(120+100) ; -120 => 120/(120+100)
    try:
        o = float(odds)
    except Exception:
        return None
    if o == 0:
        return None
    if o > 0:
        return 100.0 / (o + 100.0)
    return (-o) / ((-o) + 100.0)

# ==================== SPORTRADAR ====================
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

# ==================== BALLDONTLIE ====================
BDL_HEADERS = {"Authorization": BALLDONTLIE_API_KEY}
BDL_PREFIXES = ["/nba", ""]

BDL_MAX_RETRIES = int(os.environ.get("BDL_MAX_RETRIES", "5"))
BDL_RETRY_BASE_SEC = float(os.environ.get("BDL_RETRY_BASE_SEC", "1.5"))
BDL_PER_PAGE = int(os.environ.get("BDL_PER_PAGE", "100"))
BDL_MAX_PAGES = int(os.environ.get("BDL_MAX_PAGES", "10"))

TEAM_CACHE = None
PLAYER_NAME_CACHE = {}  # pid -> name

def _bdl_get(path: str, params=None, timeout: int = 20) -> dict:
    last_err = None
    for pref in BDL_PREFIXES:
        url = f"https://api.balldontlie.io{pref}{path}"
        for attempt in range(BDL_MAX_RETRIES):
            check_deadline(f"_bdl_get:{path}")
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

def bdl_last_n_games_stats(player_ids, season: int, n: int, stat_key: str):
    """
    stat_key: "pts" for points, "fg3m" for threes made
    Returns: pid -> [(date, stat, min)]
    """
    out = {int(pid): [] for pid in player_ids}
    if not player_ids:
        return out

    cursor = None
    pages = 0
    while pages < BDL_MAX_PAGES:
        check_deadline("bdl_last_n_games_stats")
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
            mins = _parse_minutes(row.get("min"))

            if stat_key == "pts":
                v = float(row.get("pts", 0) or 0)
            elif stat_key == "fg3m":
                v = float(row.get("fg3m", 0) or 0)
            else:
                v = 0.0

            if date:
                out[pid].append((date, v, mins))

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

# ==================== LINES PREFETCH (FAST) ====================
def _line_bounds_for_prop(prop_type: str):
    if prop_type == "points":
        return MIN_POINTS_LINE, MAX_POINTS_LINE
    # threes
    return MIN_THREES_LINE, MAX_THREES_LINE

def prefetch_today_lines(now_et: datetime):
    """
    Fetches props once for each (game_id, vendor, prop_type),
    keeps only market.type=="over_under", and returns:
      lines[(prop_type)][player_id] = best_row dict
    where best_row has line, vendor, over_odds, under_odds, game_id
    """
    game_ids = bdl_games_today_ids(now_et)
    if not game_ids:
        return {}, [], {"games": 0, "total_rows": 0, "ou_rows": 0, "milestone_rows": 0}

    lines = {pt: {} for pt in PROP_TYPES}
    counts = {"games": len(game_ids), "total_rows": 0, "ou_rows": 0, "milestone_rows": 0}

    for gid in game_ids:
        for prop_type in PROP_TYPES:
            for vendor in BOOK_VENDORS + [None]:
                check_deadline("prefetch_today_lines")
                params = {"game_id": int(gid), "prop_type": prop_type}
                if vendor:
                    params["vendors[]"] = [vendor]
                try:
                    resp = _bdl_get("/v2/odds/player_props", params=params)
                    rows = resp.get("data") or []
                except Exception:
                    rows = []

                if not rows:
                    continue

                counts["total_rows"] += len(rows)

                # optional debug sample per type
                if prop_type in DEBUG_PROP_SAMPLE_TYPES and prop_type not in DEBUG_PRINTED:
                    sample = rows[0]
                    vnm = sample.get("vendor") or "NO_VENDOR"
                    print(f"[DEBUG] SAMPLE PROP ROW ({prop_type}, vendor={vnm}): {json.dumps(sample)[:2000]}")
                    DEBUG_PRINTED.add(prop_type)

                min_line, max_line = _line_bounds_for_prop(prop_type)

                for pp in rows:
                    market = pp.get("market") or {}
                    mtype = (market.get("type") or "").lower()

                    # we ONLY model over/under
                    if mtype != "over_under":
                        if mtype == "milestone":
                            counts["milestone_rows"] += 1
                        continue

                    counts["ou_rows"] += 1

                    pid = pp.get("player_id")
                    if pid is None:
                        continue
                    try:
                        pid = int(pid)
                    except Exception:
                        continue

                    try:
                        line = float(pp.get("line_value"))
                    except Exception:
                        continue

                    if not (min_line <= line <= max_line):
                        continue

                    over_odds = market.get("over_odds")
                    under_odds = market.get("under_odds")
                    # choose the "main" line as the one closest to -110/-110 when possible
                    dist = None
                    if isinstance(over_odds, (int, float)) and isinstance(under_odds, (int, float)):
                        dist = abs(abs(float(over_odds)) - 110.0) + abs(abs(float(under_odds)) - 110.0)

                    cur = lines[prop_type].get(pid)
                    if cur is None:
                        lines[prop_type][pid] = {
                            "line": float(line),
                            "vendor": (pp.get("vendor") or (vendor or "no_vendor")),
                            "game_id": int(pp.get("game_id") or gid),
                            "over_odds": over_odds,
                            "under_odds": under_odds,
                            "dist": dist,
                        }
                    else:
                        # keep better "main" line (lower dist); fallback keep closer-to-middle line by dist None => keep existing
                        if cur.get("dist") is None and dist is not None:
                            better = True
                        elif cur.get("dist") is not None and dist is None:
                            better = False
                        elif cur.get("dist") is None and dist is None:
                            better = False
                        else:
                            better = (dist < cur.get("dist"))

                        if better:
                            lines[prop_type][pid] = {
                                "line": float(line),
                                "vendor": (pp.get("vendor") or (vendor or "no_vendor")),
                                "game_id": int(pp.get("game_id") or gid),
                                "over_odds": over_odds,
                                "under_odds": under_odds,
                                "dist": dist,
                            }

    return lines, game_ids, counts

# ==================== PROJECTION CORE ====================
def compute_projection_and_prob(games_all, line, injury_boost_pts=0.0, injury_boost_min=0.0):
    base_slice = _slice_last(games_all, BASELINE_GAMES)
    l10_slice = _slice_last(games_all, LOOKBACK_GAMES)
    l3_slice = _slice_last(games_all, SHORT_GAMES)

    base_avg, _, base_std = avg_stat_min_std(base_slice)
    l10_avg, l10_min, l10_std = avg_stat_min_std(l10_slice)
    l3_avg, _, _ = avg_stat_min_std(l3_slice)

    sigma = max(STD_FLOOR, (l10_std if l10_std > 0 else base_std if base_std > 0 else STD_FLOOR))
    proj = (W_BASE * base_avg) + (W_L10 * l10_avg) + (W_L3 * l3_avg) + (W_LINE * line)

    rate = l10_avg / max(l10_min, 1e-6)
    proj += injury_boost_pts
    proj += (injury_boost_min * rate * 0.20)

    edge = proj - line
    z = (proj - line) / sigma
    prob_over = _norm_cdf(z)
    return proj, edge, prob_over, (base_avg, l10_avg, l3_avg, l10_min, sigma, rate)

# ==================== INJURY ENGINE ====================
def build_injury_edges(team_short, injured_name, injured_status, exclude_names_lower, now_et,
                       prop_type: str, stat_key: str, today_lines_map: dict, game_ids_today: list[int]):
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

    inj_games = bdl_last_n_games_stats([injured_pid], season, BASELINE_GAMES, stat_key).get(injured_pid, [])
    ip10, im10, _ = avg_stat_min_std(_slice_last(inj_games, LOOKBACK_GAMES))
    if len(inj_games) < 3:
        return []

    status = (injured_status or "").lower()
    STATUS_MULT = {"out": 1.0, "doubtful": 0.8, "questionable": 0.55}.get(status, 0.65)

    vac_val = ip10 * STATUS_MULT
    vac_min = im10 * STATUS_MULT

    # only require vacancy for points (threes can be low but still meaningful)
    if prop_type == "points":
        if not ((vac_min >= MIN_VAC_MIN) or (vac_val >= MIN_VAC_PTS)):
            return []

    trigger_strength = min(100.0, (vac_min * 1.2 + vac_val * 1.5))

    cand_ids = [pid for pid, _ in roster_tuples]
    stats = bdl_last_n_games_stats(cand_ids, season, BASELINE_GAMES, stat_key)

    ideas = []
    for pid, nm in roster_tuples:
        check_deadline("build_injury_edges_loop")
        games = stats.get(pid, [])
        if len(games) < 6:
            continue

        # must have line today
        line_row = today_lines_map.get(int(pid))
        if not line_row:
            continue

        # strict match to today's game if you want
        if STRICT_INJURY_GAME_MATCH:
            if int(line_row.get("game_id", -1)) not in set(game_ids_today):
                continue

        line = float(line_row["line"])
        vendor_used = (line_row.get("vendor") or "no_vendor")

        # mins/rate filters
        v_l10, m_l10, _ = avg_stat_min_std(_slice_last(games, LOOKBACK_GAMES))
        if m_l10 < 10:
            continue

        # avoid “wild gap” where player average dwarfs line (often indicates wrong market/alt)
        if (v_l10 - line) > LINE_MIN_GAP:
            continue

        # trend
        m_s, m_l, rate_s, rate_l = _role_trend(games, LOOKBACK_GAMES, SHORT_GAMES)
        m_delta = m_s - m_l
        rate_delta = rate_s - rate_l

        # absorption heuristic
        absorption = 0.0
        if m_l10 >= 28:
            absorption += 0.30
        if m_l10 >= 34:
            absorption += 0.10
        if m_delta >= 2.0:
            absorption += 0.15
        if rate_delta > 0.05:
            absorption += 0.10
        absorption = min(0.65, absorption)

        # boost (units are same as stat_key)
        boost_val = min(BOOST_CAP_PTS if prop_type == "points" else 1.3, vac_val * absorption * (0.65 if prop_type == "points" else 0.50))
        boost_min = min(BOOST_CAP_MIN, vac_min * absorption * 0.25)

        proj, edge, prob_over, aux = compute_projection_and_prob(
            games_all=games,
            line=line,
            injury_boost_pts=boost_val,
            injury_boost_min=boost_min,
        )
        base_avg, l10_avg2, l3_avg, l10_min2, _, _ = aux

        if edge < MIN_EDGE or prob_over < MIN_PROB:
            continue
        if m_delta < MIN_DELTA_FLOOR and edge < (MIN_EDGE + 1.5):
            continue

        # value edge vs implied probability from odds (optional)
        value_note = ""
        ip = implied_prob_from_american(line_row.get("over_odds"))
        if ip is not None:
            val_edge = prob_over - ip
            if val_edge < VALUE_EDGE_MIN:
                continue
            value_note = f" | ValEdge +{val_edge*100:.1f}%"

        why = (
            f"TriggerStrength {trigger_strength:.0f} | Absorb {absorption:.2f}. "
            f"{injured_name} {injured_status.upper()} vacates ~{vac_val:.1f} {prop_type.title()} / {vac_min:.1f} min. "
            f"{nm} base(L{BASELINE_GAMES}) {base_avg:.1f}, L10 {l10_avg2:.1f}, L3 {l3_avg:.1f} "
            f"(mins L10 {l10_min2:.1f}). Role Δmin={m_delta:+.1f}, Δrate={rate_delta:+.2f}. "
            f"Proj {proj:.1f} vs {vendor_used} line {line:.1f} | edge +{edge:.1f} | P≈{prob_over*100:.0f}%{value_note}. "
            f"[prop_type={prop_type}]"
        )

        ideas.append({
            "section": "injury",
            "prop_type": prop_type,
            "player_name": nm,
            "player_id": int(pid),
            "line": float(line),
            "proj": float(proj),
            "edge": float(edge),
            "prob_over": float(prob_over),
            "trigger_strength": float(trigger_strength),
            "trigger": f"{injured_name} ({team_short}) {injured_status}",
            "why": why,
            "vendor": vendor_used,
            "game_id": int(line_row.get("game_id") or -1),
        })

    ideas.sort(key=lambda x: (x["trigger_strength"], x["edge"], x["prob_over"]), reverse=True)
    return ideas[:MAX_PER_MARKET]

# ==================== SLATE SCAN ENGINE ====================
def slate_scan_edges(now_et, prop_type: str, stat_key: str, today_lines_map: dict):
    if not ENABLE_SLATE_SCAN:
        return [], "ENABLE_SLATE_SCAN=0"
    if SLATE_ONLY_IN_BURST and (not _in_burst_window(now_et)):
        return [], "SLATE_ONLY_IN_BURST=1 and outside burst"

    season = _season_year(now_et)

    # choose candidates from today lines map
    pids = list(today_lines_map.keys())
    if not pids:
        return [], "No usable over_under lines in feed for this market/vendor"

    pids = pids[:SLATE_SCAN_MAX_PLAYERS]

    # batch stats pulls to keep runtime safe
    stats = {}
    batch = 60
    for i in range(0, len(pids), batch):
        check_deadline("slate_scan_stats")
        chunk = pids[i:i+batch]
        stats.update(bdl_last_n_games_stats(chunk, season, BASELINE_GAMES, stat_key))

    ideas = []
    for pid in pids:
        check_deadline("slate_scan_loop")
        games = stats.get(int(pid), [])
        if len(games) < 8:
            continue

        line_row = today_lines_map.get(int(pid))
        if not line_row:
            continue

        line = float(line_row["line"])
        vendor_used = (line_row.get("vendor") or "no_vendor")

        v_l10, m_l10, _ = avg_stat_min_std(_slice_last(games, LOOKBACK_GAMES))
        if m_l10 < 10:
            continue
        if (v_l10 - line) > LINE_MIN_GAP:
            continue

        m_s, m_l, rate_s, rate_l = _role_trend(games, LOOKBACK_GAMES, SHORT_GAMES)
        m_delta = m_s - m_l
        rate_delta = rate_s - rate_l

        proj, edge, prob_over, aux = compute_projection_and_prob(games_all=games, line=line)
        base_avg, l10_avg2, l3_avg, l10_min2, _, _ = aux

        if edge < MIN_EDGE or prob_over < MIN_PROB:
            continue
        if m_delta < MIN_DELTA_FLOOR and edge < (MIN_EDGE + 2.0):
            continue

        # value edge vs implied prob (optional)
        value_note = ""
        ip = implied_prob_from_american(line_row.get("over_odds"))
        if ip is not None:
            val_edge = prob_over - ip
            if val_edge < VALUE_EDGE_MIN:
                continue
            value_note = f" | ValEdge +{val_edge*100:.1f}%"

        name = PLAYER_NAME_CACHE.get(int(pid), f"Player {pid}")

        why = (
            f"SlateScan. base(L{BASELINE_GAMES}) {base_avg:.1f}, L10 {l10_avg2:.1f}, L3 {l3_avg:.1f} "
            f"(mins L10 {l10_min2:.1f}). Role Δmin={m_delta:+.1f}, Δrate={rate_delta:+.2f}. "
            f"Proj {proj:.1f} vs {vendor_used} line {line:.1f} | edge +{edge:.1f} | P≈{prob_over*100:.0f}%{value_note}. "
            f"[prop_type={prop_type}]"
        )

        ideas.append({
            "section": "slate",
            "prop_type": prop_type,
            "player_name": name,
            "player_id": int(pid),
            "line": float(line),
            "proj": float(proj),
            "edge": float(edge),
            "prob_over": float(prob_over),
            "trigger_strength": 0.0,
            "trigger": "No injury trigger (league-wide scan)",
            "why": why,
            "vendor": vendor_used,
            "game_id": int(line_row.get("game_id") or -1),
        })

    ideas.sort(key=lambda x: (x["edge"], x["prob_over"]), reverse=True)
    return ideas[:MAX_PER_MARKET], None

# ==================== COOLDOWN ====================
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

# ==================== MAIN ====================
def run():
    now_et = _now_et()
    ts_et = now_et.strftime("%Y-%m-%d %I:%M %p ET")
    now_ts = int(now_et.timestamp())

    print(
        f"[BOOT] ts={ts_et} TEST_MODE={int(TEST_MODE)} "
        f"PROP_TYPES={','.join(PROP_TYPES)} MIN_PER_MARKET={MIN_PER_MARKET} MAX_PER_MARKET={MAX_PER_MARKET} "
        f"MAX_TOTAL_PLAYS={MAX_TOTAL_PLAYS} BOOK_VENDORS={','.join(BOOK_VENDORS)} "
        f"ENABLE_SLATE_SCAN={int(ENABLE_SLATE_SCAN)} ENABLE_INJURY_TRIGGERS={int(ENABLE_INJURY_TRIGGERS)} "
        f"STRICT_INJURY_GAME_MATCH={int(STRICT_INJURY_GAME_MATCH)} RUN_MAX_SECONDS={RUN_MAX_SECONDS}"
    )

    if TEST_MODE:
        send_one(f"✅ NBA props agent test OK ({ts_et})")
        return

    state = load_state()
    old_players = state.get("players", {})

    # injuries (optional but recommended)
    try:
        sr = fetch_sportradar_injuries()
        new_players = parse_injuries(sr)
    except Exception as e:
        print(f"[WARN] Sportradar injuries fetch failed: {e}")
        new_players = {}

    exclude_names_lower = {_clean_name(v.get("name", "")) for v in new_players.values() if v.get("name")}

    # prefetch lines once (FAST)
    try:
        lines_by_prop, game_ids_today, counts = prefetch_today_lines(now_et)
    except TimeoutError as te:
        print(str(te))
        lines_by_prop, game_ids_today, counts = ({pt: {} for pt in PROP_TYPES}, [], {"games": 0, "total_rows": 0, "ou_rows": 0, "milestone_rows": 0})

    print(f"[COUNTS] games={counts['games']} rows_total={counts['total_rows']} ou_rows={counts['ou_rows']} milestone_rows={counts['milestone_rows']}")

    all_out = []
    notes = []

    # per prop type build outputs
    for prop_type in PROP_TYPES:
        check_deadline("per_prop_loop")

        # mapping to stat key
        if prop_type == "points":
            stat_key = "pts"
        elif prop_type in ("threes", "three_pointers_made", "threes_made"):
            stat_key = "fg3m"
        else:
            # unknown prop type => skip safely
            notes.append(f"Unknown prop_type '{prop_type}' (skip)")
            continue

        today_lines_map = lines_by_prop.get(prop_type, {}) or {}

        # injury triggers
        injury_ideas = []
        triggers = []
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

                try:
                    ideas = build_injury_edges(
                        team_short=team_short,
                        injured_name=injured_name,
                        injured_status=injured_status,
                        exclude_names_lower=exclude_names_lower | {_clean_name(injured_name)},
                        now_et=now_et,
                        prop_type=prop_type,
                        stat_key=stat_key,
                        today_lines_map=today_lines_map,
                        game_ids_today=game_ids_today
                    )
                except TimeoutError as te:
                    print(str(te))
                    ideas = []

                if ideas:
                    triggers.append(f"{injured_name} ({team_short}) {injured_status}")
                    injury_ideas.extend(ideas)

        # slate scan
        try:
            slate_ideas, slate_reason = slate_scan_edges(now_et, prop_type, stat_key, today_lines_map)
        except TimeoutError as te:
            print(str(te))
            slate_ideas, slate_reason = [], str(te)

        # combine + cooldown
        combined = injury_ideas + slate_ideas
        combined = apply_cooldown(state, combined, now_ts)

        # limit per market, but keep both injury and slate
        injury_out = sorted([i for i in combined if i["section"] == "injury"],
                            key=lambda x: (x["trigger_strength"], x["edge"], x["prob_over"]), reverse=True)[:MAX_PER_MARKET]
        slate_out = sorted([i for i in combined if i["section"] == "slate"],
                           key=lambda x: (x["edge"], x["prob_over"]), reverse=True)[:MAX_PER_MARKET]

        # enforce MIN_PER_MARKET if you want (else let it be empty)
        if MIN_PER_MARKET > 0 and (len(injury_out) + len(slate_out)) < MIN_PER_MARKET:
            # if not enough plays, we still show section + why
            pass

        # store for final message formatting
        all_out.append({
            "prop_type": prop_type,
            "triggers": triggers,
            "injury_out": injury_out,
            "slate_out": slate_out,
            "slate_reason": slate_reason,
            "lines_count": len(today_lines_map),
        })

    # finalize message
    msg = [f"💰 FanDuel Props ({ts_et})", ""]
    msg.append(f"Books: {', '.join(BOOK_VENDORS)}")
    msg.append(f"Model: MIN_EDGE={MIN_EDGE:.1f}, MIN_PROB={MIN_PROB:.2f}, ValueEdgeMin={VALUE_EDGE_MIN:.2f}")
    msg.append("")

    total_sent_candidates = []

    for block in all_out:
        prop_type = block["prop_type"]
        title = "Points" if prop_type == "points" else ("3PT Made" if prop_type in ("threes", "three_pointers_made", "threes_made") else prop_type)
        msg.append(f"🏷️ {title}")
        msg.append("")

        # injury
        msg.append("🚑 Injury-Triggered Plays:")
        if block["triggers"]:
            msg.append("Triggers:")
            for t in block["triggers"][:8]:
                msg.append(f"- {t}")
            if len(block["triggers"]) > 8:
                msg.append(f"- …and {len(block['triggers'])-8} more")
        else:
            msg.append("(No qualifying injury changes this run.)")
        msg.append("")

        if block["injury_out"]:
            for i in block["injury_out"]:
                msg.append(f"• {i['player_name']} OVER {i['line']:.1f}  (edge +{i['edge']:.1f}, P≈{i['prob_over']*100:.0f}%)")
                msg.append(f"  Trigger: {i['trigger']}")
                msg.append(f"  Why: {i['why']}")
                msg.append("")
                total_sent_candidates.append(i)
        else:
            msg.append(f"🧩 Note: No injury-based {title} plays passed filters (or no O/U lines matched vendors).")
            msg.append("")

        # slate
        msg.append("🌎 League-Wide Slate Scan (no injury required):")
        msg.append("")
        if block["slate_out"]:
            for i in block["slate_out"]:
                msg.append(f"• {i['player_name']} OVER {i['line']:.1f}  (edge +{i['edge']:.1f}, P≈{i['prob_over']*100:.0f}%)")
                msg.append(f"  Why: {i['why']}")
                msg.append("")
                total_sent_candidates.append(i)
        else:
            reason = block["slate_reason"] or "No slate plays passed filters"
            msg.append(f"🧩 Note: SlateScan empty for {title}. Reason: {reason}. LinesAvailable={block['lines_count']}.")
            msg.append("")

        msg.append("")

    # global cap across markets
    # keep top by edge*prob
    total_sent_candidates.sort(key=lambda x: (x["edge"] * x["prob_over"], x["edge"], x["prob_over"]), reverse=True)
    final_out = total_sent_candidates[:MAX_TOTAL_PLAYS]

    if final_out:
        # send full detailed message (already built)
        send_chunked("\n".join(msg).strip())
        record_sent(state, final_out, now_ts)
    else:
        if SEND_NO_EDGE_PING and _in_burst_window(now_et):
            send_one(f"🧠 No edges met filters this run. ({ts_et})")

    # save injuries state
    state["players"] = new_players
    save_state(state)

if __name__ == "__main__":
    try:
        run()
    except TimeoutError as te:
        # never hard-fail cron for timeout; just log
        print(str(te))
    except Exception as e:
        # log and exit non-zero so you can see real errors in Render logs
        raise
