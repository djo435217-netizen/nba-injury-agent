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

BOOK_VENDOR_RAW = os.environ.get("BOOK_VENDOR", "fanduel").strip().lower()
BOOK_VENDORS = [v.strip() for v in BOOK_VENDOR_RAW.split(",") if v.strip()]

PROP_TYPE = os.environ.get("PROP_TYPE", "points").strip().lower()

# Multi-horizon windows
BASELINE_GAMES = int(os.environ.get("BASELINE_GAMES", "30"))  # "season-ish" baseline
LOOKBACK_GAMES = int(os.environ.get("LOOKBACK_GAMES", "10"))   # recent
SHORT_GAMES = int(os.environ.get("SHORT_GAMES", "3"))          # trend

# Projection blend weights (must sum ~1, doesn't have to be exact)
W_BASE = float(os.environ.get("W_BASE", "0.45"))
W_L10 = float(os.environ.get("W_L10", "0.35"))
W_L3 = float(os.environ.get("W_L3", "0.10"))
W_LINE = float(os.environ.get("W_LINE", "0.10"))

# Output sizing
INJURY_TOPN = int(os.environ.get("INJURY_TOPN", "6"))
SLATE_TOPN = int(os.environ.get("SLATE_TOPN", "6"))
MAX_BET_IDEAS = int(os.environ.get("MAX_BET_IDEAS", "12"))

# Thresholds
MIN_EDGE = float(os.environ.get("MIN_EDGE", "2.5"))
MIN_PROB = float(os.environ.get("MIN_PROB", "0.62"))
STD_FLOOR = float(os.environ.get("STD_FLOOR", "5.0"))

# Guardrails
MIN_POINTS_LINE = float(os.environ.get("MIN_POINTS_LINE", "6.0"))
MAX_POINTS_LINE = float(os.environ.get("MAX_POINTS_LINE", "45.0"))
LINE_MIN_GAP = float(os.environ.get("LINE_MIN_GAP", "8.0"))        # if avg10 - line too large => probably stale/weird
MIN_DELTA_FLOOR = float(os.environ.get("MIN_DELTA_FLOOR", "-3.0"))  # minutes collapse floor

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
SLATE_SCAN_MAX_PLAYERS = int(os.environ.get("SLATE_SCAN_MAX_PLAYERS", "260"))  # cap API load

# Cooldown (prevents repeats)
BET_COOLDOWN_MIN = int(os.environ.get("BET_COOLDOWN_MIN", "180"))  # 3 hours
EDGE_JUMP_TO_RESEND = float(os.environ.get("EDGE_JUMP_TO_RESEND", "1.5"))  # resend early if edge improves

# Debug
DEBUG_POINTS_SAMPLE = os.environ.get("DEBUG_POINTS_SAMPLE", "0") == "1"


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

def avg_pts_min_std(games):
    if not games:
        return 0.0, 0.0, 0.0
    pts = [x[1] for x in games]
    mins = [x[2] for x in games]
    n = len(pts)
    pts_avg = sum(pts) / n
    min_avg = sum(mins) / n
    var = sum((p - pts_avg) ** 2 for p in pts) / max(n, 1)
    return pts_avg, min_avg, math.sqrt(var)

def _slice_last(games, n):
    if not games:
        return []
    return games[-min(len(games), n):]

def _role_trend(games):
    """
    Return: min_short, min_long, ppm_short, ppm_long
    """
    if not games:
        return 0.0, 0.0, 0.0, 0.0
    long_slice = _slice_last(games, LOOKBACK_GAMES)
    short_slice = _slice_last(games, SHORT_GAMES)

    pts_l, min_l, _ = avg_pts_min_std(long_slice)
    pts_s, min_s, _ = avg_pts_min_std(short_slice)
    ppm_l = pts_l / max(min_l, 1e-6)
    ppm_s = pts_s / max(min_s, 1e-6)
    return min_s, min_l, ppm_s, ppm_l

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


# -------------------- BALLDONTLIE (retry + caching) --------------------
BDL_HEADERS = {"Authorization": BALLDONTLIE_API_KEY}
BDL_PREFIXES = ["/nba", ""]

BDL_MAX_RETRIES = int(os.environ.get("BDL_MAX_RETRIES", "5"))
BDL_RETRY_BASE_SEC = float(os.environ.get("BDL_RETRY_BASE_SEC", "1.5"))
BDL_PER_PAGE = int(os.environ.get("BDL_PER_PAGE", "100"))
BDL_MAX_PAGES = int(os.environ.get("BDL_MAX_PAGES", "10"))

TEAM_CACHE = None
PROPS_CACHE = {}   # (game_id, vendor) -> list
DEBUG_PRINTED_ONCE = False

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

def bdl_last_n_games_stats(player_ids, season: int, n: int):
    """
    Returns dict[player_id] -> list[(date, pts, min)] length<=n
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

            game = row.get("game") or {}
            date = game.get("date")
            pts = float(row.get("pts", 0) or 0)
            mins = _parse_minutes(row.get("min"))
            if date:
                out[pid].append((date, pts, mins))

        # stop early if everyone has enough
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

def bdl_player_props_points(game_id: int, vendor: str | None):
    """
    Cached: returns list of prop rows for this game/vendor.
    """
    global DEBUG_PRINTED_ONCE
    key = (int(game_id), vendor or "")
    if key in PROPS_CACHE:
        return PROPS_CACHE[key]

    params = {"game_id": int(game_id), "prop_type": "points"}
    if vendor:
        params["vendors[]"] = [vendor]

    try:
        resp = _bdl_get("/v2/odds/player_props", params=params)
        props = resp.get("data") or []
    except Exception:
        props = []

    if DEBUG_POINTS_SAMPLE and (not DEBUG_PRINTED_ONCE) and props:
        print("[DEBUG] SAMPLE POINT PROP ROW:", json.dumps(props[0])[:2000])
        DEBUG_PRINTED_ONCE = True

    PROPS_CACHE[key] = props
    return props

def _pick_main_points_line(rows_for_player):
    """
    Prefer closest to -110/-110. Else median.
    Enforces MIN_POINTS_LINE..MAX_POINTS_LINE.
    """
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
        if line < MIN_POINTS_LINE or line > MAX_POINTS_LINE:
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

def points_line_for_player(game_id: int, player_id: int):
    """
    Find player's MAIN points line for this game using vendor preference fallback.
    """
    for v in BOOK_VENDORS + [None]:
        props = bdl_player_props_points(game_id, v)
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
        line = _pick_main_points_line(rows)
        if line is not None:
            return float(line)
    return None


# -------------------- PROJECTION CORE --------------------
def compute_projection_and_prob(games_all, line, injury_boost_pts=0.0, injury_boost_min=0.0):
    """
    Multi-horizon projection:
      - baseline window (L30 default)
      - L10
      - L3
      - line anchor
    """
    # Need enough data
    base_slice = _slice_last(games_all, BASELINE_GAMES)
    l10_slice = _slice_last(games_all, LOOKBACK_GAMES)
    l3_slice = _slice_last(games_all, SHORT_GAMES)

    base_avg, base_min, base_std = avg_pts_min_std(base_slice)
    l10_avg, l10_min, l10_std = avg_pts_min_std(l10_slice)
    l3_avg, l3_min, l3_std = avg_pts_min_std(l3_slice)

    # use L10 std primarily, fall back to baseline
    sigma = max(STD_FLOOR, (l10_std if l10_std > 0 else base_std if base_std > 0 else STD_FLOOR))

    # blended base
    proj = (W_BASE * base_avg) + (W_L10 * l10_avg) + (W_L3 * l3_avg) + (W_LINE * line)

    # ppm for minutes-based injury boost conversion
    ppm = l10_avg / max(l10_min, 1e-6)

    proj += injury_boost_pts
    proj += (injury_boost_min * ppm * 0.20)

    edge = proj - line
    z = (proj - line) / sigma
    prob_over = _norm_cdf(z)
    return proj, edge, prob_over, (base_avg, l10_avg, l3_avg, l10_min, sigma, ppm)


# -------------------- INJURY ENGINE --------------------
def build_injury_edges(team_short, injured_name, injured_status, exclude_names_lower, now_et):
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

    inj_games = bdl_last_n_games_stats([injured_pid], season, BASELINE_GAMES).get(injured_pid, [])
    ip10, im10, _ = avg_pts_min_std(_slice_last(inj_games, LOOKBACK_GAMES))
    if len(inj_games) < 3:
        return []

    status = (injured_status or "").lower()
    STATUS_MULT = {"out": 1.0, "doubtful": 0.8, "questionable": 0.55}.get(status, 0.65)

    vac_pts = ip10 * STATUS_MULT
    vac_min = im10 * STATUS_MULT
    if not ((vac_min >= MIN_VAC_MIN) or (vac_pts >= MIN_VAC_PTS)):
        return []

    trigger_strength = min(100.0, (vac_min * 1.2 + vac_pts * 1.5))

    # Gather candidate ids
    cand_ids = [pid for pid, _ in roster_tuples]
    stats = bdl_last_n_games_stats(cand_ids, season, BASELINE_GAMES)

    # Find today's games once
    game_ids = bdl_games_today_ids(now_et)
    if not game_ids:
        return []

    ideas = []
    for pid, nm in roster_tuples:
        games = stats.get(pid, [])
        if len(games) < 6:
            continue

        # role trend
        min_s, min_l, ppm_s, ppm_l = _role_trend(games)
        min_delta = min_s - min_l
        ppm_delta = ppm_s - ppm_l

        # minutes floor
        _, l10_min, _ = avg_pts_min_std(_slice_last(games, LOOKBACK_GAMES))
        if l10_min < 10:
            continue

        # absorption (0..0.65)
        absorption = 0.0
        if l10_min >= 28:
            absorption += 0.30
        if l10_min >= 34:
            absorption += 0.10
        if min_delta >= 2.0:
            absorption += 0.15
        if ppm_delta > 0.05:
            absorption += 0.10
        absorption = min(0.65, absorption)

        # Find line for today's game
        line = None
        use_gid = None
        for gid in game_ids:
            line = points_line_for_player(gid, pid)
            if line is not None:
                use_gid = gid
                break
        if line is None:
            continue

        # Baseline sanity vs L10
        l10_avg, _, _ = avg_pts_min_std(_slice_last(games, LOOKBACK_GAMES))
        if (l10_avg - line) > LINE_MIN_GAP:
            continue

        # Build injury boost
        injury_boost_pts = min(BOOST_CAP_PTS, vac_pts * absorption * 0.65)
        injury_boost_min = min(BOOST_CAP_MIN, vac_min * absorption * 0.25)

        proj, edge, prob_over, aux = compute_projection_and_prob(
            games_all=games,
            line=line,
            injury_boost_pts=injury_boost_pts,
            injury_boost_min=injury_boost_min
        )
        base_avg, l10_avg2, l3_avg, l10_min2, sigma, ppm = aux

        if edge < MIN_EDGE or prob_over < MIN_PROB:
            continue

        # minutes collapse guardrail
        if min_delta < MIN_DELTA_FLOOR and edge < (MIN_EDGE + 1.5):
            continue

        why = (
            f"TriggerStrength {trigger_strength:.0f} | Absorb {absorption:.2f}. "
            f"{injured_name} {injured_status.upper()} vacates ~{vac_pts:.1f} pts / {vac_min:.1f} min. "
            f"{nm} base(L{BASELINE_GAMES}) {base_avg:.1f}, L10 {l10_avg2:.1f}, L3 {l3_avg:.1f} "
            f"(mins L10 {l10_min2:.1f}). Role Δmin={min_delta:+.1f}, Δppm={ppm_delta:+.2f}. "
            f"Proj {proj:.1f} vs MAIN line {line:.1f} | edge +{edge:.1f} | P≈{prob_over*100:.0f}%."
        )

        ideas.append({
            "section": "injury",
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
        })

    ideas.sort(key=lambda x: (x["trigger_strength"], x["edge"], x["prob_over"]), reverse=True)
    return ideas[:INJURY_TOPN]


# -------------------- SLATE SCAN ENGINE --------------------
def slate_scan_edges(now_et):
    """
    League-wide scan of today's games (no injury required).
    Finds edges from multi-horizon projection + role trend + main-line.
    """
    if not ENABLE_SLATE_SCAN:
        return []

    if SLATE_ONLY_IN_BURST and (not _in_burst_window(now_et)):
        return []

    season = _season_year(now_et)
    game_ids = bdl_games_today_ids(now_et)
    if not game_ids:
        return []

    # Pull props and collect player_ids up to cap
    player_to_any_game = {}
    player_to_best_line = {}  # pid -> (line, gid)
    pulled = 0

    for gid in game_ids:
        # prefer vendor order; pick first vendor that returns data
        props = []
        for v in BOOK_VENDORS + [None]:
            props = bdl_player_props_points(gid, v)
            if props:
                break
        if not props:
            continue

        # group by player
        by_pid = {}
        for pp in props:
            pid = pp.get("player_id")
            if pid is None:
                continue
            pid = int(pid)
            by_pid.setdefault(pid, []).append(pp)

        for pid, rows in by_pid.items():
            if pid in player_to_best_line:
                continue
            line = _pick_main_points_line(rows)
            if line is None:
                continue
            player_to_any_game[pid] = int(gid)
            player_to_best_line[pid] = (float(line), int(gid))
            pulled += 1
            if pulled >= SLATE_SCAN_MAX_PLAYERS:
                break
        if pulled >= SLATE_SCAN_MAX_PLAYERS:
            break

    if not player_to_best_line:
        return []

    pids = list(player_to_best_line.keys())
    stats = bdl_last_n_games_stats(pids, season, BASELINE_GAMES)

    ideas = []
    for pid in pids:
        games = stats.get(pid, [])
        if len(games) < 8:
            continue

        line, gid = player_to_best_line[pid]

        # sanity: line too far below L10 avg (stale-ish)
        l10_avg, l10_min, _ = avg_pts_min_std(_slice_last(games, LOOKBACK_GAMES))
        if l10_min < 10:
            continue
        if (l10_avg - line) > LINE_MIN_GAP:
            continue

        # role trend
        min_s, min_l, ppm_s, ppm_l = _role_trend(games)
        min_delta = min_s - min_l
        ppm_delta = ppm_s - ppm_l

        proj, edge, prob_over, aux = compute_projection_and_prob(games_all=games, line=line)
        base_avg, l10_avg2, l3_avg, l10_min2, sigma, ppm = aux

        if edge < MIN_EDGE or prob_over < MIN_PROB:
            continue

        # minutes collapse guardrail (lighter for slate, but still helps)
        if min_delta < MIN_DELTA_FLOOR and edge < (MIN_EDGE + 2.0):
            continue

        why = (
            f"SlateScan. base(L{BASELINE_GAMES}) {base_avg:.1f}, L10 {l10_avg2:.1f}, L3 {l3_avg:.1f} "
            f"(mins L10 {l10_min2:.1f}). Role Δmin={min_delta:+.1f}, Δppm={ppm_delta:+.2f}. "
            f"Proj {proj:.1f} vs MAIN line {line:.1f} | edge +{edge:.1f} | P≈{prob_over*100:.0f}%."
        )

        ideas.append({
            "section": "slate",
            "player_name": f"Player {pid}",
            "player_id": pid,
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
    return ideas[:SLATE_TOPN]


# -------------------- COOLDOWN FILTER --------------------
def apply_cooldown(state, ideas, now_ts: int):
    """
    Skip plays sent recently, unless the line changed or edge improved enough.
    Keyed by (player_id, line, section).
    """
    sent = state.get("sent_bets", {}) or {}
    cooldown_sec = BET_COOLDOWN_MIN * 60

    kept = []
    for i in ideas:
        key = f"{i['section']}|{int(i['player_id'])}|{i['line']:.1f}"
        prev = sent.get(key)

        if not prev:
            kept.append(i)
            continue

        last_ts = int(prev.get("ts", 0) or 0)
        last_edge = float(prev.get("edge", 0.0) or 0.0)
        last_line = float(prev.get("line", i["line"]) or i["line"])

        # If line changed, resend
        if abs(last_line - float(i["line"])) >= 0.5:
            kept.append(i)
            continue

        # If edge improved materially, resend
        if (float(i["edge"]) - last_edge) >= EDGE_JUMP_TO_RESEND:
            kept.append(i)
            continue

        # else enforce cooldown time
        if (now_ts - last_ts) >= cooldown_sec:
            kept.append(i)

    return kept

def record_sent(state, ideas, now_ts: int):
    sent = state.get("sent_bets", {}) or {}
    for i in ideas:
        key = f"{i['section']}|{int(i['player_id'])}|{i['line']:.1f}"
        sent[key] = {"ts": now_ts, "edge": float(i["edge"]), "line": float(i["line"])}
    state["sent_bets"] = sent


# -------------------- MAIN --------------------
def run():
    now_et = _now_et()
    ts_et = now_et.strftime("%Y-%m-%d %I:%M %p ET")
    now_ts = int(now_et.timestamp())

    print(
        f"[BOOT] ts={ts_et} TEST_MODE={int(TEST_MODE)} "
        f"BASELINE_GAMES={BASELINE_GAMES} L10={LOOKBACK_GAMES} L3={SHORT_GAMES} "
        f"MIN_EDGE={MIN_EDGE} MIN_PROB={MIN_PROB} "
        f"BOOK_VENDORS={','.join(BOOK_VENDORS)} ENABLE_SLATE_SCAN={int(ENABLE_SLATE_SCAN)}"
    )

    if TEST_MODE:
        send_one(f"✅ NBA betting agent test OK ({ts_et})")
        return

    state = load_state()
    old_players = state.get("players", {})

    # ---------- Injury engine triggers ----------
    sr = fetch_sportradar_injuries()
    new_players = parse_injuries(sr)

    exclude_names_lower = {_clean_name(v.get("name", "")) for v in new_players.values() if v.get("name")}

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

        team_short = cur.get("team", "")
        injured_name = cur.get("name", "")
        injured_status = (cur.get("status") or "").strip()

        ideas = build_injury_edges(
            team_short=team_short,
            injured_name=injured_name,
            injured_status=injured_status,
            exclude_names_lower=exclude_names_lower | {_clean_name(injured_name)},
            now_et=now_et
        )
        if ideas:
            triggers.append(f"{injured_name} ({team_short}) {injured_status}")
            injury_ideas.extend(ideas)

    # ---------- Slate scan (no injuries required) ----------
    slate_ideas = slate_scan_edges(now_et)

    # Combine, then dedupe by player within each section preference:
    # Keep highest edge for same player_id across ideas.
    combined = injury_ideas + slate_ideas
    best = {}
    for i in combined:
        k = (i["section"], int(i["player_id"]))
        if (k not in best) or ((i["edge"], i["prob_over"]) > (best[k]["edge"], best[k]["prob_over"])):
            best[k] = i
    combined = list(best.values())

    # Apply cooldown
    combined = apply_cooldown(state, combined, now_ts)

    # Re-split for pretty output, and cap total
    injury_out = sorted([i for i in combined if i["section"] == "injury"],
                        key=lambda x: (x["trigger_strength"], x["edge"], x["prob_over"]), reverse=True)[:INJURY_TOPN]
    slate_out = sorted([i for i in combined if i["section"] == "slate"],
                       key=lambda x: (x["edge"], x["prob_over"]), reverse=True)[:SLATE_TOPN]

    # Final cap
    final_out = (injury_out + slate_out)[:MAX_BET_IDEAS]

    if final_out:
        msg = [f"💰 FanDuel Points Bet Ideas ({ts_et})", ""]

        if injury_out:
            msg.append("🚑 Injury-Triggered Plays:")
            if triggers:
                msg.append("Triggers:")
                for t in triggers[:8]:
                    msg.append(f"- {t}")
                if len(triggers) > 8:
                    msg.append(f"- …and {len(triggers)-8} more")
            msg.append("")
            for i in injury_out:
                msg.append(f"• {i['player_name']} OVER {i['line']:.1f}  (edge +{i['edge']:.1f}, P≈{i['prob_over']*100:.0f}%)")
                msg.append(f"  Trigger: {i['trigger']}")
                msg.append(f"  Why: {i['why']}")
                msg.append("")

        if slate_out:
            msg.append("🌎 League-Wide Slate Scan (no injury required):")
            msg.append("")
            for i in slate_out:
                msg.append(f"• {i['player_name']} OVER {i['line']:.1f}  (edge +{i['edge']:.1f}, P≈{i['prob_over']*100:.0f}%)")
                msg.append(f"  Why: {i['why']}")
                msg.append("")

        send_chunked("\n".join(msg).strip())

        # record sent plays (only what we actually sent)
        record_sent(state, final_out, now_ts)

    else:
        if SEND_NO_EDGE_PING and _in_burst_window(now_et):
            send_one(f"🧠 No edges ≥ {MIN_EDGE:.1f} and P ≥ {MIN_PROB:.2f} this run. ({ts_et})")

    # persist injury state always
    state["players"] = new_players
    save_state(state)

if __name__ == "__main__":
    run()
