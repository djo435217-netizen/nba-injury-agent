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

# Vendors: allow comma-separated fallback
BOOK_VENDOR_RAW = os.environ.get("BOOK_VENDOR", "fanduel").strip().lower()
BOOK_VENDORS = [v.strip() for v in BOOK_VENDOR_RAW.split(",") if v.strip()]

PROP_TYPE = os.environ.get("PROP_TYPE", "points").strip().lower()

EDGE_THRESHOLD = float(os.environ.get("EDGE_THRESHOLD", "2.0"))
LOOKBACK_GAMES = int(os.environ.get("LOOKBACK_GAMES", "10"))
TOPN_CANDIDATES = int(os.environ.get("TOPN_CANDIDATES", "6"))
MAX_BET_IDEAS = int(os.environ.get("MAX_BET_IDEAS", "10"))

W_PPM = float(os.environ.get("W_PPM", "1.0"))
W_MIN = float(os.environ.get("W_MIN", "0.18"))

SEND_NO_EDGE_PING = os.environ.get("SEND_NO_EDGE_PING", "0") == "1"
BURST_START_ET = os.environ.get("BURST_START_ET", "17:00").strip()
BURST_END_ET = os.environ.get("BURST_END_ET", "23:45").strip()

# Option B quality controls
MIN_PROB = float(os.environ.get("MIN_PROB", "0.62"))
MIN_EDGE = float(os.environ.get("MIN_EDGE", str(EDGE_THRESHOLD)))
STD_FLOOR = float(os.environ.get("STD_FLOOR", "5.0"))

# Role trend
ROLE_LOOKBACK_SHORT = int(os.environ.get("ROLE_LOOKBACK_SHORT", "3"))
ROLE_LOOKBACK_LONG = int(os.environ.get("ROLE_LOOKBACK_LONG", "10"))
ROLE_BOOST_MAX = float(os.environ.get("ROLE_BOOST_MAX", "0.18"))
ROLE_MIN_DELTA = float(os.environ.get("ROLE_MIN_DELTA", "2.0"))

# Injury boost caps
BOOST_CAP_PTS = float(os.environ.get("BOOST_CAP_PTS", "5.5"))
BOOST_CAP_MIN = float(os.environ.get("BOOST_CAP_MIN", "6.0"))

# Require meaningful vacancy (for trigger strength)
MIN_VAC_MIN = float(os.environ.get("MIN_VAC_MIN", "10.0"))
MIN_VAC_PTS = float(os.environ.get("MIN_VAC_PTS", "6.0"))

# Main-line sanity guard (kills weird alt lines like 3.5, 4.5, 14.5 for star players)
MIN_POINTS_LINE = float(os.environ.get("MIN_POINTS_LINE", "6.0"))
MAX_POINTS_LINE = float(os.environ.get("MAX_POINTS_LINE", "45.0"))

# Debug: print one sample points prop row (set to 1 temporarily)
DEBUG_POINTS_SAMPLE = os.environ.get("DEBUG_POINTS_SAMPLE", "0") == "1"


# -------------------- UTILS --------------------
def _now_et() -> datetime:
    return datetime.now(ET)

def _time_to_minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)

def _in_burst_window(now_et: datetime) -> bool:
    start = _time_to_minutes(BURST_START_ET)
    end = _time_to_minutes(BURST_END_ET)
    cur = now_et.hour * 60 + now_et.minute
    return start <= cur <= end

def _season_year(now_et: datetime) -> int:
    # NBA season label starting in Oct
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

def avg_pts_min_std(games):
    """
    games: list[(date, pts, min)]
    returns (avg_pts, avg_min, std_pts)
    """
    if not games:
        return 0.0, 0.0, 0.0
    pts = [x[1] for x in games]
    mins = [x[2] for x in games]
    n = len(pts)
    pts_avg = sum(pts) / n
    min_avg = sum(mins) / n
    var = sum((p - pts_avg) ** 2 for p in pts) / max(n, 1)
    std = math.sqrt(var)
    return pts_avg, min_avg, std

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"players": {}}
    try:
        with open(STATE_FILE, "r") as f:
            raw = json.load(f)
        if isinstance(raw, dict) and "players" in raw:
            return raw
        if isinstance(raw, dict):
            return {"players": raw}
        return {"players": {}}
    except Exception:
        return {"players": {}}

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

def status_in_scope(status: str) -> bool:
    return (status or "").strip().lower() in IMPACT_STATUSES


# -------------------- BALLDONTLIE (RETRY + FALLBACK) --------------------
BDL_HEADERS = {"Authorization": BALLDONTLIE_API_KEY}
BDL_PREFIXES = ["/nba", ""]  # try NBA namespace first

TEAM_CACHE = None
PROPS_CACHE = {}  # (game_id, vendor_key) -> list
DEBUG_PRINTED_ONCE = False

BDL_MAX_RETRIES = int(os.environ.get("BDL_MAX_RETRIES", "5"))
BDL_RETRY_BASE_SEC = float(os.environ.get("BDL_RETRY_BASE_SEC", "1.5"))
BDL_PER_PAGE = int(os.environ.get("BDL_PER_PAGE", "100"))
BDL_MAX_PAGES = int(os.environ.get("BDL_MAX_PAGES", "10"))

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
                    if retry_after:
                        sleep_s = float(retry_after)
                    else:
                        sleep_s = BDL_RETRY_BASE_SEC * (2 ** attempt)
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
        chunk = resp.get("data") or []
        players.extend(chunk)
        meta = resp.get("meta") or {}
        cursor = meta.get("next_cursor")
        pages += 1
        if not cursor:
            break

    out = []
    for p in players:
        team = p.get("team") or {}
        if (team.get("name") or "").strip() != team_short:
            continue
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
    out = {pid: [] for pid in player_ids}
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

        done = all(len(out[pid]) >= n for pid in player_ids)
        if done:
            break

        meta = resp.get("meta") or {}
        cursor = meta.get("next_cursor")
        pages += 1
        if not cursor:
            break

    for pid in player_ids:
        g = out[pid]
        g.sort(key=lambda x: x[0])
        out[pid] = g[-n:]
    return out

def bdl_games_today_ids(now_et: datetime):
    today = now_et.strftime("%Y-%m-%d")
    resp = _bdl_get("/v1/games", params={"dates[]": [today], "per_page": 100})
    return [int(g["id"]) for g in (resp.get("data") or []) if g.get("id") is not None]

def bdl_player_props_points(game_id: int, vendor: str | None):
    """
    Fetch points props for a game_id. Cached per run per vendor.
    """
    global DEBUG_PRINTED_ONCE

    vendor_key = vendor or ""
    cache_key = (int(game_id), vendor_key)
    if cache_key in PROPS_CACHE:
        return PROPS_CACHE[cache_key]

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

    PROPS_CACHE[cache_key] = props
    return props

def _pick_main_points_line(rows_for_player):
    """
    Choose "main" points line:
      - Prefer line whose over/under odds are closest to -110/-110
      - Else median line
    Also enforces MIN_POINTS_LINE..MAX_POINTS_LINE to kill alt junk.
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
    if len(lines) % 2 == 1:
        return lines[mid]
    return 0.5 * (lines[mid - 1] + lines[mid])

def points_line_for_player(game_id: int, player_id: int):
    """
    Find player's MAIN points line for this game by vendor preference.
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


# -------------------- SMART BETTING LOGIC (Trigger Strength + Absorption) --------------------
def _role_trend(games):
    """
    games: list[(date, pts, min)] sorted ascending
    returns: mins_short, mins_long, ppm_short, ppm_long
    """
    if not games:
        return 0.0, 0.0, 0.0, 0.0
    g = list(games)
    long_n = min(len(g), ROLE_LOOKBACK_LONG)
    short_n = min(len(g), ROLE_LOOKBACK_SHORT)

    long_slice = g[-long_n:]
    short_slice = g[-short_n:]

    pts_l, min_l, _ = avg_pts_min_std(long_slice)
    pts_s, min_s, _ = avg_pts_min_std(short_slice)

    ppm_l = pts_l / max(min_l, 1e-6)
    ppm_s = pts_s / max(min_s, 1e-6)
    return min_s, min_l, ppm_s, ppm_l

def build_team_bet_ideas(team_short, injured_name, injured_status,
                         exclude_names_lower, now_et):

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

    inj_stats = bdl_last_n_games_stats([injured_pid], season, LOOKBACK_GAMES).get(injured_pid, [])
    ip, im, _ = avg_pts_min_std(inj_stats)
    if len(inj_stats) < 3:
        return []

    # Status certainty
    status = (injured_status or "").lower()
    STATUS_MULT = {"out": 1.0, "doubtful": 0.8, "questionable": 0.55}.get(status, 0.65)

    vac_pts = ip * STATUS_MULT
    vac_min = im * STATUS_MULT

    # Require meaningful vacancy to be a “real” trigger
    if not ((vac_min >= MIN_VAC_MIN) or (vac_pts >= MIN_VAC_PTS)):
        return []

    # Trigger Strength (0–100-ish)
    # Weighted by both minutes + points; scaled modestly.
    trigger_strength = min(100.0, (vac_min * 1.2 + vac_pts * 1.5))

    # Pull stats for roster
    pids = [pid for pid, _ in roster_tuples]
    stats = bdl_last_n_games_stats(pids, season, LOOKBACK_GAMES)

    # Score candidates by absorption fit + role trend
    scored = []
    for pid, nm in roster_tuples:
        g = stats.get(pid, [])
        pts_avg, min_avg, std_pts = avg_pts_min_std(g)
        if min_avg < 10:
            continue

        min_s, min_l, ppm_s, ppm_l = _role_trend(g)
        min_delta = min_s - min_l
        ppm_delta = ppm_s - ppm_l

        ppm = pts_avg / max(min_avg, 1e-6)

        # Absorption factor (0..0.65)
        absorption = 0.0
        if min_avg >= 28:
            absorption += 0.30
        if min_avg >= 34:
            absorption += 0.10
        if min_delta >= ROLE_MIN_DELTA:
            absorption += 0.15
        if ppm_delta > 0.05:
            absorption += 0.10
        absorption = min(0.65, absorption)

        # Score (keeps your style but adds trigger strength)
        score = (W_PPM * ppm) + (W_MIN * min_avg) + (absorption * 6.0) + (trigger_strength / 35.0)

        scored.append((score, pid, nm, pts_avg, min_avg, ppm, std_pts, absorption, min_delta, ppm_delta))

    scored.sort(reverse=True)
    candidates = scored[:max(TOPN_CANDIDATES, 8)]
    if not candidates:
        return []

    # Find today’s games and lines
    game_ids = bdl_games_today_ids(now_et)
    if not game_ids:
        return []

    total_score = sum(c[0] for c in candidates) or 1.0
    ideas = []

    for score, pid, nm, pts_avg, min_avg, ppm, std_pts, absorption, min_delta, ppm_delta in candidates:
        line = None
        use_gid = None
        for gid in game_ids:
            line = points_line_for_player(gid, pid)
            if line is not None:
                use_gid = gid
                break
        if line is None:
            continue

        # Projection:
        # anchor toward line to reduce “same guy always”
        base_proj = 0.60 * pts_avg + 0.40 * line

        # Injury-driven boost (allocated by absorption)
        injury_boost_pts = min(BOOST_CAP_PTS, vac_pts * absorption * 0.65)
        injury_boost_min = min(BOOST_CAP_MIN, vac_min * absorption * 0.25)

        proj = base_proj + injury_boost_pts + (injury_boost_min * ppm * 0.20)

        # Role bonus multiplicative (changes slate-to-slate)
        role_bonus = 0.0
        if min_delta >= ROLE_MIN_DELTA:
            role_bonus += min(ROLE_BOOST_MAX, (min_delta / 10.0) * 0.10)
        if ppm_delta > 0.05:
            role_bonus += min(ROLE_BOOST_MAX, (ppm_delta / 0.25) * 0.08)
        proj *= (1.0 + min(ROLE_BOOST_MAX, role_bonus))

        edge = proj - line
        if edge < MIN_EDGE:
            continue

        sigma = max(STD_FLOOR, std_pts if std_pts > 0 else STD_FLOOR)
        z = (proj - line) / sigma
        prob_over = _norm_cdf(z)
        if prob_over < MIN_PROB:
            continue

        why = (
            f"TriggerStrength {trigger_strength:.0f} | Absorb {absorption:.2f}. "
            f"{injured_name} {injured_status.upper()} vacates ~{vac_pts:.1f} pts / {vac_min:.1f} min. "
            f"{nm} L{LOOKBACK_GAMES}: {pts_avg:.1f} pts in {min_avg:.1f} min (ppm {ppm:.2f}). "
            f"Role Δmin={min_delta:+.1f}, Δppm={ppm_delta:+.2f}. "
            f"Proj {proj:.1f} vs MAIN line {line:.1f} | edge +{edge:.1f} | P≈{prob_over*100:.0f}%."
        )

        ideas.append({
            "player_name": nm,
            "player_id": pid,
            "line": line,
            "proj": proj,
            "edge": edge,
            "prob_over": prob_over,
            "why": why,
            "trigger_strength": trigger_strength,
            "trigger": f"{injured_name} ({team_short}) {injured_status}",
            "game_id": use_gid,
        })

    ideas.sort(key=lambda x: (x["trigger_strength"], x["edge"], x["prob_over"]), reverse=True)
    return ideas[:MAX_BET_IDEAS]


# -------------------- MAIN --------------------
def run():
    now_et = _now_et()
    ts_et = now_et.strftime("%Y-%m-%d %I:%M %p ET")

    # quick boot print (helps logs)
    print(
        f"[BOOT] ts={ts_et} TEST_MODE={int(TEST_MODE)} "
        f"MIN_EDGE={MIN_EDGE} MIN_PROB={MIN_PROB} "
        f"BOOK_VENDORS={','.join(BOOK_VENDORS)}"
    )

    if TEST_MODE:
        send_one(f"✅ NBA betting agent test OK ({ts_et})")
        return

    state = load_state()
    old_players = state.get("players", {})

    sr = fetch_sportradar_injuries()
    new_players = parse_injuries(sr)

    exclude_names_lower = {_clean_name(v.get("name", "")) for v in new_players.values() if v.get("name")}

    triggers = []
    bet_ideas = []

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

        ideas = build_team_bet_ideas(
            team_short=team_short,
            injured_name=injured_name,
            injured_status=injured_status,
            exclude_names_lower=exclude_names_lower | {_clean_name(injured_name)},
            now_et=now_et
        )

        if ideas:
            triggers.append(f"{injured_name} ({team_short}) {injured_status}")

        bet_ideas.extend(ideas)

    # Dedupe by player name: keep best trigger_strength then edge
    best = {}
    for i in bet_ideas:
        k = _clean_name(i["player_name"])
        if (k not in best) or ((i["trigger_strength"], i["edge"]) > (best[k]["trigger_strength"], best[k]["edge"])):
            best[k] = i

    bet_ideas = sorted(best.values(), key=lambda x: (x["trigger_strength"], x["edge"], x["prob_over"]), reverse=True)[:MAX_BET_IDEAS]

    if bet_ideas:
        msg = [f"💰 FanDuel Points Bet Ideas ({ts_et})", ""]
        if triggers:
            msg.append("Triggers:")
            for t in triggers[:8]:
                msg.append(f"- {t}")
            if len(triggers) > 8:
                msg.append(f"- …and {len(triggers)-8} more")
            msg.append("")

        for i in bet_ideas:
            msg.append(f"• {i['player_name']} OVER {i['line']:.1f}  (edge +{i['edge']:.1f})")
            msg.append(f"  Trigger: {i['trigger']}")
            msg.append(f"  Why: {i['why']}")
            msg.append("")

        send_chunked("\n".join(msg).strip())
    else:
        if SEND_NO_EDGE_PING and _in_burst_window(now_et):
            send_one(f"🧠 No MAIN-line points edges ≥ {MIN_EDGE:.1f} and P ≥ {MIN_PROB:.2f}. ({ts_et})")

    state["players"] = new_players
    save_state(state)

if __name__ == "__main__":
    run()
