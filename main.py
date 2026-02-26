import os
import json
import re
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

# Which injury statuses should trigger bet logic
IMPACT_STATUSES_RAW = os.environ.get("IMPACT_STATUSES", "out,doubtful").strip()
IMPACT_STATUSES = {x.strip().lower() for x in IMPACT_STATUSES_RAW.split(",") if x.strip()}
IMPACT_ONLY_CHANGES = os.environ.get("IMPACT_ONLY_CHANGES", "1") == "1"

# Betting / props
BOOK_VENDOR = os.environ.get("BOOK_VENDOR", "fanduel").strip().lower()
PROP_TYPE = os.environ.get("PROP_TYPE", "points").strip().lower()  # points only for now
EDGE_THRESHOLD = float(os.environ.get("EDGE_THRESHOLD", "1.0"))     # proj - line >= this
LOOKBACK_GAMES = int(os.environ.get("LOOKBACK_GAMES", "10"))
TOPN_CANDIDATES = int(os.environ.get("TOPN_CANDIDATES", "4"))       # evaluate top N teammates
MAX_BET_IDEAS = int(os.environ.get("MAX_BET_IDEAS", "8"))

# Candidate ranking weights (points-prop)
W_PPM = float(os.environ.get("W_PPM", "1.0"))       # points-per-minute
W_MIN = float(os.environ.get("W_MIN", "0.18"))      # minutes stability

# Pre-tip burst ping if no edges (optional)
SEND_NO_EDGE_PING = os.environ.get("SEND_NO_EDGE_PING", "0") == "1"
BURST_START_ET = os.environ.get("BURST_START_ET", "17:00").strip()
BURST_END_ET = os.environ.get("BURST_END_ET", "22:30").strip()

# Safety caps
BDL_PER_PAGE = int(os.environ.get("BDL_PER_PAGE", "100"))
BDL_MAX_PAGES = int(os.environ.get("BDL_MAX_PAGES", "10"))

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
    # Feb 2026 belongs to 2025 season
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
    # Normalize suffixes + punctuation for matching
    s = (s or "").strip()
    s = re.sub(r"\.", "", s)
    s = re.sub(r"\s+", " ", s)
    s = s.replace("’", "'")
    return s.lower()

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

# -------------------- BALLDONTLIE (ROBUST ROUTING) --------------------
BDL_HEADERS = {"Authorization": BALLDONTLIE_API_KEY}
# Try NBA namespace first, then fall back to legacy.
BDL_PREFIXES = ["/nba", ""]

def _bdl_get(path: str, params: dict | None = None, timeout: int = 20) -> dict:
    """
    Tries /nba-prefixed routes first (if supported), then falls back to legacy.
    """
    last_err = None
    for pref in BDL_PREFIXES:
        url = f"https://api.balldontlie.io{pref}{path}"
        try:
            r = requests.get(url, headers=BDL_HEADERS, params=params or {}, timeout=timeout)
            if r.status_code == 404:
                last_err = f"404 {url}"
                continue
            if r.status_code != 200:
                raise RuntimeError(f"BallDontLie error {r.status_code}: {r.text[:300]}")
            return r.json()
        except Exception as e:
            last_err = str(e)
            continue
    raise RuntimeError(f"BallDontLie request failed for {path}. Last error: {last_err}")

_TEAM_CACHE = None

def bdl_team_name_to_id():
    global _TEAM_CACHE
    if _TEAM_CACHE is not None:
        return _TEAM_CACHE
    data = _bdl_get("/v1/teams", params={"per_page": 100})
    m = {}
    for t in data.get("data", []):
        nm = (t.get("name") or "").strip()  # "Hawks", "Knicks", ...
        if nm and t.get("id") is not None:
            m[nm] = int(t["id"])
    _TEAM_CACHE = m
    return _TEAM_CACHE

def bdl_active_roster(team_short: str) -> list[dict]:
    """
    Returns list of player dicts for that team, TEAM-CORRECT (hard checked).
    """
    team_map = bdl_team_name_to_id()
    team_id = team_map.get(team_short)
    if not team_id:
        return []

    players = []
    cursor = None
    pages = 0
    while pages < 5:
        params = {
            "per_page": 100,
            "team_ids[]": [team_id],  # IMPORTANT: array param
        }
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

    # HARD CHECK team correctness
    out = []
    for p in players:
        team = p.get("team") or {}
        if (team.get("name") or "").strip() != team_short:
            continue
        out.append(p)
    return out

def bdl_find_player_id_on_team(team_short: str, full_name: str) -> int | None:
    """
    Best-effort match injured player name to BDL player on that team roster.
    Handles suffixes (III), punctuation differences, etc.
    """
    roster = bdl_active_roster(team_short)
    if not roster:
        return None

    target = _clean_name(full_name)

    # Common normalization: remove suffix tokens for matching
    def strip_suffix(n: str) -> str:
        n = _clean_name(n)
        n = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", n).strip()
        n = re.sub(r"\s+", " ", n)
        return n

    t0 = strip_suffix(target)

    # Exact-ish matches first
    for p in roster:
        pid = p.get("id")
        nm = f"{p.get('first_name','')} {p.get('last_name','')}".strip()
        if not pid or not nm:
            continue
        if strip_suffix(nm) == t0:
            return int(pid)

    # Fuzzy: last name match + first initial
    # (works well for “Murphy III” vs “Trey Murphy”)
    try:
        t_parts = t0.split(" ")
        t_first = t_parts[0] if t_parts else ""
        t_last = t_parts[-1] if t_parts else ""
        for p in roster:
            pid = p.get("id")
            nm = f"{p.get('first_name','')} {p.get('last_name','')}".strip()
            if not pid or not nm:
                continue
            n0 = strip_suffix(nm)
            n_parts = n0.split(" ")
            n_first = n_parts[0] if n_parts else ""
            n_last = n_parts[-1] if n_parts else ""
            if n_last == t_last and n_first[:1] == t_first[:1]:
                return int(pid)
    except Exception:
        pass

    return None

def bdl_last_n_games_stats(player_ids: list[int], season: int, n: int) -> dict[int, list[tuple[str, float, float]]]:
    """
    Returns dict: pid -> list[(date, pts, minutes)] for last n games.
    """
    out = {pid: [] for pid in player_ids}
    if not player_ids:
        return out

    cursor = None
    pages = 0
    while pages < BDL_MAX_PAGES:
        params = {
            "per_page": min(BDL_PER_PAGE, 100),
            "seasons[]": [season],
            "player_ids[]": player_ids,
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

            game = row.get("game") or {}
            date = game.get("date")
            pts = float(row.get("pts", 0) or 0)
            mins = _parse_minutes(row.get("min"))
            if date:
                out[pid].append((date, pts, mins))

        # stop early if everyone has n
        done = True
        for pid in player_ids:
            if len(out[pid]) < n:
                done = False
                break
        if done:
            break

        meta = resp.get("meta") or {}
        cursor = meta.get("next_cursor")
        pages += 1
        if not cursor:
            break

    # finalize: sort by date and keep last n
    for pid in player_ids:
        g = out[pid]
        g.sort(key=lambda x: x[0])
        out[pid] = g[-n:]
    return out

def bdl_games_today_ids(now_et: datetime) -> list[int]:
    today = now_et.strftime("%Y-%m-%d")
    resp = _bdl_get("/v1/games", params={"dates[]": [today], "per_page": 100})
    return [int(g["id"]) for g in (resp.get("data") or []) if g.get("id") is not None]

def bdl_player_props_points(game_id: int) -> list[dict]:
    """
    Uses the odds namespace for NBA if available.
    (Docs show /nba/v2/odds and player props usage + vendors like fanduel.)  [oai_citation:1‡Balldontlie](https://www.balldontlie.io/blog/nba-prediction-markets-comparison/?utm_source=chatgpt.com)
    """
    params = {"game_id": game_id, "prop_type": "points", "vendors[]": [BOOK_VENDOR]}
    # Note: We call /v2 here and let _bdl_get try /nba + fallback
    resp = _bdl_get("/v2/odds/player_props", params=params)
    return resp.get("data") or []

def points_line_for_player(game_id: int, player_id: int) -> float | None:
    for pp in bdl_player_props_points(game_id):
        if int(pp.get("player_id", -1)) != int(player_id):
            continue
        if (pp.get("prop_type") or "").lower() != "points":
            continue
        market = pp.get("market") or {}
        if (market.get("type") or "").lower() != "over_under":
            continue
        try:
            return float(pp.get("line_value"))
        except Exception:
            return None
    return None

# -------------------- BETTING LOGIC --------------------
def avg_pts_min(games: list[tuple[str, float, float]]) -> tuple[float, float]:
    if not games:
        return 0.0, 0.0
    pts = sum(x[1] for x in games) / len(games)
    mins = sum(x[2] for x in games) / len(games)
    return pts, mins

def build_team_bet_ideas(team_short: str, injured_name: str, injured_status: str,
                         exclude_names_lower: set[str], now_et: datetime) -> list[dict]:
    """
    Produces bet ideas for this team from a single injury trigger.
    """
    season = _season_year(now_et)

    # 1) roster + stats for roster
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

    # 2) injured player's vacated pts/min from last N games (best-effort match)
    injured_pid = bdl_find_player_id_on_team(team_short, injured_name)
    vac_pts, vac_min = 12.0, 26.0  # fallback conservative
    if injured_pid is not None:
        inj_stats = bdl_last_n_games_stats([injured_pid], season, LOOKBACK_GAMES).get(injured_pid, [])
        ip, im = avg_pts_min(inj_stats)
        # only trust if we have a few games
        if len(inj_stats) >= max(3, LOOKBACK_GAMES // 3):
            vac_pts, vac_min = ip, im

    # 3) teammate stats
    pids = [pid for pid, _ in roster_tuples]
    stats = bdl_last_n_games_stats(pids, season, LOOKBACK_GAMES)

    # 4) rank candidates by "absorption ability" (ppm + minutes stability)
    scored = []
    for pid, nm in roster_tuples:
        g = stats.get(pid, [])
        pts_avg, min_avg = avg_pts_min(g)
        if min_avg < 8:  # ignore deep bench
            continue
        ppm = pts_avg / max(min_avg, 1e-6)
        score = (W_PPM * ppm) + (W_MIN * min_avg)
        scored.append((score, pid, nm, pts_avg, min_avg, ppm))

    scored.sort(reverse=True, key=lambda x: x[0])
    candidates = scored[:max(TOPN_CANDIDATES, 6)]
    if not candidates:
        return []

    # 5) locate today's game(s) for this team by checking which games have candidate props listed
    game_ids = bdl_games_today_ids(now_et)
    relevant = []
    cand_ids = {c[1] for c in candidates}
    for gid in game_ids:
        props = bdl_player_props_points(gid)
        if any(int(pp.get("player_id", -1)) in cand_ids for pp in props):
            relevant.append(gid)

    if not relevant:
        return []

    # 6) projection model: baseline pts_avg + share of vacated pts (dampened)
    total_score = sum(c[0] for c in candidates) or 1.0
    ideas = []

    for score, pid, nm, pts_avg, min_avg, ppm in candidates:
        # pull FD line (over/under line value) for tonight
        line = None
        use_gid = None
        for gid in relevant:
            line = points_line_for_player(gid, pid)
            if line is not None:
                use_gid = gid
                break
        if line is None:
            continue

        share = score / total_score
        boost_pts = vac_pts * share * 0.70  # dampening to avoid overconfidence
        # small minutes effect: if they already play a lot, they absorb more
        boost_min = vac_min * share * 0.35
        proj = pts_avg + boost_pts + (boost_min * ppm * 0.20)

        edge = proj - line
        if edge < EDGE_THRESHOLD:
            continue

        why = (
            f"{injured_name} {injured_status.upper()} → vacates ~{vac_pts:.1f} pts / {vac_min:.1f} min (L{LOOKBACK_GAMES}). "
            f"{nm} L{LOOKBACK_GAMES}: {pts_avg:.1f} pts in {min_avg:.1f} min (ppm {ppm:.2f}). "
            f"Proj {proj:.1f} vs FD line {line:.1f} (edge +{edge:.1f})."
        )

        ideas.append({
            "player_name": nm,
            "player_id": pid,
            "line": line,
            "proj": proj,
            "edge": edge,
            "why": why,
            "game_id": use_gid,
        })

    ideas.sort(key=lambda x: x["edge"], reverse=True)
    return ideas[:MAX_BET_IDEAS]

# -------------------- MAIN --------------------
def run():
    now_et = _now_et()
    ts_et = now_et.strftime("%Y-%m-%d %I:%M %p ET")

    if TEST_MODE:
        send_one(f"✅ NBA betting agent test OK ({ts_et})")
        return

    state = load_state()
    old_players = state.get("players", {})

    sr = fetch_sportradar_injuries()
    new_players = parse_injuries(sr)

    # exclude anyone currently on injury list from being recommended
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

        triggers.append(f"{injured_name} ({team_short}) {injured_status}")

        ideas = build_team_bet_ideas(
            team_short=team_short,
            injured_name=injured_name,
            injured_status=injured_status,
            exclude_names_lower=exclude_names_lower | {_clean_name(injured_name)},
            now_et=now_et
        )
        for i in ideas:
            i["trigger"] = f"{injured_name} ({team_short}) {injured_status}"
        bet_ideas.extend(ideas)

    # dedupe by player name keep best edge
    best = {}
    for i in bet_ideas:
        k = _clean_name(i["player_name"])
        if (k not in best) or (i["edge"] > best[k]["edge"]):
            best[k] = i
    bet_ideas = sorted(best.values(), key=lambda x: x["edge"], reverse=True)[:MAX_BET_IDEAS]

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
            send_one(f"🧠 No FanDuel points edges ≥ {EDGE_THRESHOLD:.1f} this run. ({ts_et})")

    state["players"] = new_players
    save_state(state)

if __name__ == "__main__":
    run()
