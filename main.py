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

IMPACT_STATUSES_RAW = os.environ.get("IMPACT_STATUSES", "out,doubtful").strip()
IMPACT_STATUSES = {x.strip().lower() for x in IMPACT_STATUSES_RAW.split(",") if x.strip()}
IMPACT_ONLY_CHANGES = os.environ.get("IMPACT_ONLY_CHANGES", "1") == "1"

BOOK_VENDOR = os.environ.get("BOOK_VENDOR", "fanduel").strip().lower()
PROP_TYPE = os.environ.get("PROP_TYPE", "points").strip().lower()
EDGE_THRESHOLD = float(os.environ.get("EDGE_THRESHOLD", "1.0"))
LOOKBACK_GAMES = int(os.environ.get("LOOKBACK_GAMES", "10"))
TOPN_CANDIDATES = int(os.environ.get("TOPN_CANDIDATES", "4"))
MAX_BET_IDEAS = int(os.environ.get("MAX_BET_IDEAS", "8"))

W_PPM = float(os.environ.get("W_PPM", "1.0"))
W_MIN = float(os.environ.get("W_MIN", "0.18"))

SEND_NO_EDGE_PING = os.environ.get("SEND_NO_EDGE_PING", "0") == "1"
BURST_START_ET = os.environ.get("BURST_START_ET", "17:00").strip()
BURST_END_ET = os.environ.get("BURST_END_ET", "22:30").strip()

# OPTION B QUALITY CONTROLS
MIN_PROB = float(os.environ.get("MIN_PROB", "0.60"))
MIN_EDGE = float(os.environ.get("MIN_EDGE", str(EDGE_THRESHOLD)))
STD_FLOOR = float(os.environ.get("STD_FLOOR", "5.0"))

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

def avg_pts_min(games):
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

# -------------------- (SPORTRADAR + BALLDONTLIE FUNCTIONS UNCHANGED) --------------------
# Keep all your original API helper functions exactly as they were
# (fetch_sportradar_injuries, parse_injuries, bdl_active_roster, etc.)

# -------------------- UPGRADED BETTING LOGIC --------------------
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
        if pid and nm and _clean_name(nm) not in exclude_names_lower:
            roster_tuples.append((int(pid), nm))

    if not roster_tuples:
        return []

    injured_pid = bdl_find_player_id_on_team(team_short, injured_name)
    vac_pts, vac_min = 12.0, 26.0

    if injured_pid:
        inj_stats = bdl_last_n_games_stats([injured_pid], season, LOOKBACK_GAMES).get(injured_pid, [])
        ip, im, _ = avg_pts_min(inj_stats)
        if len(inj_stats) >= 3:
            vac_pts, vac_min = ip, im

    # Injury certainty weighting
    status = (injured_status or "").lower()
    STATUS_MULT = {"out":1.0,"doubtful":0.8,"questionable":0.55}.get(status,0.65)
    vac_pts *= STATUS_MULT
    vac_min *= STATUS_MULT

    pids = [pid for pid, _ in roster_tuples]
    stats = bdl_last_n_games_stats(pids, season, LOOKBACK_GAMES)

    scored = []
    for pid, nm in roster_tuples:
        g = stats.get(pid, [])
        pts_avg, min_avg, std_pts = avg_pts_min(g)
        if min_avg < 10:
            continue
        ppm = pts_avg / max(min_avg, 1e-6)
        score = (W_PPM * ppm) + (W_MIN * min_avg)
        scored.append((score, pid, nm, pts_avg, min_avg, ppm, std_pts))

    scored.sort(reverse=True)
    candidates = scored[:max(TOPN_CANDIDATES, 6)]
    if not candidates:
        return []

    game_ids = bdl_games_today_ids(now_et)
    if not game_ids:
        return []

    total_score = sum(c[0] for c in candidates) or 1.0
    ideas = []

    for score, pid, nm, pts_avg, min_avg, ppm, std_pts in candidates:
        line = None
        use_gid = None
        for gid in game_ids:
            line = points_line_for_player(gid, pid)
            if line:
                use_gid = gid
                break
        if not line:
            continue

        # regression baseline
        seasonish = (0.7 * pts_avg) + (0.3 * line)
        base_proj = 0.55 * seasonish + 0.45 * pts_avg

        share = score / total_score
        boost_pts = min(5.5, vac_pts * share * 0.75)
        boost_min = min(6.0, vac_min * share * 0.35)

        proj = base_proj + boost_pts + (boost_min * ppm * 0.20)

        edge = proj - line
        if edge < MIN_EDGE:
            continue

        sigma = max(STD_FLOOR, std_pts if std_pts > 0 else STD_FLOOR)
        z = (proj - line) / sigma
        prob_over = _norm_cdf(z)

        if prob_over < MIN_PROB:
            continue

        ideas.append({
            "player_name": nm,
            "player_id": pid,
            "line": line,
            "proj": proj,
            "edge": edge,
            "game_id": use_gid,
            "prob_over": prob_over,
            "why": f"{injured_name} {injured_status.upper()} → proj {proj:.1f} vs line {line:.1f} | edge +{edge:.1f} | P(over) {prob_over*100:.0f}%"
        })

    ideas.sort(key=lambda x: (x["edge"], x["prob_over"]), reverse=True)
    return ideas[:MAX_BET_IDEAS]
