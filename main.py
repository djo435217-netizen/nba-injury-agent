import os
import json
import re
import time
import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
from twilio.rest import Client

STATE_FILE = “state.json”
ET = ZoneInfo(“America/New_York”)

# –––––––––– REQUIRED ENV ––––––––––

TWILIO_SID = os.environ[“TWILIO_ACCOUNT_SID”]
TWILIO_TOKEN = os.environ[“TWILIO_AUTH_TOKEN”]
SPORTRADAR_KEY = os.environ.get(“SPORTRADAR_API_KEY”, “”).strip()
BALLDONTLIE_API_KEY = os.environ[“BALLDONTLIE_API_KEY”].strip()

FROM_WHATSAPP = os.environ.get(“TWILIO_WHATSAPP_FROM”, “whatsapp:+14155238886”)
TO_WHATSAPP = f”whatsapp:{os.environ[‘MY_WHATSAPP_NUMBER’]}”

twilio = Client(TWILIO_SID, TWILIO_TOKEN)

# –––––––––– CONFIG ––––––––––

TEST_MODE = os.environ.get(“TEST_MODE”, “0”) == “1”
MAX_BODY_CHARS = int(os.environ.get(“MAX_BODY_CHARS”, “1500”))

BOOK_VENDOR_RAW = os.environ.get(“BOOK_VENDORS”, os.environ.get(“BOOK_VENDOR”, “fanduel”)).strip().lower()
BOOK_VENDORS = [v.strip() for v in BOOK_VENDOR_RAW.split(”,”) if v.strip()]

PROP_TYPES_RAW = os.environ.get(“PROP_TYPES”, “points”).strip().lower()
PROP_TYPES = [p.strip() for p in PROP_TYPES_RAW.split(”,”) if p.strip()]

ENABLE_INJURY_TRIGGERS = os.environ.get(“ENABLE_INJURY_TRIGGERS”, “1”) == “1”
ENABLE_SLATE_SCAN = os.environ.get(“ENABLE_SLATE_SCAN”, “1”) == “1”

# Exposure caps

MAX_PLAYS_PER_TEAM = int(os.environ.get(“MAX_PLAYS_PER_TEAM”, “2”))
MAX_PLAYS_PER_GAME = int(os.environ.get(“MAX_PLAYS_PER_GAME”, “4”))

# Card composition caps

MAX_INJURY_PLAYS = int(os.environ.get(“MAX_INJURY_PLAYS”, “4”))
MAX_LINEUPNEWS_PLAYS = int(os.environ.get(“MAX_LINEUPNEWS_PLAYS”, “3”))
MAX_SLATE_PLAYS = int(os.environ.get(“MAX_SLATE_PLAYS”, “2”))

# Consensus + steam + EV + market respect

MIN_VENDORS_FOR_CONSENSUS = int(os.environ.get(“MIN_VENDORS_FOR_CONSENSUS”, “2”))
MIN_SHARP_VENDORS = int(os.environ.get(“MIN_SHARP_VENDORS”, “1”))
SHARP_VENDORS_RAW = os.environ.get(
“SHARP_VENDORS”,
“draftkings,caesars,betmgm,bet365,pointsbet,hardrock,betparx,betway,betrivers”,
).strip().lower()
SHARP_VENDORS = {x.strip() for x in SHARP_VENDORS_RAW.split(”,”) if x.strip()}

CONSENSUS_VENDORS = SHARP_VENDORS.copy()

ENABLE_STEAM = os.environ.get(“ENABLE_STEAM”, “0”) == “1”
STEAM_MIN_SCORE = float(os.environ.get(“STEAM_MIN_SCORE”, “1.0”))
STEAM_MAX_AGE_MIN = int(os.environ.get(“STEAM_MAX_AGE_MIN”, “240”))

# Output sizing

MIN_PER_MARKET = int(os.environ.get(“MIN_PER_MARKET”, “0”))
MAX_PER_MARKET = int(os.environ.get(“MAX_PER_MARKET”, “6”))
MAX_TOTAL_PLAYS = int(os.environ.get(“MAX_TOTAL_PLAYS”, “10”))

# Windows

BASELINE_GAMES = int(os.environ.get(“BASELINE_GAMES”, “30”))
LOOKBACK_GAMES = int(os.environ.get(“LOOKBACK_GAMES”, “10”))
SHORT_GAMES = int(os.environ.get(“SHORT_GAMES”, “3”))

# Model thresholds

MIN_EDGE = float(os.environ.get(“MIN_EDGE”, “2.5”))
MIN_PROB = float(os.environ.get(“MIN_PROB”, “0.62”))

# IMPROVEMENT: Lowered STD_FLOOR from 5.0 -> 3.5

# Old floor of 5.0 was over-penalizing consistent scorers by pushing prob

# estimates artificially toward 50%, causing us to miss high-confidence overs.

STD_FLOOR = float(os.environ.get(“STD_FLOOR”, “3.5”))

# EV filter

EV_MIN = float(os.environ.get(“EV_MIN”, “0.00”))
VALUE_EDGE_MIN = float(os.environ.get(“VALUE_EDGE_MIN”, “0.00”))

# Guardrails

MIN_L10_MIN = float(os.environ.get(“MIN_L10_MIN”, “10”))
LINE_MIN_GAP = float(os.environ.get(“LINE_MIN_GAP”, “8.0”))
ROLE_DROP_MIN = float(os.environ.get(“ROLE_DROP_MIN”, “5.0”))
ROLE_DROP_RATE = float(os.environ.get(“ROLE_DROP_RATE”, “0.08”))

# Injury vacancy requirements

IMPACT_STATUSES_RAW = os.environ.get(“IMPACT_STATUSES”, “out,doubtful,questionable”).strip()
IMPACT_STATUSES = {x.strip().lower() for x in IMPACT_STATUSES_RAW.split(”,”) if x.strip()}
IMPACT_ONLY_CHANGES = os.environ.get(“IMPACT_ONLY_CHANGES”, “1”) == “1”

MIN_VAC_MIN = float(os.environ.get(“MIN_VAC_MIN”, “10.0”))
MIN_VAC_STAT = float(os.environ.get(“MIN_VAC_PTS”, os.environ.get(“MIN_VAC_STAT”, “6.0”)))
BOOST_CAP_RATE = float(os.environ.get(“BOOST_CAP_RATE”, “0.20”))
BOOST_CAP_STAT = float(os.environ.get(“BOOST_CAP_STAT”, “5.5”))
BOOST_CAP_MIN = float(os.environ.get(“BOOST_CAP_MIN”, “6.0”))

# Cooldown

BET_COOLDOWN_MIN = int(os.environ.get(“BET_COOLDOWN_MIN”, “180”))
EDGE_JUMP_TO_RESEND = float(os.environ.get(“EDGE_JUMP_TO_RESEND”, “1.5”))

# Runtime guard

RUN_MAX_SECONDS = int(os.environ.get(“RUN_MAX_SECONDS”, “170”))
STAT_BATCH_SIZE = int(os.environ.get(“STAT_BATCH_SIZE”, “90”))

DEBUG_PROP_SAMPLE_TYPES = os.environ.get(“DEBUG_PROP_SAMPLE_TYPES”, “0”).strip().lower()

# Plus odds

PLUS_ODDS_MIN = float(os.environ.get(“PLUS_ODDS_MIN”, “100”))
PLUS_ODDS_TOPN = int(os.environ.get(“PLUS_ODDS_TOPN”, “3”))

PLUS_HUNT_ENABLED = os.environ.get(“PLUS_HUNT_ENABLED”, “1”) == “1”
PLUS_HUNT_MIN_PROB = float(os.environ.get(“PLUS_HUNT_MIN_PROB”, “0.52”))
PLUS_HUNT_MIN_VALUE_EDGE = float(os.environ.get(“PLUS_HUNT_MIN_VALUE_EDGE”, “0.03”))
PLUS_HUNT_MIN_EV = float(os.environ.get(“PLUS_HUNT_MIN_EV”, “0.01”))
PLUS_HUNT_TOPN = int(os.environ.get(“PLUS_HUNT_TOPN”, “5”))

# Threes

THREES_BETA_BINOM = os.environ.get(“THREES_BETA_BINOM”, “1”) == “1”
THREES_MIN_ATT_GAMES = int(os.environ.get(“THREES_MIN_ATT_GAMES”, “8”))

# LineupExperts

LINEUPEXPERTS = os.environ.get(“LINEUPEXPERTS”, “0”) == “1”
LINEUPEXPERTS_KEY = os.environ.get(“LINEUPEXPERTS_KEY”, os.environ.get(“LINEUPEXPERTS_API_KEY”, “”)).strip()
LINEUPEXPERTS_BASE_URL = os.environ.get(“LINEUPEXPERTS_BASE_URL”, “https://api.lineupexperts.com/v1”).strip()
LINEUPEXPERTS_BASE_URL = LINEUPEXPERTS_BASE_URL.strip().strip(’”’).strip(”’”).rstrip(”/”)
LINEUPEXPERTS_TIMEOUT = int(os.environ.get(“LINEUPEXPERTS_TIMEOUT”, “12”))
LINEUPEXPERTS_MAX_ITEMS = int(os.environ.get(“LINEUPEXPERTS_MAX_ITEMS”, “200”))
NEWS_LOOKBACK_HOURS = int(os.environ.get(“NEWS_LOOKBACK_HOURS”, “36”))
NEWS_MIN_CONFIDENCE = float(os.environ.get(“NEWS_MIN_CONFIDENCE”, “0.25”))

LE_DEBUG = os.environ.get(“LE_DEBUG”, “0”) == “1”
LE_NO_TIME_FILTER = os.environ.get(“LE_NO_TIME_FILTER”, “0”) == “1”

USE_LE_MAIN_INJURY_ENGINE = os.environ.get(“USE_LE_MAIN_INJURY_ENGINE”, “1”) == “1”
LE_NEWS_ENGINE_ENABLED = os.environ.get(“LE_NEWS_ENGINE_ENABLED”, “1”) == “1”
LE_NEWS_MIN_SCORE = float(os.environ.get(“LE_NEWS_MIN_SCORE”, “0.50”))
LE_NEWS_MIN_EFFECT = float(os.environ.get(“LE_NEWS_MIN_EFFECT”, “0.03”))
LE_NEWS_TOPN = int(os.environ.get(“LE_NEWS_TOPN”, “5”))

# Final ranking weights

FINAL_SCORE_EV_W = float(os.environ.get(“FINAL_SCORE_EV_W”, “30.0”))
FINAL_SCORE_VALUE_W = float(os.environ.get(“FINAL_SCORE_VALUE_W”, “100.0”))
FINAL_SCORE_EDGE_W = float(os.environ.get(“FINAL_SCORE_EDGE_W”, “2.0”))
FINAL_SCORE_MINCONF_W = float(os.environ.get(“FINAL_SCORE_MINCONF_W”, “8.0”))
FINAL_SCORE_MATCHUP_W = float(os.environ.get(“FINAL_SCORE_MATCHUP_W”, “10.0”))
FINAL_SCORE_LE_W = float(os.environ.get(“FINAL_SCORE_LE_W”, “8.0”))
FINAL_SCORE_STABILITY_W = float(os.environ.get(“FINAL_SCORE_STABILITY_W”, “8.0”))
FINAL_SCORE_VOL_PENALTY_W = float(os.environ.get(“FINAL_SCORE_VOL_PENALTY_W”, “6.0”))

# –––––––––– NEW: CONTEXT FACTORS ––––––––––

# Home court advantage: historically ~2.5 pts for the home team player

HOME_COURT_BOOST = float(os.environ.get(“HOME_COURT_BOOST”, “1.5”))

# B2B fatigue penalty: second night of back-to-back games

B2B_PENALTY = float(os.environ.get(“B2B_PENALTY”, “2.0”))

# Pace factor weight: how much opponent pace affects projection (0.0 = off)

PACE_FACTOR_WEIGHT = float(os.environ.get(“PACE_FACTOR_WEIGHT”, “0.5”))

# Enable opponent defensive stats adjustment

ENABLE_OPP_DEF_ADJ = os.environ.get(“ENABLE_OPP_DEF_ADJ”, “1”) == “1”

# –––––––––– NEW: RECENCY WEIGHTS ––––––––––

# IMPROVEMENT: Increased L3 recency weight from 0.20 -> 0.30, reduced base from 0.45 -> 0.35

# L3 form is the strongest short-term signal for player points props.

# Hot/cold streaks are real – the market often under-adjusts for them.

PROJ_WEIGHT_BASE = float(os.environ.get(“PROJ_WEIGHT_BASE”, “0.35”))
PROJ_WEIGHT_L10 = float(os.environ.get(“PROJ_WEIGHT_L10”, “0.35”))
PROJ_WEIGHT_L3 = float(os.environ.get(“PROJ_WEIGHT_L3”, “0.30”))

# –––––––––– NEW: RESULT TRACKING ––––––––––

# Track sent plays so we can measure hit rate over time

ENABLE_RESULT_TRACKING = os.environ.get(“ENABLE_RESULT_TRACKING”, “1”) == “1”
RESULTS_FILE = os.environ.get(“RESULTS_FILE”, “results_log.json”)

# –––––––––– RUNTIME DEADLINE ––––––––––

RUN_START = time.time()

# –––––––––– PLAYER TEAM OVERRIDES ––––––––––

PLAYER_TEAM_OVERRIDES = {
“tidjane salaun”: “Hornets”,
“vit krejci”: “Hawks”,
“daniel gafford”: “Mavericks”,
“isaac okoro”: “Cavaliers”,
“lachlan olbrich”: “Bulls”,
“jaylen brown”: “Celtics”,
“stephen curry”: “Warriors”,
“draymond green”: “Warriors”,
“marcus smart”: “Grizzlies”,
“nick richards”: “Suns”,
“jaden mcdaniels”: “Timberwolves”,
}

# –––––––––– TEAM DEFENSE RATINGS ––––––––––

# Points allowed per game vs position (updated manually or via future API hook).

# Format: team_short -> {“pts_allowed_pg”: float, “pace”: float}

# pts_allowed_pg: average points allowed per game this season (league avg ~113)

# pace: possessions per 48 min (league avg ~99)

# Positive pace = faster = more possessions = more pts scoring opportunities

TEAM_DEFENSE_RATINGS: dict[str, dict] = {
“Celtics”:      {“pts_allowed_pg”: 107.5, “pace”: 97.5},
“Bucks”:        {“pts_allowed_pg”: 112.0, “pace”: 100.2},
“Knicks”:       {“pts_allowed_pg”: 110.3, “pace”: 96.8},
“Heat”:         {“pts_allowed_pg”: 109.4, “pace”: 96.3},
“76ers”:        {“pts_allowed_pg”: 113.2, “pace”: 98.1},
“Nets”:         {“pts_allowed_pg”: 117.8, “pace”: 99.4},
“Raptors”:      {“pts_allowed_pg”: 115.6, “pace”: 98.7},
“Cavaliers”:    {“pts_allowed_pg”: 108.1, “pace”: 97.2},
“Pacers”:       {“pts_allowed_pg”: 118.4, “pace”: 104.5},
“Bulls”:        {“pts_allowed_pg”: 114.9, “pace”: 99.8},
“Bucks”:        {“pts_allowed_pg”: 112.0, “pace”: 100.2},
“Pistons”:      {“pts_allowed_pg”: 116.2, “pace”: 98.9},
“Hawks”:        {“pts_allowed_pg”: 116.8, “pace”: 101.3},
“Hornets”:      {“pts_allowed_pg”: 118.1, “pace”: 100.6},
“Magic”:        {“pts_allowed_pg”: 108.9, “pace”: 96.1},
“Wizards”:      {“pts_allowed_pg”: 119.3, “pace”: 100.4},
“Thunder”:      {“pts_allowed_pg”: 106.8, “pace”: 98.3},
“Nuggets”:      {“pts_allowed_pg”: 111.7, “pace”: 97.9},
“Timberwolves”: {“pts_allowed_pg”: 108.2, “pace”: 97.6},
“Jazz”:         {“pts_allowed_pg”: 118.9, “pace”: 101.8},
“Suns”:         {“pts_allowed_pg”: 113.5, “pace”: 99.1},
“Clippers”:     {“pts_allowed_pg”: 111.1, “pace”: 97.4},
“Lakers”:       {“pts_allowed_pg”: 113.8, “pace”: 99.5},
“Warriors”:     {“pts_allowed_pg”: 113.1, “pace”: 100.1},
“Kings”:        {“pts_allowed_pg”: 116.3, “pace”: 101.9},
“Mavericks”:    {“pts_allowed_pg”: 110.4, “pace”: 98.6},
“Rockets”:      {“pts_allowed_pg”: 109.3, “pace”: 98.4},
“Spurs”:        {“pts_allowed_pg”: 119.7, “pace”: 102.1},
“Pelicans”:     {“pts_allowed_pg”: 112.6, “pace”: 98.8},
“Grizzlies”:    {“pts_allowed_pg”: 114.7, “pace”: 99.3},
“Blazers”:      {“pts_allowed_pg”: 118.5, “pace”: 101.5},
}

LEAGUE_AVG_PTS_ALLOWED = 113.5
LEAGUE_AVG_PACE = 99.0

def deadline_exceeded() -> bool:
return (time.time() - RUN_START) > RUN_MAX_SECONDS

# –––––––––– UTILS ––––––––––

def _now_et() -> datetime:
return datetime.now(ET)

def _season_year(now_et: datetime) -> int:
return now_et.year if now_et.month >= 10 else now_et.year - 1

def _parse_minutes(min_str) -> float:
if not min_str:
return 0.0
s = str(min_str)
if “:” in s:
try:
mm, ss = s.split(”:”, 1)
return float(mm) + float(ss) / 60.0
except Exception:
return 0.0
try:
return float(s)
except Exception:
return 0.0

def _clean_name(s: str) -> str:
s = (s or “”).strip()
s = re.sub(r”.”, “”, s)
s = re.sub(r”\s+”, “ “, s)
s = s.replace(”\u2019”, “’”)
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
return {“players”: {}, “sent_bets”: {}, “market”: {}}
try:
with open(STATE_FILE, “r”) as f:
raw = json.load(f)
if not isinstance(raw, dict):
return {“players”: {}, “sent_bets”: {}, “market”: {}}
raw.setdefault(“players”, {})
raw.setdefault(“sent_bets”, {})
raw.setdefault(“market”, {})
return raw
except Exception:
return {“players”: {}, “sent_bets”: {}, “market”: {}}

def save_state(state):
try:
with open(STATE_FILE, “w”) as f:
json.dump(state, f, indent=2, sort_keys=True)
except Exception:
pass

def send_one(body: str):
try:
twilio.messages.create(from_=FROM_WHATSAPP, to=TO_WHATSAPP, body=body[:MAX_BODY_CHARS])
except Exception as e:
print(f”[WARN] Twilio send failed: {type(e).**name**}: {e}”)

def send_chunked(full_text: str):
if len(full_text) <= MAX_BODY_CHARS:
send_one(full_text)
return
parts = []
remaining = full_text
while len(remaining) > MAX_BODY_CHARS:
cut = remaining.rfind(”\n”, 0, MAX_BODY_CHARS)
if cut < 200:
cut = MAX_BODY_CHARS
parts.append(remaining[:cut].rstrip())
remaining = remaining[cut:].lstrip()
if remaining:
parts.append(remaining)
total = len(parts)
for i, p in enumerate(parts, start=1):
header = f”(Part {i}/{total})\n”
if len(header) + len(p) > MAX_BODY_CHARS:
p = p[: MAX_BODY_CHARS - len(header)]
send_one(header + p)

def status_in_scope(status: str) -> bool:
return (status or “”).strip().lower() in IMPACT_STATUSES

def _chunk(lst, n):
for i in range(0, len(lst), n):
yield lst[i:i + n]

def _safe_rate(stat_avg: float, min_avg: float) -> float:
return float(stat_avg) / max(float(min_avg), 1e-6)

def _clamp(x: float, lo: float, hi: float) -> float:
return max(lo, min(hi, x))

# –––––––––– NEW: OPPONENT DEFENSIVE ADJUSTMENT ––––––––––

def opponent_defense_adjustment(opponent_team: str, prop_type: str) -> tuple[float, str]:
“””
IMPROVEMENT: Replaces the stub matchup_adjustment() that always returned 0.0.

```
Returns (multiplier_delta, note) where multiplier_delta is added to 1.0
to scale the projection. E.g. +0.06 means project 6% more pts vs weak defense.

Logic:
- Compare opponent's pts_allowed_pg to league average
- Every 5 pts above league avg = +4% scoring opportunity
- Pace adjustment: every 2 possessions above league avg = +1%
"""
if not ENABLE_OPP_DEF_ADJ or prop_type not in ("points", "pts"):
    return 0.0, "neutral"

opp = normalize_team_name(opponent_team)
def_stats = TEAM_DEFENSE_RATINGS.get(opp)
if not def_stats:
    return 0.0, "no-data"

pts_diff = def_stats["pts_allowed_pg"] - LEAGUE_AVG_PTS_ALLOWED
pace_diff = def_stats["pace"] - LEAGUE_AVG_PACE

# Scale: +5 pts allowed above avg => +4% scoring boost for the scorer
pts_factor = (pts_diff / 5.0) * 0.04
pace_factor = (pace_diff / 2.0) * 0.01 * PACE_FACTOR_WEIGHT

adj = _clamp(pts_factor + pace_factor, -0.12, 0.12)

if adj > 0.04:
    note = f"weak-def({opp},{def_stats['pts_allowed_pg']:.0f}ptAllowed)"
elif adj < -0.04:
    note = f"strong-def({opp},{def_stats['pts_allowed_pg']:.0f}ptAllowed)"
else:
    note = f"neutral-def({opp})"

return adj, note
```

def matchup_adjustment(team_name: str, player_name: str, prop_type: str) -> tuple[float, str]:
“”“Kept for backward compatibility – now delegates to opponent_defense_adjustment.”””
return opponent_defense_adjustment(team_name, prop_type)

# –––––––––– NEW: HOME/AWAY & B2B CONTEXT ––––––––––

def get_game_context(player_team: str, games_map: dict, gid: int) -> dict:
“””
IMPROVEMENT: Extract home/away status and back-to-back flags.

```
Returns dict with:
  - is_home: bool
  - opponent: str (team name)
  - home_away_boost: float (pts to add/subtract to projection)
"""
context = {"is_home": False, "opponent": "", "home_away_boost": 0.0}
if not gid or gid not in games_map:
    return context

game = games_map[gid]
home = normalize_team_name(game.get("home", ""))
away = normalize_team_name(game.get("away", ""))
pteam = normalize_team_name(player_team)

if pteam == home:
    context["is_home"] = True
    context["opponent"] = away
    context["home_away_boost"] = HOME_COURT_BOOST
elif pteam == away:
    context["is_home"] = False
    context["opponent"] = home
    context["home_away_boost"] = -HOME_COURT_BOOST * 0.5  # away slight penalty
else:
    context["opponent"] = home if pteam != home else away

return context
```

def is_back_to_back(games: list, today_str: str) -> bool:
“””
IMPROVEMENT: Detect if player played yesterday (back-to-back).
games is a list of (date_str, val, mins) tuples sorted ascending.
“””
if not games:
return False
try:
today = datetime.strptime(today_str, “%Y-%m-%d”).date()
yesterday = today - timedelta(days=1)
for g in reversed(games):
try:
gdate = datetime.strptime(g[0][:10], “%Y-%m-%d”).date()
if gdate == yesterday:
return True
if gdate < yesterday:
break
except Exception:
continue
except Exception:
pass
return False

# –––––––––– NEW: CONSISTENCY SCORE ––––––––––

def consistency_score(games, n=LOOKBACK_GAMES) -> float:
“””
IMPROVEMENT: Measures how consistently a player hits their average.
Returns 0.0-1.0. High consistency = more reliable projection.

```
Method: % of L10 games where player scored within +/-20% of their L10 avg.
This reduces the STD_FLOOR problem -- consistent players get tighter confidence.
"""
sl = _slice_last(games, n)
if len(sl) < 5:
    return 0.5
avg, _, _ = avg_stat_min_std(sl)
if avg < 1.0:
    return 0.5
threshold = avg * 0.20
hits = sum(1 for g in sl if abs(g[1] - avg) <= threshold)
return _clamp(hits / len(sl), 0.0, 1.0)
```

def adaptive_sigma(games, base_sigma: float, cons_score: float) -> float:
“””
IMPROVEMENT: Scale sigma down for consistent players.
A player with 80% consistency within +/-20% of avg gets sigma reduced by up to 25%.
This lets us have more confidence on consistent over-achievers vs the line.
“””
reduction = (cons_score - 0.5) * 0.5  # 0.0 at 50% cons, 0.25 at 100% cons
return max(STD_FLOOR, base_sigma * (1.0 - reduction))

# –––––––––– NEW: RESULT TRACKING ––––––––––

def log_play_for_tracking(play: dict, now_ts: int):
“””
IMPROVEMENT: Persist each sent play so you can later compare proj vs actual.
Load results_log.json, check actual scores daily, measure hit rate over time.

```
Schema per entry:
  key = "prop_type|pid|line|date"
  value = {proj, line, over_odds, prob_over, edge, section, sent_ts, actual_pts (fill in later)}
"""
if not ENABLE_RESULT_TRACKING:
    return
try:
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            log = json.load(f)
    else:
        log = {}

    date_str = datetime.fromtimestamp(now_ts, tz=ET).strftime("%Y-%m-%d")
    key = f"{play['prop_type']}|{play['player_id']}|{play.get('cons_line', play['line']):.1f}|{date_str}"
    log[key] = {
        "player_name": play["player_name"],
        "team": play.get("team", ""),
        "prop_type": play["prop_type"],
        "proj": round(play["proj"], 2),
        "line": play.get("cons_line", play["line"]),
        "over_odds": play["over_odds"],
        "prob_over": round(play["prob_over"], 4),
        "edge": round(play["edge"], 2),
        "ev": round(play["ev"], 4),
        "section": play["section"],
        "sent_ts": now_ts,
        "date": date_str,
        "actual_pts": None,   # Fill this in after the game
        "hit": None,          # True/False once actual is known
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(log, f, indent=2, sort_keys=True)

except Exception as e:
    print(f"[WARN] result tracking failed: {e}")
```

def get_hit_rate_summary() -> str:
“””
IMPROVEMENT: Compute running hit rate from results log.
Returns a human-readable summary string for the WhatsApp message.
“””
if not ENABLE_RESULT_TRACKING or not os.path.exists(RESULTS_FILE):
return “”
try:
with open(RESULTS_FILE, “r”) as f:
log = json.load(f)

```
    resolved = [v for v in log.values() if v.get("hit") is not None]
    if not resolved:
        return ""

    total = len(resolved)
    hits = sum(1 for v in resolved if v["hit"])
    rate = hits / total
    recent = sorted(resolved, key=lambda x: x.get("sent_ts", 0), reverse=True)[:20]
    recent_hits = sum(1 for v in recent if v["hit"])
    recent_rate = recent_hits / len(recent) if recent else 0

    return (
        f"[STATS] Model accuracy: {hits}/{total} ({rate*100:.0f}%) overall | "
        f"L20: {recent_hits}/{len(recent)} ({recent_rate*100:.0f}%)"
    )
except Exception:
    return ""
```

# –––––––––– NEW: CONFIDENCE TIER ––––––––––

def confidence_tier(edge: float, prob_over: float, ev: float, value_edge: float) -> str:
“””
IMPROVEMENT: Assign a human-readable confidence tier to each pick.
Makes WhatsApp output scannable at a glance for fast bet decisions.

```
[LOCK] Lock   = elite signal, bet with confidence
[STRONG] Strong = solid edge, good play
[LEAN] Lean   = marginal edge, smaller if anything
"""
score = 0
if edge >= 5.0:
    score += 3
elif edge >= 3.5:
    score += 2
elif edge >= 2.5:
    score += 1

if prob_over >= 0.70:
    score += 3
elif prob_over >= 0.65:
    score += 2
elif prob_over >= 0.60:
    score += 1

if ev >= 0.20:
    score += 2
elif ev >= 0.10:
    score += 1

if value_edge >= 0.08:
    score += 2
elif value_edge >= 0.04:
    score += 1

if score >= 8:
    return "[LOCK] Lock"
elif score >= 5:
    return "[STRONG] Strong"
else:
    return "[LEAN] Lean"
```

def projected_minutes(base_min, l10_min, l3_min, min_delta, injury_boost_min=0.0, le_score=0.0):
proj = (0.15 * base_min) + (0.55 * l10_min) + (0.30 * l3_min)

```
if min_delta >= 2.0:
    proj += 1.0
if min_delta >= 4.0:
    proj += 1.0

proj += min(3.5, injury_boost_min * 0.18)

if le_score > 0.5:
    proj += 1.0
elif le_score < -0.5:
    proj -= 1.0

return max(8.0, proj)
```

def minutes_confidence(proj_min: float, l10_min: float, l3_min: float) -> float:
diff = abs(proj_min - l10_min) + abs(l10_min - l3_min) * 0.5
conf = 1.0 - min(1.0, diff / 15.0)
return _clamp(conf, 0.0, 1.0)

def stability_score(edge: float, sigma: float | None) -> float:
if sigma is None:
return max(-5.0, min(5.0, edge / 2.0))
return edge / max(float(sigma), 1.0)

def volatility_penalty(sigma: float | None) -> float:
if sigma is None:
return 0.0
return max(0.0, (float(sigma) - 8.0) / 4.0)

def final_play_score(ev, value_edge, edge, min_conf, matchup_score, le_score, stab_score, vol_pen):
return (
(ev * FINAL_SCORE_EV_W)
+ (value_edge * FINAL_SCORE_VALUE_W)
+ (edge * FINAL_SCORE_EDGE_W)
+ (min_conf * FINAL_SCORE_MINCONF_W)
+ (matchup_score * FINAL_SCORE_MATCHUP_W)
+ (le_score * FINAL_SCORE_LE_W)
+ (stab_score * FINAL_SCORE_STABILITY_W)
- (vol_pen * FINAL_SCORE_VOL_PENALTY_W)
)

def player_risk_bucket(l10_min: float, sigma: float | None, prop_type: str) -> str:
if prop_type == “threes”:
return “high”
if l10_min < 18:
return “high”
if l10_min < 26:
return “medium”
if sigma is not None and sigma >= 9.0:
return “medium”
return “low”

def thresholds_for_bucket(bucket: str) -> dict:
if bucket == “high”:
return {“min_edge”: 3.5, “min_prob”: 0.66, “min_ev”: 0.04, “min_value_edge”: 0.04}
if bucket == “medium”:
return {“min_edge”: 2.8, “min_prob”: 0.63, “min_ev”: 0.02, “min_value_edge”: 0.02}
return {“min_edge”: 2.0, “min_prob”: 0.60, “min_ev”: 0.00, “min_value_edge”: 0.00}

def minutes_stability_ok(l10_min: float, l3_min: float, le_score: float) -> bool:
if abs(l3_min - l10_min) <= 3.0:
return True
if le_score >= 0.8:
return True
return False

def threes_attempt_profile_ok(games) -> bool:
l10 = _slice_last(games, LOOKBACK_GAMES)
if len(l10) < 5:
return False
att = [float(x[2]) for x in l10 if float(x[2]) > 0]
if len(att) < 5:
return False
avg_att = sum(att) / len(att)
if avg_att < 4.0:
return False
if max(att) - min(att) > 6.0:
return False
return True

# –––––––––– TEAM ALIASES ––––––––––

TEAM_ALIAS_TO_FULL = {
“atlanta hawks”: “Hawks”, “hawks”: “Hawks”,
“boston celtics”: “Celtics”, “celtics”: “Celtics”,
“brooklyn nets”: “Nets”, “nets”: “Nets”,
“charlotte hornets”: “Hornets”, “hornets”: “Hornets”,
“chicago bulls”: “Bulls”, “bulls”: “Bulls”,
“cleveland cavaliers”: “Cavaliers”, “cavaliers”: “Cavaliers”,
“dallas mavericks”: “Mavericks”, “mavericks”: “Mavericks”,
“denver nuggets”: “Nuggets”, “nuggets”: “Nuggets”,
“detroit pistons”: “Pistons”, “pistons”: “Pistons”,
“golden state warriors”: “Warriors”, “warriors”: “Warriors”,
“houston rockets”: “Rockets”, “rockets”: “Rockets”,
“indiana pacers”: “Pacers”, “pacers”: “Pacers”,
“la clippers”: “Clippers”, “los angeles clippers”: “Clippers”, “clippers”: “Clippers”,
“la lakers”: “Lakers”, “los angeles lakers”: “Lakers”, “lakers”: “Lakers”,
“memphis grizzlies”: “Grizzlies”, “grizzlies”: “Grizzlies”,
“miami heat”: “Heat”, “heat”: “Heat”,
“milwaukee bucks”: “Bucks”, “bucks”: “Bucks”,
“minnesota timberwolves”: “Timberwolves”, “timberwolves”: “Timberwolves”,
“new orleans pelicans”: “Pelicans”, “pelicans”: “Pelicans”,
“new york knicks”: “Knicks”, “knicks”: “Knicks”,
“oklahoma city thunder”: “Thunder”, “thunder”: “Thunder”,
“orlando magic”: “Magic”, “magic”: “Magic”,
“philadelphia 76ers”: “76ers”, “76ers”: “76ers”, “sixers”: “76ers”,
“phoenix suns”: “Suns”, “suns”: “Suns”,
“portland trail blazers”: “Blazers”, “trail blazers”: “Blazers”, “blazers”: “Blazers”,
“sacramento kings”: “Kings”, “kings”: “Kings”,
“san antonio spurs”: “Spurs”, “spurs”: “Spurs”,
“toronto raptors”: “Raptors”, “raptors”: “Raptors”,
“utah jazz”: “Jazz”, “jazz”: “Jazz”,
“washington wizards”: “Wizards”, “wizards”: “Wizards”,
}

def normalize_team_name(raw: str) -> str:
s = (raw or “”).strip()
if not s:
return “”
sl = s.lower()
if sl in TEAM_ALIAS_TO_FULL:
return TEAM_ALIAS_TO_FULL[sl]
for k, v in TEAM_ALIAS_TO_FULL.items():
if sl == k or sl.endswith(k) or k in sl:
return v
return s

def extract_team_from_text(text: str) -> str:
text = (text or “”).strip()
if not text:
return “”
m = re.match(r”^([A-Za-z0-9 .&’-]+?)’s\s+”, text, flags=re.I)
if m:
return normalize_team_name(m.group(1))
patterns = [
r”\b([A-Za-z0-9 .&’-]+?)’s\s+[A-Za-z .’-]+:”,
r”\b([A-Za-z0-9 .&’-]+?)\s+vs.?\s+”,
r”\bagainst\s+the\s+([A-Za-z0-9 .&’-]+)\b”,
r”\bagainst\s+([A-Za-z0-9 .&’-]+)\b”,
]
for pat in patterns:
m = re.search(pat, text, flags=re.I)
if m:
team = normalize_team_name(m.group(1))
if team:
return team
return “”

def extract_team_from_url(url: str) -> str:
u = (url or “”).lower()
if not u:
return “”
for alias, team in TEAM_ALIAS_TO_FULL.items():
slug = alias.replace(” “, “-”)
if f”/{slug}/” in u or f”/{alias.replace(’ ’, ‘’)}/” in u or alias in u:
return team
return “”

# –––––––––– SPORTRADAR ––––––––––

def fetch_sportradar_injuries():
if not SPORTRADAR_KEY:
raise RuntimeError(“No SPORTRADAR_API_KEY provided”)
url = “https://api.sportradar.com/nba/trial/v8/en/league/injuries.json”
r = requests.get(url, params={“api_key”: SPORTRADAR_KEY}, timeout=20)
if r.status_code != 200:
raise RuntimeError(f”Sportradar error {r.status_code}: {r.text[:300]}”)
ct = (r.headers.get(“Content-Type”) or “”).lower()
if “json” not in ct:
raise RuntimeError(f”Unexpected content-type: {ct}. Body: {r.text[:300]}”)
return r.json()

def parse_injuries(data):
flat_by_player = {}
for team in data.get(“teams”, []):
team_name = team.get(“name”) or team.get(“market”) or team.get(“id”, “TEAM”)
team_name = normalize_team_name(team_name)
for p in team.get(“players”, []):
injuries = p.get(“injuries”) or []
if not injuries:
continue
inj = injuries[-1]
pid = p.get(“id”)
if not pid:
continue
name = p.get(“full_name”) or f”{p.get(‘first_name’,’’)} {p.get(‘last_name’,’’)}”.strip()
status = (inj.get(“status”) or “Unknown”).strip()
detail = (inj.get(“comment”) or inj.get(“description”) or “”).strip()
flat_by_player[pid] = {“name”: name, “team”: team_name, “status”: status, “detail”: detail}
return flat_by_player

# –––––––––– BALLDONTLIE ––––––––––

BDL_HEADERS = {“Authorization”: BALLDONTLIE_API_KEY}
BDL_PREFIXES = [”/nba”, “”]

BDL_MAX_RETRIES = int(os.environ.get(“BDL_MAX_RETRIES”, “5”))
BDL_RETRY_BASE_SEC = float(os.environ.get(“BDL_RETRY_BASE_SEC”, “1.5”))
BDL_PER_PAGE = int(os.environ.get(“BDL_PER_PAGE”, “100”))
BDL_MAX_PAGES = int(os.environ.get(“BDL_MAX_PAGES”, “10”))

TEAM_CACHE = None
PLAYER_NAME_CACHE = {}
PLAYER_TEAM_CACHE = {}
PROPS_CACHE = {}

def _bdl_get(path: str, params=None, timeout: int = 20) -> dict:
last_err = None
for pref in BDL_PREFIXES:
url = f”https://api.balldontlie.io{pref}{path}”
for attempt in range(BDL_MAX_RETRIES):
try:
r = requests.get(url, headers=BDL_HEADERS, params=params or {}, timeout=timeout)
if r.status_code == 404:
last_err = f”404 {url}”
break
if r.status_code in (429, 500, 502, 503, 504):
retry_after = r.headers.get(“Retry-After”)
sleep_s = float(retry_after) if retry_after else (BDL_RETRY_BASE_SEC * (2 ** attempt))
last_err = f”{r.status_code} {r.text[:120]}”
time.sleep(min(sleep_s, 20.0))
continue
if r.status_code != 200:
raise RuntimeError(f”BallDontLie error {r.status_code}: {r.text[:300]}”)
return r.json()
except Exception as e:
last_err = str(e)
time.sleep(min(BDL_RETRY_BASE_SEC * (2 ** attempt), 20.0))
continue
raise RuntimeError(f”BallDontLie request failed for {path}. Last error: {last_err}”)

def bdl_games_today(now_et: datetime):
today = now_et.strftime(”%Y-%m-%d”)
resp = _bdl_get(”/v1/games”, params={“dates[]”: [today], “per_page”: 100})
games = resp.get(“data”) or []
out = {}
for g in games:
try:
gid = int(g[“id”])
except Exception:
continue
home = normalize_team_name(((g.get(“home_team”) or {}).get(“name”) or “”).strip())
away = normalize_team_name(((g.get(“visitor_team”) or {}).get(“name”) or “”).strip())
out[gid] = {“home”: home, “away”: away}
return out

def bdl_team_name_to_id():
global TEAM_CACHE
if TEAM_CACHE is not None:
return TEAM_CACHE
data = _bdl_get(”/v1/teams”, params={“per_page”: 100})
m = {}
for t in data.get(“data”, []):
nm = normalize_team_name((t.get(“name”) or “”).strip())
if nm and t.get(“id”) is not None:
m[nm] = int(t[“id”])
TEAM_CACHE = m
return TEAM_CACHE

def bdl_active_roster(team_short: str):
team_short = normalize_team_name(team_short)
team_map = bdl_team_name_to_id()
team_id = team_map.get(team_short)
if not team_id:
return []
players = []
cursor = None
pages = 0
while pages < 5 and (not deadline_exceeded()):
params = {“per_page”: 100, “team_ids[]”: [team_id]}
if cursor is not None:
params[“cursor”] = cursor
resp = _bdl_get(”/v1/players/active”, params=params)
players.extend(resp.get(“data”) or [])
cursor = (resp.get(“meta”) or {}).get(“next_cursor”)
pages += 1
if not cursor:
break
out = []
for p in players:
team = p.get(“team”) or {}
if normalize_team_name((team.get(“name”) or “”).strip()) == team_short:
out.append(p)
return out

def bdl_find_player_id_on_team(team_short: str, full_name: str):
roster = bdl_active_roster(team_short)
if not roster:
return None

```
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
```

def bdl_last_n_games_stats(player_ids, season: int, n: int, stat_key: str):
out = {int(pid): [] for pid in player_ids}
if not player_ids:
return out

```
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
        tname = normalize_team_name((team.get("name") or "").strip())
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
```

def bdl_last_n_games_threes(player_ids, season: int, n: int):
out = {int(pid): [] for pid in player_ids}
if not player_ids:
return out

```
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
        tname = normalize_team_name((team.get("name") or "").strip())
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
```

def bdl_fetch_props_for_game(game_id: int, vendor: str | None, prop_type: str):
key = (int(game_id), (vendor or “NO_VENDOR”), prop_type)
if key in PROPS_CACHE:
return PROPS_CACHE[key]

```
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
```

# –––––––––– PROPS ––––––––––

def build_today_props(now_et: datetime):
games_map = bdl_games_today(now_et)
game_ids = list(games_map.keys())
lines_map = {pt: {} for pt in PROP_TYPES}

```
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
```

# –––––––––– CONSENSUS ––––––––––

def _round_to_half(x: float) -> float:
return round(float(x) * 2.0) / 2.0

def consensus_line(rows):
if not rows:
return None, 0, 0

```
by_vendor = {}

for r in rows:
    v = str(r.get("vendor") or "").strip().lower()
    if not v or v not in CONSENSUS_VENDORS:
        continue
    try:
        line = float(r["line"])
    except Exception:
        continue

    if v not in by_vendor:
        by_vendor[v] = _round_to_half(line)

sharp_count = len(by_vendor)
lines = sorted(by_vendor.values())
n = len(lines)

if n == 0:
    return None, 0, 0

mid = n // 2
if n % 2 == 1:
    return float(lines[mid]), n, sharp_count
return float(0.5 * (lines[mid - 1] + lines[mid])), n, sharp_count
```

def best_offer_near_consensus(rows, cons_line: float):
if not rows or cons_line is None:
return None

```
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
```

# –––––––––– PROJECTION CORE ––––––––––

def compute_projection_components_points(games_all, line):
base_slice = _slice_last(games_all, BASELINE_GAMES)
l10_slice = _slice_last(games_all, LOOKBACK_GAMES)
l3_slice = _slice_last(games_all, SHORT_GAMES)

```
base_avg, base_min, base_std = avg_stat_min_std(base_slice)
l10_avg, l10_min, l10_std = avg_stat_min_std(l10_slice)
l3_avg, l3_min, _ = avg_stat_min_std(l3_slice)

rate_base = _safe_rate(base_avg, base_min)
rate_l10 = _safe_rate(l10_avg, l10_min)
rate_l3 = _safe_rate(l3_avg, l3_min)

raw_sigma = l10_std if l10_std > 0 else base_std if base_std > 0 else STD_FLOOR

# IMPROVEMENT: Use adaptive sigma based on consistency
# Consistent players get tighter sigma so prob_over is more decisive
cons = consistency_score(games_all)
sigma = adaptive_sigma(games_all, max(STD_FLOOR, raw_sigma), cons)

return {
    "base_avg": base_avg,
    "l10_avg": l10_avg,
    "l3_avg": l3_avg,
    "base_min": base_min,
    "l10_min": l10_min,
    "l3_min": l3_min,
    "rate_base": rate_base,
    "rate_l10": rate_l10,
    "rate_l3": rate_l3,
    "sigma": sigma,
    "raw_sigma": raw_sigma,
    "consistency": cons,
    "line": float(line),
}
```

def compute_proj_rate(rate_base, rate_l10, rate_l3):
“””
IMPROVEMENT: Use configurable recency weights (default: base 35%, L10 35%, L3 30%).
Original was base 45% / L10 35% / L3 20%, which under-weighted recent form.
Hot streaks and cold stretches matter more for player points than season averages.
“””
total = PROJ_WEIGHT_BASE + PROJ_WEIGHT_L10 + PROJ_WEIGHT_L3
w_base = PROJ_WEIGHT_BASE / total
w_l10 = PROJ_WEIGHT_L10 / total
w_l3 = PROJ_WEIGHT_L3 / total
return (w_base * rate_base) + (w_l10 * rate_l10) + (w_l3 * rate_l3)

# –––––––––– THREES ––––––––––

def _betaln(a: float, b: float) -> float:
return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

def _log_choose(n: int, k: int) -> float:
try:
n = int(n)
k = int(k)
except Exception:
return float(”-inf”)
if n < 0 or k < 0 or k > n:
return float(”-inf”)
return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

def beta_binom_pmf(k: int, n: int, a: float, b: float) -> float:
lc = _log_choose(n, k)
if not math.isfinite(lc):
return 0.0
try:
return math.exp(lc + _betaln(k + a, (n - k) + b) - _betaln(a, b))
except Exception:
return 0.0

def beta_binom_cdf(k: int, n: int, a: float, b: float) -> float:
try:
n = int(n)
k = int(k)
except Exception:
return 0.0
if n <= 0:
return 1.0 if k >= 0 else 0.0
k = max(-1, min(k, n))
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
k = int(math.floor(float(line)))
probs = []
for n_att in att_list:
n_att = max(0, int(n_att))
if n_att <= 0:
probs.append(0.0)
continue
p_le_k = beta_binom_cdf(k, n_att, a, b)
probs.append(1.0 - p_le_k)
if not probs:
return None
return sum(probs) / len(probs)

# –––––––––– STEAM ––––––––––

def steam_score(prev, cur):
try:
prev_line = float(prev.get(“line”))
cur_line = float(cur.get(“line”))
prev_over = float(prev.get(“over_odds”))
cur_over = float(cur.get(“over_odds”))
except Exception:
return 0.0
line_move = (prev_line - cur_line)
odds_move = (cur_over - prev_over) / 50.0
score = (line_move / 0.5) * 0.9 + odds_move * 0.6
return float(score)

def market_key(prop_type: str, pid: int) -> str:
return f”{prop_type}|{int(pid)}”

def remember_market(state, prop_type, pid, offer, cons_line, n_cons, now_ts):
mk = market_key(prop_type, pid)
state.setdefault(“market”, {})
state[“market”][mk] = {
“ts”: int(now_ts),
“line”: float(cons_line) if cons_line is not None else float(offer[“line”]),
“over_odds”: float(offer[“over_odds”]),
“under_odds”: float(offer[“under_odds”]),
“vendor”: str(offer.get(“vendor”, “”)),
“n_cons”: int(n_cons),
}

def get_prev_market(state, prop_type, pid, now_ts):
mk = market_key(prop_type, pid)
prev = (state.get(“market”) or {}).get(mk)
if not prev:
return None
age_min = (int(now_ts) - int(prev.get(“ts”, 0))) / 60.0
if age_min > STEAM_MAX_AGE_MIN:
return None
return prev

# –––––––––– LINEUPEXPERTS ––––––––––

def _try_parse_dt(s: str):
if not s:
return None
s = str(s).strip()
for fmt in (
“%Y-%m-%dT%H:%M:%S%z”,
“%Y-%m-%dT%H:%M:%S.%f%z”,
“%Y-%m-%dT%H:%M:%S”,
“%Y-%m-%d %H:%M:%S”,
“%b %d, %Y”,
“%b %d, %Y %I:%M %p”,
“%b %d, %Y %H:%M”,
“%b %d, %Y %I:%M%p”,
):
try:
return datetime.strptime(s, fmt)
except Exception:
continue
return None

def find_team_for_player_from_cache(player_name: str) -> str:
target = _clean_name(player_name)
if not target:
return “”
if target in PLAYER_TEAM_OVERRIDES:
return PLAYER_TEAM_OVERRIDES[target]
for pid, nm in PLAYER_NAME_CACHE.items():
if _clean_name(nm) == target:
team = normalize_team_name(PLAYER_TEAM_CACHE.get(pid, “”))
if team:
return team
return “”

def fetch_lineupexperts_news(now_et: datetime):
if not LINEUPEXPERTS or not LINEUPEXPERTS_KEY:
return []

```
url = f"{LINEUPEXPERTS_BASE_URL}/nba-NewsBySport"

try:
    r = requests.get(url, params={"key": LINEUPEXPERTS_KEY}, timeout=LINEUPEXPERTS_TIMEOUT)
except Exception as e:
    print(f"[WARN] LineupExperts request failed: {type(e).__name__}: {e}")
    return []

if LE_DEBUG:
    print(f"[LE_DEBUG] status={r.status_code} url={r.url} body_head={r.text[:220].replace(chr(10), ' ')}")

if r.status_code != 200:
    print(f"[WARN] LineupExperts HTTP {r.status_code} for {r.url}: {r.text[:300]}")
    return []

try:
    data = r.json()
except Exception:
    print(f"[WARN] LineupExperts non-JSON response: {r.text[:200]}")
    return []

items = []

if isinstance(data, list):
    items = data
elif isinstance(data, dict):
    if isinstance(data.get("data"), list):
        items = data["data"]
    else:
        for _, v in data.items():
            if not isinstance(v, dict):
                continue
            pinfo = v.get("PlayerInfo") or {}
            pname = (pinfo.get("PlayerName") or pinfo.get("Player") or "").strip()
            stories = v.get("Stories") or []
            if not isinstance(stories, list):
                continue
            for st in stories:
                if not isinstance(st, dict):
                    continue
                title = st.get("Title") or st.get("title") or ""
                surl = st.get("URL") or st.get("url") or ""
                team_hint = extract_team_from_text(title)
                if not team_hint:
                    team_hint = extract_team_from_url(surl)
                if not team_hint and pname:
                    team_hint = find_team_for_player_from_cache(pname)
                items.append({
                    "player": pname,
                    "title": title,
                    "publisher": st.get("Publisher") or st.get("publisher") or "",
                    "url": surl,
                    "date": st.get("Date") or st.get("date") or "",
                    "raw": st,
                    "team_hint": team_hint,
                })
else:
    items = []

if not isinstance(items, list):
    return []
items = items[: max(1, LINEUPEXPERTS_MAX_ITEMS)]

if LE_NO_TIME_FILTER:
    return items

cutoff = now_et - timedelta(hours=NEWS_LOOKBACK_HOURS)
out = []
for it in items:
    if not isinstance(it, dict):
        continue
    dt = None
    for k in ("date", "updated", "updated_at", "created", "created_at", "publishDate", "published", "time", "Date"):
        if k in it:
            dt = _try_parse_dt(it.get(k))
            break
    if dt is not None and dt.tzinfo is None:
        dt = dt.replace(tzinfo=ET)
    if dt is not None and dt < cutoff:
        continue
    out.append(it)

return out
```

def build_news_boost_map(news_items):
boosts = {}

```
probable_pat = re.compile(r"\b(probable|available|cleared|active|returns?|good to go)\b", re.I)
questionable_pat = re.compile(r"\b(questionable|game-time decision|gtd)\b", re.I)
doubtful_pat = re.compile(r"\b(doubtful)\b", re.I)
out_pat = re.compile(r"\b(ruled out|will miss|out for|out vs|out tonight|inactive|won't play)\b", re.I)
starter_pat = re.compile(r"\b(will start|expected to start|starting lineup|named starter|draws start)\b", re.I)
bench_pat = re.compile(r"\b(returning to bench role|coming off the bench|bench role)\b", re.I)
minutes_up_pat = re.compile(r"\b(minutes restriction lifted|minutes limit lifted|no minutes limit|workload increased|bigger role)\b", re.I)
minutes_down_pat = re.compile(r"\b(minutes restriction|minutes monitored|limited to)\b", re.I)

def push(player_name: str, boost: float, confidence: float, why: str):
    if not player_name:
        return
    k = _clean_name(player_name)
    if not k:
        return
    if confidence < NEWS_MIN_CONFIDENCE:
        return
    cur = boosts.get(k)
    score = abs(boost) * confidence
    if (cur is None) or (score > (abs(cur["boost"]) * cur["confidence"])):
        boosts[k] = {"boost": float(boost), "confidence": float(confidence), "why": why[:220]}

for it in news_items:
    try:
        title = str(it.get("title") or "").strip()
        player = str(it.get("player") or "").strip()
    except Exception:
        continue

    text = title
    if not text or not player:
        continue

    boost = 0.0
    confidence = 0.40
    why_bits = []

    if starter_pat.search(text):
        boost += 0.12
        confidence = max(confidence, 0.70)
        why_bits.append("starter-news")
    if minutes_up_pat.search(text):
        boost += 0.10
        confidence = max(confidence, 0.70)
        why_bits.append("minutes-up")
    if probable_pat.search(text):
        boost += 0.05
        confidence = max(confidence, 0.55)
        why_bits.append("positive-status")
    if questionable_pat.search(text):
        boost -= 0.08
        confidence = max(confidence, 0.60)
        why_bits.append("negative-tag")
    if doubtful_pat.search(text):
        boost -= 0.12
        confidence = max(confidence, 0.65)
        why_bits.append("doubtful")
    if out_pat.search(text):
        boost -= 0.18
        confidence = max(confidence, 0.75)
        why_bits.append("out-news")
    if minutes_down_pat.search(text):
        boost -= 0.10
        confidence = max(confidence, 0.65)
        why_bits.append("minutes-cap")
    if bench_pat.search(text):
        boost -= 0.06
        confidence = max(confidence, 0.50)
        why_bits.append("bench")

    if abs(boost) < 1e-9:
        continue

    push(player, boost, confidence, f"{'|'.join(why_bits)}: {title}")

return boosts
```

def build_news_score_map(news_items):
out = {}

```
pos_strong = re.compile(r"\b(will start|expected to start|starting lineup|draws start|named starter)\b", re.I)
pos_med = re.compile(r"\b(available|cleared|returns?|good to go|active|probable)\b", re.I)
pos_min = re.compile(r"\b(minutes restriction lifted|no minutes limit|minutes limit lifted|workload increased|bigger role)\b", re.I)
neg_status = re.compile(r"\b(questionable|doubtful|game-time decision|gtd)\b", re.I)
neg_limit = re.compile(r"\b(minutes restriction|minutes monitored|limited to)\b", re.I)
neg_bench = re.compile(r"\b(coming off the bench|bench role|returning to bench role)\b", re.I)
neg_out = re.compile(r"\b(ruled out|will miss|out for|out vs|out tonight|inactive|won't play)\b", re.I)

def push(player_name, score, why):
    k = _clean_name(player_name)
    if not k:
        return
    cur = out.get(k)
    if (cur is None) or (abs(score) > abs(cur["score"])):
        out[k] = {"score": float(score), "why": why[:220]}

for it in news_items:
    if not isinstance(it, dict):
        continue
    title = str(it.get("title") or "").strip()
    player = str(it.get("player") or "").strip()
    if not player or not title:
        continue

    text = title
    score = 0.0
    why_bits = []

    if pos_strong.search(text):
        score += 1.2
        why_bits.append("start")
    if pos_min.search(text):
        score += 0.9
        why_bits.append("mins_up")
    if pos_med.search(text):
        score += 0.5
        why_bits.append("positive_status")
    if neg_status.search(text):
        score -= 0.9
        why_bits.append("negative_tag")
    if neg_limit.search(text):
        score -= 1.0
        why_bits.append("mins_cap")
    if neg_bench.search(text):
        score -= 0.5
        why_bits.append("bench")
    if neg_out.search(text):
        score -= 1.4
        why_bits.append("out")

    if score == 0.0:
        continue
    push(player, score, f"{'|'.join(why_bits)}: {title}")

return out
```

def apply_news_to_projection(proj: float, boost_rec: dict | None, cap: float = 0.30):
if not boost_rec:
return proj, 0.0, None
b = float(boost_rec.get(“boost”, 0.0))
c = float(boost_rec.get(“confidence”, 0.0))
eff = max(-cap, min(cap, b * c))
if abs(eff) <= 0:
return proj, 0.0, None
why = boost_rec.get(“why”) or “”
return proj * (1.0 + eff), eff, why

def parse_le_injuries(news_items):
out_pat = re.compile(r”\b(ruled out|will miss|out for|out vs|out tonight|inactive|won’t play)\b”, re.I)
doubtful_pat = re.compile(r”\b(doubtful)\b”, re.I)
questionable_pat = re.compile(r”\b(questionable|game-time decision|gtd)\b”, re.I)

```
injuries = {}
for it in news_items:
    player = str(it.get("player") or "").strip()
    title = str(it.get("title") or "").strip()
    surl = str(it.get("url") or "").strip()
    team_hint = normalize_team_name(str(it.get("team_hint") or "").strip())

    if not player or not title:
        continue

    if not team_hint:
        team_hint = extract_team_from_url(surl)
    if not team_hint:
        team_hint = find_team_for_player_from_cache(player)

    status = ""
    if out_pat.search(title):
        status = "Out"
    elif doubtful_pat.search(title):
        status = "Doubtful"
    elif questionable_pat.search(title):
        status = "Questionable"

    if not status:
        continue

    k = _clean_name(player)
    injuries[k] = {"name": player, "team": team_hint, "status": status, "detail": title}

return injuries
```

# –––––––––– ENGINE HELPERS ––––––––––

def should_bad_role_filter(min_delta: float, rate_delta: float, le_score: float) -> bool:
if min_delta <= -ROLE_DROP_MIN and rate_delta <= -ROLE_DROP_RATE:
return True
if min_delta <= -(ROLE_DROP_MIN + 2.0) and le_score < -0.5:
return True
return False

# –––––––––– UNIFIED PROJECTION ENGINE ––––––––––

def build_player_projection(
games,
line: float,
prop_type: str,
injury_boost_stat: float = 0.0,
injury_boost_min: float = 0.0,
le_score: float = 0.0,
news_boosts: dict = None,
player_name: str = “”,
player_team: str = “”,
opponent_team: str = “”,
games_map: dict = None,
gid: int = 0,
today_str: str = “”,
) -> dict:
“””
IMPROVEMENT: Unified projection engine replaces duplicated logic across
build_injury_edges, slate_scan_edges, lineup_news_edges, plus_odds_hunt_edges.

```
Now includes:
- Configurable recency weights (PROJ_WEIGHT_BASE/L10/L3)
- Opponent defensive adjustment (replaces stub matchup_adjustment)
- Home/away boost
- Back-to-back penalty
- Adaptive sigma based on consistency
- Confidence tier label
"""
if prop_type == "threes":
    long_slice = _slice_last(games, LOOKBACK_GAMES)
    short_slice = _slice_last(games, SHORT_GAMES)
    base = _slice_last(games, BASELINE_GAMES)

    base_avg = sum(float(x[1]) for x in base) / max(1, len(base))
    l10_avg = sum(float(x[1]) for x in long_slice) / max(1, len(long_slice))
    l3_avg = sum(float(x[1]) for x in short_slice) / max(1, len(short_slice))
    base_min = sum(float(x[3]) for x in base) / max(1, len(base))
    l10_min = sum(float(x[3]) for x in long_slice) / max(1, len(long_slice))
    l3_min = sum(float(x[3]) for x in short_slice) / max(1, len(short_slice))

    rate_base = _safe_rate(base_avg, base_min)
    rate_l10 = _safe_rate(l10_avg, l10_min)
    rate_l3 = _safe_rate(l3_avg, l3_min)
    min_delta = l3_min - l10_min
    rate_delta = rate_l3 - rate_l10
    m10 = l10_min
    sigma = None
    consistency = 0.5
else:
    comps = compute_projection_components_points(games, line)
    base_avg = comps["base_avg"]
    l10_avg = comps["l10_avg"]
    l3_avg = comps["l3_avg"]
    base_min = comps["base_min"]
    l10_min = comps["l10_min"]
    l3_min = comps["l3_min"]
    rate_base = comps["rate_base"]
    rate_l10 = comps["rate_l10"]
    rate_l3 = comps["rate_l3"]
    min_delta = l3_min - l10_min
    rate_delta = rate_l3 - rate_l10
    m10 = l10_min
    sigma = comps["sigma"]
    consistency = comps["consistency"]

proj_min = projected_minutes(base_min, l10_min, l3_min, min_delta, injury_boost_min, le_score)
min_conf = minutes_confidence(proj_min, l10_min, l3_min)

# IMPROVEMENT: Use compute_proj_rate with configurable weights
proj_rate = compute_proj_rate(rate_base, rate_l10, rate_l3)

if prop_type == "threes" and THREES_BETA_BINOM:
    proj = proj_min * proj_rate
    proj += min(0.4, injury_boost_stat * 0.05)
else:
    proj = proj_min * proj_rate
    proj += injury_boost_stat

# Apply news boost
boost_rec = (news_boosts or {}).get(_clean_name(player_name)) if player_name else None
proj, news_eff, news_why = apply_news_to_projection(proj, boost_rec)

# IMPROVEMENT: Opponent defensive adjustment (was always 0.0 before)
matchup_score, matchup_note = opponent_defense_adjustment(opponent_team, prop_type)
proj *= (1.0 + matchup_score)

# IMPROVEMENT: Home/away boost
game_ctx = {}
home_away_note = ""
if games_map and gid and player_team:
    game_ctx = get_game_context(player_team, games_map, gid)
    ha_boost = game_ctx.get("home_away_boost", 0.0)
    if ha_boost != 0:
        proj += ha_boost
        home_away_note = f"{'home' if game_ctx.get('is_home') else 'away'}({ha_boost:+.1f}pts)"
    if not opponent_team and game_ctx.get("opponent"):
        # Re-apply matchup if we got opponent from game context
        opp = game_ctx["opponent"]
        matchup_score, matchup_note = opponent_defense_adjustment(opp, prop_type)
        proj *= (1.0 + matchup_score)

# IMPROVEMENT: Back-to-back penalty
b2b_note = ""
if today_str and is_back_to_back(games, today_str):
    proj -= B2B_PENALTY
    b2b_note = f"b2b(-{B2B_PENALTY:.1f}pts)"
    print(f"[INFO] B2B penalty applied: {player_name} -{B2B_PENALTY} pts")

# Compute edge and probability
if prop_type == "threes" and THREES_BETA_BINOM:
    edge = proj - float(line)
    prob_over = threes_prob_over_beta_binom(games, float(line))
    if prob_over is None:
        prob_over = 0.0
else:
    edge = proj - float(line)
    z = (proj - float(line)) / max(sigma, 1e-6)
    prob_over = _norm_cdf(z)

return {
    "proj": proj,
    "edge": edge,
    "prob_over": prob_over,
    "proj_rate": proj_rate,
    "proj_min": proj_min,
    "min_conf": min_conf,
    "matchup_score": matchup_score,
    "matchup_note": matchup_note,
    "home_away_note": home_away_note,
    "b2b_note": b2b_note,
    "news_eff": news_eff,
    "news_why": news_why,
    "sigma": sigma,
    "consistency": consistency,
    "base_avg": base_avg,
    "l10_avg": l10_avg,
    "l3_avg": l3_avg,
    "base_min": base_min,
    "l10_min": l10_min,
    "l3_min": l3_min,
    "rate_base": rate_base,
    "rate_l10": rate_l10,
    "rate_l3": rate_l3,
    "min_delta": min_delta,
    "rate_delta": rate_delta,
    "m10": m10,
}
```

# –––––––––– ENGINES ––––––––––

def build_injury_edges(
team_short, injured_name, injured_status, exclude_names_lower,
now_et, prop_type, lines_map_for_prop, state, now_ts,
news_boosts, news_scores, games_map=None,
):
if deadline_exceeded():
return []

```
team_short = normalize_team_name(team_short)
season = _season_year(now_et)
today_str = now_et.strftime("%Y-%m-%d")
roster = bdl_active_roster(team_short)
if not roster:
    return []

roster_tuples = []
for p in roster:
    pid = p.get("id")
    nm = f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()
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
status_mult = {"out": 1.0, "doubtful": 0.55, "questionable": 0.25}.get(status, 0.35)

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

    rows = (lines_map_for_prop or {}).get(pid, [])
    cons, n_cons, n_sharp = consensus_line(rows)
    if cons is None or n_cons < MIN_VENDORS_FOR_CONSENSUS or n_sharp < MIN_SHARP_VENDORS:
        continue

    offer = best_offer_near_consensus(rows, cons)
    if not offer:
        continue

    line = float(cons)
    le_score = float(news_scores.get(_clean_name(nm), {"score": 0.0}).get("score", 0.0))

    # Compute absorption
    m10_pre = sum(g[2] for g in _slice_last(games, LOOKBACK_GAMES)) / max(1, len(_slice_last(games, LOOKBACK_GAMES)))
    l3_pre = sum(g[2] for g in _slice_last(games, SHORT_GAMES)) / max(1, len(_slice_last(games, SHORT_GAMES)))
    min_delta_pre = l3_pre - m10_pre
    absorption = 0.0
    if m10_pre >= 28:
        absorption += 0.30
    if m10_pre >= 34:
        absorption += 0.10
    if min_delta_pre >= 2.0:
        absorption += 0.15
    if le_score > 0.5:
        absorption += 0.05
    absorption = min(0.65, absorption)

    injury_boost_stat = min(BOOST_CAP_STAT, vac_stat * absorption * 0.65)
    injury_boost_min = min(BOOST_CAP_MIN, vac_min * absorption * 0.25)

    p = build_player_projection(
        games=games,
        line=line,
        prop_type=prop_type,
        injury_boost_stat=injury_boost_stat,
        injury_boost_min=injury_boost_min,
        le_score=le_score,
        news_boosts=news_boosts,
        player_name=nm,
        player_team=PLAYER_TEAM_CACHE.get(pid, team_short),
        games_map=games_map,
        gid=int(offer.get("gid") or offer.get("game_id") or 0),
        today_str=today_str,
    )

    if p["m10"] < MIN_L10_MIN:
        continue
    if (p["l10_avg"] - line) > LINE_MIN_GAP:
        continue
    if should_bad_role_filter(p["min_delta"], p["rate_delta"], le_score):
        continue
    if not minutes_stability_ok(p["l10_min"], p["l3_min"], le_score):
        continue
    if prop_type == "threes" and not threes_attempt_profile_ok(games):
        continue

    p_over = american_to_prob(float(offer["over_odds"]))
    p_under = american_to_prob(float(offer["under_odds"]))
    p_market = p_over / max(p_over + p_under, 1e-9)

    value_edge = p["prob_over"] - p_market
    ev = ev_per_dollar(p["prob_over"], float(offer["over_odds"]))

    bucket = player_risk_bucket(p["m10"], p["sigma"], prop_type)
    thr = thresholds_for_bucket(bucket)

    if p["edge"] < thr["min_edge"] or p["prob_over"] < thr["min_prob"]:
        continue
    if value_edge < thr["min_value_edge"]:
        continue
    if ev < thr["min_ev"]:
        continue

    steam = 0.0
    if ENABLE_STEAM:
        prev = get_prev_market(state, prop_type, pid, now_ts)
        if prev:
            cur = {"line": line, "over_odds": offer["over_odds"], "under_odds": offer["under_odds"], "ts": now_ts}
            steam = steam_score(prev, cur)

    stab = stability_score(p["edge"], p["sigma"])
    vol_pen = volatility_penalty(p["sigma"])
    final_score = final_play_score(ev, value_edge, p["edge"], p["min_conf"], p["matchup_score"], le_score, stab, vol_pen)

    tier = confidence_tier(p["edge"], p["prob_over"], ev, value_edge)
    gid = int(offer.get("gid") or offer.get("game_id") or 0)
    team_name = PLAYER_TEAM_CACHE.get(pid) or team_short or ""

    context_notes = " | ".join(filter(None, [p["home_away_note"], p["b2b_note"]]))

    why = (
        f"[{tier}] TrigStrength {trigger_strength:.0f} | Absorb {absorption:.2f}. "
        f"{injured_name} {injured_status.upper()} vacates ~{vac_stat:.1f}pts / {vac_min:.1f}min. "
        f"base {p['base_avg']:.1f} | L10 {p['l10_avg']:.1f} | L3 {p['l3_avg']:.1f} "
        f"(minL10 {p['l10_min']:.1f}, projMin {p['proj_min']:.1f}, rate {p['proj_rate']:.3f}, cons {p['consistency']:.2f}). "
        f"Deltamin={p['min_delta']:+.1f} Deltarate={p['rate_delta']:+.2f}. "
        f"Proj {p['proj']:.1f} vs line {line:.1f} | "
        f"{offer['vendor']} ({int(offer['over_odds']):+d}) | "
        f"edge +{p['edge']:.1f} | P={p['prob_over']*100:.0f}% (mkt={p_market*100:.0f}%, val={value_edge:+.2f}) | "
        f"EV={ev:+.2f} | matchup={p['matchup_note']}{(' | ' + context_notes) if context_notes else ''}"
        f"{(' | LE:' + p['news_why']) if p['news_why'] else ''}"
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
        "proj": float(p["proj"]),
        "edge": float(p["edge"]),
        "prob_over": float(p["prob_over"]),
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
        "le_score": float(le_score),
        "min_conf": float(p["min_conf"]),
        "stability_score": float(stab),
        "final_score": float(final_score),
        "tier": tier,
        "consistency": float(p["consistency"]),
    })

    remember_market(state, prop_type, pid, offer, line, n_cons, now_ts)

ideas.sort(key=lambda x: (x["final_score"], x["trigger_strength"], x["ev"]), reverse=True)
return ideas
```

def slate_scan_edges(now_et, prop_type, lines_map_for_prop, state, now_ts, news_boosts, news_scores, games_map=None):
if not ENABLE_SLATE_SCAN or deadline_exceeded():
return []

```
season = _season_year(now_et)
today_str = now_et.strftime("%Y-%m-%d")
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

    line = float(cons)
    name = PLAYER_NAME_CACHE.get(int(pid), f"Player {pid}")
    le_score = float(news_scores.get(_clean_name(name), {"score": 0.0}).get("score", 0.0))
    player_team = PLAYER_TEAM_CACHE.get(int(pid), "")

    p = build_player_projection(
        games=games,
        line=line,
        prop_type=prop_type,
        le_score=le_score,
        news_boosts=news_boosts,
        player_name=name,
        player_team=player_team,
        games_map=games_map,
        gid=int(offer.get("gid") or offer.get("game_id") or 0),
        today_str=today_str,
    )

    if p["m10"] < MIN_L10_MIN:
        continue
    if should_bad_role_filter(p["min_delta"], p["rate_delta"], le_score):
        continue
    if not minutes_stability_ok(p["l10_min"], p["l3_min"], le_score):
        continue
    if prop_type == "threes" and not threes_attempt_profile_ok(games):
        continue

    p_over = american_to_prob(float(offer["over_odds"]))
    p_under = american_to_prob(float(offer["under_odds"]))
    p_market = p_over / max(p_over + p_under, 1e-9)

    value_edge = p["prob_over"] - p_market
    ev = ev_per_dollar(p["prob_over"], float(offer["over_odds"]))

    bucket = player_risk_bucket(p["m10"], p["sigma"], prop_type)
    thr = thresholds_for_bucket(bucket)

    slate_min_edge = max(thr["min_edge"], 3.5)
    slate_min_prob = max(thr["min_prob"], 0.64)
    slate_min_ev = max(thr["min_ev"], 0.03)
    slate_min_value = max(thr["min_value_edge"], 0.03)

    if p["edge"] < slate_min_edge or p["prob_over"] < slate_min_prob:
        continue
    if value_edge < slate_min_value:
        continue
    if ev < slate_min_ev:
        continue

    steam = 0.0
    if ENABLE_STEAM:
        prev = get_prev_market(state, prop_type, pid, now_ts)
        if prev:
            cur = {"line": line, "over_odds": offer["over_odds"], "under_odds": offer["under_odds"], "ts": now_ts}
            steam = steam_score(prev, cur)

    team_name = player_team
    gid = int(offer.get("gid") or offer.get("game_id") or 0)
    stab = stability_score(p["edge"], p["sigma"])
    vol_pen = volatility_penalty(p["sigma"])
    final_score = final_play_score(ev, value_edge, p["edge"], p["min_conf"], p["matchup_score"], le_score, stab, vol_pen)
    tier = confidence_tier(p["edge"], p["prob_over"], ev, value_edge)
    context_notes = " | ".join(filter(None, [p["home_away_note"], p["b2b_note"]]))

    why = (
        f"[{tier}] Slate. base {p['base_avg']:.1f} | L10 {p['l10_avg']:.1f} | L3 {p['l3_avg']:.1f} "
        f"(minL10 {p['l10_min']:.1f}, projMin {p['proj_min']:.1f}, rate {p['proj_rate']:.3f}, cons {p['consistency']:.2f}). "
        f"Deltamin={p['min_delta']:+.1f} Deltarate={p['rate_delta']:+.2f}. "
        f"Proj {p['proj']:.1f} vs line {line:.1f} | "
        f"{offer['vendor']} ({int(offer['over_odds']):+d}) | "
        f"edge +{p['edge']:.1f} | P={p['prob_over']*100:.0f}% (mkt={p_market*100:.0f}%, val={value_edge:+.2f}) | "
        f"EV={ev:+.2f} | matchup={p['matchup_note']}{(' | ' + context_notes) if context_notes else ''}"
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
        "proj": float(p["proj"]),
        "edge": float(p["edge"]),
        "prob_over": float(p["prob_over"]),
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
        "le_score": float(le_score),
        "min_conf": float(p["min_conf"]),
        "stability_score": float(stab),
        "final_score": float(final_score),
        "tier": tier,
        "consistency": float(p["consistency"]),
    })

    remember_market(state, prop_type, int(pid), offer, line, n_cons, now_ts)

ideas.sort(key=lambda x: (x["final_score"], x["ev"], x["value_edge"]), reverse=True)
return ideas
```

def lineup_news_edges(now_et, prop_type, lines_map_for_prop, state, now_ts, news_boosts, news_scores, games_map=None):
if not LE_NEWS_ENGINE_ENABLED or deadline_exceeded():
return []

```
season = _season_year(now_et)
today_str = now_et.strftime("%Y-%m-%d")
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

    name = PLAYER_NAME_CACHE.get(int(pid), f"Player {pid}")
    nm_key = _clean_name(name)
    rec = news_boosts.get(nm_key)
    ns = news_scores.get(nm_key, {"score": 0.0})

    if not rec:
        continue

    le_eff = float(rec.get("boost", 0.0)) * float(rec.get("confidence", 0.0))
    le_score = float(ns.get("score", 0.0))

    if le_eff <= 0 or le_eff < LE_NEWS_MIN_EFFECT or le_score < LE_NEWS_MIN_SCORE:
        continue

    games = stats_all.get(int(pid), [])
    if len(games) < 8:
        continue

    rows = (lines_map_for_prop or {}).get(int(pid), [])
    cons, n_cons, n_sharp = consensus_line(rows)
    if cons is None or n_cons < MIN_VENDORS_FOR_CONSENSUS:
        continue

    offer = best_offer_near_consensus(rows, cons)
    if not offer:
        continue

    line = float(cons)
    player_team = PLAYER_TEAM_CACHE.get(int(pid), "")

    p = build_player_projection(
        games=games,
        line=line,
        prop_type=prop_type,
        le_score=le_score,
        news_boosts=news_boosts,
        player_name=name,
        player_team=player_team,
        games_map=games_map,
        gid=int(offer.get("gid") or offer.get("game_id") or 0),
        today_str=today_str,
    )

    if p["m10"] < MIN_L10_MIN:
        continue
    if should_bad_role_filter(p["min_delta"], p["rate_delta"], le_score):
        continue
    if not minutes_stability_ok(p["l10_min"], p["l3_min"], le_score):
        continue
    if prop_type == "threes" and not threes_attempt_profile_ok(games):
        continue

    p_over = american_to_prob(float(offer["over_odds"]))
    p_under = american_to_prob(float(offer["under_odds"]))
    p_market = p_over / max(p_over + p_under, 1e-9)

    value_edge = p["prob_over"] - p_market
    ev = ev_per_dollar(p["prob_over"], float(offer["over_odds"]))

    bucket = player_risk_bucket(p["m10"], p["sigma"], prop_type)
    thr = thresholds_for_bucket(bucket)

    le_min_edge = max(1.8, thr["min_edge"] - 0.5)
    le_min_prob = max(0.58, thr["min_prob"] - 0.03)
    le_min_value = max(0.01, thr["min_value_edge"])
    le_min_ev = max(0.00, thr["min_ev"] - 0.01)

    if p["edge"] < le_min_edge or p["prob_over"] < le_min_prob:
        continue
    if value_edge < le_min_value or ev < le_min_ev:
        continue

    team_name = player_team
    gid = int(offer.get("gid") or offer.get("game_id") or 0)
    stab = stability_score(p["edge"], p["sigma"])
    vol_pen = volatility_penalty(p["sigma"])
    final_score = final_play_score(ev, value_edge, p["edge"], p["min_conf"], p["matchup_score"], le_score, stab, vol_pen) + 8.0
    tier = confidence_tier(p["edge"], p["prob_over"], ev, value_edge)

    why = (
        f"[{tier}] LineupNews. base {p['base_avg']:.1f} | L10 {p['l10_avg']:.1f} | L3 {p['l3_avg']:.1f} "
        f"(minL10 {p['l10_min']:.1f}, projMin {p['proj_min']:.1f}). "
        f"Proj {p['proj']:.1f} vs line {line:.1f} | "
        f"{offer['vendor']} ({int(offer['over_odds']):+d}) | "
        f"edge +{p['edge']:.1f} | P={p['prob_over']*100:.0f}% (val={value_edge:+.2f}) | "
        f"EV={ev:+.2f} | LE_boost={p['news_eff']:+.2f} | matchup={p['matchup_note']}"
    )

    ideas.append({
        "section": "lineupnews",
        "prop_type": prop_type,
        "player_name": name,
        "player_id": int(pid),
        "team": team_name,
        "gid": gid,
        "cons_line": float(line),
        "line": float(offer["line"]),
        "proj": float(p["proj"]),
        "edge": float(p["edge"]),
        "prob_over": float(p["prob_over"]),
        "market_prob": float(p_market),
        "value_edge": float(value_edge),
        "ev": float(ev),
        "vendor": str(offer["vendor"]),
        "over_odds": float(offer["over_odds"]),
        "under_odds": float(offer["under_odds"]),
        "n_cons": int(n_cons),
        "n_sharp": int(n_sharp),
        "steam": 0.0,
        "trigger_strength": 0.0,
        "trigger": "LineupExperts league news",
        "why": why,
        "le_score": float(le_score),
        "final_score": float(final_score),
        "tier": tier,
        "consistency": float(p["consistency"]),
    })

ideas.sort(key=lambda x: (x["final_score"], x["ev"], x["value_edge"]), reverse=True)
return ideas[:LE_NEWS_TOPN]
```

def plus_odds_hunt_edges(now_et, prop_type, lines_map_for_prop, state, now_ts, news_boosts, news_scores, games_map=None):
if not PLUS_HUNT_ENABLED or deadline_exceeded():
return []

```
season = _season_year(now_et)
today_str = now_et.strftime("%Y-%m-%d")
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
    if cons is None or n_cons < MIN_VENDORS_FOR_CONSENSUS:
        continue

    offer = best_offer_near_consensus(rows, cons)
    if not offer:
        continue

    try:
        if float(offer.get("over_odds", -999)) < PLUS_ODDS_MIN:
            continue
    except Exception:
        continue

    line = float(cons)
    name = PLAYER_NAME_CACHE.get(int(pid), f"Player {pid}")
    le_score = float(news_scores.get(_clean_name(name), {"score": 0.0}).get("score", 0.0))
    player_team = PLAYER_TEAM_CACHE.get(int(pid), "")

    p = build_player_projection(
        games=games,
        line=line,
        prop_type=prop_type,
        le_score=le_score,
        news_boosts=news_boosts,
        player_name=name,
        player_team=player_team,
        games_map=games_map,
        gid=int(offer.get("gid") or offer.get("game_id") or 0),
        today_str=today_str,
    )

    if p["m10"] < MIN_L10_MIN:
        continue
    if should_bad_role_filter(p["min_delta"], p["rate_delta"], le_score):
        continue
    if not minutes_stability_ok(p["l10_min"], p["l3_min"], le_score):
        continue
    if prop_type == "threes" and not threes_attempt_profile_ok(games):
        continue

    p_over = american_to_prob(float(offer["over_odds"]))
    p_under = american_to_prob(float(offer["under_odds"]))
    p_market = p_over / max(p_over + p_under, 1e-9)

    value_edge = p["prob_over"] - p_market
    ev = ev_per_dollar(p["prob_over"], float(offer["over_odds"]))

    if p["prob_over"] < PLUS_HUNT_MIN_PROB:
        continue
    if value_edge < PLUS_HUNT_MIN_VALUE_EDGE or ev < PLUS_HUNT_MIN_EV:
        continue

    role_bonus = min(0.25, (0.15 if p["min_delta"] >= 2.0 else 0.0) + (0.10 if p["rate_delta"] > 0.05 else 0.0))
    plus_score = (ev * 1.2) + (value_edge * 100.0) + (le_score * 8.0) + (role_bonus * 5.0) + (p["matchup_score"] * 5.0)
    tier = confidence_tier(p["edge"], p["prob_over"], ev, value_edge)

    why = (
        f"[{tier}] PlusHunt {offer['vendor']} ({int(offer['over_odds']):+d}) | "
        f"P={p['prob_over']*100:.0f}% (mkt={p_market*100:.0f}%, val={value_edge:+.2f}) | "
        f"EV={ev:+.2f} | matchup={p['matchup_note']} | news={le_score:+.2f}"
    )

    ideas.append({
        "section": "plus",
        "prop_type": prop_type,
        "player_name": name,
        "player_id": int(pid),
        "team": player_team,
        "gid": int(offer.get("gid") or offer.get("game_id") or 0),
        "cons_line": float(line),
        "line": float(offer["line"]),
        "proj": float(p["proj"]),
        "edge": float(p["edge"]),
        "prob_over": float(p["prob_over"]),
        "market_prob": float(p_market),
        "value_edge": float(value_edge),
        "ev": float(ev),
        "vendor": str(offer["vendor"]),
        "over_odds": float(offer["over_odds"]),
        "under_odds": float(offer["under_odds"]),
        "n_cons": int(n_cons),
        "n_sharp": int(n_sharp),
        "steam": 0.0,
        "trigger_strength": 0.0,
        "trigger": "Plus-odds hunter",
        "why": why,
        "plus_score": float(plus_score),
        "news_score": float(le_score),
        "tier": tier,
    })

    remember_market(state, prop_type, int(pid), offer, line, n_cons, now_ts)

ideas.sort(key=lambda x: x.get("plus_score", 0.0), reverse=True)
return ideas
```

# –––––––––– COOLDOWN / EXPOSURE ––––––––––

def apply_cooldown(state, ideas, now_ts: int):
sent = state.get(“sent_bets”, {}) or {}
cooldown_sec = BET_COOLDOWN_MIN * 60
kept = []
for i in ideas:
key = f”{i[‘prop_type’]}|{i[‘section’]}|{int(i[‘player_id’])}|{float(i.get(‘cons_line’, i.get(‘line’, 0.0))):.1f}”
prev = sent.get(key)
if not prev:
kept.append(i)
continue
last_ts = int(prev.get(“ts”, 0) or 0)
last_edge = float(prev.get(“edge”, 0.0) or 0.0)
if “edge” in i and (float(i.get(“edge”, 0.0)) - last_edge) >= EDGE_JUMP_TO_RESEND:
kept.append(i)
continue
if (now_ts - last_ts) >= cooldown_sec:
kept.append(i)
return kept

def record_sent(state, ideas, now_ts: int):
sent = state.get(“sent_bets”, {}) or {}
for i in ideas:
key = f”{i[‘prop_type’]}|{i[‘section’]}|{int(i[‘player_id’])}|{float(i.get(‘cons_line’, i.get(‘line’, 0.0))):.1f}”
sent[key] = {“ts”: now_ts, “edge”: float(i.get(“edge”, 0.0))}
state[“sent_bets”] = sent

def apply_exposure_caps(ideas):
if not ideas:
return ideas
team_ct = {}
game_ct = {}
out = []
for it in ideas:
team = (it.get(“team”) or “”).strip().lower()
gid = int(it.get(“gid”) or 0)
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

# –––––––––– WHATSAPP CARD FORMATTER ––––––––––

def format_play_card(play: dict, idx: int) -> str:
“””
IMPROVEMENT: Clean, scannable WhatsApp card format.

```
Each card is designed to be read in 3 seconds:
Line 1: Player, over line, tier label
Line 2: Key numbers -- projection, edge, probability, EV
Line 3: Context -- matchup, home/away, news signal
"""
tier = play.get("tier", "")
name = play["player_name"]
line = play["cons_line"]
proj = play["proj"]
edge = play["edge"]
prob = play["prob_over"] * 100
ev = play["ev"]
over_odds = int(play["over_odds"])
vendor = play["vendor"]
team = play.get("team", "")

# Extract key context from why string -- keep it short
matchup_hint = ""
why = play.get("why", "")
m = re.search(r"matchup=([^\s|]+)", why)
if m:
    matchup_hint = m.group(1)

b2b = "[WARN] B2B" if "b2b(" in why else ""
le_note = ""
if play.get("le_score", 0) > 0.5:
    le_note = "[NEWS]+"
elif play.get("le_score", 0) < -0.5:
    le_note = "[NEWS]-"

consistency = play.get("consistency", 0)
cons_note = f"[AIM]{consistency:.0%}" if consistency > 0 else ""

card = [
    f"{idx}. {tier} {name} ({team}) OVER {line:.1f}",
    f"   Proj {proj:.1f} | edge +{edge:.1f} | P={prob:.0f}% | EV={ev:+.2f}",
    f"   {vendor} {over_odds:+d} | def:{matchup_hint} {cons_note} {le_note} {b2b}".rstrip(),
]
return "\n".join(card)
```

# –––––––––– MAIN ––––––––––

def run():
now_et = _now_et()
ts_et = now_et.strftime(”%Y-%m-%d %I:%M %p ET”)
now_ts = int(now_et.timestamp())
today_str = now_et.strftime(”%Y-%m-%d”)

```
print(
    f"[BOOT] ts={ts_et} TEST_MODE={int(TEST_MODE)} "
    f"PROP_TYPES={','.join(PROP_TYPES)} STD_FLOOR={STD_FLOOR} "
    f"PROJ_WEIGHTS=base{PROJ_WEIGHT_BASE}/l10{PROJ_WEIGHT_L10}/l3{PROJ_WEIGHT_L3} "
    f"HOME_BOOST={HOME_COURT_BOOST} B2B_PENALTY={B2B_PENALTY} OPP_DEF={int(ENABLE_OPP_DEF_ADJ)}"
)

if TEST_MODE:
    send_one(f"[STRONG] NBA betting agent v2 test OK ({ts_et})")
    return

state = load_state()
old_players = state.get("players", {})

lines_map, games_map = build_today_props(now_et)

season = _season_year(now_et)
all_prop_pids = []
for pt in PROP_TYPES:
    all_prop_pids.extend(list(lines_map.get(pt, {}).keys()))
all_prop_pids = list({int(x) for x in all_prop_pids})

if all_prop_pids:
    for chunk_ids in _chunk(all_prop_pids, STAT_BATCH_SIZE):
        if deadline_exceeded():
            break
        try:
            bdl_last_n_games_stats(chunk_ids, season, max(LOOKBACK_GAMES, 8), "pts")
        except Exception as e:
            print(f"[WARN] warmup stats failed: {e}")

    if "threes" in PROP_TYPES and THREES_BETA_BINOM:
        for chunk_ids in _chunk(all_prop_pids, STAT_BATCH_SIZE):
            if deadline_exceeded():
                break
            try:
                bdl_last_n_games_threes(chunk_ids, season, max(LOOKBACK_GAMES, 8))
            except Exception as e:
                print(f"[WARN] warmup threes failed: {e}")

news_items = fetch_lineupexperts_news(now_et) if LINEUPEXPERTS else []
news_boosts = build_news_boost_map(news_items) if news_items else {}
news_scores = build_news_score_map(news_items) if news_items else {}

if LINEUPEXPERTS:
    print(f"[INFO] LineupExperts news_items={len(news_items)} boosts={len(news_boosts)}")

# ---- Injury engine ----
new_players = {}
triggers = []
injury_ideas_all = []

if ENABLE_INJURY_TRIGGERS and (not deadline_exceeded()):
    if USE_LE_MAIN_INJURY_ENGINE and LINEUPEXPERTS:
        le_injuries = parse_le_injuries(news_items)
        exclude_names_lower = {k for k in le_injuries.keys()}

        for _, cur in le_injuries.items():
            if deadline_exceeded():
                break
            if not status_in_scope(cur.get("status", "")):
                continue

            team_short = normalize_team_name(cur.get("team", ""))
            injured_name = cur.get("name", "")
            injured_status = (cur.get("status") or "").strip()

            if not team_short:
                team_short = find_team_for_player_from_cache(injured_name)
            if not team_short:
                continue

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
                    news_boosts=news_boosts,
                    news_scores=news_scores,
                    games_map=games_map,
                )
                if ideas:
                    got_any = True
                    injury_ideas_all.extend(ideas)

            if got_any:
                triggers.append(f"{injured_name} ({team_short}) {injured_status}")

        new_players = {k: {"name": v["name"], "team": v["team"], "status": v["status"], "detail": v["detail"]} for k, v in le_injuries.items()}

    else:
        try:
            sr = fetch_sportradar_injuries()
            parsed = parse_injuries(sr)
        except Exception as e:
            print(f"[WARN] Sportradar injuries failed: {e}")
            parsed = {}

        exclude_names_lower = {_clean_name(v.get("name", "")) for v in parsed.values() if v.get("name")}

        for pid, cur in parsed.items():
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

            team_short = normalize_team_name(cur.get("team", ""))
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
                    news_boosts=news_boosts,
                    news_scores=news_scores,
                    games_map=games_map,
                )
                if ideas:
                    got_any = True
                    injury_ideas_all.extend(ideas)

            if got_any:
                triggers.append(f"{injured_name} ({team_short}) {injured_status}")

        new_players = parsed

# ---- Slate scan ----
slate_ideas_all = []
if ENABLE_SLATE_SCAN and (not deadline_exceeded()):
    for pt in PROP_TYPES:
        if deadline_exceeded():
            break
        slate_ideas_all.extend(
            slate_scan_edges(now_et, pt, lines_map.get(pt, {}), state=state, now_ts=now_ts,
                             news_boosts=news_boosts, news_scores=news_scores, games_map=games_map)
        )

# ---- Lineup news edges ----
lineup_news_ideas_all = []
if LE_NEWS_ENGINE_ENABLED and (not deadline_exceeded()):
    for pt in PROP_TYPES:
        if deadline_exceeded():
            break
        lineup_news_ideas_all.extend(
            lineup_news_edges(now_et, pt, lines_map.get(pt, {}), state=state, now_ts=now_ts,
                              news_boosts=news_boosts, news_scores=news_scores, games_map=games_map)
        )
lineup_news_ideas_all = lineup_news_ideas_all[:LE_NEWS_TOPN]

# ---- Plus odds ----
plus_ideas_all = []
if ENABLE_SLATE_SCAN and PLUS_HUNT_ENABLED and (not deadline_exceeded()):
    for pt in PROP_TYPES:
        if deadline_exceeded():
            break
        plus_ideas_all.extend(
            plus_odds_hunt_edges(now_et, pt, lines_map.get(pt, {}), state=state, now_ts=now_ts,
                                 news_boosts=news_boosts, news_scores=news_scores, games_map=games_map)
        )
plus_ideas_all = plus_ideas_all[:PLUS_HUNT_TOPN]

# ---- Merge and deduplicate ----
combined = injury_ideas_all + slate_ideas_all + lineup_news_ideas_all
best = {}
for i in combined:
    k = (i["prop_type"], int(i["player_id"]))
    score = (float(i.get("final_score", 0.0)), float(i.get("ev", 0.0)), float(i.get("value_edge", 0.0)), float(i.get("edge", 0.0)))
    if (k not in best) or (score > best[k][0]):
        best[k] = (score, i)

combined = [v[1] for v in best.values()]
combined = apply_cooldown(state, combined, now_ts)

out_by_market = {}
for pt in PROP_TYPES:
    inj = sorted([x for x in combined if x["prop_type"] == pt and x["section"] == "injury"],
                 key=lambda x: (x["final_score"], x["trigger_strength"], x["ev"]), reverse=True)
    slt = sorted([x for x in combined if x["prop_type"] == pt and x["section"] == "slate"],
                 key=lambda x: (x["final_score"], x["ev"], x["value_edge"]), reverse=True)
    lne = sorted([x for x in combined if x["prop_type"] == pt and x["section"] == "lineupnews"],
                 key=lambda x: (x["final_score"], x["ev"], x["value_edge"]), reverse=True)

    picks = inj[:MAX_INJURY_PLAYS] + lne[:MAX_LINEUPNEWS_PLAYS] + slt[:MAX_SLATE_PLAYS]
    picks = picks[:MAX_PER_MARKET]
    out_by_market[pt] = picks

final_out = []
for pt in PROP_TYPES:
    final_out.extend(out_by_market.get(pt, []))
final_out = final_out[:MAX_TOTAL_PLAYS]
final_out = apply_exposure_caps(final_out)

capped_by_market = {pt: [] for pt in PROP_TYPES}
for it in final_out:
    capped_by_market[it["prop_type"]].append(it)
out_by_market = capped_by_market

# Track results
if ENABLE_RESULT_TRACKING:
    for play in final_out:
        log_play_for_tracking(play, now_ts)

# ---- Build WhatsApp message ----
if final_out:
    msg = []
    hit_rate_str = get_hit_rate_summary()

    # IMPROVEMENT: Clean summary card at top for fast scanning
    msg.append(f"[NBA] NBA PROPS -- {ts_et}")
    if hit_rate_str:
        msg.append(hit_rate_str)
    msg.append(f"[LOCK]=Lock  [STRONG]=Strong  [LEAN]=Lean")
    msg.append("-" * 30)
    msg.append("")

    # Quick-scan card for each play
    for idx, play in enumerate(final_out, 1):
        msg.append(format_play_card(play, idx))
    msg.append("")
    msg.append("-" * 30)

    # Injury context
    if triggers:
        msg.append("")
        msg.append("\u1f691 Injury triggers:")
        for t in triggers[:6]:
            msg.append(f"  - {t}")
        msg.append("")

    # Detail section for each play
    msg.append("[LIST] DETAIL:")
    msg.append("")
    for pt in PROP_TYPES:
        picks = out_by_market.get(pt, [])
        if not picks:
            continue
        label = "Points" if pt == "points" else ("3PT" if pt in ("threes", "three_pointers_made") else pt)
        msg.append(f"[ {label} ]")

        for i in picks:
            msg.append(f"- {i['player_name']} OVER {i['cons_line']:.1f}")
            msg.append(f"  {i['why']}")
            msg.append("")

    # Plus odds section
    plus_bucket = [x for x in final_out if x.get("over_odds", -999) >= PLUS_ODDS_MIN]
    plus_bucket.sort(key=lambda x: (x["ev"], x["value_edge"]), reverse=True)
    plus_bucket = plus_bucket[:PLUS_ODDS_TOPN]

    if plus_bucket or plus_ideas_all:
        msg.append("[PLUS] PLUS ODDS:")
        for i in (plus_bucket + [x for x in plus_ideas_all if x not in plus_bucket])[:PLUS_ODDS_TOPN]:
            msg.append(f"- {i['player_name']} OVER {i['cons_line']:.1f} ({i['vendor']} {int(i['over_odds']):+d}) P={i['prob_over']*100:.0f}% EV={i['ev']:+.2f}")
        msg.append("")

    msg.append(f"[CAPS] Caps: team<={MAX_PLAYS_PER_TEAM}, game<={MAX_PLAYS_PER_GAME}")
    send_chunked("\n".join(msg).strip())

    record_sent(state, final_out, now_ts)
else:
    print("[INFO] No plays cleared thresholds this run.")

state["players"] = new_players
save_state(state)
```

if **name** == “**main**”:
run()
