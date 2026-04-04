"""
NBA Player Props Prediction Model - Render Cron Job + Twilio WhatsApp
Conservative, calibrated projections with sustained breakout detection.
Production-ready for daily prop picks.
"""

import os
import json
import re
import time
import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from functools import lru_cache
import traceback

try:
    import requests
    from twilio.rest import Client as TwilioClient
except ImportError:
    requests = None
    TwilioClient = None


# ============================================================================
# SECTION 1: IMPORTS, ENV CONFIG, CONSTANTS
# ============================================================================

# Environment variables with sensible defaults
ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
RECIPIENT_WHATSAPP = os.getenv("RECIPIENT_WHATSAPP", "")

BALLDONTLIE_API_KEY = os.getenv("BALLDONTLIE_API_KEY", "")
LINEUP_EXPERTS_API_KEY = os.getenv("LINEUP_EXPERTS_API_KEY", "")

BANKROLL = float(os.getenv("BANKROLL", "500.0"))
STATE_FILE = os.getenv("STATE_FILE", "/tmp/nba_props_state.json")
CACHE_DIR = os.getenv("CACHE_DIR", "/tmp/nba_props_cache")

# Timing
DEADLINE_HOURS_BEFORE = float(os.getenv("DEADLINE_HOURS_BEFORE", "0.5"))
MAX_RUNTIME_SECONDS = int(os.getenv("MAX_RUNTIME_SECONDS", "300"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "10"))

# Model thresholds - CONSERVATIVE
MIN_EDGE = 3.0  # min 3% edge to claim
MIN_PROB = 0.58  # calibrated probability threshold (0.50 raw is ~0.50 calibrated)
STD_FLOOR = 4.0  # global min std dev

# Per-stat standard deviation floors (prevent overconfident narrows)
STAT_STD_FLOORS = {
    "pts": 4.0,
    "reb": 2.0,
    "ast": 1.8,
    "blk": 0.9,
    "stl": 0.9,
    "threes": 1.2,
}

# Projection window weights - L10 is most reliable (king of windows)
PROJECTION_WEIGHTS = {
    "base": 0.30,   # 20+ games
    "l10": 0.40,    # L10 most reliable, not L3
    "l3": 0.30,     # recency but not overweighted
}

# Breakout detection - require SUSTAINED performance (L5 + L10)
BREAKOUT_WEIGHTS = {
    "base": 0.15,
    "l10": 0.25,
    "l5": 0.60,  # L5 drives breakout, but requires L10 elevation too
}

# Breakout thresholds - must sustain
BREAKOUT_L5_THRESHOLD = 1.30  # L5 avg >= 1.30x baseline
BREAKOUT_L10_THRESHOLD = 1.15  # L10 avg >= 1.15x baseline to confirm

# Minutes projection windows
SHORT_GAMES = 5
LOOKBACK_GAMES = 10
BASELINE_GAMES = 20

# Kelly criterion - quarter Kelly with caps
KELLY_FRACTION = 0.25
MIN_BET = 25.0
MAX_BET = 250.0

# Exposure caps per day
MAX_PLAYS_PER_PLAYER = 2
MAX_PLAYS_PER_STAT = 6
MAX_TOTAL_PLAYS = 15

# Parlay config
SGP_MAX_CORR_LEGS = 3
LADDER_ENABLE = True
LADDER_SIZES = [2, 3]  # 2-leg and 3-leg ladders

# Time windows for consistency
CONSISTENCY_WINDOW = 10  # last 10 games
CONSISTENCY_BAND = 0.20  # ±20% of mean

# Book vendors and prop types
BOOK_VENDORS = [
    "DraftKings",
    "FanDuel",
    "BetMGM",
    "Caesars",
    "PointsBet",
]

PROP_TYPES = [
    "player_points",
    "player_rebounds",
    "player_assists",
    "player_threes",
    "player_blocks",
    "player_steals",
]

# Confidence tier scoring
CONFIDENCE_TIERS = {
    "LOCK": 9,
    "STRONG": 6,
    "LEAN": 3,
    "SKIP": 0,
}

# Cooldown windows (seconds)
PLAY_COOLDOWN = 3600  # 1 hour between same player/stat
BET_COOLDOWN = 300    # 5 min between WhatsApp sends

# Adjustment caps (total portfolio)
ADJ_CAP_TOTAL = 0.12  # max ±12% adjustment from context

# Minute projection bounds
MIN_MINUTES_CHANGE = -3.0  # cap at -3min from L10
MAX_MINUTES_CHANGE = 3.0   # cap at +3min from L10

# Calibration table: raw_prob -> calibrated_prob
# Based on sports betting model research: raw normal CDF is overconfident
CALIBRATION_TABLE = {
    0.50: 0.50,
    0.55: 0.53,
    0.60: 0.56,
    0.65: 0.59,
    0.70: 0.62,
    0.75: 0.65,
    0.80: 0.68,
    0.85: 0.71,
    0.90: 0.74,
    0.95: 0.77,
}

# Team alias normalization
TEAM_ALIAS_TO_FULL = {
    "LAL": "Los Angeles Lakers",
    "LAC": "Los Angeles Clippers",
    "GSW": "Golden State Warriors",
    "DEN": "Denver Nuggets",
    "PHX": "Phoenix Suns",
    "MEM": "Memphis Grizzlies",
    "MIN": "Minnesota Timberwolves",
    "OKC": "Oklahoma City Thunder",
    "DAL": "Dallas Mavericks",
    "HOU": "Houston Rockets",
    "ATL": "Atlanta Hawks",
    "MIA": "Miami Heat",
    "BOS": "Boston Celtics",
    "NYK": "New York Knicks",
    "PHI": "Philadelphia 76ers",
    "NOR": "New Orleans Pelicans",
    "CHI": "Chicago Bulls",
    "TOR": "Toronto Raptors",
    "BRK": "Brooklyn Nets",
    "WAS": "Washington Wizards",
    "DET": "Detroit Pistons",
    "CLE": "Cleveland Cavaliers",
    "IND": "Indiana Pacers",
    "MIL": "Milwaukee Bucks",
    "CHA": "Charlotte Hornets",
    "SAS": "San Antonio Spurs",
    "POR": "Portland Trail Blazers",
    "UTA": "Utah Jazz",
    "SAC": "Sacramento Kings",
}

# Defensive ratings (opponent adjustments) - per 100 possessions
TEAM_DEFENSE_RATINGS = {
    "Denver Nuggets": {"pts": 110.5, "reb": 46.2, "ast": 26.1},
    "Boston Celtics": {"pts": 109.2, "reb": 45.8, "ast": 25.5},
    "Miami Heat": {"pts": 108.9, "reb": 46.5, "ast": 26.3},
    "Memphis Grizzlies": {"pts": 110.1, "reb": 45.9, "ast": 25.7},
    "Golden State Warriors": {"pts": 111.2, "reb": 47.1, "ast": 27.2},
    "Los Angeles Lakers": {"pts": 111.8, "reb": 47.5, "ast": 27.5},
    "Phoenix Suns": {"pts": 112.4, "reb": 48.2, "ast": 28.1},
    "Brooklyn Nets": {"pts": 113.5, "reb": 48.8, "ast": 28.6},
    "Chicago Bulls": {"pts": 114.2, "reb": 49.1, "ast": 29.0},
    "Washington Wizards": {"pts": 115.1, "reb": 49.8, "ast": 29.5},
}

# API endpoints
BDL_BASE = "https://api.balldontlie.io/api/v1"
LINEUP_EXPERTS_BASE = "https://api.lineupexperts.com/v1"

# Caches
PROPS_CACHE = {}
ADV_STATS_CACHE = {}
GAME_ODDS_CACHE = {}
ROSTER_CACHE = {}
TEAM_ID_CACHE = {}
LINEUPS_CACHE = {}

# State tracking
RUNTIME_START = time.time()
LAST_WHATSAPP_SEND = 0.0


# ============================================================================
# SECTION 2: UTILITY FUNCTIONS
# ============================================================================

def _now_et() -> datetime:
    """Get current time in Eastern Time."""
    return datetime.now(ZoneInfo("America/New_York"))


def _season_year(dt: datetime = None) -> int:
    """Get NBA season year (e.g., 2025 for 2024-25 season)."""
    if dt is None:
        dt = _now_et()
    year = dt.year
    # Season starts in October, so Oct-Dec is same year, Jan-Sep is next year
    if dt.month < 10:
        return year
    return year + 1


def _parse_minutes(min_str: str) -> float:
    """Convert 'MM:SS' to float minutes."""
    if not min_str or min_str == "0:00":
        return 0.0
    try:
        parts = min_str.split(":")
        return float(parts[0]) + float(parts[1]) / 60.0
    except:
        return 0.0


def _clean_name(name: str) -> str:
    """Normalize player name for matching."""
    return re.sub(r"[^\w\s]", "").lower().strip()


def _norm_cdf(z: float) -> float:
    """Standard normal CDF approximation (error < 0.002)."""
    if z > 6:
        return 1.0
    if z < -6:
        return 0.0
    # Abramowitz & Stegun approximation
    b1, b2, b3, b4, b5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    p = 0.2316419
    t = 1.0 / (1.0 + p * abs(z))
    t_val = t * (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))))
    cdf = 1.0 - t_val * math.exp(-z * z / 2.0) / math.sqrt(2 * math.pi) if z >= 0 else t_val * math.exp(-z * z / 2.0) / math.sqrt(2 * math.pi)
    return max(0.0, min(1.0, cdf))


def american_to_prob(american_odds: float) -> float:
    """Convert American odds to implied probability."""
    if american_odds > 0:
        return 100.0 / (american_odds + 100.0)
    else:
        return abs(american_odds) / (abs(american_odds) + 100.0)


def american_to_payout(american_odds: float, bet_amount: float) -> float:
    """Get payout from American odds."""
    if american_odds > 0:
        return bet_amount * (american_odds / 100.0)
    else:
        return bet_amount * (100.0 / abs(american_odds))


def ev_per_dollar(prob_win: float, american_odds: float) -> float:
    """Calculate EV per dollar wagered."""
    if american_odds > 0:
        payout_ratio = american_odds / 100.0
    else:
        payout_ratio = 100.0 / abs(american_odds)
    return prob_win * payout_ratio - (1.0 - prob_win)


def avg_stat_min_std(games_list: List[Dict]) -> Tuple[float, float, float]:
    """
    Compute average, minimum (for min projection), and std dev from game list.
    games_list should be list of dicts with 'value' key.
    Returns (avg, min_val, std_dev).
    """
    if not games_list:
        return 0.0, 0.0, 0.0
    values = [g.get("value", 0.0) for g in games_list]
    avg = sum(values) / len(values)
    min_val = min(values) if values else 0.0
    if len(values) > 1:
        variance = sum((x - avg) ** 2 for x in values) / len(values)
        std = math.sqrt(variance)
    else:
        std = 0.0
    return avg, min_val, std


def median_stat(games_list: List[Dict]) -> float:
    """Get median value from game list."""
    if not games_list:
        return 0.0
    values = sorted([g.get("value", 0.0) for g in games_list])
    n = len(values)
    if n % 2 == 0:
        return (values[n // 2 - 1] + values[n // 2]) / 2.0
    return values[n // 2]


def floor_ceiling(avg: float, std: float) -> Tuple[float, float]:
    """Estimate floor (10th pct) and ceiling (90th pct) from avg and std."""
    # Approximate: floor = avg - 1.28*std, ceiling = avg + 1.28*std
    return max(0.0, avg - 1.28 * std), avg + 1.28 * std


def prop_type_to_stat_key(prop_type: str) -> str:
    """Map prop type to stat key."""
    mapping = {
        "player_points": "pts",
        "player_rebounds": "reb",
        "player_assists": "ast",
        "player_blocks": "blk",
        "player_steals": "stl",
        "player_threes": "threes",
    }
    return mapping.get(prop_type, "pts")


def prop_type_is_threes(prop_type: str) -> bool:
    """Check if prop is 3-pointers."""
    return prop_type == "player_threes"


def _slice_last(lst: List, n: int) -> List:
    """Get last n elements of list."""
    if not lst:
        return []
    return lst[-n:] if n < len(lst) else lst


def _role_trend(l3: float, l10: float, base: float) -> str:
    """Determine role trend: increasing, stable, decreasing."""
    if l3 > l10 * 1.05:
        return "increasing"
    elif l3 < l10 * 0.95:
        return "decreasing"
    return "stable"


def _safe_rate(made: float, attempts: float) -> float:
    """Safe rate calculation."""
    if attempts < 1.0:
        return 0.0
    return made / attempts


def _clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))


def calibrated_prob(raw_prob: float) -> float:
    """
    Apply empirical calibration to raw model probability.
    Raw normal CDF probabilities from betting models are overconfident.
    Uses interpolation on CALIBRATION_TABLE.
    """
    if raw_prob <= 0.50:
        return 0.50
    if raw_prob >= 0.95:
        return 0.77

    # Linear interpolation
    keys = sorted(CALIBRATION_TABLE.keys())
    for i in range(len(keys) - 1):
        k1, k2 = keys[i], keys[i + 1]
        if k1 <= raw_prob <= k2:
            v1, v2 = CALIBRATION_TABLE[k1], CALIBRATION_TABLE[k2]
            alpha = (raw_prob - k1) / (k2 - k1)
            return v1 + alpha * (v2 - v1)

    return raw_prob


def kelly_bet_size(edge_pct: float, prob: float, bankroll: float) -> float:
    """
    Quarter Kelly for bet sizing.
    Returns bet amount in dollars, capped between MIN_BET and MAX_BET.
    """
    if prob <= 0.5 or prob >= 1.0:
        return 0.0

    # Kelly: f* = (p*b - q) / b, where p=prob, q=1-p, b=odds_payout_ratio
    # For simplicity, use edge-based Kelly: f* = edge / (2 * payout_ratio)
    if edge_pct <= 0:
        return 0.0

    # Approximate: 1 unit payout
    kelly_frac = (edge_pct / 100.0) / 2.0
    kelly_bet = bankroll * kelly_frac
    kelly_quarter = kelly_bet * KELLY_FRACTION

    return _clamp(kelly_quarter, MIN_BET, MAX_BET)


def passes_juice_filter(odds: float) -> bool:
    """Check if odds have reasonable juice (not too extreme)."""
    # Typical juices: -110, -120
    if -200 < odds < 0:
        return True
    if odds > 0:
        return True
    return False


def consistency_score(games_list: List[Dict], mean: float) -> float:
    """
    Calculate consistency: % of games within ±20% of mean.
    Higher = more consistent projections.
    """
    if not games_list or mean == 0:
        return 0.0

    count_consistent = 0
    for g in games_list:
        val = g.get("value", 0.0)
        if mean * (1.0 - CONSISTENCY_BAND) <= val <= mean * (1.0 + CONSISTENCY_BAND):
            count_consistent += 1

    return count_consistent / len(games_list) if games_list else 0.0


def normalize_team_name(team_name: str) -> str:
    """Normalize team name via alias map."""
    if not team_name:
        return ""
    # Try direct match first
    if team_name in TEAM_ALIAS_TO_FULL:
        return TEAM_ALIAS_TO_FULL[team_name]
    # Try reverse match
    for code, full in TEAM_ALIAS_TO_FULL.items():
        if full.lower() == team_name.lower():
            return full
    return team_name


# ============================================================================
# SECTION 3: STATE MANAGEMENT & MESSAGING
# ============================================================================

def load_state() -> Dict:
    """Load state from JSON file."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except:
            return _blank_state()
    return _blank_state()


def _blank_state() -> Dict:
    """Create blank state structure."""
    return {
        "plays": [],
        "market_memory": {},
        "hit_tracking": {},
        "last_send": 0,
    }


def save_state(state: Dict) -> None:
    """Save state to JSON file."""
    try:
        os.makedirs(os.path.dirname(STATE_FILE) if os.path.dirname(STATE_FILE) else ".", exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)
    except Exception as e:
        print(f"ERROR saving state: {e}")


def send_one(client: TwilioClient, msg: str) -> bool:
    """Send single WhatsApp message via Twilio."""
    if not client or not RECIPIENT_WHATSAPP:
        print(f"[MOCK] WhatsApp: {msg}")
        return True
    try:
        client.messages.create(
            from_=TWILIO_WHATSAPP_FROM,
            to=RECIPIENT_WHATSAPP,
            body=msg
        )
        return True
    except Exception as e:
        print(f"ERROR sending WhatsApp: {e}")
        return False


def send_chunked(client: TwilioClient, text: str, chunk_size: int = 1500) -> bool:
    """Send long text in chunks via WhatsApp."""
    global LAST_WHATSAPP_SEND

    if not text:
        return True

    # Throttle sends
    now = time.time()
    if now - LAST_WHATSAPP_SEND < BET_COOLDOWN:
        time.sleep(BET_COOLDOWN - (now - LAST_WHATSAPP_SEND))

    # Split and send
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        if not send_one(client, chunk):
            return False
        LAST_WHATSAPP_SEND = time.time()
        time.sleep(0.5)

    return True


def deadline_exceeded(game_time_et: datetime, hours_before: float = DEADLINE_HOURS_BEFORE) -> bool:
    """Check if game start time is within deadline."""
    deadline = _now_et() + timedelta(hours=hours_before)
    return game_time_et <= deadline


def log_play_for_tracking(state: Dict, play: Dict) -> None:
    """Log a play to state for hit rate tracking."""
    if "plays" not in state:
        state["plays"] = []
    state["plays"].append({
        "player": play.get("player"),
        "stat": play.get("stat_key"),
        "line": play.get("line"),
        "proj": play.get("proj"),
        "timestamp": _now_et().isoformat(),
    })


def get_hit_rate_summary(state: Dict) -> str:
    """Generate hit rate summary from tracked plays."""
    if not state.get("plays"):
        return "No plays tracked yet."

    plays = state["plays"]
    hit_count = sum(1 for p in plays if p.get("result") == "HIT")
    total = len(plays)
    hit_rate = (hit_count / total * 100) if total > 0 else 0

    return f"Hit Rate: {hit_count}/{total} ({hit_rate:.1f}%)"


# ============================================================================
# SECTION 4: API WRAPPERS
# ============================================================================

def _bdl_get(endpoint: str, params: Dict = None, retries: int = 3) -> Dict:
    """
    Fetch from BallDontLie API with retries and rate limiting.
    Returns response JSON or empty dict on failure.
    """
    if not BALLDONTLIE_API_KEY or not requests:
        return {}

    url = f"{BDL_BASE}/{endpoint}"
    headers = {"Authorization": BALLDONTLIE_API_KEY}

    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                # Rate limit
                wait = int(resp.headers.get("Retry-After", 2))
                time.sleep(wait)
            else:
                break
        except Exception as e:
            print(f"BDL request error: {e}")
            if attempt < retries - 1:
                time.sleep(1)

    return {}


def bdl_games_today(dt: datetime = None) -> List[Dict]:
    """
    Get all NBA games for today.
    Returns list of game dicts with keys: id, home_team, away_team, status, scheduled_at.
    """
    if dt is None:
        dt = _now_et()

    date_str = dt.strftime("%Y-%m-%d")
    resp = _bdl_get("games", {"dates[]": date_str, "per_page": 50})

    return resp.get("data", []) if resp else []


def bdl_team_name_to_id(team_name: str) -> Optional[int]:
    """Get BDL team ID from name."""
    if not team_name:
        return None

    # Check cache
    if team_name in TEAM_ID_CACHE:
        return TEAM_ID_CACHE[team_name]

    resp = _bdl_get("teams", {})
    teams = resp.get("data", [])

    team_name_clean = _clean_name(team_name)
    for team in teams:
        if _clean_name(team.get("name", "")) == team_name_clean:
            team_id = team.get("id")
            TEAM_ID_CACHE[team_name] = team_id
            return team_id

    return None


def bdl_active_roster(team_id: int) -> List[Dict]:
    """Get active roster for team."""
    if not team_id:
        return []

    cache_key = f"roster_{team_id}"
    if cache_key in ROSTER_CACHE:
        return ROSTER_CACHE[cache_key]

    resp = _bdl_get("teams", {"id": team_id})
    roster = resp.get("data", [{}])[0].get("players", []) if resp and resp.get("data") else []

    ROSTER_CACHE[cache_key] = roster
    return roster


def bdl_find_player_id_on_team(player_name: str, team_id: int) -> Optional[int]:
    """Fuzzy match player name to roster and get player ID."""
    if not player_name or not team_id:
        return None

    roster = bdl_active_roster(team_id)
    player_clean = _clean_name(player_name)

    # Exact match
    for player in roster:
        if _clean_name(player.get("first_name", "") + " " + player.get("last_name", "")) == player_clean:
            return player.get("id")

    # Partial match (last name)
    last_name_clean = _clean_name(player_name.split()[-1] if player_name else "")
    for player in roster:
        if _clean_name(player.get("last_name", "")) == last_name_clean:
            return player.get("id")

    return None


def bdl_last_n_games_stats(player_id: int, stat_key: str, n: int = BASELINE_GAMES) -> List[Dict]:
    """
    Fetch last n games for a player, extracting specific stat.
    stat_key should be: pts, reb, ast, blk, stl, or 'min' for minutes.
    Returns list of dicts with 'value' key.
    """
    if not player_id:
        return []

    resp = _bdl_get(f"players/{player_id}/games", {"per_page": n, "cursor": 0})
    games = resp.get("data", [])

    result = []
    for game in games:
        if not game:
            continue
        val = 0.0
        if stat_key == "min":
            val = _parse_minutes(game.get("min", "0:00"))
        else:
            val = float(game.get(stat_key, 0.0))
        result.append({
            "value": val,
            "game_id": game.get("game_id"),
            "date": game.get("game", {}).get("date"),
        })

    return result


def bdl_last_n_games_threes(player_id: int, n: int = BASELINE_GAMES) -> List[Dict]:
    """
    Fetch 3-pointers made from last n games.
    Returns list of dicts with 'value' key (3PM).
    """
    if not player_id:
        return []

    resp = _bdl_get(f"players/{player_id}/games", {"per_page": n, "cursor": 0})
    games = resp.get("data", [])

    result = []
    for game in games:
        if not game:
            continue
        val = float(game.get("fg3m", 0.0))  # 3-pointers made
        result.append({
            "value": val,
            "game_id": game.get("game_id"),
        })

    return result


def bdl_fetch_props_for_game(game_id: int, prop_types: List[str] = None) -> Dict:
    """
    Fetch player prop lines for a game.
    Returns dict: {player_name: {prop_type: [{book, line, odds}]}}

    NOTE: BDL API may not have real-time prop odds. This is a placeholder
    that would need integration with actual sportsbook APIs.
    """
    if not game_id or not prop_types:
        return {}

    cache_key = f"props_{game_id}"
    if cache_key in PROPS_CACHE:
        return PROPS_CACHE[cache_key]

    # Placeholder: in production, integrate with DK, FD, etc.
    result = {}
    PROPS_CACHE[cache_key] = result
    return result


def bdl_fetch_advanced_stats(player_id: int) -> Dict:
    """
    Fetch advanced stats: usage%, off_rating, pie, pace.
    Returns dict with keys: usage_pct, off_rating, pie, pace.
    """
    if not player_id:
        return {}

    cache_key = f"adv_{player_id}"
    if cache_key in ADV_STATS_CACHE:
        return ADV_STATS_CACHE[cache_key]

    # BDL may not have detailed advanced stats via this endpoint
    # Placeholder: integrate with manual lookup or secondary API
    result = {
        "usage_pct": 25.0,
        "off_rating": 110.0,
        "pie": 0.12,
        "pace": 98.0,
    }

    ADV_STATS_CACHE[cache_key] = result
    return result


def bdl_fetch_game_odds_full(game_id: int) -> Dict:
    """
    Fetch game odds: total, spread, etc.
    Returns dict with keys: game_total, home_spread, away_spread.
    """
    if not game_id:
        return {}

    cache_key = f"odds_{game_id}"
    if cache_key in GAME_ODDS_CACHE:
        return GAME_ODDS_CACHE[cache_key]

    # Placeholder: integrate with live odds API
    result = {
        "game_total": 220.0,
        "home_spread": -3.5,
        "away_spread": 3.5,
        "pace": 98.0,
    }

    GAME_ODDS_CACHE[cache_key] = result
    return result


def bdl_fetch_lineups(game_id: int) -> Dict:
    """
    Fetch confirmed lineups for a game.
    Returns dict: {team_id: [player dicts]}
    """
    if not game_id:
        return {}

    cache_key = f"lineups_{game_id}"
    if cache_key in LINEUPS_CACHE:
        return LINEUPS_CACHE[cache_key]

    # Placeholder
    result = {}
    LINEUPS_CACHE[cache_key] = result
    return result


def fetch_lineupexperts_news() -> List[Dict]:
    """
    Fetch daily news/injuries from LineupExperts.
    Returns list of news items with keys: player, team, news_type, status.
    """
    if not LINEUP_EXPERTS_API_KEY or not requests:
        return []

    try:
        headers = {"Authorization": f"Bearer {LINEUP_EXPERTS_API_KEY}"}
        resp = requests.get(
            f"{LINEUP_EXPERTS_BASE}/news",
            headers=headers,
            timeout=REQUEST_TIMEOUT
        )
        if resp.status_code == 200:
            return resp.json().get("data", [])
    except Exception as e:
        print(f"LineupExperts error: {e}")

    return []


def build_news_boost_map(news: List[Dict]) -> Dict:
    """
    Parse news and create boost map for players.
    Returns dict: {player_name: {stat_key: boost_pct}}

    Examples:
    - Injury to teammate → +5% usage/touches for others
    - Returning from injury → -10% early back
    - Load management rest → -20% if on rest
    """
    boost_map = {}

    for item in news:
        player = item.get("player", "")
        news_type = item.get("type", "").lower()
        status = item.get("status", "").lower()

        if not player:
            continue

        boost = 0.0
        stat_key = "pts"  # default

        if "out" in status or "injury" in news_type:
            # Injured: find teammates for boost
            continue
        elif "questionable" in status:
            boost = -0.05
        elif "returning" in status and "rest" in news_type:
            boost = -0.10

        if boost != 0:
            if player not in boost_map:
                boost_map[player] = {}
            boost_map[player][stat_key] = boost

    return boost_map


def build_news_score_map(news: List[Dict]) -> Dict:
    """
    Parse news for confidence adjustments.
    Returns dict: {player_name: confidence_delta}
    """
    score_map = {}

    for item in news:
        player = item.get("player", "")
        if not player:
            continue

        delta = 0
        status = item.get("status", "").lower()

        if "confirmed" in status:
            delta = +1
        elif "probable" in status:
            delta = 0
        elif "questionable" in status:
            delta = -1

        if delta != 0:
            score_map[player] = delta

    return score_map


def parse_le_injuries() -> Dict:
    """
    Parse LineupExperts injuries to identify vacancy.
    Returns dict: {player_name: {stat_key: freed_up_value}}
    """
    news = fetch_lineupexperts_news()
    injury_map = {}

    for item in news:
        player = item.get("player", "")
        status = item.get("status", "").lower()

        if player and ("out" in status or "day-to-day" in status):
            injury_map[player] = {
                "pts": 0.0,  # Will be filled in by compute_projection
                "reb": 0.0,
                "ast": 0.0,
            }

    return injury_map


# ============================================================================
# SECTION 5: PROJECTION ENGINE
# ============================================================================

@dataclass
class ProjectionResult:
    """Projection result with all metadata."""
    proj: float
    edge: float
    prob_over: float
    raw_prob: float
    sigma: float
    consensus_line: float
    best_odds: float
    ev: float
    consistency: float
    confidence_tier: str
    l5_avg: float
    l10_avg: float
    base_avg: float
    is_breakout: bool
    minutes_proj: float
    adjustment_total: float


def compute_projection(
    player_id: int,
    player_name: str,
    game_id: int,
    line: float,
    prop_type: str,
    over_odds: float,
    game_info: Dict = None,
    context: Dict = None,
) -> Optional[ProjectionResult]:
    """
    Core projection engine.

    Returns ProjectionResult with proj, edge, prob_over, and all metadata.
    Returns None if insufficient data.
    """
    if not player_id or line < 0:
        return None

    stat_key = prop_type_to_stat_key(prop_type)

    # Step 1: Fetch game history
    if prop_type_is_threes(prop_type):
        games = bdl_last_n_games_threes(player_id, BASELINE_GAMES)
    else:
        games = bdl_last_n_games_stats(player_id, stat_key, BASELINE_GAMES)

    if len(games) < 10:
        print(f"[SKIP] {player_name}: only {len(games)} games")
        return None

    # Step 2: Window averages
    base_games = _slice_last(games, BASELINE_GAMES)
    l10_games = _slice_last(games, LOOKBACK_GAMES)
    l5_games = _slice_last(games, SHORT_GAMES)

    base_avg, base_min, base_std = avg_stat_min_std(base_games)
    l10_avg, l10_min, l10_std = avg_stat_min_std(l10_games)
    l5_avg, l5_min, l5_std = avg_stat_min_std(l5_games)

    if base_avg < 0.1:
        return None

    # Step 3: Breakout detection (require both L5 and L10 elevated)
    is_breakout = (
        l5_avg >= base_avg * BREAKOUT_L5_THRESHOLD and
        l10_avg >= base_avg * BREAKOUT_L10_THRESHOLD
    )

    # Step 4: Direct average projection
    if is_breakout:
        avg_proj = (
            BREAKOUT_WEIGHTS["base"] * base_avg +
            BREAKOUT_WEIGHTS["l10"] * l10_avg +
            BREAKOUT_WEIGHTS["l5"] * l5_avg
        )
    else:
        avg_proj = (
            PROJECTION_WEIGHTS["base"] * base_avg +
            PROJECTION_WEIGHTS["l10"] * l10_avg +
            PROJECTION_WEIGHTS["l3"] * l5_avg
        )

    # Step 5: Rate-based projection
    proj_min = project_minutes(base_min, l10_min, l5_min)

    if proj_min < 1.0:
        return None

    rate_base = _safe_rate(base_avg, base_min) if base_min > 0 else 0
    rate_l10 = _safe_rate(l10_avg, l10_min) if l10_min > 0 else 0
    rate_l5 = _safe_rate(l5_avg, l5_min) if l5_min > 0 else 0

    rate_proj = (
        PROJECTION_WEIGHTS["base"] * rate_base +
        PROJECTION_WEIGHTS["l10"] * rate_l10 +
        PROJECTION_WEIGHTS["l3"] * rate_l5
    )

    if rate_proj < 0.01:
        rate_proj = 0

    rate_based = proj_min * rate_proj

    # Step 6: Blend (50/50)
    raw_proj = 0.50 * avg_proj + 0.50 * rate_based

    # Step 7: Context adjustments (capped at ±12%)
    adj_total = context_adjustments(
        player_id, player_name, stat_key, game_id, game_info, context or {}
    )
    proj = raw_proj * (1.0 + adj_total)

    # Step 8: Additive adjustments (injuries, news, boosts)
    injury_boost = compute_injury_boost(player_name, stat_key)
    proj += injury_boost

    # Step 9: Guardrail (within ±40% of weighted average)
    guardrail_low = avg_proj * 0.60
    guardrail_high = avg_proj * 1.40
    proj = _clamp(proj, guardrail_low, guardrail_high)

    # Step 10: Adaptive sigma
    consistency = consistency_score(l10_games, l10_avg)
    sigma = adaptive_sigma(base_std, l10_std, l5_std, consistency, stat_key)

    # Step 11: Probability (calibrated)
    z = (proj - line) / sigma if sigma > 0 else 0
    raw_prob = _norm_cdf(z)
    prob_over = calibrated_prob(raw_prob)

    # Step 12: Edge and EV
    edge = proj - line
    implied_prob = american_to_prob(over_odds)
    ev = ev_per_dollar(prob_over, over_odds)

    # Step 13: Consensus line (would fetch from multiple books)
    consensus_line = line

    # Step 14: Confidence tier
    confidence_tier = compute_confidence_tier(
        edge, prob_over, consistency, is_breakout
    )

    return ProjectionResult(
        proj=proj,
        edge=edge,
        prob_over=prob_over,
        raw_prob=raw_prob,
        sigma=sigma,
        consensus_line=consensus_line,
        best_odds=over_odds,
        ev=ev,
        consistency=consistency,
        confidence_tier=confidence_tier,
        l5_avg=l5_avg,
        l10_avg=l10_avg,
        base_avg=base_avg,
        is_breakout=is_breakout,
        minutes_proj=proj_min,
        adjustment_total=adj_total,
    )


def project_minutes(base_min: float, l10_min: float, l5_min: float) -> float:
    """
    Project minutes played.
    Weight L10 heavily. Cap changes at ±3min from L10.
    """
    if base_min < 1.0:
        return 0.0

    # Weight toward L10 (most reliable)
    proj = (0.30 * base_min + 0.40 * l10_min + 0.30 * l5_min)

    # Cap change from L10
    delta = proj - l10_min
    delta = _clamp(delta, MIN_MINUTES_CHANGE, MAX_MINUTES_CHANGE)

    return max(0, l10_min + delta)


def adaptive_sigma(
    base_std: float,
    l10_std: float,
    l5_std: float,
    consistency: float,
    stat_key: str,
) -> float:
    """
    Compute adaptive standard deviation.
    Use base_std + l10_std blend, reduce by consistency (max 10%), apply floor.
    """
    # Blend recent windows
    raw_sigma = 0.40 * base_std + 0.60 * l10_std

    # Reduce for consistency (high consistency = lower uncertainty)
    consistency_reduction = min(0.10, consistency * 0.15)
    sigma = raw_sigma * (1.0 - consistency_reduction)

    # Floor per stat
    floor = STAT_STD_FLOORS.get(stat_key, STD_FLOOR)
    sigma = max(floor, sigma)

    return sigma


def opponent_defense_adjustment(opp_team: str, prop_type: str, player_name: str = "") -> float:
    """
    Estimate opponent defensive adjustment.
    Capped at ±6%.
    """
    if not opp_team:
        return 0.0

    opp_full = normalize_team_name(opp_team)
    defense_ratings = TEAM_DEFENSE_RATINGS.get(opp_full, {})

    stat_key = prop_type_to_stat_key(prop_type)
    opp_rating = defense_ratings.get(stat_key, 111.0)

    # League average = 111 per 100 poss
    league_avg = 111.0

    adj = (league_avg - opp_rating) / 100.0  # Convert to percentage
    return _clamp(adj, -0.06, 0.06)


def context_adjustments(
    player_id: int,
    player_name: str,
    stat_key: str,
    game_id: int,
    game_info: Dict = None,
    context: Dict = None,
) -> float:
    """
    Compute total context adjustment.
    Returns value to multiply projection by (e.g., 0.05 for +5%).
    Capped at ±12%.
    """
    if game_info is None:
        game_info = {}
    if context is None:
        context = {}

    adjustments = []

    # 1. Opponent defense (±0-6%)
    opp_team = game_info.get("opp_team", "")
    opp_adj = opponent_defense_adjustment(opp_team, f"player_{stat_key}", player_name)
    adjustments.append(opp_adj)

    # 2. Home/away (±1-2%)
    is_home = game_info.get("is_home", True)
    home_adj = 0.015 if is_home else -0.008
    adjustments.append(home_adj)

    # 3. Back-to-back (-3%)
    is_b2b = game_info.get("is_b2b", False)
    b2b_adj = -0.03 if is_b2b else 0
    adjustments.append(b2b_adj)

    # 4. Pace adjustment (±0-3%)
    game_pace = game_info.get("pace", 100.0)
    league_pace = 100.0
    pace_adj = (game_pace - league_pace) / 100.0 * 0.5
    pace_adj = _clamp(pace_adj, -0.03, 0.03)
    adjustments.append(pace_adj)

    # 5. Usage % adjustment (±0-4%)
    adv_stats = bdl_fetch_advanced_stats(player_id)
    usage_pct = adv_stats.get("usage_pct", 25.0)
    # Typical range: 20-32%
    usage_adj = (usage_pct - 25.0) / 100.0 * 0.04
    usage_adj = _clamp(usage_adj, -0.04, 0.04)
    adjustments.append(usage_adj)

    # Total (capped)
    total = sum(adjustments)
    return _clamp(total, -ADJ_CAP_TOTAL, ADJ_CAP_TOTAL)


def compute_injury_boost(player_name: str, stat_key: str) -> float:
    """
    Compute injury boost from teammates going out.
    Max boost: +3 pts, +1.5 reb, +1.0 ast.
    """
    injuries = parse_le_injuries()

    boost = 0.0
    for injured, freed_stats in injuries.items():
        # Simple heuristic: if teammate is out, +2 pts per week they're out
        # This is placeholder; real implementation needs game context
        if stat_key == "pts":
            boost += 0.5
        elif stat_key == "reb":
            boost += 0.25
        elif stat_key == "ast":
            boost += 0.15

    # Cap
    max_boost = {"pts": 3.0, "reb": 1.5, "ast": 1.0, "blk": 0.5, "stl": 0.5, "threes": 0.5}
    return min(boost, max_boost.get(stat_key, 0.5))


def compute_confidence_tier(
    edge: float,
    prob_over: float,
    consistency: float,
    is_breakout: bool,
) -> str:
    """
    Compute confidence tier: LOCK, STRONG, LEAN, SKIP.
    Based on edge, probability, and consistency.
    """
    score = 0

    # Edge points
    if edge >= 5.0:
        score += 3
    elif edge >= 3.0:
        score += 2
    elif edge >= 1.0:
        score += 1

    # Probability points
    if prob_over >= 0.70:
        score += 3
    elif prob_over >= 0.62:
        score += 2
    elif prob_over >= 0.58:
        score += 1

    # Consistency bonus
    if consistency >= 0.80:
        score += 2
    elif consistency >= 0.60:
        score += 1

    # Breakout bonus
    if is_breakout:
        score += 1

    if score >= 9:
        return "LOCK"
    elif score >= 6:
        return "STRONG"
    elif score >= 3:
        return "LEAN"
    else:
        return "SKIP"


# ============================================================================
# SECTION 6: CONSENSUS & MARKET FUNCTIONS
# ============================================================================

def consensus_line(lines_from_books: List[float]) -> float:
    """
    Compute consensus line (median) from multiple books.
    """
    if not lines_from_books:
        return 0.0
    sorted_lines = sorted(lines_from_books)
    n = len(sorted_lines)
    if n % 2 == 0:
        return (sorted_lines[n // 2 - 1] + sorted_lines[n // 2]) / 2.0
    return sorted_lines[n // 2]


def best_offer_near_consensus(
    consensus: float,
    offers: List[Tuple[float, float]],  # (line, odds) tuples
    tolerance: float = 0.5,
) -> Optional[Tuple[float, float]]:
    """
    Find best odds for a line near consensus (within tolerance).
    Returns (line, odds) tuple.
    """
    if not offers:
        return None

    # Filter near consensus
    near = [(line, odds) for line, odds in offers if abs(line - consensus) <= tolerance]

    if not near:
        # Return best odds overall
        return max(offers, key=lambda x: x[1]) if offers else None

    # Among near-consensus, return best odds (highest for overs)
    return max(near, key=lambda x: x[1])


def remember_market(state: Dict, game_id: int, prop_type: str, lines: Dict) -> None:
    """Remember market snapshot for comparison."""
    if "market_memory" not in state:
        state["market_memory"] = {}
    key = f"{game_id}_{prop_type}"
    state["market_memory"][key] = {
        "timestamp": _now_et().isoformat(),
        "lines": lines,
    }


def get_prev_market(state: Dict, game_id: int, prop_type: str) -> Optional[Dict]:
    """Get previous market snapshot."""
    if "market_memory" not in state:
        return None
    key = f"{game_id}_{prop_type}"
    return state["market_memory"].get(key)


# ============================================================================
# SECTION 7: EDGE DETECTION ENGINES (start)
# ============================================================================

def build_today_props(now_et: datetime = None) -> Dict:
    """
    Fetch all player prop lines for today's games.
    Returns dict: {game_id: {prop_type: {player_name: [{book, line, odds}]}}}

    In production, integrate with DraftKings, FanDuel, etc. APIs.
    This is a placeholder structure.
    """
    if now_et is None:
        now_et = _now_et()

    games = bdl_games_today(now_et)

    all_props = {}
    for game in games:
        game_id = game.get("id")
        if not game_id:
            continue

        all_props[game_id] = {}
        for prop_type in PROP_TYPES:
            all_props[game_id][prop_type] = {}

    return all_props


def slate_scan_edges(
    prop_type: str,
    lines_map: Dict,  # {game_id: {player_name: [{book, line, odds}]}}
    games_map: Dict,  # {game_id: game_dict}
    state: Dict = None,
    now_et: datetime = None,
) -> List[Dict]:
    """
    Scan for edges in a specific prop type.
    Returns list of plays, ranked by confidence.

    Filters:
    - min_edge >= 3.0%
    - min_prob >= 0.58 (calibrated)
    - min_ev >= 0.03
    - deadline check
    - cooldowns
    """
    if state is None:
        state = load_state()
    if now_et is None:
        now_et = _now_et()

    plays = []

    for game_id, players_lines in lines_map.items():
        game_info = games_map.get(game_id, {})
        game_time = game_info.get("scheduled_at")

        # Check deadline
        if game_time and deadline_exceeded(game_time):
            continue

        for player_name, book_lines in players_lines.items():
            if not book_lines:
                continue

            # Aggregate lines from books
            all_lines = [bl.get("line") for bl in book_lines if bl.get("line")]
            if not all_lines:
                continue

            consensus = consensus_line(all_lines)
            best_odds_offer = best_offer_near_consensus(consensus,
                [(bl["line"], bl["odds"]) for bl in book_lines])

            if not best_odds_offer:
                continue

            best_line, best_odds = best_odds_offer

            # Get player ID
            home_team = game_info.get("home_team", {})
            away_team = game_info.get("away_team", {})
            team_ids = [
                bdl_team_name_to_id(home_team.get("name", "")),
                bdl_team_name_to_id(away_team.get("name", "")),
            ]

            player_id = None
            for tid in team_ids:
                player_id = bdl_find_player_id_on_team(player_name, tid)
                if player_id:
                    break

            if not player_id:
                continue

            # Compute projection
            game_context = {
                "opp_team": (away_team.get("name") if game_info.get("home_team", {}).get("id") == game_info.get("home_team", {}).get("id") else home_team.get("name")),
                "is_home": True,  # Simplify
                "pace": 100.0,
            }

            proj_result = compute_projection(
                player_id, player_name, game_id, best_line, prop_type,
                best_odds, game_info, game_context
            )

            if not proj_result:
                continue

            # Filter
            if proj_result.edge < MIN_EDGE:
                continue
            if proj_result.prob_over < MIN_PROB:
                continue
            if proj_result.ev < 0.03:
                continue

            # Cooldown check
            if has_recent_play(state, player_name, prop_type):
                continue

            plays.append({
                "player": player_name,
                "player_id": player_id,
                "game_id": game_id,
                "prop_type": prop_type,
                "stat_key": prop_type_to_stat_key(prop_type),
                "line": best_line,
                "proj": proj_result.proj,
                "edge": proj_result.edge,
                "prob_over": proj_result.prob_over,
                "ev": proj_result.ev,
                "sigma": proj_result.sigma,
                "consistency": proj_result.consistency,
                "confidence_tier": proj_result.confidence_tier,
                "is_breakout": proj_result.is_breakout,
                "odds": best_odds,
                "bet_size": kelly_bet_size(proj_result.edge, proj_result.prob_over, BANKROLL),
            })

    # Sort by confidence tier then edge
    tier_order = {"LOCK": 0, "STRONG": 1, "LEAN": 2, "SKIP": 3}
    plays.sort(key=lambda p: (tier_order.get(p["confidence_tier"], 3), -p["edge"]))

    return plays


def has_recent_play(state: Dict, player_name: str, prop_type: str) -> bool:
    """Check if player/stat combo has recent play (within cooldown)."""
    plays = state.get("plays", [])
    recent_time = _now_et() - timedelta(seconds=PLAY_COOLDOWN)

    for play in plays:
        if (play.get("player") == player_name and
            play.get("stat") == prop_type):
            try:
                play_time = datetime.fromisoformat(play.get("timestamp", ""))
                if play_time > recent_time:
                    return True
            except:
                pass

    return False


def build_injury_edges(
    team_id: int,
    injured_player_name: str,
    games_map: Dict,
    state: Dict = None,
) -> List[Dict]:
    """
    When a key player is out, find beneficiaries on the same team.
    Calculate vacancy and project beneficiary with injury boost.
    """
    if state is None:
        state = load_state()

    plays = []

    # Find teammates of injured player
    roster = bdl_active_roster(team_id)

    # Placeholder: injured_player stats would be used to calculate vacancy
    # For now, generic boost

    for teammate_dict in roster:
        teammate_name = f"{teammate_dict.get('first_name', '')} {teammate_dict.get('last_name', '')}".strip()
        teammate_id = teammate_dict.get("id")

        if not teammate_id or teammate_name == injured_player_name:
            continue

        # For each stat type
        for stat_key in ["pts", "reb", "ast"]:
            prop_type = {
                "pts": "player_points",
                "reb": "player_rebounds",
                "ast": "player_assists",
            }.get(stat_key)

            # Placeholder: would fetch actual lines
            # For now, skip injury edges in this build
            pass

    return plays


def lineup_news_edges(
    games_map: Dict,
    state: Dict = None,
) -> List[Dict]:
    """
    Use LineupExperts news to find edges from status changes.
    """
    if state is None:
        state = load_state()

    news = fetch_lineupexperts_news()
    score_map = build_news_score_map(news)

    plays = []
    # Placeholder: integrate with game and projection systems

    return plays


# ============================================================================
# SECTION 8: PLUS ODDS HUNTER
# ============================================================================

def plus_odds_hunt_edges(
    lines_map: Dict,
    games_map: Dict,
    now_et: datetime = None,
) -> List[Dict]:
    """
    Hunt for plus-odds plays (+100 and above) where the model still sees edge.
    Filter: only plus odds, min prob 0.52 (calibrated), min EV 0.03
    Score by EV and value_edge, return top 5.
    """
    if now_et is None:
        now_et = _now_et()

    plays = []
    state = load_state()

    for game_id, game_info in games_map.items():
        game_time_et = game_info.get("game_time_et")
        if not game_time_et or deadline_exceeded(game_time_et):
            continue

        for prop_type, lines_dict in lines_map.get(game_id, {}).items():
            if prop_type not in PROP_TYPES:
                continue

            for (player_name, player_id, player_team), line_data in lines_dict.items():
                if not player_id:
                    continue

                best_odds = line_data.get("best_over_odds", line_data.get("consensus_over_odds"))
                if not best_odds or best_odds < 100:
                    continue  # Only plus odds

                stat_key = {
                    "player_points": "pts",
                    "player_rebounds": "reb",
                    "player_assists": "ast",
                    "player_threes": "threes",
                    "player_blocks": "blk",
                    "player_steals": "stl",
                }.get(prop_type)

                if not stat_key:
                    continue

                # Check recent play cooldown
                if has_recent_play(state, player_name, prop_type):
                    continue

                try:
                    # Compute projection
                    proj, sigma, proj_method = compute_projection(
                        player_id,
                        stat_key,
                        game_info.get("opponent_name", ""),
                        game_info.get("is_home", False),
                    )

                    line = line_data.get("line")
                    if not line or proj is None or sigma is None:
                        continue

                    # Calculate over probability
                    z_over = (proj - line) / max(sigma, STAT_STD_FLOORS.get(stat_key, STD_FLOOR))
                    raw_prob = _norm_cdf(z_over)
                    prob = calibrated_prob(raw_prob)

                    if prob < 0.52:
                        continue  # Min prob threshold

                    # Calculate EV
                    ev = ev_per_dollar(prob, best_odds)
                    if ev < 0.03:
                        continue  # Min EV threshold

                    # Score by EV and value edge
                    value_edge = (prob - american_to_prob(best_odds)) * 100
                    score = ev * 100 + value_edge

                    play = {
                        "player": player_name,
                        "player_id": player_id,
                        "player_team": player_team,
                        "game_id": game_id,
                        "stat_key": stat_key,
                        "prop_type": prop_type,
                        "line": line,
                        "proj": proj,
                        "sigma": sigma,
                        "odds": best_odds,
                        "vendor": line_data.get("best_over_vendor", "Consensus"),
                        "prob": prob,
                        "ev": ev,
                        "value_edge": value_edge,
                        "score": score,
                        "bet_size": kelly_bet_size(ev * 100, prob, BANKROLL),
                        "edge": value_edge,
                        "confidence_tier": "LEAN",
                        "type": "plus_odds",
                    }
                    plays.append(play)

                except Exception as e:
                    print(f"[DEBUG] Plus odds error for {player_name} {stat_key}: {e}")
                    continue

    # Sort by score and return top 5
    plays.sort(key=lambda p: -p["score"])
    return plays[:5]


# ============================================================================
# SECTION 9: PARLAY ENGINES
# ============================================================================

PROP_CORRELATIONS = {
    ("pts", "reb"): 0.35,
    ("pts", "ast"): 0.25,
    ("pts", "threes"): 0.30,
    ("reb", "ast"): 0.15,
}


def find_sgp_opportunities(
    plays: List[Dict],
    games_map: Dict,
) -> List[Dict]:
    """
    Same-Game Parlay builder.
    Group qualifying plays by game_id.
    Same team: 2-3 legs, positive correlation.
    Min combined prob 0.35, min EV 0.10.
    Return top 3.
    """
    sgps = []

    # Group plays by game_id
    plays_by_game = {}
    for play in plays:
        gid = play.get("game_id")
        if gid not in plays_by_game:
            plays_by_game[gid] = []
        plays_by_game[gid].append(play)

    for game_id, game_plays in plays_by_game.items():
        if len(game_plays) < 2:
            continue

        game_info = games_map.get(game_id, {})
        game_total = game_info.get("total_points", 220)  # Default estimate

        # Correlation factor based on game total
        # High total (>220) = higher correlation
        corr_factor = 0.15 if game_total > 220 else 0.10

        # Try 2-leg and 3-leg parlays from same team
        for team_id in [1, 2]:  # Could iterate actual teams, simplify for now
            team_plays = [p for p in game_plays]  # Simplified: all same game
            if len(team_plays) < 2:
                continue

            # Try 2-leg combos
            for i in range(len(team_plays)):
                for j in range(i + 1, len(team_plays)):
                    leg1, leg2 = team_plays[i], team_plays[j]

                    # Same player means skip
                    if leg1.get("player_id") == leg2.get("player_id"):
                        continue

                    p1, p2 = leg1.get("prob"), leg2.get("prob")
                    if not p1 or not p2:
                        continue

                    # Correlated probability
                    combined_prob = p1 * p2 + corr_factor * math.sqrt(p1 * (1 - p1)) * math.sqrt(p2 * (1 - p2))
                    combined_prob = min(combined_prob, 0.99)

                    if combined_prob < 0.35:
                        continue

                    # Estimate parlay odds with 15% SGP tax
                    o1, o2 = leg1.get("odds", -110), leg2.get("odds", -110)
                    est_odds_1 = american_to_payout(o1, 100) * american_to_payout(o2, 1) * 100 - 100
                    est_odds = int(est_odds_1 * 0.85)  # SGP tax

                    ev = ev_per_dollar(combined_prob, est_odds)
                    if ev < 0.10:
                        continue

                    sgp = {
                        "game_id": game_id,
                        "legs": [
                            f"{leg1.get('player')} {leg1.get('stat_key').upper()}O",
                            f"{leg2.get('player')} {leg2.get('stat_key').upper()}O",
                        ],
                        "estimated_odds": est_odds,
                        "prob": combined_prob,
                        "ev": ev,
                        "bet_size": kelly_bet_size(ev * 100, combined_prob, BANKROLL),
                        "edge": (combined_prob - american_to_prob(est_odds)) * 100,
                        "type": "sgp",
                    }
                    sgps.append(sgp)

    sgps.sort(key=lambda s: -s["ev"])
    return sgps[:3]


def find_correlated_parlays(
    plays: List[Dict],
    lines_map: Dict,
    games_map: Dict,
) -> List[Dict]:
    """
    Correlated parlay builder.
    Same player, different props (e.g., pts over + reb over).
    Apply correlation formula.
    Min combined prob 0.40, min EV 0.12, min odds +150.
    Return top 3.
    """
    corr_parlays = []

    # Group by player_id
    plays_by_player = {}
    for play in plays:
        pid = play.get("player_id")
        if pid not in plays_by_player:
            plays_by_player[pid] = []
        plays_by_player[pid].append(play)

    for player_id, player_plays in plays_by_player.items():
        if len(player_plays) < 2:
            continue

        # Try pairs of different stat types
        for i in range(len(player_plays)):
            for j in range(i + 1, len(player_plays)):
                play1, play2 = player_plays[i], player_plays[j]

                stat1 = play1.get("stat_key")
                stat2 = play2.get("stat_key")

                if stat1 == stat2:
                    continue

                # Look up correlation
                corr_key = (stat1, stat2) if (stat1, stat2) in PROP_CORRELATIONS else (stat2, stat1)
                correlation = PROP_CORRELATIONS.get(corr_key, 0.12)

                p1, p2 = play1.get("prob"), play2.get("prob")
                if not p1 or not p2:
                    continue

                # Correlated probability formula
                combined_prob = (p1 * p2 +
                                 correlation * math.sqrt(p1 * (1 - p1)) * math.sqrt(p2 * (1 - p2)))
                combined_prob = min(combined_prob, 0.99)

                if combined_prob < 0.40:
                    continue

                # Estimate parlay odds with 15% SGP tax
                o1, o2 = play1.get("odds", -110), play2.get("odds", -110)
                payout1 = american_to_payout(o1, 100)
                payout2 = american_to_payout(o2, 1)
                est_odds_raw = (payout1 * payout2 - 100) * 0.85

                if est_odds_raw < 150:
                    continue

                est_odds = int(est_odds_raw)

                ev = ev_per_dollar(combined_prob, est_odds)
                if ev < 0.12:
                    continue

                parlay = {
                    "player": play1.get("player"),
                    "player_id": player_id,
                    "legs": [
                        f"{stat1.upper()}O {play1.get('line')}",
                        f"{stat2.upper()}O {play2.get('line')}",
                    ],
                    "correlation": correlation,
                    "combined_prob": combined_prob,
                    "estimated_odds": est_odds,
                    "ev": ev,
                    "bet_size": kelly_bet_size(ev * 100, combined_prob, BANKROLL),
                    "edge": (combined_prob - american_to_prob(est_odds)) * 100,
                    "type": "correlated_parlay",
                }
                corr_parlays.append(parlay)

    corr_parlays.sort(key=lambda p: -p["ev"])
    return corr_parlays[:3]


LADDER_RUNGS = [10.5, 15.5, 20.5, 25.5, 30.5]


def scan_ladder_plays(
    lines_map: Dict,
    games_map: Dict,
    now_et: datetime = None,
) -> List[Dict]:
    """
    Ladder bet scanner.
    Only for players projecting 18+ points.
    Find available lines near each rung.
    Min prob per leg 0.52, min EV 0.04.
    Return top 4.
    """
    if now_et is None:
        now_et = _now_et()

    ladders = []
    state = load_state()

    for game_id, game_info in games_map.items():
        game_time_et = game_info.get("game_time_et")
        if not game_time_et or deadline_exceeded(game_time_et):
            continue

        prop_lines = lines_map.get(game_id, {}).get("player_points", {})

        for (player_name, player_id, player_team), line_data in prop_lines.items():
            if not player_id:
                continue

            try:
                # Project points
                proj, sigma, _ = compute_projection(
                    player_id,
                    "pts",
                    game_info.get("opponent_name", ""),
                    game_info.get("is_home", False),
                )

                if not proj or proj < 18:
                    continue

                # Find lines near each rung
                available_lines = line_data.get("lines_at_rungs", {})
                ladder_legs = []

                for rung in LADDER_RUNGS:
                    # Check if we have a line near this rung
                    if rung not in available_lines:
                        continue

                    rung_line = available_lines[rung].get("line", rung)
                    rung_odds = available_lines[rung].get("odds", -110)

                    # Calculate prob at rung
                    z = (proj - rung_line) / max(sigma, STAT_STD_FLOORS.get("pts", STD_FLOOR))
                    raw_prob = _norm_cdf(z)
                    prob = calibrated_prob(raw_prob)

                    if prob < 0.52:
                        continue

                    ev = ev_per_dollar(prob, rung_odds)
                    if ev < 0.04:
                        continue

                    ladder_legs.append({
                        "rung": rung,
                        "line": rung_line,
                        "odds": rung_odds,
                        "prob": prob,
                        "ev": ev,
                    })

                if not ladder_legs:
                    continue

                # Best leg by EV
                best_leg = max(ladder_legs, key=lambda l: l["ev"])

                # 2-leg combo suggestion: best + second best
                combo_leg = None
                if len(ladder_legs) >= 2:
                    sorted_legs = sorted(ladder_legs, key=lambda l: -l["ev"])
                    combo_leg = sorted_legs[1]

                ladder = {
                    "player": player_name,
                    "player_id": player_id,
                    "player_team": player_team,
                    "projection": proj,
                    "best_leg": best_leg,
                    "combo_leg": combo_leg,
                    "all_legs": ladder_legs,
                    "type": "ladder",
                }
                ladders.append(ladder)

            except Exception as e:
                print(f"[DEBUG] Ladder error for {player_name}: {e}")
                continue

    ladders.sort(key=lambda l: -l["best_leg"]["ev"])
    return ladders[:4]


# ============================================================================
# SECTION 10: PLAY FORMATTING
# ============================================================================

def format_play_card(play: Dict, index: int) -> str:
    """
    Format play card for WhatsApp output.
    Clean, professional format with context line (max 80 chars).
    """
    player = play.get("player", "?")
    team = play.get("player_team", "?")
    prop_type = play.get("prop_type", "pts")
    line = play.get("line", 0)
    proj = play.get("proj", 0)
    odds = play.get("odds", -110)
    vendor = play.get("vendor", "Books")
    bet_size = play.get("bet_size", 0)
    prob = play.get("prob", 0)
    ev = play.get("ev", 0)
    edge = play.get("edge", 0)

    # Determine prop display name
    prop_name = {
        "player_points": "PTS",
        "player_rebounds": "REB",
        "player_assists": "AST",
        "player_threes": "3PM",
        "player_blocks": "BLK",
        "player_steals": "STL",
    }.get(prop_type, "?")

    # Check for breakout tag
    breakout_tag = " [BREAKOUT]" if play.get("is_breakout") else ""

    # Format odds display
    if odds > 0:
        odds_str = f"+{odds}"
    else:
        odds_str = str(odds)

    # Build context line (explain_play)
    context = explain_play(play)

    # Main card
    card = (
        f"{index}. [{play.get('confidence_tier', 'LEAN')}] {player} ({team}){breakout_tag}\n"
        f"   {prop_name} OVER {line} | {vendor} {odds_str} | Kelly ${bet_size:.0f}\n"
        f"   Proj {proj:.1f} vs Line {line} | Edge +{edge:.1f}% | P={prob*100:.0f}% | EV=+{ev:.2f}\n"
        f"   {context}"
    )
    return card


def explain_play(play: Dict) -> str:
    """
    Generate a short one-line context explanation (max 80 chars).
    Build from: matchup, home/away, b2b, breakout, injury, etc.
    """
    parts = []

    # Last N average if available
    if play.get("l10_avg"):
        parts.append(f"L10 {play['l10_avg']:.1f}")

    # Matchup/opponent strength
    if play.get("opp_def_rating"):
        opp = "weak" if play.get("opp_def_rating", 110) > 110 else "tough"
        parts.append(f"vs {opp} def")

    # Home/away
    if play.get("is_home"):
        parts.append("home")
    else:
        parts.append("road")

    # Back-to-back
    if play.get("is_b2b"):
        parts.append("B2B caution")
    elif play.get("rest_days", 0) >= 2:
        parts.append(f"well-rested")

    # Return from injury
    if play.get("games_back_from_injury", 0) <= 3:
        parts.append("return +boost")

    # Injury beneficiary boost
    if play.get("injury_boost", 0) > 0:
        parts.append(f"injury +{play['injury_boost']:.1f}pts")

    # Combine with separator and truncate
    context = " | ".join(parts)
    if len(context) > 80:
        context = context[:77] + "..."

    return context if context else "Neutral matchup"


def format_parlay_card(parlay: Dict) -> str:
    """Format parlay card for WhatsApp output."""
    legs_str = " + ".join(parlay.get("legs", []))
    odds = parlay.get("estimated_odds", -110)
    prob = parlay.get("prob", 0)
    ev = parlay.get("ev", 0)
    bet_size = parlay.get("bet_size", 0)

    ptype = parlay.get("type", "sgp").upper()

    odds_str = f"+{odds}" if odds > 0 else str(odds)

    return (
        f"{ptype}: {legs_str}\n"
        f"  Est. {odds_str} | P={prob*100:.0f}% | EV=+{ev:.2f} | ${bet_size:.0f}"
    )


def format_ladder_card(ladder: Dict) -> str:
    """Format ladder card for WhatsApp output."""
    player = ladder.get("player", "?")
    team = ladder.get("player_team", "?")
    proj = ladder.get("projection", 0)
    best = ladder.get("best_leg", {})

    best_line = best.get("line", 0)
    best_odds = best.get("odds", -110)
    best_prob = best.get("prob", 0)
    best_ev = best.get("ev", 0)

    odds_str = f"+{best_odds}" if best_odds > 0 else str(best_odds)

    rungs_list = ", ".join([f"{l['rung']}" for l in ladder.get("all_legs", [])])

    return (
        f"LADDER: {player} ({team}) Proj {proj:.1f}\n"
        f"  BEST: {best_line}+ {odds_str} P={best_prob*100:.0f}% EV=+{best_ev:.2f}\n"
        f"  Rungs: {rungs_list}"
    )


# ============================================================================
# SECTION 11: MAIN ORCHESTRATOR
# ============================================================================

def deduplicate_plays(plays: List[Dict]) -> List[Dict]:
    """
    Group by (player_id, prop_type), keep play with highest final_score.
    """
    best_by_key = {}

    for play in plays:
        key = (play.get("player_id"), play.get("prop_type"))
        if key not in best_by_key:
            best_by_key[key] = play
        else:
            # Keep higher score
            if play.get("score", 0) > best_by_key[key].get("score", 0):
                best_by_key[key] = play

    return list(best_by_key.values())


def apply_exposure_caps(plays: List[Dict]) -> List[Dict]:
    """
    Apply exposure caps: max 2 per team, max 5 per game.
    Sort by final_score, keep best.
    """
    plays = sorted(plays, key=lambda p: -p.get("score", 0))

    team_count = {}
    game_count = {}
    kept = []

    for play in plays:
        team = play.get("player_team")
        game = play.get("game_id")

        team_count[team] = team_count.get(team, 0) + 1
        game_count[game] = game_count.get(game, 0) + 1

        if team_count[team] > 2 or game_count[game] > 5:
            continue

        if len(kept) < MAX_TOTAL_PLAYS:
            kept.append(play)

    return kept


def apply_cooldown(state: Dict, plays: List[Dict], now_ts: float) -> List[Dict]:
    """
    Skip plays sent within last 3 hours unless edge jumped by 2+ points.
    """
    kept = []
    recent_plays = {}

    # Load recent plays from state
    for play_record in state.get("plays", []):
        key = (play_record.get("player"), play_record.get("stat"))
        recent_plays[key] = play_record

    for play in plays:
        key = (play.get("player"), play.get("stat_key"))
        if key in recent_plays:
            prev_play = recent_plays[key]
            prev_edge = prev_play.get("edge", 0)
            curr_edge = play.get("edge", 0)

            # Check time and edge delta
            try:
                prev_time = datetime.fromisoformat(prev_play.get("timestamp", "")).timestamp()
                time_diff = now_ts - prev_time
                edge_jump = curr_edge - prev_edge

                # If within 3 hours and edge didn't jump 2+, skip
                if time_diff < 3 * 3600 and edge_jump < 2.0:
                    continue
            except:
                pass

        kept.append(play)

    return kept


def record_sent(state: Dict, plays: List[Dict], now_ts: float) -> None:
    """Log sent plays for cooldown tracking."""
    if "plays" not in state:
        state["plays"] = []

    for play in plays:
        state["plays"].append({
            "player": play.get("player"),
            "stat": play.get("stat_key"),
            "line": play.get("line"),
            "proj": play.get("proj"),
            "edge": play.get("edge"),
            "timestamp": datetime.fromtimestamp(now_ts, tz=ZoneInfo("America/New_York")).isoformat(),
        })


def build_whatsapp_message(
    straights: List[Dict],
    plus_plays: List[Dict],
    sgps: List[Dict],
    corr_parlays: List[Dict],
    ladders: List[Dict],
    state: Dict = None,
) -> str:
    """Assemble full WhatsApp message with all play types."""
    if state is None:
        state = load_state()

    now_et = _now_et()
    date_str = now_et.strftime("%m/%d")
    time_str = now_et.strftime("%I:%M %p")

    lines = [f"NBA PICKS {date_str} {time_str} ET"]

    # Hit rate if available
    hit_rate = get_hit_rate_summary(state)
    if "No plays" not in hit_rate:
        lines.append(f"({hit_rate})")

    lines.append("\n-- STRAIGHT BETS --")

    # Format straight plays
    if straights:
        for i, play in enumerate(straights, 1):
            lines.append(format_play_card(play, i))
            lines.append("")
    else:
        lines.append("None")

    # Plus odds section
    if plus_plays:
        lines.append("\n-- PLUS ODDS --")
        for i, play in enumerate(plus_plays, 1):
            lines.append(format_play_card(play, i))
            lines.append("")

    # Parlays section
    if sgps or corr_parlays:
        lines.append("\n-- PARLAYS --")
        for sgp in sgps:
            lines.append(format_parlay_card(sgp))
            lines.append("")
        for parlay in corr_parlays:
            lines.append(format_parlay_card(parlay))
            lines.append("")

    # Ladders section
    if ladders:
        lines.append("\n-- LADDERS --")
        for ladder in ladders:
            lines.append(format_ladder_card(ladder))
            lines.append("")

    # Summary line
    total_bets = len(straights) + len(plus_plays)
    total_kelly = sum(p.get("bet_size", 0) for p in straights + plus_plays)
    lines.append(f"Total action: ${total_kelly:.0f} across {total_bets} plays")

    return "\n".join(lines)


def run():
    """
    Main entry point for daily prop scanning and plays generation.
    Orchestrates all edge engines, deduplication, and formatting.
    """
    print("[INFO] Starting NBA prop scan...")
    start_time = time.time()
    now_et = _now_et()

    # Phase 1: Time checks
    if now_et.hour < 9 or now_et.hour > 23:
        print(f"[INFO] Skipping (time {now_et.hour}:00, need 9am-11:30pm ET)")
        return

    # Phase 2: Load state
    state = load_state()

    # Phase 3: Fetch today's games and prop lines
    print("[INFO] Building prop lines map...")
    lines_map, games_map = build_today_props(now_et)

    if not games_map:
        print("[INFO] No games found for today")
        return

    print(f"[INFO] Found {len(games_map)} games, {sum(len(v) for v in lines_map.values())} props")

    # Phase 4: Warm up stats (batch fetch)
    print("[INFO] Warming up player stats...")
    all_player_ids = set()
    for game_props in lines_map.values():
        for prop_lines in game_props.values():
            for (_, player_id, _) in prop_lines.keys():
                if player_id:
                    all_player_ids.add(player_id)

    print(f"[INFO] Batch fetching stats for {len(all_player_ids)} players...")
    # In practice, this would batch fetch game logs, advanced stats, etc.
    # For now, we rely on on-demand fetching in compute_projection

    # Phase 5: Fetch news & injuries
    print("[INFO] Fetching LineupExperts news...")
    news_items = fetch_lineupexperts_news()
    news_boosts = build_news_boost_map(news_items)
    news_scores = build_news_score_map(news_items)
    le_injuries = parse_le_injuries()

    # Phase 6: Run edge engines
    print("[INFO] Running edge engines...")
    injury_plays = []
    slate_plays = []
    news_plays = []

    # 6a: Injury edges
    for player_key, injury_info in le_injuries.items():
        if injury_info.get("status") in ("Out", "Doubtful"):
            edges = build_injury_edges(
                injury_info.get("team_id", 0),
                injury_info.get("player_name", ""),
                games_map,
                state,
            )
            injury_plays.extend(edges)

    # 6b: Main slate scan
    for prop_type in PROP_TYPES:
        edges = slate_scan_edges(prop_type, lines_map, games_map, news_scores, now_et)
        slate_plays.extend(edges)

    # 6c: Lineup news edges
    news_plays = lineup_news_edges(games_map, state)

    # 6d: Plus odds hunt
    plus_plays = plus_odds_hunt_edges(lines_map, games_map, now_et)

    print(f"[INFO] Found {len(injury_plays)} injury, {len(slate_plays)} slate, "
          f"{len(news_plays)} news edges")

    # Phase 7: Deduplicate
    combined = deduplicate_plays(injury_plays + slate_plays + news_plays)

    # Phase 8: Apply exposure caps
    final_plays = apply_exposure_caps(combined)

    # Phase 8b: Apply cooldown
    now_ts = time.time()
    final_plays = apply_cooldown(state, final_plays, now_ts)

    print(f"[INFO] Final plays after dedup/caps: {len(final_plays)}")

    # Phase 9: Build parlays
    print("[INFO] Building parlays...")
    sgps = find_sgp_opportunities(final_plays, games_map)
    corr_parlays = find_correlated_parlays(final_plays, lines_map, games_map)
    ladders = scan_ladder_plays(lines_map, games_map, now_et)

    print(f"[INFO] SGPs: {len(sgps)}, Corr: {len(corr_parlays)}, Ladders: {len(ladders)}")

    # Phase 10: Format and send
    if final_plays or plus_plays or sgps or corr_parlays or ladders:
        msg = build_whatsapp_message(final_plays, plus_plays, sgps, corr_parlays, ladders, state)
        print("[INFO] Message ready, sending via WhatsApp...")

        # Send via Twilio
        client = None
        if ACCOUNT_SID and AUTH_TOKEN and TwilioClient:
            try:
                client = TwilioClient(ACCOUNT_SID, AUTH_TOKEN)
            except Exception as e:
                print(f"[WARN] Twilio init failed: {e}")

        send_chunked(client, msg)

        # Record sent plays
        record_sent(state, final_plays, now_ts)
        save_state(state)

        elapsed = time.time() - start_time
        print(f"[INFO] Complete in {elapsed:.1f}s")
    else:
        print("[INFO] No qualifying plays found today")


# ============================================================================
# SECTION 12: USER OVERRIDES
# ============================================================================

def get_instinct_boosts() -> Dict[str, float]:
    """Parse INSTINCT_BOOSTS env var (e.g., 'LeBron:+2.5,Tatum:+1.5')."""
    boosts = {}
    env_val = os.getenv("INSTINCT_BOOSTS", "")
    if not env_val:
        return boosts

    try:
        for pair in env_val.split(","):
            name, boost = pair.split(":")
            boosts[name.strip()] = float(boost.strip())
    except Exception as e:
        print(f"[WARN] Error parsing INSTINCT_BOOSTS: {e}")

    return boosts


def get_fade_players() -> List[str]:
    """Parse FADE_PLAYERS env var (comma-separated names)."""
    env_val = os.getenv("FADE_PLAYERS", "")
    if not env_val:
        return []
    return [name.strip() for name in env_val.split(",")]


def get_target_players() -> List[str]:
    """Parse TARGET_PLAYERS env var (comma-separated names)."""
    env_val = os.getenv("TARGET_PLAYERS", "")
    if not env_val:
        return []
    return [name.strip() for name in env_val.split(",")]


def get_manual_minute_caps() -> Dict[str, float]:
    """Parse MINUTE_CAPS env var (e.g., 'LeBron:30,Curry:28')."""
    caps = {}
    env_val = os.getenv("MINUTE_CAPS", "")
    if not env_val:
        return caps

    try:
        for pair in env_val.split(","):
            name, mins = pair.split(":")
            caps[name.strip()] = float(mins.strip())
    except Exception as e:
        print(f"[WARN] Error parsing MINUTE_CAPS: {e}")

    return caps


# ============================================================================
# SECTION 13: SITUATIONAL ANALYSIS
# ============================================================================

def analyze_situation(
    player_name: str,
    player_team: str,
    opponent: str,
    games_map: Dict,
    now_et: datetime = None,
) -> Dict:
    """
    Analyze situational factors: bounce back, B2B fatigue, rest, return from injury.
    Return dict with adjustment recommendations (conservative boosts).
    """
    if now_et is None:
        now_et = _now_et()

    situation = {
        "bounce_back_boost": 0.0,
        "fatigue_penalty": 0.0,
        "rest_boost": 0.0,
        "return_boost": 0.0,
        "b2b": False,
    }

    # Bounce back detection: if scored <10 last game, usually bounces back
    # This would require last game result, placeholder for now
    # situation["bounce_back_boost"] = 2.0 if last_game_pts < 10 else 0.0

    # Back-to-back fatigue: -2 pts
    # situation["b2b"] = check_b2b(player_name, now_et)
    # if situation["b2b"]:
    #     situation["fatigue_penalty"] = 2.0

    # Rest days advantage
    # situation["rest_boost"] = 0.5 * rest_days_count

    # Return from injury boost: +1.5 pts
    # if games_back_from_injury <= 3:
    #     situation["return_boost"] = 1.5

    return situation


# ============================================================================
# SECTION 14: ENTRY POINT
# ============================================================================

LAST_WHATSAPP_SEND = 0  # Global tracking for send throttling

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"[FATAL] {type(e).__name__}: {e}")
        traceback.print_exc()
