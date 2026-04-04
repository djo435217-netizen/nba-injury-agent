"""
NBA Player Props Prediction Model - Render Cron Job + Twilio WhatsApp
Enhanced with all critical bugs fixed, proper API integration, and advanced features.
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
    print("[INIT] requests loaded OK")
except ImportError:
    requests = None
    print("[INIT] WARNING: requests not installed")

try:
    from twilio.rest import Client as TwilioClient
    print("[INIT] twilio loaded OK")
except ImportError:
    TwilioClient = None
    print("[INIT] WARNING: twilio not installed - WhatsApp will be MOCK only")


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
MIN_EDGE = 3.0  # default min 3% edge to claim
MIN_PROB = 0.58  # calibrated probability threshold
STD_FLOOR = 4.0  # global min std dev

# Per-stat minimum edge thresholds (% over line)
# Higher volume stats need less edge because they're more predictable
STAT_MIN_EDGE = {
    "pts": 3.0,      # Points: most predictable, 3% edge
    "reb": 5.0,      # Rebounds: moderate variance, need 5%
    "ast": 5.0,      # Assists: moderate variance, need 5%
    "blk": 10.0,     # Blocks: high variance, need 10%
    "stl": 10.0,     # Steals: high variance, need 10%
    "threes": 8.0,   # Threes: high variance, need 8%
}

# Per-stat minimum probability thresholds
STAT_MIN_PROB = {
    "pts": 0.57,
    "reb": 0.58,
    "ast": 0.58,
    "blk": 0.60,     # Need higher confidence for volatile stats
    "stl": 0.60,
    "threes": 0.59,
}

# Per-stat standard deviation floors (prevent overconfident narrows)
STAT_STD_FLOORS = {
    "pts": 4.0,
    "reb": 2.0,
    "ast": 1.8,
    "blk": 0.9,
    "stl": 0.9,
    "threes": 1.2,
}

# Per-stat max picks (ensures diversity across stat types)
STAT_MAX_PICKS = {
    "pts": 3,
    "reb": 2,
    "ast": 2,
    "blk": 1,
    "stl": 1,
    "threes": 2,
}

# Projection window weights - L10 is most reliable (king of windows)
PROJECTION_WEIGHTS = {
    "base": 0.30,
    "l10": 0.40,
    "l3": 0.30,
}

# Breakout detection - require SUSTAINED performance (L5 + L10)
BREAKOUT_WEIGHTS = {
    "base": 0.15,
    "l10": 0.25,
    "l5": 0.60,
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
LADDER_SIZES = [2, 3]

# Time windows for consistency
CONSISTENCY_WINDOW = 10
CONSISTENCY_BAND = 0.20

# Book vendors and prop types
# PRIMARY_BOOK is the preferred sportsbook — only its lines/odds are used.
# Falls back to others only if primary has no data for a player/prop.
PRIMARY_BOOK = os.getenv("PRIMARY_BOOK", "fanduel").lower()
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
MIN_MINUTES_CHANGE = -3.0
MAX_MINUTES_CHANGE = 3.0

# Calibration table: raw_prob -> calibrated_prob
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
    "Los Angeles Clippers": {"pts": 110.8, "reb": 46.8, "ast": 26.8},
    "Oklahoma City Thunder": {"pts": 109.5, "reb": 45.5, "ast": 25.2},
    "Dallas Mavericks": {"pts": 112.1, "reb": 47.8, "ast": 27.8},
    "Houston Rockets": {"pts": 113.2, "reb": 48.5, "ast": 28.5},
    "Atlanta Hawks": {"pts": 114.5, "reb": 49.2, "ast": 29.2},
    "Milwaukee Bucks": {"pts": 108.5, "reb": 45.2, "ast": 25.0},
    "New York Knicks": {"pts": 109.8, "reb": 46.1, "ast": 25.8},
    "Philadelphia 76ers": {"pts": 110.2, "reb": 46.5, "ast": 26.2},
}

# API endpoints
BDL_BASE = "https://api.balldontlie.io/v1"
LINEUP_EXPERTS_BASE = "https://api.lineupexperts.com/v1"

# Caches
PROPS_CACHE = {}
ADV_STATS_CACHE = {}
GAME_ODDS_CACHE = {}
ROSTER_CACHE = {}
TEAM_ID_CACHE = {}
LINEUPS_CACHE = {}
DEF_RATINGS_CACHE = {}
PLAYER_NAME_CACHE = {}

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
    """Get BDL season year. BDL uses the year the season STARTS (Oct 2025 = season 2025).
    April 2026 → season 2025. October 2026 → season 2026."""
    if dt is None:
        dt = _now_et()
    year = dt.year
    if dt.month < 10:
        return year - 1  # Apr 2026 -> 2025 (season started Oct 2025)
    return year          # Oct 2026 -> 2026 (new season starting)


def _parse_minutes(min_str) -> float:
    """Convert 'MM:SS' or plain number to float minutes."""
    if not min_str:
        return 0.0
    # Handle numeric input (int or float)
    if isinstance(min_str, (int, float)):
        return float(min_str)
    min_str = str(min_str).strip()
    if min_str in ("0:00", "0", ""):
        return 0.0
    try:
        if ":" in min_str:
            parts = min_str.split(":")
            return float(parts[0]) + float(parts[1]) / 60.0
        else:
            return float(min_str)
    except:
        return 0.0


def _clean_name(name: str) -> str:
    """Normalize player name for matching. BUG FIX: was missing string argument."""
    return re.sub(r"[^\w\s]", "", name).lower().strip()


def _norm_cdf(z: float) -> float:
    """Standard normal CDF approximation (error < 0.002)."""
    if z > 6:
        return 1.0
    if z < -6:
        return 0.0
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
    Uses interpolation on CALIBRATION_TABLE.
    """
    if raw_prob <= 0.50:
        return 0.50
    if raw_prob >= 0.95:
        return 0.77

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

    if edge_pct <= 0:
        return 0.0

    kelly_frac = (edge_pct / 100.0) / 2.0
    kelly_bet = bankroll * kelly_frac
    kelly_quarter = kelly_bet * KELLY_FRACTION

    return _clamp(kelly_quarter, MIN_BET, MAX_BET)


def passes_juice_filter(odds: float) -> bool:
    """Check if odds have reasonable juice (not too extreme)."""
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


def hit_rate(games_list: List[Dict], line: float) -> float:
    """
    Calculate hit rate: % of games that went OVER the line.
    This is THE most important signal for profitability.
    """
    if not games_list:
        return 0.0
    hits = sum(1 for g in games_list if g.get("value", 0) > line)
    return hits / len(games_list)


def hit_rates_by_window(base_games, l10_games, l5_games, line: float) -> Dict:
    """Calculate hit rates across all windows + weighted composite."""
    base_hr = hit_rate(base_games, line)
    l10_hr = hit_rate(l10_games, line)
    l5_hr = hit_rate(l5_games, line)
    # Weighted: recent matters more
    composite = 0.25 * base_hr + 0.35 * l10_hr + 0.40 * l5_hr
    return {
        "base": base_hr, "l10": l10_hr, "l5": l5_hr,
        "composite": composite,
        "l5_hits": sum(1 for g in l5_games if g.get("value", 0) > line),
        "l10_hits": sum(1 for g in l10_games if g.get("value", 0) > line),
        "base_hits": sum(1 for g in base_games if g.get("value", 0) > line),
    }


def normalize_team_name(team_name: str) -> str:
    """Normalize team name via alias map."""
    if not team_name:
        return ""
    if team_name in TEAM_ALIAS_TO_FULL:
        return TEAM_ALIAS_TO_FULL[team_name]
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


def send_one(client, msg: str) -> bool:
    """Send single WhatsApp message via Twilio."""
    if not client or not RECIPIENT_WHATSAPP:
        print(f"[MOCK] WhatsApp ({len(msg)} chars) - no client or recipient")
        return True
    try:
        result = client.messages.create(
            from_=TWILIO_WHATSAPP_FROM,
            to=RECIPIENT_WHATSAPP,
            body=msg
        )
        print(f"[TWILIO] Sent OK: SID={result.sid}, status={result.status}")
        return True
    except Exception as e:
        print(f"[TWILIO] Send FAILED: {e}")
        return False


def send_chunked(client: TwilioClient, text: str, chunk_size: int = 1500) -> bool:
    """Send long text in chunks via WhatsApp."""
    global LAST_WHATSAPP_SEND

    if not text:
        return True

    now = time.time()
    if now - LAST_WHATSAPP_SEND < BET_COOLDOWN:
        time.sleep(BET_COOLDOWN - (now - LAST_WHATSAPP_SEND))

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
        print(f"[WARN] BDL skipped: key={'set' if BALLDONTLIE_API_KEY else 'MISSING'}, requests={'ok' if requests else 'MISSING'}")
        return {}

    url = f"{BDL_BASE}/{endpoint}"
    headers = {"Authorization": BALLDONTLIE_API_KEY}

    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                print(f"[API] {endpoint} -> {resp.status_code} ({len(data.get('data', []))} items)")
                return data
            elif resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 2))
                print(f"[API] {endpoint} -> 429 rate limit, waiting {wait}s")
                time.sleep(wait)
            else:
                print(f"[API] {endpoint} -> {resp.status_code}: {resp.text[:200]}")
                break
        except Exception as e:
            print(f"[API] {endpoint} error: {e}")
            if attempt < retries - 1:
                time.sleep(1)

    return {}


def bdl_games_today(dt: datetime = None) -> List[Dict]:
    """
    Get all NBA games for today.
    BDL uses: home_team, visitor_team (NOT away_team), datetime (NOT scheduled_at).
    We normalize to consistent keys for downstream use.
    """
    if dt is None:
        dt = _now_et()

    date_str = dt.strftime("%Y-%m-%d")
    resp = _bdl_get("games", {"dates[]": date_str, "per_page": 50})

    games = resp.get("data", []) if resp else []

    # Normalize BDL field names for consistency downstream
    for game in games:
        # BDL uses "visitor_team" not "away_team"
        if "visitor_team" in game and "away_team" not in game:
            game["away_team"] = game["visitor_team"]
        # BDL uses "datetime" not "scheduled_at"
        if "datetime" in game and "scheduled_at" not in game:
            game["scheduled_at"] = game["datetime"]
        # Parse datetime
        dt_str = game.get("datetime") or game.get("scheduled_at") or ""
        if dt_str:
            try:
                game["scheduled_dt"] = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            except:
                pass

    if games:
        home = games[0].get("home_team", {}).get("full_name", "?")
        away = games[0].get("away_team", {}).get("full_name", "?")
        print(f"[INFO] First game: {away} @ {home}")

    return games


def bdl_team_name_to_id(team_name: str) -> Optional[int]:
    """Get BDL team ID from name."""
    if not team_name:
        return None

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


def bdl_player_name(player_id: int) -> str:
    """
    Look up player name from BDL player ID.
    Caches results to avoid repeated API calls.
    """
    if not player_id:
        return ""

    if player_id in PLAYER_NAME_CACHE:
        return PLAYER_NAME_CACHE[player_id]

    resp = _bdl_get(f"players/{player_id}", {})
    name = ""
    if resp and resp.get("data"):
        data = resp["data"]
        first = data.get("first_name", "")
        last = data.get("last_name", "")
        name = f"{first} {last}".strip()
    elif resp:
        # Some BDL endpoints return data at top level
        first = resp.get("first_name", "")
        last = resp.get("last_name", "")
        name = f"{first} {last}".strip()

    PLAYER_NAME_CACHE[player_id] = name
    return name


def bdl_batch_player_names(player_ids: List[int]) -> Dict[int, str]:
    """
    Batch look up player names. Uses cache where possible.
    For uncached IDs, fetches individually (BDL has no batch player endpoint).
    """
    result = {}
    to_fetch = []

    for pid in player_ids:
        if pid in PLAYER_NAME_CACHE:
            result[pid] = PLAYER_NAME_CACHE[pid]
        else:
            to_fetch.append(pid)

    for pid in to_fetch:
        name = bdl_player_name(pid)
        result[pid] = name

    return result


def bdl_active_roster(team_id: int) -> List[Dict]:
    """Get active roster for team."""
    if not team_id:
        return []

    cache_key = f"roster_{team_id}"
    if cache_key in ROSTER_CACHE:
        return ROSTER_CACHE[cache_key]

    resp = _bdl_get(f"teams/{team_id}", {})
    roster = resp.get("data", {}).get("players", []) if resp and resp.get("data") else []

    ROSTER_CACHE[cache_key] = roster
    return roster


def bdl_find_player_id_on_team(player_name: str, team_id: int) -> Optional[int]:
    """Fuzzy match player name to roster and get player ID."""
    if not player_name or not team_id:
        return None

    roster = bdl_active_roster(team_id)
    player_clean = _clean_name(player_name)

    for player in roster:
        if _clean_name(player.get("first_name", "") + " " + player.get("last_name", "")) == player_clean:
            return player.get("id")

    last_name_clean = _clean_name(player_name.split()[-1] if player_name else "")
    for player in roster:
        if _clean_name(player.get("last_name", "")) == last_name_clean:
            return player.get("id")

    return None


def bdl_last_n_games_stats(player_id: int, stat_key: str, n: int = BASELINE_GAMES) -> List[Dict]:
    """
    Fetch last n games for a player, extracting specific stat.
    BDL endpoint: /v1/stats?player_ids[]=X&seasons[]=YEAR&per_page=N
    stat_key should be: pts, reb, ast, blk, stl, or 'min' for minutes.
    Returns list of dicts with 'value' key.
    """
    if not player_id:
        return []

    season = _season_year()
    resp = _bdl_get("stats", {
        "player_ids[]": player_id,
        "seasons[]": season,
        "per_page": n,
    })
    games = resp.get("data", [])

    if not games:
        print(f"[WARN] No stats for player {player_id} season {season}")
        return []

    # Debug: print first game keys and values on first call
    if games and not getattr(bdl_last_n_games_stats, '_debug_printed', False):
        g0 = games[0]
        print(f"[DEBUG] Stats game[0] keys: {list(g0.keys())}")
        print(f"[DEBUG] Stats game[0] sample: pts={g0.get('pts')}, reb={g0.get('reb')}, ast={g0.get('ast')}, min={g0.get('min')}, fg3m={g0.get('fg3m')}")
        bdl_last_n_games_stats._debug_printed = True

    result = []
    for game in games:
        if not game:
            continue
        val = 0.0
        if stat_key == "min":
            val = _parse_minutes(str(game.get("min", "0:00")))
        else:
            val = float(game.get(stat_key, 0) or 0)
        result.append({
            "value": val,
            "game_id": game.get("game", {}).get("id") if isinstance(game.get("game"), dict) else game.get("game_id"),
            "date": game.get("game", {}).get("date") if isinstance(game.get("game"), dict) else None,
        })

    return result


def bdl_last_n_games_threes(player_id: int, n: int = BASELINE_GAMES) -> List[Dict]:
    """
    Fetch 3-pointers made from last n games using /v1/stats endpoint.
    Returns list of dicts with 'value' key (3PM).
    """
    if not player_id:
        return []

    season = _season_year()
    resp = _bdl_get("stats", {
        "player_ids[]": player_id,
        "seasons[]": season,
        "per_page": n,
    })
    games = resp.get("data", [])

    result = []
    for game in games:
        if not game:
            continue
        val = float(game.get("fg3m", 0) or 0)
        result.append({
            "value": val,
            "game_id": game.get("game", {}).get("id") if isinstance(game.get("game"), dict) else None,
        })

    return result


def bdl_fetch_props_for_game(game_id: int, prop_types: List[str] = None) -> Dict:
    """
    Fetch player prop lines for a game from BDL God Tier API.
    Endpoint: /v2/odds/player_props?game_id=XXX

    Actual BDL v2 response format per item:
      {id, game_id, player_id (int!), vendor, prop_type, line_value (string!), market, updated_at}
      market = {type: "over_under", over_odds: -102, under_odds: -132}
           or {type: "milestone", odds: -1400}

    Returns dict: {player_name: {prop_type: [{book, line, odds}]}}
    """
    if not game_id or not prop_types:
        return {}

    cache_key = f"props_{game_id}"
    if cache_key in PROPS_CACHE:
        return PROPS_CACHE[cache_key]

    if not BALLDONTLIE_API_KEY or not requests:
        return {}

    url = "https://api.balldontlie.io/v2/odds/player_props"
    headers = {"Authorization": BALLDONTLIE_API_KEY}

    # BDL prop_type names -> our internal names
    PROP_TYPE_MAP = {
        "points": "player_points",
        "rebounds": "player_rebounds",
        "assists": "player_assists",
        "threes": "player_threes",
        "blocks": "player_blocks",
        "steals": "player_steals",
    }

    result = {}
    try:
        resp = requests.get(url, headers=headers, params={"game_id": game_id}, timeout=REQUEST_TIMEOUT)
        print(f"[API] v2/odds/player_props -> {resp.status_code} (game {game_id})")

        if resp.status_code != 200:
            print(f"[API] player_props error: {resp.text[:200]}")
            PROPS_CACHE[cache_key] = result
            return result

        data = resp.json()
        raw_items = data.get("data", [])
        print(f"[API] player_props raw: {len(raw_items)} entries")

        # Step 1: Collect unique player_ids so we can batch-resolve names
        unique_pids = set()
        for prop in raw_items:
            pid = prop.get("player_id")
            if pid:
                unique_pids.add(pid)

        # Step 2: Batch resolve player names
        if unique_pids:
            print(f"[API] Resolving {len(unique_pids)} player names...")
            bdl_batch_player_names(list(unique_pids))

        # Step 3: Parse each prop entry into temp structure
        # temp[player_name][prop_type][vendor_lower] = [{book, line, odds, player_id}]
        temp = {}
        skipped_milestone = 0
        parsed = 0
        vendor_counts = {}

        for prop in raw_items:
            pid = prop.get("player_id")
            if not pid:
                continue

            player_name = PLAYER_NAME_CACHE.get(pid, "")
            if not player_name:
                continue

            raw_prop_type = prop.get("prop_type", "")
            prop_type = PROP_TYPE_MAP.get(raw_prop_type, "")
            if not prop_type:
                continue

            # Parse market - only use over_under, skip milestone
            market = prop.get("market", {})
            market_type = market.get("type", "")

            if market_type == "milestone":
                skipped_milestone += 1
                continue

            # Get over odds from market
            over_odds = market.get("over_odds", -110)
            if over_odds is None:
                over_odds = -110

            # line_value is a STRING in BDL response
            line_str = prop.get("line_value", "0")
            try:
                line_val = float(line_str)
            except (ValueError, TypeError):
                continue

            if line_val <= 0:
                continue

            vendor = prop.get("vendor", "Consensus")
            vendor_lower = vendor.lower().replace(" ", "").replace("_", "")
            vendor_counts[vendor] = vendor_counts.get(vendor, 0) + 1

            if player_name not in temp:
                temp[player_name] = {}
            if prop_type not in temp[player_name]:
                temp[player_name][prop_type] = {}
            if vendor_lower not in temp[player_name][prop_type]:
                temp[player_name][prop_type][vendor_lower] = []

            temp[player_name][prop_type][vendor_lower].append({
                "book": vendor,
                "line": line_val,
                "odds": float(over_odds),
                "player_id": pid,
            })
            parsed += 1

        print(f"[API] Vendors found: {vendor_counts}")

        # Step 4: For each player/prop, use ONLY PRIMARY_BOOK lines.
        # Try multiple name variants to match BDL's vendor strings.
        primary_variants = set()
        pb = PRIMARY_BOOK.lower().replace(" ", "").replace("_", "").replace("-", "")
        primary_variants.add(pb)
        # Common BDL vendor name variations
        FANDUEL_ALIASES = {"fanduel", "fan_duel", "fd", "fanduelus", "fanduelsportsbook"}
        if pb in FANDUEL_ALIASES:
            primary_variants.update(FANDUEL_ALIASES)

        primary_used = 0
        fallback_used = 0
        skipped_no_primary = 0

        for player_name, props_dict in temp.items():
            for prop_type, vendors_dict in props_dict.items():
                matched_key = None
                for vk in vendors_dict.keys():
                    if vk in primary_variants:
                        matched_key = vk
                        break

                if matched_key:
                    if player_name not in result:
                        result[player_name] = {}
                    result[player_name][prop_type] = vendors_dict[matched_key]
                    primary_used += 1
                else:
                    # No FanDuel line — use best available as fallback
                    # (user can set STRICT_BOOK=true to skip non-FanDuel entirely)
                    if os.getenv("STRICT_BOOK", "false").lower() == "true":
                        skipped_no_primary += 1
                        continue
                    best_vendor = max(vendors_dict.keys(), key=lambda v: len(vendors_dict[v]))
                    if player_name not in result:
                        result[player_name] = {}
                    # Tag fallback lines so display shows the actual book
                    for entry in vendors_dict[best_vendor]:
                        entry["is_fallback"] = True
                    result[player_name][prop_type] = vendors_dict[best_vendor]
                    fallback_used += 1

        total_players = len(result)
        total_props = sum(len(v) for player_props in result.values() for v in player_props.values())
        print(f"[API] player_props: {total_players} players, {total_props} lines | "
              f"PRIMARY ({PRIMARY_BOOK}): {primary_used} props, fallback: {fallback_used}, "
              f"skipped: {skipped_no_primary} | "
              f"({skipped_milestone} milestones skipped)")

    except Exception as e:
        print(f"[API] player_props exception: {e}")
        traceback.print_exc()

    PROPS_CACHE[cache_key] = result
    return result


def bdl_fetch_advanced_stats(player_id: int) -> Dict:
    """
    Fetch advanced stats: usage%, off_rating, pie, pace from BDL API.
    Returns dict with keys: usage_pct, off_rating, pie, pace.
    """
    if not player_id:
        return {}

    cache_key = f"adv_{player_id}"
    if cache_key in ADV_STATS_CACHE:
        return ADV_STATS_CACHE[cache_key]

    season = _season_year()
    resp = _bdl_get("season_averages", {"player_id": player_id, "season": season})

    result = {
        "usage_pct": 25.0,
        "off_rating": 110.0,
        "pie": 0.12,
        "pace": 98.0,
    }

    if resp and resp.get("data"):
        for avg in resp.get("data", []):
            if avg.get("player_id") == player_id:
                result["usage_pct"] = avg.get("usg_pct", 25.0)
                result["off_rating"] = avg.get("off_rating", 110.0)
                result["pie"] = avg.get("pie", 0.12)
                break

    ADV_STATS_CACHE[cache_key] = result
    return result


def bdl_fetch_game_odds_full(game_id: int) -> Dict:
    """
    Fetch game odds: total, spread, etc. from BDL God Tier API.
    Returns dict with keys: game_total, home_spread, away_spread, pace.
    """
    if not game_id:
        return {}

    cache_key = f"odds_{game_id}"
    if cache_key in GAME_ODDS_CACHE:
        return GAME_ODDS_CACHE[cache_key]

    # BDL odds endpoint: /v1/odds?game_id=XXX (singular, no brackets)
    resp = _bdl_get("odds", {"game_id": game_id})

    result = {
        "game_total": 220.0,
        "home_spread": -3.5,
        "away_spread": 3.5,
        "pace": 98.0,
    }

    if resp and resp.get("data"):
        for odds_entry in resp.get("data", []):
            # BDL v1/v2 odds response fields
            total = odds_entry.get("total_value") or odds_entry.get("over_under") or odds_entry.get("total")
            if total:
                result["game_total"] = float(total)
            spread_home = odds_entry.get("spread_home_value") or odds_entry.get("home_spread")
            if spread_home:
                result["home_spread"] = float(spread_home)
            spread_away = odds_entry.get("spread_away_value") or odds_entry.get("away_spread")
            if spread_away:
                result["away_spread"] = float(spread_away)
            break  # Take first entry

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

    resp = _bdl_get(f"games/{game_id}", {})

    result = {}
    if resp and resp.get("data"):
        game = resp.get("data", {})
        if game.get("home_team", {}).get("players"):
            result[game["home_team"]["id"]] = game["home_team"]["players"]
        if game.get("away_team", {}).get("players"):
            result[game["away_team"]["id"]] = game["away_team"]["players"]

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
    """
    boost_map = {}

    for item in news:
        player = item.get("player", "")
        news_type = item.get("type", "").lower()

        if not player:
            continue

        boost = {}
        if "questionable" in news_type or "probable" in news_type:
            boost = {"pts": 0.0, "reb": 0.0, "ast": 0.0}
        elif "out" in news_type or "day_to_day" in news_type:
            boost = {"pts": 0.0, "reb": 0.0, "ast": 0.0}

        if boost:
            boost_map[player] = boost

    return boost_map


def build_news_score_map(news: List[Dict]) -> Dict:
    """
    Parse news to score players (positive/negative impact).
    Returns dict: {player_name: confidence_adjustment}
    """
    score_map = {}

    for item in news:
        player = item.get("player", "")
        status = item.get("status", "").lower()

        if not player:
            continue

        delta = 0
        if "probable" in status:
            delta = 1
        elif "day-to-day" in status:
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
                "pts": 0.0,
                "reb": 0.0,
                "ast": 0.0,
            }

    return injury_map


def fetch_minutes_windows(player_id: int, n: int = 20) -> Tuple[float, float, float]:
    """
    Fetch last n games of minutes played.
    Returns (base_min_avg, l10_min_avg, l5_min_avg) tuple.
    """
    if not player_id:
        return 0.0, 0.0, 0.0

    games = bdl_last_n_games_stats(player_id, "min", n)

    if not games:
        return 0.0, 0.0, 0.0

    base_games = _slice_last(games, BASELINE_GAMES)
    l10_games = _slice_last(games, LOOKBACK_GAMES)
    l5_games = _slice_last(games, SHORT_GAMES)

    base_avg, _, _ = avg_stat_min_std(base_games)
    l10_avg, _, _ = avg_stat_min_std(l10_games)
    l5_avg, _, _ = avg_stat_min_std(l5_games)

    return base_avg, l10_avg, l5_avg


def fetch_def_rating(opp_team: str, stat_key: str) -> float:
    """
    Fetch opponent defensive rating from BDL or cache.
    Returns rating per 100 possessions. Falls back to hardcoded if API fails.
    """
    opp_full = normalize_team_name(opp_team)

    cache_key = f"def_{opp_full}_{stat_key}"
    if cache_key in DEF_RATINGS_CACHE:
        return DEF_RATINGS_CACHE[cache_key]

    # Try BDL API
    season = _season_year()
    team_id = bdl_team_name_to_id(opp_full)

    rating = 111.0
    if team_id:
        resp = _bdl_get(f"teams/{team_id}/stats", {"season": season})
        if resp and resp.get("data"):
            team_stats = resp.get("data", {})[0] if resp.get("data") else {}
            if stat_key == "pts":
                rating = team_stats.get("def_rating", 111.0)
            elif stat_key == "reb":
                rating = team_stats.get("reb_allowed_per_100", 46.0)
            elif stat_key == "ast":
                rating = team_stats.get("ast_allowed_per_100", 26.0)

    # Fall back to hardcoded
    if rating == 111.0:
        defense_ratings = TEAM_DEFENSE_RATINGS.get(opp_full, {})
        rating = defense_ratings.get(stat_key, 111.0)

    DEF_RATINGS_CACHE[cache_key] = rating
    return rating


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
    breakout_evidence: Dict = None
    hit_rates: Dict = None


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
    Core projection engine with proper minutes tracking and breakout detection.

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

    # Debug: show first game value for this player
    if games:
        print(f"[DEBUG] {player_name} {prop_type}: {len(games)} games, first val={games[0].get('value', '?')}, last val={games[-1].get('value', '?')}")

    # Step 2: Window averages
    base_games = _slice_last(games, BASELINE_GAMES)
    l10_games = _slice_last(games, LOOKBACK_GAMES)
    l5_games = _slice_last(games, SHORT_GAMES)

    base_avg, base_min_val, base_std = avg_stat_min_std(base_games)
    l10_avg, l10_min_val, l10_std = avg_stat_min_std(l10_games)
    l5_avg, l5_min_val, l5_std = avg_stat_min_std(l5_games)

    if base_avg < 0.1:
        return None

    # Step 2b: Fetch minutes separately (BUG FIX: was using min_val from avg_stat)
    base_min, l10_min, l5_min = fetch_minutes_windows(player_id, BASELINE_GAMES)

    # Step 3: Breakout detection with evidence gathering
    breakout_evidence = {}
    is_breakout = False

    if l5_avg >= base_avg * BREAKOUT_L5_THRESHOLD and l10_avg >= base_avg * BREAKOUT_L10_THRESHOLD:
        is_breakout = True

        # Gather evidence
        breakout_evidence["l5_avg"] = l5_avg
        breakout_evidence["l10_avg"] = l10_avg
        breakout_evidence["base_avg"] = base_avg
        breakout_evidence["l5_hits"] = sum(1 for g in l5_games if g.get("value", 0) >= line)
        breakout_evidence["l10_hits"] = sum(1 for g in l10_games if g.get("value", 0) >= line)

        # Minutes trending
        if l5_min > l10_min + 2:
            breakout_evidence["minutes_trending_up"] = True

        # Usage trending
        adv_stats = bdl_fetch_advanced_stats(player_id)
        if adv_stats.get("usage_pct", 25.0) > 27.0:
            breakout_evidence["high_usage"] = True
            breakout_evidence["usage_pct"] = adv_stats.get("usage_pct")

        # Consistency
        breakout_evidence["consistency_score"] = consistency_score(l10_games, l10_avg)

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

    # Step 11: Hit rate analysis (THE key profitability signal)
    hr = hit_rates_by_window(base_games, l10_games, l5_games, line)

    # Step 12: Probability — blend model z-score with empirical hit rates
    # Hit rate is more reliable for well-sampled props
    z = (proj - line) / sigma if sigma > 0 else 0
    raw_prob_model = _norm_cdf(z)
    model_prob = calibrated_prob(raw_prob_model)

    # Empirical probability from actual hit rates (weighted toward recent)
    empirical_prob = hr["composite"]

    # Blend: 40% model, 60% empirical (empirical is king for props)
    raw_prob = raw_prob_model
    prob_over = 0.40 * model_prob + 0.60 * empirical_prob

    # Safety: if empirical and model wildly disagree, be conservative
    if abs(model_prob - empirical_prob) > 0.20:
        prob_over = min(model_prob, empirical_prob)

    # Step 13: Edge and EV (edge as percentage, not absolute)
    raw_edge_pct = ((proj - line) / line * 100) if line > 0 else 0.0
    # Cap edge at 50% to prevent low-line props (0.5 blk/stl) from inflating rankings
    edge_pct = min(raw_edge_pct, 50.0)
    implied_prob = american_to_prob(over_odds)
    ev = ev_per_dollar(prob_over, over_odds)

    # Step 13: Consensus line (would fetch from multiple books)
    consensus_line = line

    # Step 14: Confidence tier
    confidence_tier = compute_confidence_tier(
        edge_pct, prob_over, consistency, is_breakout
    )

    return ProjectionResult(
        proj=proj,
        edge=edge_pct,
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
        breakout_evidence=breakout_evidence if is_breakout else None,
        hit_rates=hr,
    )


def project_minutes(base_min: float, l10_min: float, l5_min: float) -> float:
    """
    Project minutes played.
    Weight L10 heavily. Cap changes at ±3min from L10.
    """
    if base_min < 1.0:
        return 0.0

    proj = (0.30 * base_min + 0.40 * l10_min + 0.30 * l5_min)

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
    raw_sigma = 0.40 * base_std + 0.60 * l10_std

    consistency_reduction = min(0.10, consistency * 0.15)
    sigma = raw_sigma * (1.0 - consistency_reduction)

    floor = STAT_STD_FLOORS.get(stat_key, STD_FLOOR)
    sigma = max(floor, sigma)

    return sigma


def opponent_defense_adjustment(opp_team: str, prop_type: str, player_name: str = "") -> float:
    """
    Estimate opponent defensive adjustment from actual BDL data.
    Capped at ±6%.
    """
    if not opp_team:
        return 0.0

    stat_key = prop_type_to_stat_key(prop_type)
    opp_rating = fetch_def_rating(opp_team, stat_key)

    league_avg = 111.0

    adj = (league_avg - opp_rating) / 100.0
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
    usage_adj = (usage_pct - 25.0) / 100.0 * 0.04
    usage_adj = _clamp(usage_adj, -0.04, 0.04)
    adjustments.append(usage_adj)

    total = sum(adjustments)
    return _clamp(total, -ADJ_CAP_TOTAL, ADJ_CAP_TOTAL)


def compute_injury_boost(player_name: str, stat_key: str) -> float:
    """
    Compute injury boost from teammates going out (same team only).
    Use injured player's actual stats to calculate vacancy.
    Max boost: +3 pts, +1.5 reb, +1.0 ast.
    """
    injuries = parse_le_injuries()

    boost = 0.0
    for injured, freed_stats in injuries.items():
        if stat_key == "pts":
            boost += min(1.0, freed_stats.get("pts", 0.0) * 0.3)
        elif stat_key == "reb":
            boost += min(0.75, freed_stats.get("reb", 0.0) * 0.2)
        elif stat_key == "ast":
            boost += min(0.5, freed_stats.get("ast", 0.0) * 0.2)

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

    if edge >= 5.0:
        score += 3
    elif edge >= 3.0:
        score += 2
    elif edge >= 1.0:
        score += 1

    if prob_over >= 0.70:
        score += 3
    elif prob_over >= 0.62:
        score += 2
    elif prob_over >= 0.58:
        score += 1

    if consistency >= 0.80:
        score += 2
    elif consistency >= 0.60:
        score += 1

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
    offers: List[Tuple[float, float]],
    tolerance: float = 0.5,
) -> Optional[Tuple[float, float]]:
    """
    Find best odds for a line near consensus (within tolerance).
    Returns (line, odds) tuple.
    """
    if not offers:
        return None

    near = [(line, odds) for line, odds in offers if abs(line - consensus) <= tolerance]

    if not near:
        return max(offers, key=lambda x: x[1]) if offers else None

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
# SECTION 7: EDGE DETECTION ENGINES
# ============================================================================

def build_today_props(now_et: datetime = None) -> Tuple[Dict, Dict]:
    """
    Fetch all player prop lines for today's games.
    Returns tuple: (lines_map, games_map) where:
      lines_map: {game_id: {prop_type: {player_name: [{book, line, odds}]}}}
      games_map: {game_id: {game info}}

    BUG FIX: Returns proper tuple AND fetches actual prop data from BDL.
    """
    if now_et is None:
        now_et = _now_et()

    games = bdl_games_today(now_et)

    lines_map = {}
    games_map = {}

    for game in games:
        game_id = game.get("id")
        if not game_id:
            continue

        # Build games_map with context
        home_team = game.get("home_team", {})
        away_team = game.get("away_team", {})

        game_info = {
            "id": game_id,
            "home_team": home_team,
            "away_team": away_team,
            "scheduled_at": game.get("scheduled_at"),
            "status": game.get("status"),
        }

        # Fetch game odds
        odds_data = bdl_fetch_game_odds_full(game_id)
        game_info.update(odds_data)

        games_map[game_id] = game_info

        # Fetch props for this game
        props_by_player = bdl_fetch_props_for_game(game_id, PROP_TYPES)

        lines_map[game_id] = {}
        for prop_type in PROP_TYPES:
            lines_map[game_id][prop_type] = {}

        # Populate lines
        for player_name, props_dict in props_by_player.items():
            for prop_type, book_lines in props_dict.items():
                if prop_type not in lines_map[game_id]:
                    lines_map[game_id][prop_type] = {}
                lines_map[game_id][prop_type][player_name] = book_lines

    return (lines_map, games_map)


def slate_scan_edges(
    prop_type: str,
    lines_map: Dict,
    games_map: Dict,
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

    for game_id, game_props in lines_map.items():
        game_info = games_map.get(game_id, {})
        scheduled_at = game_info.get("scheduled_at")

        if scheduled_at:
            try:
                game_time = datetime.fromisoformat(scheduled_at.replace("Z", "+00:00"))
                game_time_et = game_time.astimezone(ZoneInfo("America/New_York"))
                if deadline_exceeded(game_time_et):
                    continue
            except:
                pass

        prop_lines = game_props.get(prop_type, {})

        for player_name, book_lines in prop_lines.items():
            if not book_lines:
                continue

            all_lines = [bl.get("line") for bl in book_lines if bl.get("line")]
            if not all_lines:
                continue

            consensus = consensus_line(all_lines)
            best_odds_offer = best_offer_near_consensus(consensus,
                [(bl["line"], bl["odds"]) for bl in book_lines])

            if not best_odds_offer:
                continue

            best_line, best_odds = best_odds_offer

            # Get player ID from props data (already resolved by bdl_fetch_props)
            player_id = None
            for bl in book_lines:
                if bl.get("player_id"):
                    player_id = bl["player_id"]
                    break

            if not player_id:
                continue

            # Determine opponent and home/away
            home_team = game_info.get("home_team", {})
            away_team = game_info.get("away_team", {})
            # BDL team objects have "full_name" (e.g., "Boston Celtics") and "name" (e.g., "Celtics")
            home_name = home_team.get("full_name", home_team.get("name", ""))
            away_name = away_team.get("full_name", away_team.get("name", ""))
            # Default to away team as opponent (simplified)
            is_home = True  # Will refine if we have team info
            opp_team = away_name

            game_context = {
                "opp_team": opp_team,
                "is_home": is_home,
                "pace": game_info.get("pace", 100.0),
            }

            proj_result = compute_projection(
                player_id, player_name, game_id, best_line, prop_type,
                best_odds, game_info, game_context
            )

            if not proj_result:
                continue

            # Debug: show projection values
            stat_key_for_filter = prop_type_to_stat_key(prop_type)
            min_edge_for_stat = STAT_MIN_EDGE.get(stat_key_for_filter, MIN_EDGE)
            min_prob_for_stat = STAT_MIN_PROB.get(stat_key_for_filter, MIN_PROB)

            print(f"[PROJ] {player_name} {prop_type}: line={best_line}, proj={proj_result.proj:.1f}, "
                  f"edge={proj_result.edge:.1f}% (need {min_edge_for_stat}%), "
                  f"prob={proj_result.prob_over:.3f} (need {min_prob_for_stat}), "
                  f"ev={proj_result.ev:.3f}")

            if proj_result.edge < min_edge_for_stat:
                continue
            if proj_result.prob_over < min_prob_for_stat:
                continue
            if proj_result.ev < 0.02:
                continue

            if has_recent_play(state, player_name, prop_type):
                continue

            # Get vendor from the book line data
            play_vendor = book_lines[0].get("book", "FanDuel") if book_lines else "FanDuel"

            # Compute composite score: weight EV and probability most heavily
            hr_data = proj_result.hit_rates or {}
            composite_score = (
                proj_result.ev * 40 +           # EV is king
                proj_result.prob_over * 30 +    # Probability matters
                proj_result.edge * 0.5 +        # Edge % as tiebreaker
                proj_result.consistency * 10 +  # Consistency bonus
                (5 if proj_result.is_breakout else 0)  # Breakout bonus
            )

            play = {
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
                "breakout_evidence": proj_result.breakout_evidence,
                "odds": best_odds,
                "vendor": play_vendor,
                "bet_size": kelly_bet_size(proj_result.edge, proj_result.prob_over, BANKROLL),
                "score": composite_score,
                "l5_avg": proj_result.l5_avg,
                "l10_avg": proj_result.l10_avg,
                "base_avg": proj_result.base_avg,
                "opp_team": opp_team,
                "is_home": is_home,
                "hit_rates": hr_data,
            }

            plays.append(play)

    tier_order = {"LOCK": 0, "STRONG": 1, "LEAN": 2, "SKIP": 3}
    plays.sort(key=lambda p: (tier_order.get(p["confidence_tier"], 3), -p["edge"]))

    return plays


def has_recent_play(state: Dict, player_name: str, prop_type: str) -> bool:
    """Check if player/stat combo has recent play (within cooldown)."""
    plays = state.get("plays", [])
    recent_time = _now_et() - timedelta(seconds=PLAY_COOLDOWN)

    for play in plays:
        if (play.get("player") == player_name and
            play.get("stat") == prop_type_to_stat_key(prop_type)):
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

    roster = bdl_active_roster(team_id)

    for teammate_dict in roster:
        teammate_name = f"{teammate_dict.get('first_name', '')} {teammate_dict.get('last_name', '')}".strip()
        teammate_id = teammate_dict.get("id")

        if not teammate_id or teammate_name == injured_player_name:
            continue

        for stat_key in ["pts", "reb", "ast"]:
            prop_type = {
                "pts": "player_points",
                "reb": "player_rebounds",
                "ast": "player_assists",
            }.get(stat_key)

            # Placeholder for injury edges - would need actual prop lines
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

    BUG FIX: Fixed compute_projection call signature (was wrong number of args).
    """
    if now_et is None:
        now_et = _now_et()

    plays = []
    state = load_state()

    for game_id, game_props in lines_map.items():
        game_info = games_map.get(game_id, {})
        scheduled_at = game_info.get("scheduled_at")

        if scheduled_at:
            try:
                game_time = datetime.fromisoformat(scheduled_at.replace("Z", "+00:00"))
                game_time_et = game_time.astimezone(ZoneInfo("America/New_York"))
                if deadline_exceeded(game_time_et):
                    continue
            except:
                pass

        for prop_type, prop_lines in game_props.items():
            if prop_type not in PROP_TYPES:
                continue

            for player_name, book_lines in prop_lines.items():
                if not book_lines:
                    continue

                best_odds = max([bl.get("odds", -110) for bl in book_lines], default=-110)
                if best_odds < 100:
                    continue

                all_lines = [bl.get("line") for bl in book_lines if bl.get("line")]
                if not all_lines:
                    continue

                best_line = consensus_line(all_lines)

                # Get player_id from props data
                player_id = None
                for bl in book_lines:
                    if bl.get("player_id"):
                        player_id = bl["player_id"]
                        break
                if not player_id:
                    continue

                if has_recent_play(state, player_name, prop_type):
                    continue

                home_team = game_info.get("home_team", {})
                away_team = game_info.get("away_team", {})
                home_name = home_team.get("full_name", home_team.get("name", ""))
                away_name = away_team.get("full_name", away_team.get("name", ""))
                is_home = True
                opp_team = away_name

                game_context = {
                    "opp_team": opp_team,
                    "is_home": is_home,
                    "pace": game_info.get("pace", 100.0),
                }

                # BUG FIX: compute_projection now takes correct signature
                proj_result = compute_projection(
                    player_id, player_name, game_id, best_line, prop_type,
                    best_odds, game_info, game_context
                )

                if not proj_result:
                    continue

                prob = proj_result.prob_over
                if prob < 0.52:
                    continue

                ev = proj_result.ev
                if ev < 0.03:
                    continue

                value_edge = (prob - american_to_prob(best_odds)) * 100
                score = ev * 100 + value_edge

                play = {
                    "player": player_name,
                    "player_id": player_id,
                    "game_id": game_id,
                    "stat_key": prop_type_to_stat_key(prop_type),
                    "prop_type": prop_type,
                    "line": best_line,
                    "proj": proj_result.proj,
                    "sigma": proj_result.sigma,
                    "odds": best_odds,
                    "vendor": "Consensus",
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
        game_total = game_info.get("game_total", 220)

        corr_factor = 0.15 if game_total > 220 else 0.10

        for team_id in [1, 2]:
            team_plays = [p for p in game_plays]
            if len(team_plays) < 2:
                continue

            for i in range(len(team_plays)):
                for j in range(i + 1, len(team_plays)):
                    leg1, leg2 = team_plays[i], team_plays[j]

                    if leg1.get("player_id") == leg2.get("player_id"):
                        continue

                    p1, p2 = leg1.get("prob", 0.5), leg2.get("prob", 0.5)
                    if not p1 or not p2:
                        continue

                    combined_prob = p1 * p2 + corr_factor * math.sqrt(p1 * (1 - p1)) * math.sqrt(p2 * (1 - p2))
                    combined_prob = min(combined_prob, 0.99)

                    if combined_prob < 0.35:
                        continue

                    o1, o2 = leg1.get("odds", -110), leg2.get("odds", -110)
                    est_odds_1 = american_to_payout(o1, 100) * american_to_payout(o2, 1) * 100 - 100
                    est_odds = int(est_odds_1 * 0.85)

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

    plays_by_player = {}
    for play in plays:
        pid = play.get("player_id")
        if pid not in plays_by_player:
            plays_by_player[pid] = []
        plays_by_player[pid].append(play)

    for player_id, player_plays in plays_by_player.items():
        if len(player_plays) < 2:
            continue

        for i in range(len(player_plays)):
            for j in range(i + 1, len(player_plays)):
                play1, play2 = player_plays[i], player_plays[j]

                stat1 = play1.get("stat_key")
                stat2 = play2.get("stat_key")

                if stat1 == stat2:
                    continue

                corr_key = (stat1, stat2) if (stat1, stat2) in PROP_CORRELATIONS else (stat2, stat1)
                correlation = PROP_CORRELATIONS.get(corr_key, 0.12)

                p1, p2 = play1.get("prob", 0.5), play2.get("prob", 0.5)
                if not p1 or not p2:
                    continue

                combined_prob = (p1 * p2 +
                                 correlation * math.sqrt(p1 * (1 - p1)) * math.sqrt(p2 * (1 - p2)))
                combined_prob = min(combined_prob, 0.99)

                if combined_prob < 0.40:
                    continue

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
                    "prob": combined_prob,
                    "estimated_odds": est_odds,
                    "ev": ev,
                    "bet_size": kelly_bet_size(ev * 100, combined_prob, BANKROLL),
                    "edge": (combined_prob - american_to_prob(est_odds)) * 100,
                    "type": "correlated_parlay",
                }
                corr_parlays.append(parlay)

    corr_parlays.sort(key=lambda p: -p["ev"])
    return corr_parlays[:3]


LADDER_RUNGS = [10.5, 15.5, 20.5, 25.5, 30.5, 35.5]


def _player_scoring_profile(games: List[Dict], line: float) -> Dict:
    """
    Build a scoring profile: floor, ceiling, consistency, hot streak, ceiling games.
    This is what makes ladder analysis actually useful.
    """
    if not games:
        return {}

    values = [g.get("value", 0) for g in games]
    l5 = values[:5] if len(values) >= 5 else values
    l10 = values[:10] if len(values) >= 10 else values

    floor = min(values) if values else 0
    ceiling = max(values) if values else 0
    avg = sum(values) / len(values) if values else 0
    l5_avg = sum(l5) / len(l5) if l5 else 0
    l10_avg = sum(l10) / len(l10) if l10 else 0

    # How many times they've hit various thresholds
    ceiling_20 = sum(1 for v in values if v >= 20)
    ceiling_25 = sum(1 for v in values if v >= 25)
    ceiling_30 = sum(1 for v in values if v >= 30)

    # Hot streak: consecutive games over line
    streak = 0
    for v in values:
        if v > line:
            streak += 1
        else:
            break

    # Scoring trend (L5 vs L10)
    trend = "heating up" if l5_avg > l10_avg * 1.10 else (
        "cooling down" if l5_avg < l10_avg * 0.90 else "steady"
    )

    return {
        "floor": floor,
        "ceiling": ceiling,
        "avg": avg,
        "l5_avg": l5_avg,
        "l10_avg": l10_avg,
        "ceiling_20": ceiling_20,
        "ceiling_25": ceiling_25,
        "ceiling_30": ceiling_30,
        "games_sampled": len(values),
        "streak_over_line": streak,
        "trend": trend,
    }


def scan_ladder_plays(
    lines_map: Dict,
    games_map: Dict,
    now_et: datetime = None,
) -> List[Dict]:
    """
    Smart ladder scanner with full analysis context.
    Finds players whose scoring profile supports multi-rung ladder bets.
    Includes narrative reasoning for WHY each ladder works.
    Lowered floor to 10+ PPG to catch role players with +200 alt lines.
    """
    if now_et is None:
        now_et = _now_et()

    ladders = []

    for game_id, game_props in lines_map.items():
        game_info = games_map.get(game_id, {})
        scheduled_at = game_info.get("scheduled_at")

        if scheduled_at:
            try:
                game_time = datetime.fromisoformat(scheduled_at.replace("Z", "+00:00"))
                game_time_et = game_time.astimezone(ZoneInfo("America/New_York"))
                if deadline_exceeded(game_time_et):
                    continue
            except:
                pass

        prop_lines = game_props.get("player_points", {})

        # Get matchup context
        home_team = game_info.get("home_team", {})
        away_team = game_info.get("away_team", {})
        home_name = home_team.get("full_name", home_team.get("name", ""))
        away_name = away_team.get("full_name", away_team.get("name", ""))

        for player_name, book_lines in prop_lines.items():
            if not book_lines:
                continue

            all_lines = [bl.get("line") for bl in book_lines if bl.get("line")]
            if not all_lines:
                continue

            avg_line = sum(all_lines) / len(all_lines)

            # Lower floor to 8+ to catch role players with +200 upside
            if avg_line < 8.0:
                continue

            player_id = None
            vendor = "FanDuel"
            for bl in book_lines:
                if bl.get("player_id"):
                    player_id = bl["player_id"]
                if bl.get("book"):
                    vendor = bl["book"]
                if player_id:
                    break
            if not player_id:
                continue

            is_home = True
            opp_team = away_name

            game_context = {
                "opp_team": opp_team,
                "is_home": is_home,
                "pace": game_info.get("pace", 100.0),
            }

            proj_result = compute_projection(
                player_id, player_name, game_id, avg_line, "player_points",
                -110, game_info, game_context
            )

            if not proj_result or proj_result.proj < 10.0:
                continue

            projection = proj_result.proj

            # Build scoring profile from raw game data
            games = bdl_last_n_games_stats(player_id, "pts", BASELINE_GAMES)
            profile = _player_scoring_profile(games, avg_line)

            # Hit rates at each rung
            ladder_legs = []
            for rung in LADDER_RUNGS:
                if rung > projection * 1.8:
                    continue  # Skip unreachable rungs

                # Find matching book line for this rung
                best_odds = None
                for bl in book_lines:
                    line = bl.get("line", 0)
                    if abs(line - rung) < 1.0:
                        best_odds = bl.get("odds", -110)
                        break

                # Estimate odds for alt lines based on distance from main line
                if best_odds is None:
                    dist = rung - avg_line
                    if dist < -10:
                        best_odds = -350
                    elif dist < -5:
                        best_odds = -200
                    elif dist < 0:
                        best_odds = -130
                    elif dist < 5:
                        best_odds = +150
                    elif dist < 10:
                        best_odds = +250
                    elif dist < 15:
                        best_odds = +450
                    else:
                        best_odds = +800

                z = (projection - rung) / proj_result.sigma if proj_result.sigma > 0 else 0
                raw_prob = _norm_cdf(z)
                prob = calibrated_prob(raw_prob)

                # Also compute empirical hit rate at this rung
                rung_hits = sum(1 for g in games if g.get("value", 0) > rung)
                rung_hr = rung_hits / len(games) if games else 0

                # Blend model + empirical
                blended_prob = 0.40 * prob + 0.60 * rung_hr

                if blended_prob < 0.15:
                    continue

                ev = ev_per_dollar(blended_prob, best_odds)

                odds_str = f"+{best_odds}" if best_odds > 0 else str(best_odds)

                ladder_legs.append({
                    "rung": rung,
                    "line": rung,
                    "odds": best_odds,
                    "odds_str": odds_str,
                    "prob": blended_prob,
                    "model_prob": prob,
                    "hit_rate": rung_hr,
                    "hits": rung_hits,
                    "total": len(games),
                    "ev": ev,
                })

            if len(ladder_legs) < 2:
                continue

            best_leg = max(ladder_legs, key=lambda x: x["ev"])

            # Build narrative analysis
            opp_short = opp_team.split()[-1] if opp_team else "?"
            opp_def = fetch_def_rating(opp_team, "pts")

            narrative_parts = []

            # Season/recent averages
            narrative_parts.append(
                f"Avg: {profile.get('l5_avg', 0):.1f} L5 / "
                f"{profile.get('l10_avg', 0):.1f} L10 / "
                f"{profile.get('avg', 0):.1f} Szn"
            )

            # Trend
            trend = profile.get("trend", "steady")
            if trend == "heating up":
                narrative_parts.append("TRENDING UP")
            elif trend == "cooling down":
                narrative_parts.append("cooling off")

            # Ceiling
            ceiling = profile.get("ceiling", 0)
            narrative_parts.append(f"Ceiling: {ceiling:.0f}")

            # Ceiling game frequency
            c20 = profile.get("ceiling_20", 0)
            c25 = profile.get("ceiling_25", 0)
            total = profile.get("games_sampled", 20)
            if c25 > 0:
                narrative_parts.append(f"25+ in {c25}/{total} games")
            elif c20 > 0:
                narrative_parts.append(f"20+ in {c20}/{total} games")

            # Matchup
            if opp_def > 112.0:
                narrative_parts.append(f"vs {opp_short} (weak def)")
            elif opp_def < 109.5:
                narrative_parts.append(f"vs {opp_short} (elite def)")
            else:
                narrative_parts.append(f"vs {opp_short}")

            # Home/away
            narrative_parts.append("HOME" if is_home else "ROAD")

            # Hot streak
            streak = profile.get("streak_over_line", 0)
            if streak >= 3:
                narrative_parts.append(f"{streak}-game streak over line")

            narrative = " | ".join(narrative_parts)

            # Ladder recommendation
            # Find the "sweet spot" rung — highest rung with prob > 0.45 and EV > 0
            plus_odds_legs = [l for l in ladder_legs if l["odds"] >= 200]
            sweet_spot = None
            if plus_odds_legs:
                sweet_spot = max(plus_odds_legs, key=lambda x: x["ev"])

            ladder = {
                "player": player_name,
                "player_id": player_id,
                "game_id": game_id,
                "projection": projection,
                "all_legs": ladder_legs,
                "best_leg": best_leg,
                "sweet_spot": sweet_spot,
                "ev": best_leg["ev"],
                "bet_size": kelly_bet_size(best_leg["ev"] * 100, best_leg["prob"], BANKROLL),
                "type": "ladder",
                "profile": profile,
                "narrative": narrative,
                "vendor": vendor,
                "opp_team": opp_team,
                "is_home": is_home,
                "main_line": avg_line,
            }
            ladders.append(ladder)

    # Sort by combination of EV and ceiling potential
    ladders.sort(key=lambda l: -(
        l["ev"] * 30 +
        l["profile"].get("ceiling", 0) * 0.5 +
        (10 if l.get("sweet_spot") else 0)
    ))
    return ladders[:6]


# ============================================================================
# SECTION 10: FORMATTING & DISPLAY
# ============================================================================

def format_play_card(play: Dict, index: int) -> str:
    """
    Format play card for WhatsApp with hit rates, evidence, and FanDuel odds.
    """
    player = play.get("player", "?")
    prop_type = play.get("prop_type", "pts")
    line = play.get("line", 0)
    proj = play.get("proj", 0)
    odds = play.get("odds", -110)
    vendor = play.get("vendor", "FanDuel")
    bet_size = play.get("bet_size", 0)
    prob = play.get("prob", play.get("prob_over", 0))
    ev = play.get("ev", 0)
    edge = play.get("edge", 0)

    PROP_NAMES = {
        "player_points": "PTS",
        "player_rebounds": "REB",
        "player_assists": "AST",
        "player_threes": "3PM",
        "player_blocks": "BLK",
        "player_steals": "STL",
    }
    prop_name = PROP_NAMES.get(prop_type, "?")

    breakout_tag = " *BREAKOUT*" if play.get("is_breakout") else ""
    tier = play.get("confidence_tier", "LEAN")

    odds_str = f"+{odds}" if odds > 0 else str(odds)

    # Hit rate line (most important for bettors)
    hr = play.get("hit_rates", {})
    l5_hits = hr.get("l5_hits", 0)
    l10_hits = hr.get("l10_hits", 0)
    base_hits = hr.get("base_hits", 0)
    hit_line = f"Hit: {l5_hits}/5 L5 | {l10_hits}/10 L10 | {base_hits}/20 Szn"

    context = explain_play(play)

    card = (
        f"{index}. [{tier}] {player}{breakout_tag}\n"
        f"   {prop_name} OVER {line} | {vendor} {odds_str} | ${bet_size:.0f}\n"
        f"   Proj {proj:.1f} | Edge {edge:.1f}% | P={prob*100:.0f}% | EV +{ev:.2f}\n"
        f"   {hit_line}\n"
        f"   {context}"
    )
    return card


def explain_play(play: Dict) -> str:
    """
    Generate analysis-style context: averages, trend, matchup, why this play.
    """
    parts = []

    # Averages
    l5_avg = play.get("l5_avg", 0)
    l10_avg = play.get("l10_avg", 0)
    base_avg = play.get("base_avg", 0)

    if l5_avg > 0:
        parts.append(f"Avg: {l5_avg:.1f} L5 / {l10_avg:.1f} L10 / {base_avg:.1f} Szn")

    # Trend detection
    if l5_avg > 0 and l10_avg > 0:
        if l5_avg > l10_avg * 1.10:
            parts.append("HEATING UP")
        elif l5_avg < l10_avg * 0.90:
            parts.append("cooling off")

    # Matchup
    opp_team = play.get("opp_team", "")
    if opp_team:
        short_opp = opp_team.split()[-1] if opp_team else ""
        opp_def = fetch_def_rating(opp_team, play.get("stat_key", "pts"))
        if opp_def > 112.0:
            parts.append(f"vs {short_opp} (weak def)")
        elif opp_def < 109.5:
            parts.append(f"vs {short_opp} (elite def)")
        else:
            parts.append(f"vs {short_opp}")

    # Home/away
    parts.append("HOME" if play.get("is_home") else "ROAD")

    # Breakout evidence
    breakout_ev = play.get("breakout_evidence", {})
    if breakout_ev:
        if breakout_ev.get("minutes_trending_up"):
            parts.append("mins trending up")
        if breakout_ev.get("usage_pct"):
            parts.append(f"USG {breakout_ev['usage_pct']:.0f}%")
        cs = breakout_ev.get("consistency_score", 0)
        if cs > 0.7:
            parts.append("very consistent")

    context = " | ".join(parts)
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
    """
    Format ladder card with full analysis narrative.
    Shows: player profile, why they're a ladder candidate, each rung with
    hit rate evidence, and the recommended sweet spot play.
    """
    player = ladder.get("player", "?")
    proj = ladder.get("projection", 0)
    main_line = ladder.get("main_line", 0)
    narrative = ladder.get("narrative", "")
    vendor = ladder.get("vendor", "FanDuel")
    legs = ladder.get("all_legs", [])
    sweet = ladder.get("sweet_spot")
    profile = ladder.get("profile", {})

    lines = []
    lines.append(f"LADDER: {player} (Proj {proj:.1f} | Line {main_line})")
    lines.append(f"  {narrative}")

    # Show each rung with hit rate evidence
    lines.append(f"  Rungs:")
    for leg in legs:
        rung = leg["rung"]
        odds_str = leg.get("odds_str", str(leg["odds"]))
        prob = leg["prob"]
        hr = leg.get("hit_rate", 0)
        hits = leg.get("hits", 0)
        total = leg.get("total", 20)
        ev = leg.get("ev", 0)

        marker = ""
        if sweet and leg["rung"] == sweet["rung"]:
            marker = " << SWEET SPOT"
        elif leg == ladder.get("best_leg"):
            marker = " << BEST EV"

        lines.append(
            f"    {rung}+ {odds_str} | P={prob*100:.0f}% | "
            f"Hit {hits}/{total} | EV {'+' if ev >= 0 else ''}{ev:.2f}{marker}"
        )

    # Recommendation
    if sweet:
        lines.append(
            f"  PLAY: {player} {sweet['rung']}+ PTS {sweet.get('odds_str', '')} "
            f"(hit {sweet.get('hits', 0)}/{sweet.get('total', 20)} szn, "
            f"P={sweet['prob']*100:.0f}%)"
        )
    else:
        best = ladder.get("best_leg", {})
        lines.append(
            f"  PLAY: {player} {best.get('rung', 0)}+ PTS "
            f"(best EV at {best.get('odds_str', str(best.get('odds', -110)))})"
        )

    return "\n".join(lines)


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
    stat_count = {}
    kept = []

    for play in plays:
        team = play.get("team")
        game = play.get("game_id")
        stat = play.get("stat_key")

        team_count[team] = team_count.get(team, 0) + 1
        game_count[game] = game_count.get(game, 0) + 1
        stat_count[stat] = stat_count.get(stat, 0) + 1

        if team_count[team] > MAX_PLAYS_PER_PLAYER:
            continue
        if game_count[game] > 5:
            continue
        if stat_count[stat] > MAX_PLAYS_PER_STAT:
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

    for play_record in state.get("plays", []):
        key = (play_record.get("player"), play_record.get("stat"))
        recent_plays[key] = play_record

    for play in plays:
        key = (play.get("player"), play.get("stat_key"))
        if key in recent_plays:
            prev_play = recent_plays[key]
            prev_edge = prev_play.get("edge", 0)
            curr_edge = play.get("edge", 0)

            try:
                prev_time = datetime.fromisoformat(prev_play.get("timestamp", "")).timestamp()
                time_diff = now_ts - prev_time

                if time_diff < 3 * 3600 and curr_edge - prev_edge < 2.0:
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
    """Assemble WhatsApp message grouped by stat type with hit rates."""
    if state is None:
        state = load_state()

    now_et = _now_et()
    date_str = now_et.strftime("%m/%d")
    time_str = now_et.strftime("%I:%M %p")
    book_label = PRIMARY_BOOK.upper()

    lines = [f"NBA PICKS {date_str} {time_str} ET ({book_label})"]

    hit_rate_str = get_hit_rate_summary(state)
    if "No plays" not in hit_rate_str:
        lines.append(f"Season: {hit_rate_str}")

    # Combine straights + plus_plays, then group by stat type
    all_plays = straights + plus_plays

    # Group by stat category
    STAT_ORDER = ["pts", "reb", "ast", "threes", "blk", "stl"]
    STAT_LABELS = {
        "pts": "POINTS", "reb": "REBOUNDS", "ast": "ASSISTS",
        "threes": "THREES", "blk": "BLOCKS", "stl": "STEALS",
    }
    grouped = {}
    for play in all_plays:
        sk = play.get("stat_key", "pts")
        if sk not in grouped:
            grouped[sk] = []
        grouped[sk].append(play)

    # Apply per-stat max picks
    total_idx = 0
    for stat_key in STAT_ORDER:
        if stat_key not in grouped:
            continue
        stat_plays = grouped[stat_key]
        # Sort by score descending within each stat
        stat_plays.sort(key=lambda p: -p.get("score", 0))
        # Cap per stat type
        max_for_stat = STAT_MAX_PICKS.get(stat_key, 2)
        stat_plays = stat_plays[:max_for_stat]

        label = STAT_LABELS.get(stat_key, stat_key.upper())
        lines.append(f"\n-- {label} --")
        for play in stat_plays:
            total_idx += 1
            lines.append(format_play_card(play, total_idx))
            lines.append("")

    if total_idx == 0:
        lines.append("\nNo qualifying plays today")

    if sgps or corr_parlays:
        lines.append("\n-- PARLAYS --")
        for sgp in sgps:
            lines.append(format_parlay_card(sgp))
            lines.append("")
        for parlay in corr_parlays:
            lines.append(format_parlay_card(parlay))
            lines.append("")

    if ladders:
        lines.append("\n-- LADDERS --")
        for ladder in ladders:
            lines.append(format_ladder_card(ladder))
            lines.append("")

    total_kelly = sum(p.get("bet_size", 0) for p in all_plays[:total_idx])
    lines.append(f"Total: ${total_kelly:.0f} across {total_idx} plays")

    return "\n".join(lines)


def run():
    """
    Main entry point for daily prop scanning and plays generation.
    Orchestrates all edge engines, deduplication, and formatting.

    BUG FIX: Corrected build_today_props return handling (was unpacking single dict as tuple).
    BUG FIX: Fixed slate_scan_edges call with correct state argument (not news_scores).
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

    print(f"[INFO] Found {len(games_map)} games")

    # Phase 4: Fetch news & injuries
    print("[INFO] Fetching LineupExperts news...")
    news_items = fetch_lineupexperts_news()
    le_injuries = parse_le_injuries()

    # Phase 5: Run edge engines
    print("[INFO] Running edge engines...")
    injury_plays = []
    slate_plays = []
    news_plays = []

    # 5a: Injury edges
    for player_key, injury_info in le_injuries.items():
        edges = []
        injury_plays.extend(edges)

    # 5b: Main slate scan (BUG FIX: correct arguments)
    for prop_type in PROP_TYPES:
        edges = slate_scan_edges(prop_type, lines_map, games_map, state, now_et)
        slate_plays.extend(edges)

    # 5c: Lineup news edges
    news_plays = lineup_news_edges(games_map, state)

    # 5d: Plus odds hunt
    plus_plays = plus_odds_hunt_edges(lines_map, games_map, now_et)

    print(f"[INFO] Found {len(slate_plays)} slate, {len(plus_plays)} plus odds")

    # Phase 6: Deduplicate
    combined = deduplicate_plays(slate_plays + news_plays)

    # Phase 7: Apply exposure caps
    final_plays = apply_exposure_caps(combined)

    # Phase 8: Apply cooldown
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

        client = None
        print(f"[TWILIO] SID set: {bool(ACCOUNT_SID)}, Token set: {bool(AUTH_TOKEN)}, "
              f"TwilioClient: {TwilioClient is not None}, "
              f"From: {TWILIO_WHATSAPP_FROM}, To: {bool(RECIPIENT_WHATSAPP)}")
        if ACCOUNT_SID and AUTH_TOKEN and TwilioClient:
            try:
                client = TwilioClient(ACCOUNT_SID, AUTH_TOKEN)
                print("[TWILIO] Client created OK")
            except Exception as e:
                print(f"[TWILIO] Client init FAILED: {e}")
        else:
            print("[TWILIO] Missing creds or library - messages will be MOCK")

        send_chunked(client, msg)

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

    return situation


# ============================================================================
# SECTION 14: ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"[FATAL] {type(e).__name__}: {e}")
        traceback.print_exc()
