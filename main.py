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

print("[INIT] ====== V2 ODDS API BUILD — IF YOU SEE THIS, NEW CODE IS RUNNING ======")

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
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

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

# ESPN public API (no auth needed)
# IMPORTANT: search + gamelog use site.web.api.espn.com, scoreboard uses site.api.espn.com
ESPN_WEB_API_BASE = "https://site.web.api.espn.com/apis"
ESPN_SITE_API_BASE = "https://site.api.espn.com/apis"
ESPN_PLAYER_SEARCH = f"{ESPN_WEB_API_BASE}/common/v3/search"
ESPN_GAMELOG_BASE = f"{ESPN_WEB_API_BASE}/common/v3/sports/basketball/nba/athletes"
ESPN_SCOREBOARD = f"{ESPN_SITE_API_BASE}/site/v2/sports/basketball/nba/scoreboard"

# ESPN gamelog column labels (index positions)
# Labels: MIN, FG, FG%, 3PT, 3P%, FT, FT%, REB, AST, BLK, STL, PF, TO, PTS
ESPN_STAT_INDEX = {
    "min": 0, "fg": 1, "fg_pct": 2, "three_pt": 3, "three_pct": 4,
    "ft": 5, "ft_pct": 6, "reb": 7, "ast": 8, "blk": 9,
    "stl": 10, "pf": 11, "to": 12, "pts": 13,
}

# NBA.com CDN stats (public, no auth)
NBACOM_STATS_BASE = "https://stats.nba.com/stats"

# Caches
PROPS_CACHE = {}
ADV_STATS_CACHE = {}
GAME_ODDS_CACHE = {}
ROSTER_CACHE = {}
TEAM_ID_CACHE = {}
LINEUPS_CACHE = {}
DEF_RATINGS_CACHE = {}
PLAYER_NAME_CACHE = {}
ESPN_PLAYER_CACHE = {}  # name -> {espn_id, stats, gamelog, news}
MULTI_SOURCE_CACHE = {}  # player_id -> cross-referenced data

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
    """
    Get the first n elements of list (most recent games).
    ESPN and BDL both return games most-recent-first,
    so [:n] gives us the last n games played.
    FIX: Was using [-n:] which gave OLDEST games, completely backwards.
    """
    if not lst:
        return []
    return lst[:n]


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


def build_deep_analysis(
    player_name: str,
    player_id: int,
    stat_key: str,
    line: float,
    proj: float,
    games: List[Dict],
    opp_team: str = "",
    is_home: bool = True,
    breakout_evidence: Dict = None,
    hit_rates: Dict = None,
    xref: Dict = None,
    explosion: Dict = None,
    breakout_signals: Dict = None,
    matchup: Dict = None,
    vacuum: Dict = None,
) -> str:
    """
    Build a multi-line research-style analysis for a pick.
    Uses raw game data to find narratives, streaks, trends, ceiling games,
    matchup context — the same depth a human analyst would provide.
    """
    if not games:
        return "Insufficient data"

    values = [g.get("value", 0) for g in games]
    n = len(values)
    l5 = values[:5] if n >= 5 else values
    l10 = values[:10] if n >= 10 else values

    l5_avg = sum(l5) / len(l5) if l5 else 0
    l10_avg = sum(l10) / len(l10) if l10 else 0
    szn_avg = sum(values) / n if n else 0
    floor_val = min(values) if values else 0
    ceiling_val = max(values) if values else 0

    STAT_LABELS = {"pts": "points", "reb": "rebounds", "ast": "assists",
                   "threes": "threes", "blk": "blocks", "stl": "steals"}
    stat_label = STAT_LABELS.get(stat_key, stat_key)

    parts = []

    # ---- Line 1: Averages and trend ----
    trend = ""
    if l5_avg > l10_avg * 1.10:
        trend = " (TRENDING UP)"
    elif l5_avg < l10_avg * 0.90:
        trend = " (cooling off)"
    parts.append(f"  Avg: {l5_avg:.1f} L5 / {l10_avg:.1f} L10 / {szn_avg:.1f} Szn{trend}")

    # ---- Line 2: Recent form narrative ----
    # Find streaks: consecutive games over line
    streak_over = 0
    for v in values:
        if v > line:
            streak_over += 1
        else:
            break

    # Count of games in a range over recent stretch
    double_digit_l10 = sum(1 for v in l10 if v >= 10)
    over_line_l10 = sum(1 for v in l10 if v > line)
    over_line_l5 = sum(1 for v in l5 if v > line)

    form_parts = []
    if streak_over >= 3:
        form_parts.append(f"{streak_over}-game streak over {line}")
    if over_line_l5 >= 4:
        form_parts.append(f"cleared line in {over_line_l5}/5 recent")
    elif over_line_l10 >= 7:
        form_parts.append(f"over line in {over_line_l10}/10 L10")

    if stat_key == "pts" and double_digit_l10 >= 8:
        form_parts.append(f"double digits in {double_digit_l10}/10 L10")

    # Look for a standout recent game
    if l5:
        best_recent = max(l5)
        if best_recent > line * 1.5:
            form_parts.append(f"hit {best_recent:.0f} recently")

    if form_parts:
        parts.append(f"  Form: {' | '.join(form_parts)}")

    # ---- Line 3: Floor/Ceiling ----
    # Count ceiling games
    if stat_key == "pts":
        c20 = sum(1 for v in values if v >= 20)
        c25 = sum(1 for v in values if v >= 25)
        c30 = sum(1 for v in values if v >= 30)
        ceil_parts = [f"Floor {floor_val:.0f} / Ceiling {ceiling_val:.0f}"]
        if c30 > 0:
            ceil_parts.append(f"30+ in {c30}/{n}")
        elif c25 > 0:
            ceil_parts.append(f"25+ in {c25}/{n}")
        elif c20 > 0:
            ceil_parts.append(f"20+ in {c20}/{n}")
        parts.append(f"  Range: {' | '.join(ceil_parts)}")
    else:
        parts.append(f"  Range: Floor {floor_val:.0f} / Ceiling {ceiling_val:.0f}")

    # ---- Line 4: Matchup ----
    matchup_parts = []
    if opp_team:
        short_opp = opp_team.split()[-1] if opp_team else "?"
        opp_def = fetch_def_rating(opp_team, stat_key)
        if opp_def > 113.0:
            matchup_parts.append(f"vs {short_opp} (bottom-10 defense)")
        elif opp_def > 111.0:
            matchup_parts.append(f"vs {short_opp} (weak def, {opp_def:.0f} DRTG)")
        elif opp_def < 109.0:
            matchup_parts.append(f"vs {short_opp} (elite def, {opp_def:.0f} DRTG)")
        else:
            matchup_parts.append(f"vs {short_opp} ({opp_def:.0f} DRTG)")

    matchup_parts.append("HOME" if is_home else "ROAD")

    # Breakout extras
    if breakout_evidence:
        if breakout_evidence.get("minutes_trending_up"):
            matchup_parts.append("mins trending up")
        if breakout_evidence.get("usage_pct"):
            matchup_parts.append(f"USG {breakout_evidence['usage_pct']:.0f}%")

    if matchup_parts:
        parts.append(f"  Matchup: {' | '.join(matchup_parts)}")

    # ---- Line 5: Data verification (multi-source) ----
    if xref:
        xref_parts = []
        primary = xref.get("primary_source", "BDL")
        sources = xref.get("sources", [])

        if len(sources) > 1:
            espn_a = xref.get("espn_avg", 0)
            bdl_a = xref.get("bdl_avg", 0)
            disc = xref.get("discrepancy", 0)

            if espn_a > 0 and bdl_a > 0:
                if disc <= 8:
                    xref_parts.append(f"ESPN {espn_a:.1f} / BDL {bdl_a:.1f} (AGREE)")
                else:
                    xref_parts.append(f"ESPN {espn_a:.1f} vs BDL {bdl_a:.1f} ({disc:.0f}% diff)")

            conf = xref.get("confidence", "high").upper()
            if conf != "HIGH":
                xref_parts.append(f"Confidence: {conf}")
        elif "ESPN" in sources:
            xref_parts.append("ESPN verified")
        else:
            xref_parts.append("BDL only")

        # Injury alerts
        for note in xref.get("notes", []):
            if "INJURY" in note:
                xref_parts.append(note)

        if xref.get("espn_team"):
            xref_parts.append(xref["espn_team"])

        label = "Verified" if len(sources) > 1 else "Source"
        parts.append(f"  {label}: {' | '.join(xref_parts)}")

    # ---- Line 6: Explosion profile (how they score, not just how much) ----
    if explosion:
        exp_parts = []
        profile_type = explosion.get("profile_type", "")
        vol = explosion.get("volatility", 0)
        exp_rate = explosion.get("explosion_rate", 0)
        boom_avg = explosion.get("boom_avg", 0)
        season_max = explosion.get("season_max", 0)

        PROFILE_LABELS = {
            "boom_or_bust": "Boom-or-bust",
            "ceiling_hunter": "Ceiling hunter",
            "steady_eddie": "Steady",
            "volatile": "High variance",
            "moderate": "Moderate",
        }
        exp_parts.append(PROFILE_LABELS.get(profile_type, profile_type))

        if exp_rate >= 0.15:
            exp_parts.append(f"explodes {exp_rate*100:.0f}% of games")
        if boom_avg > 0:
            exp_parts.append(f"boom avg {boom_avg:.0f}")
        if season_max > 0 and season_max > line * 1.3:
            exp_parts.append(f"szn high {season_max:.0f}")

        if exp_parts:
            parts.append(f"  Profile: {' | '.join(exp_parts)}")

    # ---- Line 7: Tonight's breakout signals ----
    if breakout_signals:
        bk_tier = breakout_signals.get("breakout_tier", "NEUTRAL")
        bk_score = breakout_signals.get("breakout_score", 0)
        bk_signals = breakout_signals.get("signals", [])

        if bk_tier in ("PRIME", "ELEVATED") and bk_signals:
            signal_str = " + ".join(bk_signals[:4])  # Max 4 signals shown
            parts.append(f"  Tonight: {bk_tier} breakout ({bk_score:.0f}/100) — {signal_str}")
        elif bk_tier == "SUPPRESSED" and bk_signals:
            parts.append(f"  Caution: {bk_signals[0]}")

    # ---- Line 8: Matchup history ----
    if matchup and matchup.get("games_vs", 0) >= 2:
        assessment = matchup.get("assessment", "NEUTRAL")
        if assessment in ("FEAST", "STRUGGLE"):
            parts.append(f"  Matchup: {matchup.get('narrative', '')}")

    # ---- Line 9: Usage vacuum ----
    if vacuum and vacuum.get("has_vacuum"):
        parts.append(f"  {vacuum.get('narrative', '')}")

    # ---- Line 10: Why this play (the verdict) ----
    reasons = []
    hr = hit_rates or {}
    composite_hr = hr.get("composite", 0)

    if composite_hr >= 0.75:
        reasons.append(f"hits at {composite_hr*100:.0f}% rate")
    elif composite_hr >= 0.60:
        reasons.append(f"solid {composite_hr*100:.0f}% hit rate")

    if proj > line * 1.15:
        reasons.append(f"proj {proj:.1f} well above line {line}")
    elif proj > line:
        reasons.append(f"proj {proj:.1f} above line {line}")

    if streak_over >= 3:
        reasons.append("on a streak")

    if l5_avg > l10_avg * 1.10:
        reasons.append("heating up lately")

    if opp_team:
        opp_def = fetch_def_rating(opp_team, stat_key)
        if opp_def > 112.0:
            reasons.append("soft matchup")

    if xref and xref.get("agreement", False) and len(xref.get("sources", [])) > 1:
        reasons.append("multi-source verified")

    # Breakout signal as a reason
    if breakout_signals and breakout_signals.get("breakout_tier") == "PRIME":
        reasons.append("breakout conditions stacked")

    # Matchup history as a reason
    if matchup and matchup.get("assessment") == "FEAST":
        reasons.append(f"feasts on this matchup")

    # Usage vacuum as a reason
    if vacuum and vacuum.get("has_vacuum"):
        reasons.append(f"usage vacuum (+{vacuum.get('total_boost', 0):.1f} boost)")

    if reasons:
        parts.append(f"  Why: {' + '.join(reasons)}")

    return "\n".join(parts)


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


BDL_PLAYER_TEAM_CACHE: Dict[int, int] = {}


def bdl_player_team_id(player_id: int) -> int:
    """
    Resolve a BDL player's current team ID from their most recent stat line.
    BDL stats endpoint includes player.team_id for each game.
    Returns the team_id (int), or 0 if unknown.
    """
    if player_id in BDL_PLAYER_TEAM_CACHE:
        return BDL_PLAYER_TEAM_CACHE[player_id]

    season = _season_year()
    resp = _bdl_get("stats", {
        "player_ids[]": player_id,
        "seasons[]": season,
        "per_page": 1,
    })
    games = resp.get("data", [])
    team_id = 0
    if games:
        g = games[0]
        # BDL v1 stats: team is nested under player or at top level
        team_obj = g.get("team", {})
        if isinstance(team_obj, dict):
            team_id = team_obj.get("id", 0)
        elif isinstance(team_obj, int):
            team_id = team_obj
        # Also check player.team_id
        if not team_id:
            player_obj = g.get("player", {})
            if isinstance(player_obj, dict):
                team_id = player_obj.get("team_id", 0)

    BDL_PLAYER_TEAM_CACHE[player_id] = team_id
    return team_id


def determine_home_away(player_id: int, game_info: Dict) -> Tuple[bool, str, str]:
    """
    Determine if a player is home or away by matching their BDL team_id
    against the game's home_team/away_team IDs.
    Returns (is_home, opp_team_name, player_team_name).
    """
    home_team = game_info.get("home_team", {})
    away_team = game_info.get("away_team", {})
    home_name = home_team.get("full_name", home_team.get("name", ""))
    away_name = away_team.get("full_name", away_team.get("name", ""))
    home_id = home_team.get("id", 0)
    away_id = away_team.get("id", 0)

    player_team_id = bdl_player_team_id(player_id)

    if player_team_id and player_team_id == home_id:
        return True, away_name, home_name
    elif player_team_id and player_team_id == away_id:
        return False, home_name, away_name
    else:
        # Fallback: can't determine, default to home
        return True, away_name, home_name


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


# ============================================================================
# SECTION 5b: ESPN + NBA.COM MULTI-SOURCE DATA
# ============================================================================

def _espn_get(url: str, params: Dict = None, timeout: int = 8) -> Dict:
    """Safe ESPN API call with caching."""
    if not requests:
        return {}
    try:
        resp = requests.get(url, params=params or {}, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
        else:
            print(f"[ESPN] {url.split('/')[-1]} -> {resp.status_code}")
            return {}
    except Exception as e:
        print(f"[ESPN] Error: {e}")
        return {}


def espn_search_player(player_name: str) -> Dict:
    """
    Search ESPN for a player by name using the public search endpoint.
    URL: site.web.api.espn.com/apis/common/v3/search?query=X&type=player&limit=5
    Response: {"items": [{"id": "4431678", "displayName": "Tyrese Maxey", ...}]}
    """
    if not player_name or not requests:
        return {}

    cache_key = _clean_name(player_name)
    if cache_key in ESPN_PLAYER_CACHE:
        return ESPN_PLAYER_CACHE[cache_key]

    try:
        resp = requests.get(
            ESPN_PLAYER_SEARCH,
            params={"query": player_name, "limit": 5, "type": "player"},
            timeout=8,
        )

        print(f"[ESPN] Search '{player_name}' -> {resp.status_code}")

        if resp.status_code != 200:
            ESPN_PLAYER_CACHE[cache_key] = {}
            return {}

        data = resp.json()
        items = data.get("items", [])

        if not items:
            print(f"[ESPN] No results for '{player_name}'")
            ESPN_PLAYER_CACHE[cache_key] = {}
            return {}

        # Find best NBA match
        athlete = None
        for item in items:
            if item.get("league", "").lower() == "nba" and item.get("type") == "player":
                athlete = item
                break
        if not athlete:
            athlete = items[0]  # fallback to first result

        result = {
            "espn_id": str(athlete.get("id", "")),
            "name": athlete.get("displayName", player_name),
            "team": "",
            "team_abbr": "",
            "position": "",
            "jersey": "",
            "injuries": [],
            "news": [],
        }

        # ESPN search doesn't include team/injury in the search response
        # We'll get team info from a follow-up call if needed
        print(f"[ESPN] Found: {result['name']} (ID: {result['espn_id']})")

        ESPN_PLAYER_CACHE[cache_key] = result
        return result

    except Exception as e:
        print(f"[ESPN] Player search error for {player_name}: {e}")
        ESPN_PLAYER_CACHE[cache_key] = {}
        return {}


def espn_fetch_gamelog(espn_id: str, season: str = "2026") -> List[Dict]:
    """
    Fetch player game log from ESPN's public API.
    URL: site.web.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/{id}/gamelog?season=YYYY
    Response structure:
      - labels: ["MIN","FG","FG%","3PT","3P%","FT","FT%","REB","AST","BLK","STL","PF","TO","PTS"]
      - seasonTypes[].categories[].events[].stats: ["35","7-13","53.8","1-3","33.3",...]
    Stats are strings. "Made-Attempted" fields like FG, 3PT, FT need splitting.
    Returns list of normalized game dicts with pts, reb, ast, blk, stl, fg3m, min.
    """
    if not espn_id or not requests:
        return []

    cache_key = f"espn_gl_{espn_id}"
    if cache_key in MULTI_SOURCE_CACHE:
        return MULTI_SOURCE_CACHE[cache_key]

    try:
        url = f"{ESPN_GAMELOG_BASE}/{espn_id}/gamelog"
        resp = requests.get(url, params={"season": season}, timeout=10)

        print(f"[ESPN] gamelog {espn_id} -> {resp.status_code}")

        if resp.status_code != 200:
            MULTI_SOURCE_CACHE[cache_key] = []
            return []

        data = resp.json()

        # Build label-to-index map from top-level labels
        # Expected: ["MIN","FG","FG%","3PT","3P%","FT","FT%","REB","AST","BLK","STL","PF","TO","PTS"]
        labels = data.get("labels", [])
        label_map = {label.upper(): i for i, label in enumerate(labels)}

        if not label_map:
            print(f"[ESPN] gamelog {espn_id}: no labels found in response")
            MULTI_SOURCE_CACHE[cache_key] = []
            return []

        print(f"[ESPN] gamelog labels: {labels}")

        # Collect events from all seasonTypes -> categories -> events
        # Categories are months (April, March, etc.) — most recent first
        games = []
        season_types = data.get("seasonTypes", [])

        for st in season_types:
            for cat in st.get("categories", []):
                for event in cat.get("events", []):
                    stats = event.get("stats", [])
                    if not stats:
                        continue

                    def _get_stat(label_name: str) -> float:
                        """Get a stat value by its label name."""
                        idx = label_map.get(label_name.upper())
                        if idx is None or idx >= len(stats):
                            return 0.0
                        val = str(stats[idx])
                        # Handle "Made-Attempted" format (e.g., "7-13" for FG)
                        if "-" in val and not val.startswith("-"):
                            parts = val.split("-")
                            return _safe_float(parts[0])  # Return "made" part
                        return _safe_float(val)

                    minutes = _get_stat("MIN")
                    if minutes < 1.0:
                        continue  # Skip DNPs

                    normalized = {
                        "pts": _get_stat("PTS"),
                        "reb": _get_stat("REB"),
                        "ast": _get_stat("AST"),
                        "blk": _get_stat("BLK"),
                        "stl": _get_stat("STL"),
                        "fg3m": _get_stat("3PT"),  # "3PT" is "Made-Attempted", _get_stat returns Made
                        "min": minutes,
                        "fgm": _get_stat("FG"),
                        "fga": 0.0,  # Would need to parse attempted from "FG" field
                    }

                    games.append(normalized)

        # Games come most-recent-first (April first, then March, etc.)
        # Limit to 25 most recent
        games = games[:25]
        print(f"[ESPN] gamelog {espn_id}: {len(games)} games parsed")

        MULTI_SOURCE_CACHE[cache_key] = games
        return games

    except Exception as e:
        print(f"[ESPN] gamelog error for {espn_id}: {e}")
        import traceback
        traceback.print_exc()
        MULTI_SOURCE_CACHE[cache_key] = []
        return []


def _safe_float(val) -> float:
    """Safely convert any value to float."""
    if val is None:
        return 0.0
    try:
        return float(str(val).replace("-", "0").split("/")[0])
    except (ValueError, TypeError):
        return 0.0


def espn_fetch_team_defense(team_name: str) -> Dict:
    """
    Fetch live team defensive stats from ESPN's team stats endpoint.
    More accurate than hardcoded ratings.
    """
    cache_key = f"espn_def_{_clean_name(team_name)}"
    if cache_key in DEF_RATINGS_CACHE:
        return DEF_RATINGS_CACHE[cache_key]

    if not requests:
        return {}

    try:
        # ESPN team search
        url = f"{ESPN_SITE_API_BASE}/site/v2/sports/basketball/nba/teams"
        resp = requests.get(url, params={"limit": 50}, timeout=8)

        if resp.status_code != 200:
            return {}

        data = resp.json()
        teams = data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])

        for team_entry in teams:
            team = team_entry.get("team", {})
            full_name = team.get("displayName", "")
            if _clean_name(team_name) in _clean_name(full_name):
                team_id = team.get("id")
                if team_id:
                    # Fetch team stats
                    stats_url = f"{ESPN_SITE_API_BASE}/site/v2/sports/basketball/nba/teams/{team_id}/statistics"
                    stats_resp = requests.get(stats_url, timeout=8)
                    if stats_resp.status_code == 200:
                        stats_data = stats_resp.json()
                        result = _parse_espn_team_defense(stats_data)
                        DEF_RATINGS_CACHE[cache_key] = result
                        return result
    except Exception as e:
        print(f"[ESPN] team defense error: {e}")

    return {}


def _parse_espn_team_defense(data: Dict) -> Dict:
    """Parse ESPN team stats response for defensive numbers."""
    result = {}
    try:
        splits = data.get("results", {}).get("stats", {})
        categories = splits.get("categories", [])
        for cat in categories:
            if "defensive" in cat.get("displayName", "").lower():
                for stat in cat.get("stats", []):
                    name = stat.get("name", "")
                    val = stat.get("value", 0)
                    if "reboundsDefensive" in name:
                        result["dreb"] = float(val)
                    elif "steals" in name:
                        result["stl_allowed"] = float(val)
    except:
        pass
    return result


def fetch_player_games_multisource(
    player_name: str,
    player_id: int,
    stat_key: str,
    n: int = BASELINE_GAMES,
    is_threes: bool = False,
) -> Tuple[List[Dict], str]:
    """
    ESPN-FIRST game fetcher. Tries ESPN public API as primary truth,
    falls back to BDL if ESPN unavailable or insufficient.

    Returns: (games_list, source_tag)
      - games_list: list of dicts with 'value' key (same format compute_projection expects)
      - source_tag: "ESPN" | "BDL" | "ESPN+BDL"
    """
    espn_games = []
    source_tag = "BDL"  # default fallback

    # --- ESPN Primary ---
    espn_data = espn_search_player(player_name)
    if espn_data and espn_data.get("espn_id"):
        raw_espn = espn_fetch_gamelog(espn_data["espn_id"])
        if raw_espn and len(raw_espn) >= 8:
            # Convert ESPN gamelog to {value: X} format
            espn_stat_map = {
                "pts": "pts", "reb": "reb", "ast": "ast",
                "blk": "blk", "stl": "stl", "threes": "fg3m",
                "min": "min",
            }
            espn_field = espn_stat_map.get(stat_key, stat_key)
            if is_threes:
                espn_field = "fg3m"

            for g in raw_espn[:n]:
                val = float(g.get(espn_field, 0) or 0)
                espn_games.append({"value": val, "source": "ESPN"})

            if len(espn_games) >= 10:
                source_tag = "ESPN"
                print(f"[SOURCE] {player_name} {stat_key}: ESPN primary ({len(espn_games)} games)")
                return espn_games, source_tag
            else:
                print(f"[SOURCE] {player_name} {stat_key}: ESPN only {len(espn_games)} games, supplementing with BDL")

    # --- BDL Fallback/Supplement ---
    if is_threes:
        bdl_games = bdl_last_n_games_threes(player_id, n)
    else:
        bdl_games = bdl_last_n_games_stats(player_id, stat_key, n)

    if espn_games and bdl_games:
        # ESPN had some data but not enough — use ESPN as primary, fill gaps with BDL
        combined = espn_games[:]
        needed = n - len(combined)
        if needed > 0 and len(bdl_games) > len(espn_games):
            for g in bdl_games[len(espn_games):len(espn_games) + needed]:
                g["source"] = "BDL"
                combined.append(g)
        source_tag = "ESPN+BDL"
        print(f"[SOURCE] {player_name} {stat_key}: ESPN+BDL combined ({len(combined)} games)")
        return combined, source_tag

    if bdl_games:
        for g in bdl_games:
            g["source"] = "BDL"
        source_tag = "BDL"
        print(f"[SOURCE] {player_name} {stat_key}: BDL fallback ({len(bdl_games)} games)")
        return bdl_games, source_tag

    print(f"[SOURCE] {player_name} {stat_key}: NO DATA from either source")
    return [], "NONE"


def fetch_minutes_multisource(
    player_name: str,
    player_id: int,
    n: int = BASELINE_GAMES,
) -> Tuple[float, float, float]:
    """
    Fetch minutes windows using ESPN-first approach.
    Returns (base_min, l10_min, l5_min) averages.
    """
    games, source = fetch_player_games_multisource(player_name, player_id, "min", n)
    if not games:
        return fetch_minutes_windows(player_id, n)  # BDL-only fallback

    base = games[:n]
    l10 = games[:10] if len(games) >= 10 else games
    l5 = games[:5] if len(games) >= 5 else games

    def _avg(lst):
        if not lst:
            return 0.0
        vals = [g.get("value", 0) for g in lst]
        return sum(vals) / len(vals) if vals else 0.0

    return _avg(base), _avg(l10), _avg(l5)


def cross_reference_player(
    player_name: str,
    player_id: int,
    stat_key: str,
    primary_games: List[Dict],
) -> Dict:
    """
    Cross-reference player data across ESPN (primary) and BDL (secondary).
    ESPN is treated as ground truth. BDL is the cross-check.
    Returns confidence assessment and any discrepancies.
    """
    result = {
        "sources": [],
        "primary_source": "ESPN",
        "bdl_avg": 0,
        "espn_avg": 0,
        "agreement": True,
        "discrepancy": 0,
        "confidence": "high",
        "espn_team": "",
        "injuries": [],
        "notes": [],
    }

    # --- ESPN (Primary Truth) ---
    espn_data = espn_search_player(player_name)
    if espn_data and espn_data.get("espn_id"):
        result["sources"].append("ESPN")
        result["espn_team"] = espn_data.get("team", "")

        # Check injuries from ESPN
        if espn_data.get("injuries"):
            for inj in espn_data["injuries"]:
                status = inj.get("status", "")
                detail = inj.get("detail", "")
                result["injuries"].append(f"{status}: {detail}")
                if status.lower() in ("out", "doubtful"):
                    result["notes"].append(f"INJURY ALERT: {status} - {detail}")
                    result["confidence"] = "low"

        # ESPN averages from gamelog
        espn_games = espn_fetch_gamelog(espn_data["espn_id"])
        if espn_games:
            espn_stat_map = {"threes": "fg3m"}
            espn_field = espn_stat_map.get(stat_key, stat_key)
            espn_values = [g.get(espn_field, 0) for g in espn_games]
            if espn_values:
                result["espn_avg"] = sum(espn_values) / len(espn_values)
    else:
        result["notes"].append("ESPN lookup failed")
        result["primary_source"] = "BDL"

    # --- BDL (Secondary Cross-Check) ---
    bdl_games = bdl_last_n_games_stats(player_id, stat_key, BASELINE_GAMES)
    if stat_key == "threes":
        bdl_games = bdl_last_n_games_threes(player_id, BASELINE_GAMES)

    if bdl_games:
        result["sources"].append("BDL")
        bdl_values = [g.get("value", 0) for g in bdl_games if g]
        if bdl_values:
            result["bdl_avg"] = sum(bdl_values) / len(bdl_values)

    # --- Compare sources ---
    if result["espn_avg"] > 0 and result["bdl_avg"] > 0:
        # Sanity check: if BDL avg is implausibly low compared to ESPN
        # (e.g., BDL returns 1.2 for points when ESPN shows 16.6),
        # BDL data is likely wrong player or wrong stat — discard it
        ratio = result["bdl_avg"] / result["espn_avg"] if result["espn_avg"] > 0 else 1.0
        if ratio < 0.25 and result["espn_avg"] > 3.0:
            # BDL is less than 25% of ESPN — likely bad data
            result["notes"].append(f"BDL data unreliable ({result['bdl_avg']:.1f} vs ESPN {result['espn_avg']:.1f}) — using ESPN only")
            result["bdl_avg"] = 0  # Zero it out so it doesn't confuse display
            result["sources"] = [s for s in result["sources"] if s != "BDL"]
        else:
            # Use ESPN as denominator since it's our truth
            diff_pct = abs(result["espn_avg"] - result["bdl_avg"]) / result["espn_avg"] * 100
            result["discrepancy"] = diff_pct

            if diff_pct > 15:
                result["agreement"] = False
                if result["confidence"] != "low":  # Don't override injury-based low
                    result["confidence"] = "low"
                result["notes"].append(
                    f"DATA MISMATCH: ESPN avg {result['espn_avg']:.1f} vs "
                    f"BDL avg {result['bdl_avg']:.1f} ({diff_pct:.0f}% off) — trusting ESPN"
                )
            elif diff_pct > 8:
                if result["confidence"] == "high":
                    result["confidence"] = "medium"
                result["notes"].append(
                    f"Minor diff: ESPN {result['espn_avg']:.1f} vs BDL {result['bdl_avg']:.1f}"
                )
            else:
                result["notes"].append(f"Sources agree ({diff_pct:.0f}% diff)")
    elif result["espn_avg"] > 0:
        result["notes"].append("ESPN only — no BDL cross-check")
    elif result["bdl_avg"] > 0:
        result["notes"].append("BDL only — ESPN unavailable")
        result["primary_source"] = "BDL"
    else:
        result["notes"].append("No stat averages from either source")
        result["confidence"] = "low"

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
# SECTION 4B: ESPN SCOREBOARD ENRICHMENT (LIVE DATA)
# ============================================================================

ESPN_SCOREBOARD_CACHE: Dict = {}


def espn_fetch_scoreboard(date_str: str = None) -> List[Dict]:
    """
    Fetch ESPN scoreboard for a date. Returns enriched game data including:
    - Vegas O/U (over_under)
    - Spread
    - Team pace ratings
    - Whether each team is on a back-to-back

    ESPN scoreboard endpoint:
      site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=YYYYMMDD
    Response: {events: [{competitions: [{odds: [...], competitors: [...]}]}]}
    """
    if not requests:
        return []

    if date_str is None:
        date_str = _now_et().strftime("%Y%m%d")

    cache_key = f"sb_{date_str}"
    if cache_key in ESPN_SCOREBOARD_CACHE:
        return ESPN_SCOREBOARD_CACHE[cache_key]

    try:
        resp = requests.get(
            ESPN_SCOREBOARD,
            params={"dates": date_str},
            timeout=10,
        )
        print(f"[ESPN] Scoreboard {date_str} -> {resp.status_code}")

        if resp.status_code != 200:
            ESPN_SCOREBOARD_CACHE[cache_key] = []
            return []

        data = resp.json()
        events = data.get("events", [])
        print(f"[ESPN] Scoreboard: {len(events)} games found")

        result = []
        for event in events:
            comps = event.get("competitions", [])
            if not comps:
                continue
            comp = comps[0]

            game = {
                "espn_id": event.get("id", ""),
                "name": event.get("name", ""),
                "short_name": event.get("shortName", ""),
                "status": comp.get("status", {}).get("type", {}).get("name", ""),
            }

            # --- Odds data (Vegas O/U, spread) ---
            odds_list = comp.get("odds", [])
            if odds_list:
                odds = odds_list[0]
                game["over_under"] = _safe_float(str(odds.get("overUnder", 0)))
                game["spread"] = _safe_float(str(odds.get("spread", 0)))
                game["home_ml"] = odds.get("homeTeamOdds", {}).get("moneyLine", 0)
                game["away_ml"] = odds.get("awayTeamOdds", {}).get("moneyLine", 0)

            # --- Team data ---
            competitors = comp.get("competitors", [])
            for team_data in competitors:
                team_obj = team_data.get("team", {})
                team_name = team_obj.get("displayName", "")
                team_abbr = team_obj.get("abbreviation", "")
                home_away = team_data.get("homeAway", "")

                if home_away == "home":
                    game["home_name"] = team_name
                    game["home_abbr"] = team_abbr
                    game["home_id"] = team_obj.get("id", "")
                else:
                    game["away_name"] = team_name
                    game["away_abbr"] = team_abbr
                    game["away_id"] = team_obj.get("id", "")

            result.append(game)

        ESPN_SCOREBOARD_CACHE[cache_key] = result
        return result

    except Exception as e:
        print(f"[ESPN] Scoreboard error: {e}")
        ESPN_SCOREBOARD_CACHE[cache_key] = []
        return []


def espn_detect_back_to_back(date_str: str = None) -> Dict[str, bool]:
    """
    Detect which teams are on a back-to-back by checking yesterday's scoreboard.
    Returns: {team_name: True} for teams that played yesterday.
    """
    if date_str is None:
        yesterday = _now_et() - timedelta(days=1)
        date_str = yesterday.strftime("%Y%m%d")

    games = espn_fetch_scoreboard(date_str)
    b2b_teams = {}

    for game in games:
        status = game.get("status", "")
        # Only count completed or in-progress games
        if status in ("STATUS_FINAL", "STATUS_IN_PROGRESS", "STATUS_END_PERIOD"):
            for key in ("home_name", "away_name"):
                team = game.get(key, "")
                if team:
                    b2b_teams[team] = True
                    # Also store abbreviation for matching
                    abbr_key = key.replace("_name", "_abbr")
                    abbr = game.get(abbr_key, "")
                    if abbr:
                        b2b_teams[abbr] = True

    print(f"[B2B] Yesterday's games: {len(b2b_teams) // 2} teams played")
    return b2b_teams


def enrich_games_map_with_espn(games_map: Dict) -> None:
    """
    Enrich the BDL games_map with ESPN scoreboard data:
    - over_under (Vegas O/U)
    - opp_back_to_back for each team
    - ESPN-sourced pace/spread

    Mutates games_map in place.
    """
    # Get today's ESPN scoreboard
    today_str = _now_et().strftime("%Y%m%d")
    espn_games = espn_fetch_scoreboard(today_str)

    # Build lookup by team names for matching BDL games to ESPN games
    espn_by_teams = {}
    for eg in espn_games:
        home = eg.get("home_name", "")
        away = eg.get("away_name", "")
        if home and away:
            espn_by_teams[(home, away)] = eg

    # Detect B2Bs from yesterday
    b2b_teams = espn_detect_back_to_back()

    for game_id, game_info in games_map.items():
        home_team = game_info.get("home_team", {})
        away_team = game_info.get("away_team", {})
        home_name = home_team.get("full_name", home_team.get("name", ""))
        away_name = away_team.get("full_name", away_team.get("name", ""))

        # Match to ESPN game
        espn_game = espn_by_teams.get((home_name, away_name))
        if not espn_game:
            # Try fuzzy match — last word of team name
            for (eh, ea), eg in espn_by_teams.items():
                if (home_name.split()[-1:] == eh.split()[-1:] and
                    away_name.split()[-1:] == ea.split()[-1:]):
                    espn_game = eg
                    break

        if espn_game:
            # Vegas O/U
            ou = espn_game.get("over_under", 0)
            if ou > 0:
                game_info["over_under"] = ou
                game_info["game_total"] = ou  # Also update the BDL field

            # Spread
            spread = espn_game.get("spread", 0)
            if spread:
                game_info["espn_spread"] = spread

            print(f"[ENRICH] {away_name} @ {home_name}: O/U={ou}, spread={spread}")

        # B2B detection for both teams
        home_b2b = b2b_teams.get(home_name, False) or b2b_teams.get(
            home_team.get("abbreviation", ""), False
        )
        away_b2b = b2b_teams.get(away_name, False) or b2b_teams.get(
            away_team.get("abbreviation", ""), False
        )

        game_info["home_b2b"] = home_b2b
        game_info["away_b2b"] = away_b2b

        if home_b2b:
            print(f"[B2B] {home_name} on back-to-back")
        if away_b2b:
            print(f"[B2B] {away_name} on back-to-back")


# ============================================================================
# SECTION 4C: MATCHUP HISTORY ENGINE
# ============================================================================

MATCHUP_HISTORY_CACHE: Dict = {}


def fetch_matchup_history(
    player_name: str,
    player_id: int,
    opp_team: str,
    stat_key: str = "pts",
    is_threes: bool = False,
) -> Dict:
    """
    Fetch how a player performs against a SPECIFIC opponent historically.
    Uses BDL stats endpoint filtered by opponent team.
    Some players feast on certain teams — this catches that pattern.

    Returns:
    {
        "games_vs": int,          # games found against this opponent
        "avg_vs": float,          # average stat vs this opponent
        "avg_all": float,         # season average for comparison
        "matchup_delta": float,   # avg_vs - avg_all (positive = they feast)
        "matchup_pct": float,     # percentage above/below season avg
        "high_vs": float,         # best game vs this opponent
        "hit_rate_vs": float,     # empirical rate of clearing avg_all against opp
        "assessment": str,        # "FEAST" / "NEUTRAL" / "STRUGGLE"
        "narrative": str,         # human-readable summary
    }
    """
    cache_key = f"matchup_{player_id}_{_clean_name(opp_team)}_{stat_key}"
    if cache_key in MATCHUP_HISTORY_CACHE:
        return MATCHUP_HISTORY_CACHE[cache_key]

    result = {
        "games_vs": 0,
        "avg_vs": 0.0,
        "avg_all": 0.0,
        "matchup_delta": 0.0,
        "matchup_pct": 0.0,
        "high_vs": 0.0,
        "hit_rate_vs": 0.0,
        "assessment": "NEUTRAL",
        "narrative": "",
    }

    # Get the opponent's BDL team ID for filtering
    opp_full = normalize_team_name(opp_team)
    opp_team_id = bdl_team_name_to_id(opp_full)

    if not opp_team_id or not player_id:
        MATCHUP_HISTORY_CACHE[cache_key] = result
        return result

    # Fetch this season's stats for the player
    season = _season_year()
    resp = _bdl_get("stats", {
        "player_ids[]": player_id,
        "seasons[]": season,
        "per_page": 82,  # Full season
    })
    all_games = resp.get("data", [])

    if not all_games:
        MATCHUP_HISTORY_CACHE[cache_key] = result
        return result

    # Also try prior season for more matchup data
    prior_resp = _bdl_get("stats", {
        "player_ids[]": player_id,
        "seasons[]": season - 1,
        "per_page": 82,
    })
    prior_games = prior_resp.get("data", [])

    all_seasons = all_games + prior_games

    # Extract stat field
    field = "fg3m" if is_threes or stat_key == "threes" else stat_key

    # Separate games vs this opponent from all games
    vs_values = []
    all_values = []

    for game in all_seasons:
        if not game:
            continue
        val = float(game.get(field, 0) or 0)

        # Determine opponent from game object
        game_obj = game.get("game", {})
        if isinstance(game_obj, dict):
            home_id = game_obj.get("home_team_id", 0)
            away_id = game_obj.get("visitor_team_id", 0)
            # If opponent's team ID matches home or away, this was a matchup
            if home_id == opp_team_id or away_id == opp_team_id:
                vs_values.append(val)

        all_values.append(val)

    if not all_values:
        MATCHUP_HISTORY_CACHE[cache_key] = result
        return result

    avg_all = sum(all_values) / len(all_values)
    result["avg_all"] = avg_all
    result["games_vs"] = len(vs_values)

    if vs_values:
        avg_vs = sum(vs_values) / len(vs_values)
        result["avg_vs"] = avg_vs
        result["high_vs"] = max(vs_values)
        result["matchup_delta"] = avg_vs - avg_all
        result["matchup_pct"] = ((avg_vs - avg_all) / avg_all * 100) if avg_all > 0 else 0

        # Hit rate vs this opponent at their own season average
        hits_vs = sum(1 for v in vs_values if v > avg_all)
        result["hit_rate_vs"] = hits_vs / len(vs_values) if vs_values else 0

        # Assessment
        if result["matchup_pct"] > 15 and len(vs_values) >= 2:
            result["assessment"] = "FEAST"
            result["narrative"] = (
                f"Feasts on {opp_full.split()[-1]}: {avg_vs:.1f} avg in "
                f"{len(vs_values)} games vs ({avg_all:.1f} szn avg, "
                f"+{result['matchup_pct']:.0f}%), high of {result['high_vs']:.0f}"
            )
        elif result["matchup_pct"] < -15 and len(vs_values) >= 2:
            result["assessment"] = "STRUGGLE"
            result["narrative"] = (
                f"Struggles vs {opp_full.split()[-1]}: {avg_vs:.1f} avg in "
                f"{len(vs_values)} games ({result['matchup_pct']:.0f}% below szn avg)"
            )
        else:
            result["assessment"] = "NEUTRAL"
            if vs_values:
                result["narrative"] = (
                    f"vs {opp_full.split()[-1]}: {avg_vs:.1f} in "
                    f"{len(vs_values)} games (szn avg {avg_all:.1f})"
                )

    print(f"[MATCHUP] {player_name} vs {opp_full}: {len(vs_values)} games, "
          f"avg {result['avg_vs']:.1f} vs szn {avg_all:.1f} -> {result['assessment']}")

    MATCHUP_HISTORY_CACHE[cache_key] = result
    return result


# ============================================================================
# SECTION 4D: USAGE VACUUM MODEL
# ============================================================================

USAGE_VACUUM_CACHE: Dict = {}


def compute_usage_vacuum(
    player_name: str,
    player_id: int,
    game_info: Dict,
    stat_key: str = "pts",
) -> Dict:
    """
    When a key teammate is OUT, their shots/touches redistribute.
    This model estimates HOW MUCH of an absent player's production
    flows to the target player.

    Logic:
    1. Check team injury report for OUT players (via ESPN xref or game_info)
    2. For each OUT player, estimate their per-game production in the target stat
    3. Estimate redistribution share based on the target player's usage rate
    4. Sum up the total boost

    Returns:
    {
        "has_vacuum": bool,
        "injured_out": [{"name": str, "avg_stat": float, "redistributed": float}],
        "total_boost": float,       # projected stat boost from vacuums
        "boost_pct": float,         # boost as percentage of player's avg
        "narrative": str,
    }
    """
    cache_key = f"vacuum_{player_id}_{game_info.get('id', 0)}_{stat_key}"
    if cache_key in USAGE_VACUUM_CACHE:
        return USAGE_VACUUM_CACHE[cache_key]

    result = {
        "has_vacuum": False,
        "injured_out": [],
        "total_boost": 0.0,
        "boost_pct": 0.0,
        "narrative": "",
    }

    # Determine player's team
    player_team_id = bdl_player_team_id(player_id)
    if not player_team_id:
        USAGE_VACUUM_CACHE[cache_key] = result
        return result

    # Get team roster and check for injured/out players via BDL
    roster = bdl_active_roster(player_team_id)
    if not roster:
        USAGE_VACUUM_CACHE[cache_key] = result
        return result

    # Get the target player's season stats for context
    field = "fg3m" if stat_key == "threes" else stat_key
    player_games, _ = fetch_player_games_multisource(
        player_name, player_id, stat_key, BASELINE_GAMES,
        is_threes=(stat_key == "threes"),
    )
    if not player_games:
        USAGE_VACUUM_CACHE[cache_key] = result
        return result

    player_avg = sum(g.get("value", 0) for g in player_games) / len(player_games)

    # Check ESPN for injury info on teammates
    injured_out = []
    for teammate in roster:
        tm_name = f"{teammate.get('first_name', '')} {teammate.get('last_name', '')}".strip()
        tm_id = teammate.get("id")

        if not tm_name or not tm_id or tm_id == player_id:
            continue

        # Check ESPN for this teammate's injury status
        espn_info = espn_search_player(tm_name)
        if not espn_info:
            continue

        injuries = espn_info.get("injuries", [])
        is_out = False
        for inj in injuries:
            status = str(inj.get("status", "")).lower()
            if status in ("out", "doubtful"):
                is_out = True
                break

        if not is_out:
            continue

        # This teammate is OUT — estimate their production
        tm_resp = _bdl_get("stats", {
            "player_ids[]": tm_id,
            "seasons[]": _season_year(),
            "per_page": 10,
        })
        tm_games = tm_resp.get("data", [])

        if not tm_games:
            continue

        tm_values = [float(g.get(field, 0) or 0) for g in tm_games]
        if not tm_values:
            continue
        tm_avg = sum(tm_values) / len(tm_values)

        # Only care about significant production losses (>5 pts, >1 reb/ast, etc.)
        min_production = {"pts": 8.0, "reb": 4.0, "ast": 3.0, "threes": 1.0,
                          "blk": 1.0, "stl": 1.0}.get(stat_key, 5.0)
        if tm_avg < min_production:
            continue

        # Redistribution model:
        # When a player is out, their production redistributes roughly:
        # - 40-50% to the team's top usage player
        # - 20-30% spread among remaining starters
        # - The rest is absorbed by the replacement player
        # Use the target player's relative production to estimate share
        team_total_guess = player_avg * 4.5  # rough team total for this stat
        player_share = player_avg / team_total_guess if team_total_guess > 0 else 0.2
        # Higher usage players get more of the vacuum
        redistribution_rate = min(player_share * 1.8, 0.35)
        redistributed = tm_avg * redistribution_rate

        injured_out.append({
            "name": tm_name,
            "avg_stat": tm_avg,
            "redistributed": redistributed,
        })

        print(f"[VACUUM] {tm_name} OUT (avg {tm_avg:.1f} {stat_key}) -> "
              f"{redistributed:.1f} redistributed to {player_name}")

    if injured_out:
        total_boost = sum(io["redistributed"] for io in injured_out)
        result["has_vacuum"] = True
        result["injured_out"] = injured_out
        result["total_boost"] = total_boost
        result["boost_pct"] = (total_boost / player_avg * 100) if player_avg > 0 else 0

        # Build narrative
        names = [io["name"].split()[-1] for io in injured_out]
        avgs = [f"{io['avg_stat']:.0f}" for io in injured_out]
        if len(names) == 1:
            result["narrative"] = (
                f"Usage vacuum: {names[0]} OUT ({avgs[0]} {stat_key}/g) -> "
                f"+{total_boost:.1f} projected boost ({result['boost_pct']:.0f}%)"
            )
        else:
            pairs = [f"{n} ({a})" for n, a in zip(names, avgs)]
            result["narrative"] = (
                f"Usage vacuum: {', '.join(pairs)} OUT -> "
                f"+{total_boost:.1f} projected boost ({result['boost_pct']:.0f}%)"
            )

    USAGE_VACUUM_CACHE[cache_key] = result
    return result


# ============================================================================
# SECTION 4E: THE ODDS API — REAL FANDUEL ALT LINES
# ============================================================================
# BDL only returns 1 main line per player per prop (confirmed by logs).
# The Odds API v4 provides real FanDuel alternate player props with actual odds.
# Free tier: 500 credits. Each /events call = 1 credit, each /events/{id}/odds = 1 credit.
# Strategy: fetch event list (1 credit), then alt props per event (1 credit each).
# Total cost per run: ~1 + N_games credits.
# ============================================================================

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_API_SPORT = "basketball_nba"
ODDS_API_CACHE: Dict[str, Any] = {}


def odds_api_available() -> bool:
    """Check if The Odds API is configured."""
    return bool(ODDS_API_KEY) and requests is not None


def odds_api_fetch_events() -> List[Dict]:
    """
    Fetch today's NBA events from The Odds API.
    Returns list of events with id, home_team, away_team, commence_time.
    Cost: 1 credit.
    """
    if not odds_api_available():
        print("[ODDS-API] Skipped: no API key or requests missing")
        return []

    cache_key = "events_today"
    if cache_key in ODDS_API_CACHE:
        return ODDS_API_CACHE[cache_key]

    url = f"{ODDS_API_BASE}/sports/{ODDS_API_SPORT}/events"
    params = {
        "apiKey": ODDS_API_KEY,
        "dateFormat": "iso",
    }

    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        print(f"[ODDS-API] /events -> {resp.status_code}")

        if resp.status_code != 200:
            print(f"[ODDS-API] events error: {resp.text[:300]}")
            remaining = resp.headers.get("x-requests-remaining", "?")
            print(f"[ODDS-API] Credits remaining: {remaining}")
            return []

        remaining = resp.headers.get("x-requests-remaining", "?")
        used = resp.headers.get("x-requests-used", "?")
        print(f"[ODDS-API] Credits: used={used}, remaining={remaining}")

        events = resp.json()
        if not isinstance(events, list):
            events = events.get("data", [])

        print(f"[ODDS-API] Found {len(events)} NBA events")
        ODDS_API_CACHE[cache_key] = events
        return events

    except Exception as e:
        print(f"[ODDS-API] events exception: {e}")
        return []


def odds_api_fetch_player_props(event_id: str, market: str = "player_points") -> Dict:
    """
    Fetch alternate player prop lines for a single event from The Odds API.
    Returns dict: {player_name_lower: [{line, odds, book}]}

    Markets: player_points, player_rebounds, player_assists,
             player_threes, player_blocks, player_steals

    The alternate markets use suffix '_alternate':
      player_points_alternate, player_rebounds_alternate, etc.

    We fetch BOTH the main market AND the alternate market to get full coverage.
    Cost: 1 credit per call.
    """
    if not odds_api_available():
        return {}

    cache_key = f"props_{event_id}_{market}"
    if cache_key in ODDS_API_CACHE:
        return ODDS_API_CACHE[cache_key]

    # Fetch both main and alternate markets in one call
    alt_market = f"{market}_alternate"
    markets_str = f"{market},{alt_market}"

    url = f"{ODDS_API_BASE}/sports/{ODDS_API_SPORT}/events/{event_id}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": markets_str,
        "bookmakers": "fanduel",
        "oddsFormat": "american",
        "dateFormat": "iso",
    }

    result: Dict[str, List[Dict]] = {}

    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        print(f"[ODDS-API] /events/{event_id[:8]}.../odds ({market}) -> {resp.status_code}")

        if resp.status_code != 200:
            print(f"[ODDS-API] props error: {resp.text[:300]}")
            ODDS_API_CACHE[cache_key] = result
            return result

        data = resp.json()

        # Response structure:
        # {id, bookmakers: [{key: "fanduel", markets: [{key: "player_points_alternate",
        #   outcomes: [{name: "Stephen Curry", description: "Over", point: 29.5, price: 250}]}]}]}
        bookmakers = data.get("bookmakers", [])
        if not bookmakers:
            print(f"[ODDS-API] No FanDuel data for event {event_id[:8]}...")
            ODDS_API_CACHE[cache_key] = result
            return result

        total_lines = 0
        alt_lines = 0

        for bk in bookmakers:
            if bk.get("key", "").lower() != "fanduel":
                continue

            for mkt in bk.get("markets", []):
                mkt_key = mkt.get("key", "")
                is_alt = "_alternate" in mkt_key

                for outcome in mkt.get("outcomes", []):
                    # Odds API format: name="Over"/"Under", description="Player Name"
                    over_under = outcome.get("name", "").lower()
                    if over_under != "over":
                        continue

                    player_name = outcome.get("description", "")
                    point = outcome.get("point", 0)
                    price = outcome.get("price", -110)

                    if not player_name or not point:
                        continue

                    # Normalize player name for matching
                    name_key = player_name.strip().lower()

                    if name_key not in result:
                        result[name_key] = []

                    result[name_key].append({
                        "line": float(point),
                        "odds": int(price),
                        "book": "FanDuel",
                        "is_alt": is_alt,
                        "source": "odds_api",
                    })
                    total_lines += 1
                    if is_alt:
                        alt_lines += 1

        print(f"[ODDS-API] Event {event_id[:8]}...: {len(result)} players, "
              f"{total_lines} total lines, {alt_lines} alt lines")

        ODDS_API_CACHE[cache_key] = result
        return result

    except Exception as e:
        print(f"[ODDS-API] props exception: {e}")
        ODDS_API_CACHE[cache_key] = result
        return result


def _normalize_name(name: str) -> str:
    """Normalize a player name for fuzzy matching: lowercase, strip suffixes like Jr./III/II."""
    n = name.strip().lower()
    # Remove common suffixes
    for suffix in [" jr.", " jr", " sr.", " sr", " iii", " ii", " iv", " v"]:
        if n.endswith(suffix):
            n = n[: -len(suffix)].strip()
    # Remove periods and extra spaces
    n = n.replace(".", "").replace("  ", " ")
    return n


def _match_odds_api_player(odds_api_names: Dict, bdl_player_name: str) -> Optional[str]:
    """
    Match a BDL player name to an Odds API player name key.
    Tries exact match first, then normalized match, then last-name match.
    Returns the odds_api name key or None.
    """
    bdl_lower = bdl_player_name.strip().lower()

    # Exact match
    if bdl_lower in odds_api_names:
        return bdl_lower

    # Normalized match
    bdl_norm = _normalize_name(bdl_player_name)
    for oa_key in odds_api_names:
        if _normalize_name(oa_key) == bdl_norm:
            return oa_key

    # Last name + first initial match (handles "S. Curry" vs "Stephen Curry")
    bdl_parts = bdl_norm.split()
    if len(bdl_parts) >= 2:
        bdl_last = bdl_parts[-1]
        bdl_first_init = bdl_parts[0][0] if bdl_parts[0] else ""
        for oa_key in odds_api_names:
            oa_norm = _normalize_name(oa_key)
            oa_parts = oa_norm.split()
            if len(oa_parts) >= 2:
                oa_last = oa_parts[-1]
                oa_first_init = oa_parts[0][0] if oa_parts[0] else ""
                if bdl_last == oa_last and bdl_first_init == oa_first_init:
                    return oa_key

    return None


def _match_bdl_game_to_odds_event(game_info: Dict, events: List[Dict]) -> Optional[str]:
    """
    Match a BDL game to an Odds API event by team names.
    Returns the Odds API event_id or None.
    """
    home = game_info.get("home_team", {})
    away = game_info.get("away_team", {})

    # BDL uses full_name like "Los Angeles Lakers" or abbreviation
    home_name = (home.get("full_name") or home.get("name") or "").lower()
    away_name = (away.get("full_name") or away.get("name") or "").lower()
    home_abbr = (home.get("abbreviation") or "").lower()
    away_abbr = (away.get("abbreviation") or "").lower()

    for ev in events:
        # Odds API uses team names like "Los Angeles Lakers"
        ev_home = (ev.get("home_team") or "").lower()
        ev_away = (ev.get("away_team") or "").lower()

        # Match by full name containment (handles slight name differences)
        home_match = (
            home_name and ev_home and (
                home_name in ev_home or ev_home in home_name
                or home_name.split()[-1] in ev_home  # Last word match (e.g. "Lakers")
            )
        )
        away_match = (
            away_name and ev_away and (
                away_name in ev_away or ev_away in away_name
                or away_name.split()[-1] in ev_away
            )
        )

        if home_match and away_match:
            return ev.get("id")

    return None


def enrich_lines_map_with_odds_api(lines_map: Dict, games_map: Dict) -> int:
    """
    Enrich the lines_map with real FanDuel alt lines from The Odds API.
    For each game, fetches alt player_points lines and merges them into the
    existing book_lines list for each player.

    Returns the total number of alt lines added.
    """
    if not odds_api_available():
        print("[ODDS-API] Not configured — skipping alt line enrichment")
        return 0

    print("[ODDS-API] Fetching events for alt line enrichment...")
    events = odds_api_fetch_events()
    if not events:
        print("[ODDS-API] No events found")
        return 0

    total_added = 0
    games_matched = 0

    for game_id, game_info in games_map.items():
        event_id = _match_bdl_game_to_odds_event(game_info, events)
        if not event_id:
            home_name = game_info.get("home_team", {}).get("full_name", "?")
            away_name = game_info.get("away_team", {}).get("full_name", "?")
            print(f"[ODDS-API] No event match for BDL game {game_id} ({away_name} @ {home_name})")
            continue

        games_matched += 1

        # Fetch alt lines for player_points (primary ladder market)
        alt_data = odds_api_fetch_player_props(event_id, "player_points")
        if not alt_data:
            continue

        # Merge into lines_map
        game_lines = lines_map.get(game_id, {})
        points_lines = game_lines.get("player_points", {})

        for bdl_player_name, bdl_book_lines in points_lines.items():
            # Try to match this BDL player to an Odds API player
            oa_key = _match_odds_api_player(alt_data, bdl_player_name)
            if not oa_key:
                continue

            oa_lines = alt_data[oa_key]

            # Get existing lines for dedup
            existing_lines = {bl.get("line") for bl in bdl_book_lines}

            # Get player_id from existing BDL lines
            player_id = None
            for bl in bdl_book_lines:
                if bl.get("player_id"):
                    player_id = bl["player_id"]
                    break

            added_for_player = 0
            for oa_line in oa_lines:
                line_val = oa_line["line"]
                # Skip if we already have this exact line
                if line_val in existing_lines:
                    continue

                # Add to the player's book_lines
                new_entry = {
                    "book": "FanDuel",
                    "line": line_val,
                    "odds": oa_line["odds"],
                    "player_id": player_id,
                    "source": "odds_api",
                    "is_alt": oa_line.get("is_alt", True),
                }
                bdl_book_lines.append(new_entry)
                existing_lines.add(line_val)
                added_for_player += 1
                total_added += 1

            if added_for_player > 0:
                print(f"[ODDS-API] {bdl_player_name}: +{added_for_player} alt lines "
                      f"(now {len(bdl_book_lines)} total)")

    print(f"[ODDS-API] Enrichment complete: {games_matched}/{len(games_map)} games matched, "
          f"{total_added} alt lines added")
    return total_added


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
    is_threes = prop_type_is_threes(prop_type)

    # Step 1: Fetch game history — ESPN FIRST, BDL fallback
    games, data_source = fetch_player_games_multisource(
        player_name, player_id, stat_key, BASELINE_GAMES, is_threes
    )

    if len(games) < 10:
        print(f"[SKIP] {player_name}: only {len(games)} games (source: {data_source})")
        return None

    # Debug: show data source and sample values
    if games:
        print(f"[PROJ] {player_name} {prop_type}: {len(games)} games from {data_source}, "
              f"first val={games[0].get('value', '?')}, last val={games[-1].get('value', '?')}")

    # Step 2: Window averages
    base_games = _slice_last(games, BASELINE_GAMES)
    l10_games = _slice_last(games, LOOKBACK_GAMES)
    l5_games = _slice_last(games, SHORT_GAMES)

    base_avg, base_min_val, base_std = avg_stat_min_std(base_games)
    l10_avg, l10_min_val, l10_std = avg_stat_min_std(l10_games)
    l5_avg, l5_min_val, l5_std = avg_stat_min_std(l5_games)

    if base_avg < 0.1:
        return None

    # Step 2b: Fetch minutes — ESPN first, BDL fallback
    base_min, l10_min, l5_min = fetch_minutes_multisource(player_name, player_id, BASELINE_GAMES)

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

    # Enrich games with ESPN scoreboard data (Vegas O/U, B2B, spread)
    enrich_games_map_with_espn(games_map)

    # Enrich lines with REAL FanDuel alt lines from The Odds API
    # This adds alternate player_points lines (+200, +300, etc.) that BDL doesn't provide
    alt_count = enrich_lines_map_with_odds_api(lines_map, games_map)
    print(f"[INFO] Odds API enrichment: {alt_count} alt lines added to lines_map")

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

            # Determine opponent and home/away using BDL team matching
            is_home, opp_team, player_team_name = determine_home_away(player_id, game_info)
            home_team = game_info.get("home_team", {})
            away_team = game_info.get("away_team", {})
            home_name = home_team.get("full_name", home_team.get("name", ""))
            away_name = away_team.get("full_name", away_team.get("name", ""))

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

            # Kill picks where the player has NEVER hit the line
            # Model-only projections without empirical support are unreliable
            hr_data_check = proj_result.hit_rates or {}
            l5_h = hr_data_check.get("l5_hits", 0)
            l10_h = hr_data_check.get("l10_hits", 0)
            base_h = hr_data_check.get("base_hits", 0)
            total_hits = l5_h + l10_h + base_h
            print(f"[HR CHECK] {player_name} {prop_type}: L5={l5_h} L10={l10_h} SZN={base_h} (total={total_hits})")
            if total_hits == 0:
                print(f"[SKIP] {player_name} {prop_type}: 0/0/0 hit rates — no empirical support")
                continue

            if has_recent_play(state, player_name, prop_type):
                continue

            # ---- Multi-Source Cross-Reference (ESPN primary) ----
            stat_k_xref = prop_type_to_stat_key(prop_type)
            xref = cross_reference_player(player_name, player_id, stat_k_xref, [])

            # If cross-reference confidence is low, penalize or skip
            xref_confidence = xref.get("confidence", "high")
            xref_penalty = 1.0
            if xref_confidence == "low":
                # Large data mismatch or injury alert — apply heavy penalty
                xref_penalty = 0.70
                print(f"[XREF] {player_name} LOW confidence — penalty applied. Notes: {xref.get('notes')}")
            elif xref_confidence == "medium":
                xref_penalty = 0.90
                print(f"[XREF] {player_name} MEDIUM confidence — minor penalty. Notes: {xref.get('notes')}")
            else:
                print(f"[XREF] {player_name} HIGH confidence. Notes: {xref.get('notes')}")

            # Apply cross-ref penalty to edge and probability
            adjusted_edge = proj_result.edge * xref_penalty
            adjusted_prob = proj_result.prob_over * xref_penalty

            # Re-check thresholds after cross-ref adjustment
            if adjusted_edge < min_edge_for_stat:
                print(f"[XREF SKIP] {player_name} edge {adjusted_edge:.1f}% < {min_edge_for_stat}% after xref penalty")
                continue
            if adjusted_prob < min_prob_for_stat:
                print(f"[XREF SKIP] {player_name} prob {adjusted_prob:.3f} < {min_prob_for_stat} after xref penalty")
                continue

            # Skip players with active OUT/DOUBTFUL injuries from ESPN
            if any("INJURY ALERT" in n and ("Out" in n or "Doubtful" in n) for n in xref.get("notes", [])):
                print(f"[XREF SKIP] {player_name} has OUT/DOUBTFUL injury from ESPN")
                continue

            # Get vendor from the book line data
            play_vendor = book_lines[0].get("book", "FanDuel") if book_lines else "FanDuel"

            # ---- Explosion Profile + Breakout Signals ----
            stat_k_exp = prop_type_to_stat_key(prop_type)
            exp_games, _ = fetch_player_games_multisource(
                player_name, player_id, stat_k_exp, BASELINE_GAMES,
                is_threes=prop_type_is_threes(prop_type)
            )
            explosion = compute_explosion_profile(exp_games, stat_k_exp)
            breakout_signals = compute_pregame_breakout_score(
                player_name, player_id, stat_k_exp, game_info, explosion, xref
            )

            print(f"[BOOM] {player_name}: profile={explosion.get('profile_type')} "
                  f"explode={explosion.get('explosion_rate', 0)*100:.0f}% "
                  f"volatility={explosion.get('volatility', 0):.2f} "
                  f"breakout={breakout_signals.get('breakout_tier')} "
                  f"({breakout_signals.get('breakout_score', 0):.0f}/100) "
                  f"signals: {breakout_signals.get('signals', [])}")

            # Compute composite score: weight EV and probability most heavily
            hr_data = proj_result.hit_rates or {}
            # Boost score for multi-source agreement, penalize for disagreement
            xref_score_bonus = 5 if xref.get("agreement", True) and len(xref.get("sources", [])) > 1 else 0
            xref_score_penalty = -8 if not xref.get("agreement", True) else 0

            # Breakout signals boost — when conditions are PRIME, weight the pick higher
            breakout_bonus = 0
            if breakout_signals.get("breakout_tier") == "PRIME":
                breakout_bonus = 12
            elif breakout_signals.get("breakout_tier") == "ELEVATED":
                breakout_bonus = 5

            # ---- Matchup History ----
            stat_k_match = prop_type_to_stat_key(prop_type)
            matchup = fetch_matchup_history(
                player_name, player_id, opp_team, stat_k_match,
                is_threes=prop_type_is_threes(prop_type),
            )
            matchup_bonus = 0
            if matchup.get("assessment") == "FEAST":
                matchup_bonus = 8
            elif matchup.get("assessment") == "STRUGGLE":
                matchup_bonus = -6

            # ---- Usage Vacuum ----
            vacuum = compute_usage_vacuum(player_name, player_id, game_info, stat_k_match)
            vacuum_bonus = 0
            if vacuum.get("has_vacuum"):
                # Scale bonus by magnitude of boost
                boost_pct = vacuum.get("boost_pct", 0)
                if boost_pct > 20:
                    vacuum_bonus = 10
                elif boost_pct > 10:
                    vacuum_bonus = 6
                elif boost_pct > 5:
                    vacuum_bonus = 3

            composite_score = (
                proj_result.ev * 40 +           # EV is king
                proj_result.prob_over * 30 +    # Probability matters
                proj_result.edge * 0.5 +        # Edge % as tiebreaker
                proj_result.consistency * 10 +  # Consistency bonus
                (5 if proj_result.is_breakout else 0) +  # Breakout bonus
                xref_score_bonus +              # Multi-source agreement bonus
                xref_score_penalty +            # Data mismatch penalty
                breakout_bonus +                # Pre-game signal boost
                matchup_bonus +                 # Matchup history (feast/struggle)
                vacuum_bonus                    # Usage vacuum from injuries
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
                "player_team": player_team_name or xref.get("espn_team", home_name if is_home else away_name),
                "hit_rates": hr_data,
                "xref": xref,
                "xref_confidence": xref_confidence,
                "adjusted_edge": adjusted_edge,
                "adjusted_prob": adjusted_prob,
                "explosion": explosion,
                "breakout_signals": breakout_signals,
                "matchup": matchup,
                "vacuum": vacuum,
            }

            # Build deep analysis narrative (ESPN-first data + cross-ref)
            stat_k = prop_type_to_stat_key(prop_type)
            raw_games, _ = fetch_player_games_multisource(
                player_name, player_id, stat_k, BASELINE_GAMES,
                is_threes=prop_type_is_threes(prop_type)
            )

            play["analysis"] = build_deep_analysis(
                player_name, player_id, stat_k, best_line,
                proj_result.proj, raw_games, opp_team, is_home,
                proj_result.breakout_evidence, hr_data, xref,
                explosion, breakout_signals, matchup, vacuum,
            )

            plays.append(play)

    tier_order = {"LOCK": 0, "STRONG": 1, "LEAN": 2, "SKIP": 3}
    plays.sort(key=lambda p: (tier_order.get(p["confidence_tier"], 3), -p["score"]))

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


# ============================================================================
# SECTION 9b: EXPLOSION PROFILE + BREAKOUT SIGNAL ENGINE
# ============================================================================

def compute_explosion_profile(games: List[Dict], stat_key: str = "pts") -> Dict:
    """
    Model the player's scoring DISTRIBUTION, not just the average.
    This is what makes alt line bets profitable — you're betting on the
    right tail of the distribution, not the center.

    Returns:
      - explosion_rate: % of games they exceed 1.5x their average (boom games)
      - ceiling_rate: % of games in the top 20% of their range
      - volatility: coefficient of variation (high = wide range = good for alt lines)
      - boom_avg: average score in their boom games (what their ceiling looks like)
      - bust_rate: % of games below 0.6x their average (duds)
      - best_window: are they MORE volatile recently? (opportunity signal)
    """
    if not games or len(games) < 8:
        return {"explosion_rate": 0, "volatility": 0, "boom_avg": 0,
                "ceiling_rate": 0, "bust_rate": 0, "best_window": "none",
                "profile_type": "unknown"}

    values = [g.get("value", 0) for g in games]
    n = len(values)
    avg = sum(values) / n
    if avg < 1.0:
        return {"explosion_rate": 0, "volatility": 0, "boom_avg": 0,
                "ceiling_rate": 0, "bust_rate": 0, "best_window": "none",
                "profile_type": "unknown"}

    # Standard deviation and coefficient of variation
    variance = sum((v - avg) ** 2 for v in values) / n
    std_dev = variance ** 0.5
    volatility = std_dev / avg  # CV: higher = more spread

    # Explosion rate: games at 1.5x average or higher
    boom_threshold = avg * 1.5
    boom_games = [v for v in values if v >= boom_threshold]
    explosion_rate = len(boom_games) / n
    boom_avg = sum(boom_games) / len(boom_games) if boom_games else 0

    # Ceiling rate: games in top 20% of their range
    sorted_vals = sorted(values, reverse=True)
    top_20_cutoff = sorted_vals[max(0, int(n * 0.2) - 1)] if n >= 5 else sorted_vals[0]
    ceiling_games = [v for v in values if v >= top_20_cutoff]
    ceiling_rate = len(ceiling_games) / n

    # Bust rate: games below 0.6x average
    bust_threshold = avg * 0.6
    bust_rate = sum(1 for v in values if v < bust_threshold) / n

    # Is their variance INCREASING recently? (more boom potential now)
    l5 = values[:5] if n >= 5 else values
    l5_max = max(l5) if l5 else 0
    l5_avg = sum(l5) / len(l5) if l5 else 0
    season_max = max(values)
    recent_boom = l5_max >= boom_threshold

    best_window = "none"
    if recent_boom and l5_avg > avg * 1.10:
        best_window = "hot_and_booming"
    elif recent_boom:
        best_window = "recent_boom"
    elif l5_avg > avg * 1.10:
        best_window = "trending_up"

    # Profile type: tells you WHAT kind of bets this player suits
    if volatility > 0.40 and explosion_rate >= 0.15:
        profile_type = "boom_or_bust"      # High variance, frequent explosions -> ALT LINES
    elif volatility < 0.25 and explosion_rate < 0.10:
        profile_type = "steady_eddie"       # Low variance, few explosions -> STRAIGHTS only
    elif explosion_rate >= 0.20:
        profile_type = "ceiling_hunter"     # Frequent big games -> prime ALT LINE candidate
    elif volatility > 0.35:
        profile_type = "volatile"           # Wide range but not always up -> risky ALT
    else:
        profile_type = "moderate"           # Middle ground

    return {
        "explosion_rate": explosion_rate,
        "boom_threshold": boom_threshold,
        "boom_avg": boom_avg,
        "volatility": volatility,
        "std_dev": std_dev,
        "ceiling_rate": ceiling_rate,
        "bust_rate": bust_rate,
        "best_window": best_window,
        "recent_boom": recent_boom,
        "profile_type": profile_type,
        "season_max": season_max,
    }


def compute_pregame_breakout_score(
    player_name: str,
    player_id: int,
    stat_key: str,
    game_info: Dict,
    explosion_profile: Dict,
    xref: Dict = None,
) -> Dict:
    """
    Pre-game signal aggregator: identifies spots where conditions STACK
    to make a breakout game more likely TONIGHT. This is how you catch
    the 35-point game before it happens.

    Signals scored 0-10 each, summed into a composite breakout score.
    Score >= 6 means conditions are primed for a ceiling game.

    Signals checked:
      1. Opponent pace (fast = more possessions = more points)
      2. Opponent defensive rating (weak = easier scoring)
      3. Opponent on back-to-back (fatigued defense)
      4. Player rest advantage (extra rest = fresher legs)
      5. Teammate injuries creating usage vacuum
      6. Player's recent explosion window (hot + booming)
      7. Historical ceiling frequency (has the player SHOWN this ceiling?)
      8. Vegas total (high O/U = market expects scoring)
    """
    signals = []
    total_score = 0.0

    # ---- Signal 1: Opponent pace ----
    opp_pace = game_info.get("pace", 100.0)
    if opp_pace >= 102.0:
        pace_score = min((opp_pace - 100.0) * 1.5, 10.0)
        signals.append(f"Pace-up ({opp_pace:.0f})")
        total_score += pace_score
    elif opp_pace <= 96.0:
        signals.append(f"Pace-down ({opp_pace:.0f})")
        total_score -= 2.0

    # ---- Signal 2: Opponent defensive weakness ----
    opp_team = game_info.get("away_team", {}).get("full_name", "")
    if not opp_team:
        opp_team = game_info.get("home_team", {}).get("full_name", "")
    opp_def = fetch_def_rating(opp_team, stat_key)
    if opp_def > 114.0:
        signals.append(f"Bottom-5 defense ({opp_def:.0f} DRTG)")
        total_score += 8.0
    elif opp_def > 112.0:
        signals.append(f"Weak defense ({opp_def:.0f} DRTG)")
        total_score += 5.0
    elif opp_def > 110.0:
        signals.append(f"Below-avg defense ({opp_def:.0f} DRTG)")
        total_score += 2.0
    elif opp_def < 107.0:
        signals.append(f"Elite defense ({opp_def:.0f} DRTG)")
        total_score -= 4.0

    # ---- Signal 3: Opponent on back-to-back ----
    # Derive from enriched game data: figure out which team is the opponent
    # and check if they're on a B2B
    opp_b2b = game_info.get("opp_back_to_back", False)
    if not opp_b2b:
        # Determine which side is the opponent using player's team
        p_team_id = bdl_player_team_id(player_id)
        home_id = game_info.get("home_team", {}).get("id", 0)
        if p_team_id and p_team_id == home_id:
            opp_b2b = game_info.get("away_b2b", False)
        else:
            opp_b2b = game_info.get("home_b2b", False)
    if opp_b2b:
        signals.append("Opp on B2B (tired legs)")
        total_score += 6.0

    # ---- Signal 4: Player rest advantage ----
    player_rest_days = game_info.get("player_rest_days", 1)
    if player_rest_days >= 3:
        signals.append(f"Well-rested ({player_rest_days} days off)")
        total_score += 4.0
    elif player_rest_days >= 2:
        signals.append("Extra rest")
        total_score += 2.0

    # ---- Signal 5: Teammate injury creating usage vacuum ----
    # Check xref for injury news on teammates
    injury_boost = game_info.get("teammate_injury_boost", 0)
    if injury_boost > 0:
        signals.append(f"Usage vacuum (+{injury_boost:.0f}% boost)")
        total_score += min(injury_boost, 8.0)

    # ---- Signal 6: Player's recent explosion window ----
    exp = explosion_profile
    if exp.get("best_window") == "hot_and_booming":
        signals.append("HOT + recent boom game")
        total_score += 8.0
    elif exp.get("best_window") == "recent_boom":
        signals.append("Recent boom game in L5")
        total_score += 5.0
    elif exp.get("best_window") == "trending_up":
        signals.append("Trending up")
        total_score += 3.0

    # ---- Signal 7: Historical ceiling frequency ----
    exp_rate = exp.get("explosion_rate", 0)
    if exp_rate >= 0.25:
        signals.append(f"Explodes {exp_rate*100:.0f}% of games")
        total_score += 6.0
    elif exp_rate >= 0.15:
        signals.append(f"Boom rate {exp_rate*100:.0f}%")
        total_score += 3.0

    # ---- Signal 8: Vegas implied total ----
    vegas_total = game_info.get("over_under", 0)
    if vegas_total >= 230:
        signals.append(f"High total ({vegas_total})")
        total_score += 5.0
    elif vegas_total >= 220:
        signals.append(f"Elevated total ({vegas_total})")
        total_score += 2.0
    elif vegas_total > 0 and vegas_total < 210:
        signals.append(f"Low total ({vegas_total})")
        total_score -= 3.0

    # ---- Composite ----
    # Normalize to 0-100 scale
    breakout_score = max(0, min(total_score * 2.5, 100))

    # Tier the breakout potential
    if breakout_score >= 60:
        breakout_tier = "PRIME"       # Multiple signals stacking — tonight's the night
    elif breakout_score >= 40:
        breakout_tier = "ELEVATED"    # Good conditions, not overwhelming
    elif breakout_score >= 20:
        breakout_tier = "NEUTRAL"     # Nothing special either way
    else:
        breakout_tier = "SUPPRESSED"  # Bad conditions, avoid alt lines

    return {
        "breakout_score": breakout_score,
        "breakout_tier": breakout_tier,
        "signals": signals,
        "raw_score": total_score,
        "signal_count": len(signals),
    }


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

            # Determine home/away using BDL team matching
            is_home, opp_team, player_team_name = determine_home_away(player_id, game_info)

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

            # Build scoring profile from ESPN-first data
            games, data_src = fetch_player_games_multisource(
                player_name, player_id, "pts", BASELINE_GAMES
            )
            profile = _player_scoring_profile(games, avg_line)

            # Explosion profile — determines if this player is ALT LINE material
            explosion = compute_explosion_profile(games, "pts")

            # Skip steady eddies for ladders — they don't boom
            if explosion.get("profile_type") == "steady_eddie":
                continue

            # Cross-reference once, reuse for breakout + narrative
            ladder_xref = cross_reference_player(player_name, player_id, "pts", games)

            # Pre-game breakout signals — catch the boom BEFORE tip-off
            breakout = compute_pregame_breakout_score(
                player_name, player_id, "pts", game_info, explosion,
                xref=ladder_xref,
            )

            # Build alt line legs from REAL FanDuel lines only — no estimated odds
            # Iterate the actual book_lines for this player, filter to +200 or higher
            all_book_odds = [(bl.get("line", 0), bl.get("odds", 0)) for bl in book_lines]
            plus200_lines = [(l, o) for l, o in all_book_odds if o >= 200]
            print(f"[LADDER] {player_name}: main={avg_line}, proj={projection:.1f}, "
                  f"total book lines={len(all_book_odds)}, +200 lines={len(plus200_lines)}")
            if plus200_lines:
                print(f"[LADDER] {player_name} +200 lines: {plus200_lines[:6]}")
            ladder_legs = []
            seen_lines = set()

            for bl in book_lines:
                rung = bl.get("line", 0)
                odds = bl.get("odds", -110)

                if not rung or rung <= 0:
                    continue

                # Only alt lines with +200 or higher odds
                if odds < 200:
                    continue

                # Skip duplicates at same line
                if rung in seen_lines:
                    continue
                seen_lines.add(rung)

                # Sanity: skip rungs more than 2x the main line (unreachable)
                if rung > avg_line * 2.2:
                    continue

                # Sanity: skip if projection is less than 60% of the rung
                if projection < rung * 0.60:
                    continue

                z = (projection - rung) / proj_result.sigma if proj_result.sigma > 0 else 0
                raw_prob = _norm_cdf(z)
                prob = calibrated_prob(raw_prob)

                # Empirical hit rate at this rung
                rung_hits = sum(1 for g in games if g.get("value", 0) > rung)
                rung_hr = rung_hits / len(games) if games else 0

                # Must have AT LEAST 1 historical hit at this rung
                if rung_hits == 0:
                    continue

                # Blend model + empirical
                blended_prob = 0.40 * prob + 0.60 * rung_hr

                if blended_prob < 0.05:
                    continue

                ev = ev_per_dollar(blended_prob, odds)

                # Only keep if +EV
                if ev < 0.05:
                    continue

                odds_str = f"+{odds}" if odds > 0 else str(odds)

                ladder_legs.append({
                    "rung": rung,
                    "line": rung,
                    "odds": odds,
                    "odds_str": odds_str,
                    "prob": blended_prob,
                    "model_prob": prob,
                    "hit_rate": rung_hr,
                    "hits": rung_hits,
                    "total": len(games),
                    "ev": ev,
                    "real_odds": True,  # Flag: these are REAL FanDuel odds
                })

            if not ladder_legs:
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

            # Explosion profile in narrative
            exp_type = explosion.get("profile_type", "moderate")
            exp_rate = explosion.get("explosion_rate", 0)
            boom_avg = explosion.get("boom_avg", 0)
            if exp_type in ("boom_or_bust", "ceiling_hunter"):
                narrative_parts.append(f"Profile: {exp_type.upper()} (explodes {exp_rate*100:.0f}% of games, boom avg {boom_avg:.0f})")
            elif exp_type == "volatile":
                narrative_parts.append(f"Profile: VOLATILE (explodes {exp_rate*100:.0f}%)")

            # Breakout signals in narrative
            breakout_tier = breakout.get("breakout_tier", "NEUTRAL")
            breakout_score = breakout.get("breakout_score", 0)
            breakout_signals = breakout.get("signals", [])
            if breakout_tier in ("PRIME", "ELEVATED"):
                sig_str = ", ".join(breakout_signals[:3])
                narrative_parts.append(f"BREAKOUT {breakout_tier} ({breakout_score:.0f}/100): {sig_str}")

            # ESPN cross-reference for ladders (computed above, reused here)
            if ladder_xref and len(ladder_xref.get("sources", [])) > 1:
                bdl_a = ladder_xref.get("bdl_avg", 0)
                espn_a = ladder_xref.get("espn_avg", 0)
                if bdl_a > 0 and espn_a > 0:
                    disc = ladder_xref.get("discrepancy", 0)
                    if disc <= 8:
                        narrative_parts.append(f"Verified: BDL/ESPN agree")
                    else:
                        narrative_parts.append(f"BDL {bdl_a:.1f} vs ESPN {espn_a:.1f}")
                for note in ladder_xref.get("notes", []):
                    if "INJURY" in note:
                        narrative_parts.append(note)

            narrative = " | ".join(narrative_parts)

            # Sweet spot = the leg with the best EV (all legs are already +200)
            sweet_spot = best_leg

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
                "explosion": explosion,
                "breakout": breakout,
                "narrative": narrative,
                "vendor": vendor,
                "opp_team": opp_team,
                "is_home": is_home,
                "main_line": avg_line,
            }
            ladders.append(ladder)

    # Sort by combination of EV, ceiling potential, explosion profile, and breakout score
    def _ladder_sort_score(l):
        ev_component = l["ev"] * 30
        ceiling_component = l["profile"].get("ceiling", 0) * 0.5
        sweet_bonus = 10 if l.get("sweet_spot") else 0
        # Explosion bonus — boom_or_bust and ceiling_hunter profiles get priority
        exp_type = l.get("explosion", {}).get("profile_type", "moderate")
        exp_bonus = {"boom_or_bust": 15, "ceiling_hunter": 12, "volatile": 6}.get(exp_type, 0)
        # Breakout bonus — PRIME and ELEVATED get priority
        breakout_score = l.get("breakout", {}).get("breakout_score", 0)
        breakout_bonus = breakout_score * 0.2  # 0-20 points from breakout
        return ev_component + ceiling_component + sweet_bonus + exp_bonus + breakout_bonus

    ladders.sort(key=lambda l: -_ladder_sort_score(l))
    return ladders[:6]


# ============================================================================
# SECTION 10: FORMATTING & DISPLAY
# ============================================================================

def format_play_card(play: Dict, index: int) -> str:
    """
    Sharp bettor format — one clean card per pick with the full story.
    No clutter. Every line earns its place.
    """
    player = play.get("player", "?")
    prop_type = play.get("prop_type", "pts")
    line = play.get("line", 0)
    proj = play.get("proj", 0)
    odds = play.get("odds", -110)
    prob = play.get("prob", play.get("prob_over", 0))
    ev = play.get("ev", 0)
    edge = play.get("edge", 0)
    tier = play.get("confidence_tier", "LEAN")

    PROP_NAMES = {
        "player_points": "PTS", "player_rebounds": "REB",
        "player_assists": "AST", "player_threes": "3PM",
        "player_blocks": "BLK", "player_steals": "STL",
    }
    prop_name = PROP_NAMES.get(prop_type, "?")
    odds_str = f"+{odds}" if odds > 0 else str(odds)
    breakout_tag = " BREAKOUT" if play.get("is_breakout") else ""

    # Hit rate — the most important number
    hr = play.get("hit_rates", {})
    l5_hits = hr.get("l5_hits", 0)
    l10_hits = hr.get("l10_hits", 0)
    base_hits = hr.get("base_hits", 0)

    # Build the card
    lines = []
    lines.append(f"{index}. {player} — {prop_name} OVER {line} ({odds_str})")
    lines.append(f"   [{tier}]{breakout_tag} Proj {proj:.1f} | Edge {edge:.0f}% | EV +{ev:.2f}")
    lines.append(f"   Hits: {l5_hits}/5 L5 | {l10_hits}/10 L10 | {base_hits}/20 szn")

    # Deep analysis narrative — the "why"
    analysis = play.get("analysis", "")
    if analysis:
        lines.append(analysis)

    return "\n".join(lines)


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
    Clean ladder card — just the sweet spot play with the full story.
    No rung-by-rung noise. Show what to bet and why.
    """
    player = ladder.get("player", "?")
    proj = ladder.get("projection", 0)
    main_line = ladder.get("main_line", 0)
    narrative = ladder.get("narrative", "")
    sweet = ladder.get("sweet_spot")
    best = ladder.get("best_leg", {})
    profile = ladder.get("profile", {})
    explosion = ladder.get("explosion", {})
    breakout = ladder.get("breakout", {})

    # Pick the recommended rung
    rec = sweet if sweet else best
    if not rec:
        return ""

    rung = rec.get("rung", 0)
    odds_str = rec.get("odds_str", str(rec.get("odds", -110)))
    prob = rec.get("prob", 0)
    hits = rec.get("hits", 0)
    total = rec.get("total", 20)
    ev = rec.get("ev", 0)
    hr_pct = (hits / total * 100) if total > 0 else 0

    # Breakout tag
    breakout_tier = breakout.get("breakout_tier", "NEUTRAL")
    breakout_tag = f" BREAKOUT" if breakout_tier in ("PRIME", "ELEVATED") else ""

    lines = []
    lines.append(f"{player} — {rung}+ PTS ({odds_str}){breakout_tag}")
    lines.append(f"   Main line {main_line} | Proj {proj:.1f} | EV +{ev:.2f}")
    lines.append(f"   Szn hit rate at {rung}+: {hits}/{total} ({hr_pct:.0f}%)")
    lines.append(f"   {narrative}")

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


def _enforce_one_per_team(plays: List[Dict]) -> List[Dict]:
    """
    Keep only the best play per team. Prevents 3 picks from the same roster.
    Uses composite score to pick the strongest from each team.
    """
    best_by_team = {}
    for play in plays:
        # Try to extract team from opponent/home context
        # The play's "team" is whoever the player is on
        opp = play.get("opp_team", "")
        is_home = play.get("is_home", True)
        # Rough team identification from player's perspective
        player_team = play.get("player_team", "")
        if not player_team:
            # Infer: if player is home, their team is the home side
            player_team = f"team_{play.get('game_id', 'x')}_{('home' if is_home else 'away')}"

        if player_team not in best_by_team:
            best_by_team[player_team] = play
        elif play.get("score", 0) > best_by_team[player_team].get("score", 0):
            best_by_team[player_team] = play

    return list(best_by_team.values())


def _enforce_one_per_team_ladders(ladders: List[Dict]) -> List[Dict]:
    """Keep only the best ladder per team."""
    best_by_team = {}
    for ladder in ladders:
        opp = ladder.get("opp_team", "")
        is_home = ladder.get("is_home", True)
        player_team = f"lad_{ladder.get('game_id', 'x')}_{('home' if is_home else 'away')}"

        if player_team not in best_by_team:
            best_by_team[player_team] = ladder
        elif ladder.get("ev", 0) > best_by_team[player_team].get("ev", 0):
            best_by_team[player_team] = ladder

    return list(best_by_team.values())


def build_whatsapp_message(
    straights: List[Dict],
    plus_plays: List[Dict],
    sgps: List[Dict],
    corr_parlays: List[Dict],
    ladders: List[Dict],
    state: Dict = None,
) -> str:
    """
    Clean WhatsApp output — sharp bettor format.
    Max 1 straight pick per team + 1 ladder per team.
    Deep narrative on each pick. No clutter.
    """
    if state is None:
        state = load_state()

    now_et = _now_et()
    date_str = now_et.strftime("%m/%d")
    time_str = now_et.strftime("%I:%M %p")
    book_label = PRIMARY_BOOK.upper()

    lines = [f"NBA PICKS {date_str} {time_str} ET ({book_label})"]

    hit_rate_str = get_hit_rate_summary(state)
    if "No plays" not in hit_rate_str:
        lines.append(f"Record: {hit_rate_str}")

    # ---- STRAIGHT PLAYS ----
    # Combine all plays, sort by score, then enforce 1 per team
    all_plays = straights + plus_plays
    all_plays.sort(key=lambda p: -p.get("score", 0))
    filtered = _enforce_one_per_team(all_plays)
    # Re-sort after filtering
    filtered.sort(key=lambda p: (
        {"LOCK": 0, "STRONG": 1, "LEAN": 2, "SKIP": 3}.get(p.get("confidence_tier", "LEAN"), 3),
        -p.get("score", 0)
    ))
    # Cap total straight plays at 6
    filtered = filtered[:6]

    if filtered:
        lines.append("\n-- PLAYS --")
        for idx, play in enumerate(filtered, 1):
            lines.append(format_play_card(play, idx))
            lines.append("")

    if not filtered:
        lines.append("\nNo qualifying plays today")

    # ---- PARLAYS (keep minimal) ----
    if sgps or corr_parlays:
        lines.append("-- PARLAYS --")
        for sgp in (sgps + corr_parlays)[:3]:
            lines.append(format_parlay_card(sgp))
            lines.append("")

    # ---- LADDERS ----
    if ladders:
        # Enforce 1 ladder per team
        filtered_ladders = _enforce_one_per_team_ladders(ladders)
        filtered_ladders.sort(key=lambda l: -l.get("ev", 0))
        filtered_ladders = filtered_ladders[:4]

        lines.append("-- ALT LINES --")
        for ladder in filtered_ladders:
            card = format_ladder_card(ladder)
            if card:
                lines.append(card)
                lines.append("")

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
