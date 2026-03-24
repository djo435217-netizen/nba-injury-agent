import os
import json
import re
import time
import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
from twilio.rest import Client

STATE_FILE = "state.json"
ET = ZoneInfo("America/New_York")

# -------------------- REQUIRED ENV --------------------
TWILIO_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
SPORTRADAR_KEY = os.environ.get("SPORTRADAR_API_KEY", "").strip()
BALLDONTLIE_API_KEY = os.environ["BALLDONTLIE_API_KEY"].strip()

FROM_WHATSAPP = os.environ.get("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
TO_WHATSAPP = f"whatsapp:{os.environ['MY_WHATSAPP_NUMBER']}"

twilio = Client(TWILIO_SID, TWILIO_TOKEN)

# -------------------- CONFIG --------------------
TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
MAX_BODY_CHARS = int(os.environ.get("MAX_BODY_CHARS", "1500"))

BOOK_VENDOR_RAW = os.environ.get("BOOK_VENDORS", os.environ.get("BOOK_VENDOR", "fanduel,draftkings,fanatics,caesars")).strip().lower()
BOOK_VENDORS = [v.strip() for v in BOOK_VENDOR_RAW.split(",") if v.strip()]

# Prop types -- add rebounds,assists to unlock correlated parlays
# The correlated parlay engine needs both pts AND reb/ast lines to build combos
# If PROP_TYPES only has points, correlated parlays will still work but
# need to fetch reb/ast separately (slower) -- better to include them here
PROP_TYPES_RAW = os.environ.get("PROP_TYPES", "points,threes,rebounds,assists").strip().lower()
PROP_TYPES = [p.strip() for p in PROP_TYPES_RAW.split(",") if p.strip()]

ENABLE_INJURY_TRIGGERS = os.environ.get("ENABLE_INJURY_TRIGGERS", "1") == "1"
ENABLE_SLATE_SCAN = os.environ.get("ENABLE_SLATE_SCAN", "1") == "1"

# Exposure caps
MAX_PLAYS_PER_TEAM = int(os.environ.get("MAX_PLAYS_PER_TEAM", "2"))
MAX_PLAYS_PER_GAME = int(os.environ.get("MAX_PLAYS_PER_GAME", "6"))

# Card composition caps
MAX_INJURY_PLAYS = int(os.environ.get("MAX_INJURY_PLAYS", "6"))
MAX_LINEUPNEWS_PLAYS = int(os.environ.get("MAX_LINEUPNEWS_PLAYS", "4"))
MAX_SLATE_PLAYS = int(os.environ.get("MAX_SLATE_PLAYS", "6"))

# Consensus + steam + EV + market respect
MIN_VENDORS_FOR_CONSENSUS = int(os.environ.get("MIN_VENDORS_FOR_CONSENSUS", "1"))
MIN_SHARP_VENDORS = int(os.environ.get("MIN_SHARP_VENDORS", "0"))
SHARP_VENDORS_RAW = os.environ.get(
    "SHARP_VENDORS",
    "draftkings,caesars,fanatics,fanduel,betmgm,bet365,pointsbet,hardrock,betparx,betway,betrivers,rebet",
).strip().lower()
SHARP_VENDORS = {x.strip() for x in SHARP_VENDORS_RAW.split(",") if x.strip()}

CONSENSUS_VENDORS = SHARP_VENDORS.copy()

ENABLE_STEAM = os.environ.get("ENABLE_STEAM", "0") == "1"
STEAM_MIN_SCORE = float(os.environ.get("STEAM_MIN_SCORE", "1.0"))
STEAM_MAX_AGE_MIN = int(os.environ.get("STEAM_MAX_AGE_MIN", "240"))

# Output sizing
MIN_PER_MARKET = int(os.environ.get("MIN_PER_MARKET", "0"))
MAX_PER_MARKET = int(os.environ.get("MAX_PER_MARKET", "10"))
MAX_TOTAL_PLAYS = int(os.environ.get("MAX_TOTAL_PLAYS", "12"))

# Windows
BASELINE_GAMES = int(os.environ.get("BASELINE_GAMES", "20"))
LOOKBACK_GAMES = int(os.environ.get("LOOKBACK_GAMES", "8"))
SHORT_GAMES = int(os.environ.get("SHORT_GAMES", "3"))

# Model thresholds
MIN_EDGE = float(os.environ.get("MIN_EDGE", "2.5"))
MIN_PROB = float(os.environ.get("MIN_PROB", "0.62"))
# IMPROVEMENT: Lowered STD_FLOOR from 5.0 -> 3.5
# Old floor of 5.0 was over-penalizing consistent scorers by pushing prob
# estimates artificially toward 50%, causing us to miss high-confidence overs.
STD_FLOOR = float(os.environ.get("STD_FLOOR", "3.5"))

# EV filter
EV_MIN = float(os.environ.get("EV_MIN", "0.00"))
VALUE_EDGE_MIN = float(os.environ.get("VALUE_EDGE_MIN", "0.00"))

# Guardrails
MIN_L10_MIN = float(os.environ.get("MIN_L10_MIN", "8"))
LINE_MIN_GAP = float(os.environ.get("LINE_MIN_GAP", "12.0"))
ROLE_DROP_MIN = float(os.environ.get("ROLE_DROP_MIN", "5.0"))
ROLE_DROP_RATE = float(os.environ.get("ROLE_DROP_RATE", "0.08"))

# Injury vacancy requirements
IMPACT_STATUSES_RAW = os.environ.get("IMPACT_STATUSES", "out,doubtful,questionable").strip()
IMPACT_STATUSES = {x.strip().lower() for x in IMPACT_STATUSES_RAW.split(",") if x.strip()}
IMPACT_ONLY_CHANGES = os.environ.get("IMPACT_ONLY_CHANGES", "1") == "1"

MIN_VAC_MIN = float(os.environ.get("MIN_VAC_MIN", "10.0"))
MIN_VAC_STAT = float(os.environ.get("MIN_VAC_PTS", os.environ.get("MIN_VAC_STAT", "6.0")))
BOOST_CAP_RATE = float(os.environ.get("BOOST_CAP_RATE", "0.20"))
BOOST_CAP_STAT = float(os.environ.get("BOOST_CAP_STAT", "5.5"))
BOOST_CAP_MIN = float(os.environ.get("BOOST_CAP_MIN", "6.0"))

# Cooldown
BET_COOLDOWN_MIN = int(os.environ.get("BET_COOLDOWN_MIN", "180"))
EDGE_JUMP_TO_RESEND = float(os.environ.get("EDGE_JUMP_TO_RESEND", "1.5"))

# Runtime guard
RUN_MAX_SECONDS = int(os.environ.get("RUN_MAX_SECONDS", "250"))
STAT_BATCH_SIZE = int(os.environ.get("STAT_BATCH_SIZE", "90"))

DEBUG_PROP_SAMPLE_TYPES = os.environ.get("DEBUG_PROP_SAMPLE_TYPES", "0").strip().lower()

# Plus odds
PLUS_ODDS_MIN = float(os.environ.get("PLUS_ODDS_MIN", "-105"))
PLUS_ODDS_TOPN = int(os.environ.get("PLUS_ODDS_TOPN", "5"))

PLUS_HUNT_ENABLED = os.environ.get("PLUS_HUNT_ENABLED", "1") == "1"
PLUS_HUNT_MIN_PROB = float(os.environ.get("PLUS_HUNT_MIN_PROB", "0.54"))
PLUS_HUNT_MIN_VALUE_EDGE = float(os.environ.get("PLUS_HUNT_MIN_VALUE_EDGE", "0.02"))
PLUS_HUNT_MIN_EV = float(os.environ.get("PLUS_HUNT_MIN_EV", "0.01"))
PLUS_HUNT_TOPN = int(os.environ.get("PLUS_HUNT_TOPN", "7"))

# ---- CORRELATED PARLAY ENGINE ----
# Goal: find 2-leg parlays where both legs are positively correlated
# Types:
#   1. Same player, two props (pts + reb, pts + ast) -- strongest correlation
#   2. Same team, two players in high-total game -- moderate correlation
#   3. Breakout player + team total -- if player goes off, team likely wins big
#
# The edge: correlated legs should NOT be priced as independent events
# A player scoring 25pts is much more likely to also grab 7 rebounds
# Books use SGP correlation tax but often underprice strong correlations
ENABLE_CORR_PARLAY = os.environ.get("ENABLE_CORR_PARLAY", "1") == "1"
CORR_PARLAY_MIN_PROB = float(os.environ.get("CORR_PARLAY_MIN_PROB", "0.52"))
CORR_PARLAY_MIN_EV = float(os.environ.get("CORR_PARLAY_MIN_EV", "0.15"))
CORR_PARLAY_MIN_ODDS = int(os.environ.get("CORR_PARLAY_MIN_ODDS", "150"))
CORR_PARLAY_MAX_LEGS = int(os.environ.get("CORR_PARLAY_MAX_LEGS", "3"))
CORR_PARLAY_TOPN = int(os.environ.get("CORR_PARLAY_TOPN", "4"))
CORR_PARLAY_BET = float(os.environ.get("CORR_PARLAY_BET", "100"))  # flat bet per parlay

# ---- SITUATIONAL EDGE ENGINE ----
# These are educated guess factors that books systematically underprice
# No complex math -- just real basketball situations that create edge
ENABLE_SITUATION_ENGINE = os.environ.get("ENABLE_SITUATION_ENGINE", "1") == "1"

# Revenge game boost -- player vs team that traded/cut them
REVENGE_BOOST = float(os.environ.get("REVENGE_BOOST", "3.5"))

# Schedule fatigue -- 4th game in 6 nights, road game, late arrival city
FATIGUE_PENALTY = float(os.environ.get("FATIGUE_PENALTY", "2.5"))

# Bounce back game -- player scored under 10 in last game, historically bounces back
BOUNCE_BACK_BOOST = float(os.environ.get("BOUNCE_BACK_BOOST", "2.8"))

# Spotlight game -- nationally televised, player historically performs better
SPOTLIGHT_BOOST = float(os.environ.get("SPOTLIGHT_BOOST", "1.5"))

# Back from injury -- first 3 games back, books underestimate return form
RETURN_BOOST = float(os.environ.get("RETURN_BOOST", "2.0"))

# Nationally televised games today
NATIONAL_TV_TEAMS = os.environ.get("NATIONAL_TV_TEAMS", "").strip().lower()

# ---- BREAKOUT DETECTOR ----
# Catches role players having breakout games or on hot streaks
# When L3 avg is significantly above baseline, weight it much more heavily
BREAKOUT_MIN_RATIO = float(os.environ.get("BREAKOUT_MIN_RATIO", "1.25"))  # L3 must be 25% above base
BREAKOUT_L3_WEIGHT = float(os.environ.get("BREAKOUT_L3_WEIGHT", "0.65"))  # in breakout mode L3 gets 65% weight
BREAKOUT_BASE_WEIGHT = float(os.environ.get("BREAKOUT_BASE_WEIGHT", "0.15"))  # base gets only 15%
BREAKOUT_L10_WEIGHT = float(os.environ.get("BREAKOUT_L10_WEIGHT", "0.20"))  # L10 gets 20%
BREAKOUT_MIN_GAMES = int(os.environ.get("BREAKOUT_MIN_GAMES", "3"))  # need at least 3 L3 games

# Role expansion -- when player's L3 minutes >> L10 minutes (getting more playing time)
ROLE_EXP_MIN_DELTA = float(os.environ.get("ROLE_EXP_MIN_DELTA", "4.0"))  # 4+ extra min in L3 vs L10

# ---- LADDER BETS ----
# Ladders: player scores 10+, 20+, 30+ points sequentially
# Best edge is in the MIDDLE leg (20+) -- books often underprice it
# FanDuel posts these as separate over_under markets at specific milestone lines
ENABLE_LADDER_SCAN = os.environ.get("ENABLE_LADDER_SCAN", "1") == "1"
LADDER_MIN_PROJ = float(os.environ.get("LADDER_MIN_PROJ", "18.0"))  # only chase ladders for high scorers
LADDER_MIN_PROB = float(os.environ.get("LADDER_MIN_PROB", "0.55"))  # minimum prob for any ladder leg
LADDER_MIN_EV = float(os.environ.get("LADDER_MIN_EV", "0.05"))  # minimum EV per leg
LADDER_LEGS = [10.5, 15.5, 20.5, 25.5, 30.5]  # standard ladder rungs
LADDER_TOPN = int(os.environ.get("LADDER_TOPN", "5"))

# ---- HIGH ODDS HUNTER (+250 and above) ----
# Specifically hunts FanDuel alternate lines at +250 or better
# These are the best value plays -- high payout, model says high probability
HIGH_ODDS_HUNT_ENABLED = os.environ.get("HIGH_ODDS_HUNT_ENABLED", "1") == "1"
HIGH_ODDS_MIN = float(os.environ.get("HIGH_ODDS_MIN", "250"))
HIGH_ODDS_MIN_PROB = float(os.environ.get("HIGH_ODDS_MIN_PROB", "0.45"))
HIGH_ODDS_MIN_EV = float(os.environ.get("HIGH_ODDS_MIN_EV", "0.05"))
HIGH_ODDS_TOPN = int(os.environ.get("HIGH_ODDS_TOPN", "5"))
HIGH_ODDS_VENDOR = os.environ.get("HIGH_ODDS_VENDOR", "fanduel").strip().lower()

# Injury boost for high odds -- when star is out, backup player alternate lines
# are often posted late and at inflated plus odds
INJURY_HIGH_ODDS_BOOST = float(os.environ.get("INJURY_HIGH_ODDS_BOOST", "0.12"))

# Threes
THREES_BETA_BINOM = os.environ.get("THREES_BETA_BINOM", "1") == "1"
THREES_MIN_ATT_GAMES = int(os.environ.get("THREES_MIN_ATT_GAMES", "8"))

# LineupExperts
LINEUPEXPERTS = os.environ.get("LINEUPEXPERTS", "0") == "1"
LINEUPEXPERTS_KEY = os.environ.get("LINEUPEXPERTS_KEY", os.environ.get("LINEUPEXPERTS_API_KEY", "")).strip()
LINEUPEXPERTS_BASE_URL = os.environ.get("LINEUPEXPERTS_BASE_URL", "https://api.lineupexperts.com/v1").strip()
LINEUPEXPERTS_BASE_URL = LINEUPEXPERTS_BASE_URL.strip().strip('"').strip("'").rstrip("/")
LINEUPEXPERTS_TIMEOUT = int(os.environ.get("LINEUPEXPERTS_TIMEOUT", "12"))
LINEUPEXPERTS_MAX_ITEMS = int(os.environ.get("LINEUPEXPERTS_MAX_ITEMS", "200"))
NEWS_LOOKBACK_HOURS = int(os.environ.get("NEWS_LOOKBACK_HOURS", "36"))
NEWS_MIN_CONFIDENCE = float(os.environ.get("NEWS_MIN_CONFIDENCE", "0.25"))

LE_DEBUG = os.environ.get("LE_DEBUG", "0") == "1"
LE_NO_TIME_FILTER = os.environ.get("LE_NO_TIME_FILTER", "0") == "1"

USE_LE_MAIN_INJURY_ENGINE = os.environ.get("USE_LE_MAIN_INJURY_ENGINE", "1") == "1"
LE_NEWS_ENGINE_ENABLED = os.environ.get("LE_NEWS_ENGINE_ENABLED", "1") == "1"
LE_NEWS_MIN_SCORE = float(os.environ.get("LE_NEWS_MIN_SCORE", "0.30"))
LE_NEWS_MIN_EFFECT = float(os.environ.get("LE_NEWS_MIN_EFFECT", "0.02"))
LE_NEWS_TOPN = int(os.environ.get("LE_NEWS_TOPN", "5"))

# Final ranking weights
FINAL_SCORE_EV_W = float(os.environ.get("FINAL_SCORE_EV_W", "30.0"))
FINAL_SCORE_VALUE_W = float(os.environ.get("FINAL_SCORE_VALUE_W", "100.0"))
FINAL_SCORE_EDGE_W = float(os.environ.get("FINAL_SCORE_EDGE_W", "2.0"))
FINAL_SCORE_MINCONF_W = float(os.environ.get("FINAL_SCORE_MINCONF_W", "8.0"))
FINAL_SCORE_MATCHUP_W = float(os.environ.get("FINAL_SCORE_MATCHUP_W", "10.0"))
# FINAL_SCORE_LE_W moved to SGP config section above
FINAL_SCORE_STABILITY_W = float(os.environ.get("FINAL_SCORE_STABILITY_W", "8.0"))
FINAL_SCORE_VOL_PENALTY_W = float(os.environ.get("FINAL_SCORE_VOL_PENALTY_W", "6.0"))

# -------------------- NEW: CONTEXT FACTORS --------------------
# Home court advantage: historically ~2.5 pts for the home team player
HOME_COURT_BOOST = float(os.environ.get("HOME_COURT_BOOST", "1.5"))
# B2B fatigue penalty: second night of back-to-back games
B2B_PENALTY = float(os.environ.get("B2B_PENALTY", "2.0"))
# Rest advantage: pts boost per extra rest day beyond 1 (capped at 3 days)
REST_BOOST_PER_DAY = float(os.environ.get("REST_BOOST_PER_DAY", "0.5"))
# Game total thresholds for pace scaling
GAME_TOTAL_HIGH = float(os.environ.get("GAME_TOTAL_HIGH", "228.0"))
GAME_TOTAL_LOW = float(os.environ.get("GAME_TOTAL_LOW", "212.0"))
GAME_TOTAL_BOOST = float(os.environ.get("GAME_TOTAL_BOOST", "0.04"))
# Blowout risk: if spread >= this, penalize favorite star players in 4th qtr
BLOWOUT_SPREAD_MIN = float(os.environ.get("BLOWOUT_SPREAD_MIN", "12.0"))
BLOWOUT_PENALTY = float(os.environ.get("BLOWOUT_PENALTY", "1.5"))
# Usage rate weight in projection (0=off, 1=full weight)
USAGE_RATE_WEIGHT = float(os.environ.get("USAGE_RATE_WEIGHT", "0.15"))
# Closing line value tracking
ENABLE_CLV_TRACKING = os.environ.get("ENABLE_CLV_TRACKING", "1") == "1"

# ---- KELLY CRITERION BET SIZING ----
# Set your bankroll in Render environment as BANKROLL=2000
# Quarter Kelly is standard -- aggressive enough to grow, safe enough to survive
BANKROLL = float(os.environ.get("BANKROLL", "2000"))
KELLY_FRACTION = float(os.environ.get("KELLY_FRACTION", "0.25"))  # 1/4 Kelly
MIN_BET = float(os.environ.get("MIN_BET", "50"))
MAX_BET = float(os.environ.get("MAX_BET", "300"))
ENABLE_BET_SIZING = os.environ.get("ENABLE_BET_SIZING", "1") == "1"

# ---- ODDS FILTER ----
# Only show plays at this price or better (-105 = accept up to -105, reject -110+)
MAX_JUICE = float(os.environ.get("MAX_JUICE", "-110"))

# ---- SAME GAME PARLAY ----
ENABLE_SGP = os.environ.get("ENABLE_SGP", "1") == "1"
SGP_MIN_PLAYS = int(os.environ.get("SGP_MIN_PLAYS", "2"))
SGP_MAX_PLAYS = int(os.environ.get("SGP_MAX_PLAYS", "3"))
SGP_MIN_COMBINED_EV = float(os.environ.get("SGP_MIN_COMBINED_EV", "0.08"))

# Star player injury threshold -- injuries to these avg pts trigger bigger boosts
STAR_PLAYER_MIN_AVG = float(os.environ.get("STAR_PLAYER_MIN_AVG", "20.0"))
STAR_VACANCY_MULT = float(os.environ.get("STAR_VACANCY_MULT", "1.8"))

# LE news weight in final score -- increase to surface news-driven plays more
# Was 8.0, now 20.0 -- lineup news is one of the strongest edges we have
FINAL_SCORE_LE_W = float(os.environ.get("FINAL_SCORE_LE_W", "20.0"))
# Pace factor weight: how much opponent pace affects projection (0.0 = off)
PACE_FACTOR_WEIGHT = float(os.environ.get("PACE_FACTOR_WEIGHT", "0.5"))
# Enable opponent defensive stats adjustment
ENABLE_OPP_DEF_ADJ = os.environ.get("ENABLE_OPP_DEF_ADJ", "1") == "1"

# -------------------- NEW: RECENCY WEIGHTS --------------------
# IMPROVEMENT: Increased L3 recency weight from 0.20 -> 0.30, reduced base from 0.45 -> 0.35
# L3 form is the strongest short-term signal for player points props.
# Hot/cold streaks are real -- the market often under-adjusts for them.
PROJ_WEIGHT_BASE = float(os.environ.get("PROJ_WEIGHT_BASE", "0.35"))
PROJ_WEIGHT_L10 = float(os.environ.get("PROJ_WEIGHT_L10", "0.35"))
PROJ_WEIGHT_L3 = float(os.environ.get("PROJ_WEIGHT_L3", "0.30"))

# -------------------- NEW: RESULT TRACKING --------------------
# Track sent plays so we can measure hit rate over time
ENABLE_RESULT_TRACKING = os.environ.get("ENABLE_RESULT_TRACKING", "1") == "1"
RESULTS_FILE = os.environ.get("RESULTS_FILE", "results_log.json")

# -------------------- RUNTIME DEADLINE --------------------
RUN_START = time.time()

# -------------------- PLAYER TEAM OVERRIDES --------------------
PLAYER_TEAM_OVERRIDES = {
    "tidjane salaun": "Hornets",
    "vit krejci": "Hawks",
    "daniel gafford": "Mavericks",
    "isaac okoro": "Cavaliers",
    "lachlan olbrich": "Bulls",
    "jaylen brown": "Celtics",
    "stephen curry": "Warriors",
    "draymond green": "Warriors",
    "marcus smart": "Grizzlies",
    "nick richards": "Suns",
    "jaden mcdaniels": "Timberwolves",
}

# -------------------- TEAM DEFENSE RATINGS --------------------
# Points allowed per game vs position (updated manually or via future API hook).
# Format: team_short -> {"pts_allowed_pg": float, "pace": float}
# pts_allowed_pg: average points allowed per game this season (league avg ~113)
# pace: possessions per 48 min (league avg ~99)
# Positive pace = faster = more possessions = more pts scoring opportunities
# pts_vs_pg = pts allowed to point guards per game
# pts_vs_sg = pts allowed to shooting guards
# pts_vs_sf = pts allowed to small forwards
# pts_vs_pf = pts allowed to power forwards
# pts_vs_c  = pts allowed to centers
# League avg per position: PG=27, SG=24, SF=22, PF=22, C=24
TEAM_DEFENSE_RATINGS: dict[str, dict] = {
    "Celtics":      {"pts_allowed_pg": 107.5, "pace": 97.5,  "pts_vs_pg": 24.1, "pts_vs_sg": 21.8, "pts_vs_sf": 19.9, "pts_vs_pf": 20.1, "pts_vs_c": 21.6},
    "Bucks":        {"pts_allowed_pg": 112.0, "pace": 100.2, "pts_vs_pg": 26.8, "pts_vs_sg": 23.5, "pts_vs_sf": 22.1, "pts_vs_pf": 22.4, "pts_vs_c": 23.2},
    "Knicks":       {"pts_allowed_pg": 110.3, "pace": 96.8,  "pts_vs_pg": 25.4, "pts_vs_sg": 22.7, "pts_vs_sf": 21.3, "pts_vs_pf": 21.8, "pts_vs_c": 22.9},
    "Heat":         {"pts_allowed_pg": 109.4, "pace": 96.3,  "pts_vs_pg": 25.1, "pts_vs_sg": 22.3, "pts_vs_sf": 20.8, "pts_vs_pf": 21.2, "pts_vs_c": 22.4},
    "76ers":        {"pts_allowed_pg": 113.2, "pace": 98.1,  "pts_vs_pg": 27.3, "pts_vs_sg": 24.1, "pts_vs_sf": 22.6, "pts_vs_pf": 22.9, "pts_vs_c": 23.8},
    "Nets":         {"pts_allowed_pg": 117.8, "pace": 99.4,  "pts_vs_pg": 29.2, "pts_vs_sg": 25.8, "pts_vs_sf": 24.3, "pts_vs_pf": 24.7, "pts_vs_c": 25.6},
    "Raptors":      {"pts_allowed_pg": 115.6, "pace": 98.7,  "pts_vs_pg": 28.1, "pts_vs_sg": 24.8, "pts_vs_sf": 23.4, "pts_vs_pf": 23.8, "pts_vs_c": 24.7},
    "Cavaliers":    {"pts_allowed_pg": 108.1, "pace": 97.2,  "pts_vs_pg": 24.8, "pts_vs_sg": 22.1, "pts_vs_sf": 20.6, "pts_vs_pf": 20.9, "pts_vs_c": 22.1},
    "Pacers":       {"pts_allowed_pg": 118.4, "pace": 104.5, "pts_vs_pg": 29.6, "pts_vs_sg": 26.2, "pts_vs_sf": 24.7, "pts_vs_pf": 25.1, "pts_vs_c": 26.0},
    "Bulls":        {"pts_allowed_pg": 114.9, "pace": 99.8,  "pts_vs_pg": 27.8, "pts_vs_sg": 24.6, "pts_vs_sf": 23.1, "pts_vs_pf": 23.5, "pts_vs_c": 24.4},
    "Pistons":      {"pts_allowed_pg": 116.2, "pace": 98.9,  "pts_vs_pg": 28.4, "pts_vs_sg": 25.1, "pts_vs_sf": 23.7, "pts_vs_pf": 24.1, "pts_vs_c": 25.0},
    "Hawks":        {"pts_allowed_pg": 116.8, "pace": 101.3, "pts_vs_pg": 28.7, "pts_vs_sg": 25.4, "pts_vs_sf": 23.9, "pts_vs_pf": 24.3, "pts_vs_c": 25.2},
    "Hornets":      {"pts_allowed_pg": 118.1, "pace": 100.6, "pts_vs_pg": 29.3, "pts_vs_sg": 25.9, "pts_vs_sf": 24.4, "pts_vs_pf": 24.8, "pts_vs_c": 25.7},
    "Magic":        {"pts_allowed_pg": 108.9, "pace": 96.1,  "pts_vs_pg": 25.2, "pts_vs_sg": 22.4, "pts_vs_sf": 21.0, "pts_vs_pf": 21.3, "pts_vs_c": 22.5},
    "Wizards":      {"pts_allowed_pg": 119.3, "pace": 100.4, "pts_vs_pg": 29.8, "pts_vs_sg": 26.4, "pts_vs_sf": 24.9, "pts_vs_pf": 25.3, "pts_vs_c": 26.2},
    "Thunder":      {"pts_allowed_pg": 106.8, "pace": 98.3,  "pts_vs_pg": 23.8, "pts_vs_sg": 21.2, "pts_vs_sf": 19.7, "pts_vs_pf": 20.0, "pts_vs_c": 21.3},
    "Nuggets":      {"pts_allowed_pg": 111.7, "pace": 97.9,  "pts_vs_pg": 26.4, "pts_vs_sg": 23.4, "pts_vs_sf": 22.0, "pts_vs_pf": 22.3, "pts_vs_c": 23.1},
    "Timberwolves": {"pts_allowed_pg": 108.2, "pace": 97.6,  "pts_vs_pg": 24.9, "pts_vs_sg": 22.1, "pts_vs_sf": 20.7, "pts_vs_pf": 21.0, "pts_vs_c": 22.2},
    "Jazz":         {"pts_allowed_pg": 118.9, "pace": 101.8, "pts_vs_pg": 29.5, "pts_vs_sg": 26.1, "pts_vs_sf": 24.6, "pts_vs_pf": 25.0, "pts_vs_c": 25.9},
    "Suns":         {"pts_allowed_pg": 113.5, "pace": 99.1,  "pts_vs_pg": 27.5, "pts_vs_sg": 24.3, "pts_vs_sf": 22.8, "pts_vs_pf": 23.2, "pts_vs_c": 24.1},
    "Clippers":     {"pts_allowed_pg": 111.1, "pace": 97.4,  "pts_vs_pg": 26.2, "pts_vs_sg": 23.2, "pts_vs_sf": 21.8, "pts_vs_pf": 22.1, "pts_vs_c": 23.0},
    "Lakers":       {"pts_allowed_pg": 113.8, "pace": 99.5,  "pts_vs_pg": 27.6, "pts_vs_sg": 24.4, "pts_vs_sf": 22.9, "pts_vs_pf": 23.3, "pts_vs_c": 24.2},
    "Warriors":     {"pts_allowed_pg": 113.1, "pace": 100.1, "pts_vs_pg": 27.3, "pts_vs_sg": 24.1, "pts_vs_sf": 22.7, "pts_vs_pf": 23.0, "pts_vs_c": 23.9},
    "Kings":        {"pts_allowed_pg": 116.3, "pace": 101.9, "pts_vs_pg": 28.5, "pts_vs_sg": 25.2, "pts_vs_sf": 23.7, "pts_vs_pf": 24.1, "pts_vs_c": 25.0},
    "Mavericks":    {"pts_allowed_pg": 110.4, "pace": 98.6,  "pts_vs_pg": 26.1, "pts_vs_sg": 23.1, "pts_vs_sf": 21.7, "pts_vs_pf": 22.0, "pts_vs_c": 22.9},
    "Rockets":      {"pts_allowed_pg": 109.3, "pace": 98.4,  "pts_vs_pg": 25.7, "pts_vs_sg": 22.8, "pts_vs_sf": 21.4, "pts_vs_pf": 21.7, "pts_vs_c": 22.6},
    "Spurs":        {"pts_allowed_pg": 119.7, "pace": 102.1, "pts_vs_pg": 30.1, "pts_vs_sg": 26.6, "pts_vs_sf": 25.1, "pts_vs_pf": 25.5, "pts_vs_c": 26.4},
    "Pelicans":     {"pts_allowed_pg": 112.6, "pace": 98.8,  "pts_vs_pg": 26.9, "pts_vs_sg": 23.8, "pts_vs_sf": 22.4, "pts_vs_pf": 22.7, "pts_vs_c": 23.6},
    "Grizzlies":    {"pts_allowed_pg": 114.7, "pace": 99.3,  "pts_vs_pg": 27.9, "pts_vs_sg": 24.7, "pts_vs_sf": 23.2, "pts_vs_pf": 23.6, "pts_vs_c": 24.5},
    "Blazers":      {"pts_allowed_pg": 118.5, "pace": 101.5, "pts_vs_pg": 29.4, "pts_vs_sg": 26.0, "pts_vs_sf": 24.5, "pts_vs_pf": 24.9, "pts_vs_c": 25.8},
}

LEAGUE_AVG_PTS_ALLOWED = 113.5
LEAGUE_AVG_PACE = 99.0
# League avg pts allowed per position
LEAGUE_AVG_VS_POS = {"pg": 27.0, "sg": 24.0, "sf": 22.0, "pf": 22.0, "c": 24.0}

# Player position map -- add players as you learn their positions
# Format: cleaned_name -> position (pg/sg/sf/pf/c)
PLAYER_POSITION_MAP: dict[str, str] = {
    "donovan mitchell": "sg",
    "de aaron fox": "pg",
    "deaaron fox": "pg",
    "stephen curry": "pg",
    "lebron james": "sf",
    "kevin durant": "sf",
    "giannis antetokounmpo": "pf",
    "joel embiid": "c",
    "nikola jokic": "c",
    "luka doncic": "pg",
    "jayson tatum": "sf",
    "jaylen brown": "sg",
    "damian lillard": "pg",
    "trae young": "pg",
    "shai gilgeous alexander": "pg",
    "devin booker": "sg",
    "anthony edwards": "sg",
    "tyrese haliburton": "pg",
    "bam adebayo": "c",
    "karl-anthony towns": "c",
    "karl anthony towns": "c",
    "pascal siakam": "pf",
    "julius randle": "pf",
    "zion williamson": "pf",
    "brandon ingram": "sf",
    "ja morant": "pg",
    "desmond bane": "sg",
    "jaren jackson jr": "pf",
    "cade cunningham": "pg",
    "lauri markkanen": "pf",
    "walker kessler": "c",
    "victor wembanyama": "c",
    "harrison barnes": "sf",
    "domantas sabonis": "c",
    "de aaron fox": "pg",
    "keegan murray": "sf",
    "coby white": "pg",
    "zach lavine": "sg",
    "nikola vucevic": "c",
    "scottie barnes": "pf",
    "rg3": "c",
    "evan mobley": "c",
    "darius garland": "pg",
    "jalen brunson": "pg",
    "og anunoby": "sf",
    "mikal bridges": "sg",
    "anthony davis": "c",
    "austin reaves": "sg",
    "tyrese maxey": "pg",
    "paul george": "sf",
    "kawhi leonard": "sf",
    "james harden": "pg",
}


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
    s = s.replace("\u2019", "'")
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


def prop_type_to_stat_key(prop_type: str) -> str:
    """
    Map prop type to the correct BallDontLie stat field.
    This is the critical fix -- each prop type must use its own stat history.

    points  -> pts
    threes  -> fg3m (handled separately via bdl_last_n_games_threes)
    rebounds -> reb (total rebounds = offensive + defensive)
    assists  -> ast
    blocks   -> blk
    steals   -> stl
    turnovers -> turnover
    """
    mapping = {
        "points": "pts",
        "pts": "pts",
        "rebounds": "reb",
        "reb": "reb",
        "total_rebounds": "reb",
        "assists": "ast",
        "ast": "ast",
        "blocks": "blk",
        "blk": "blk",
        "steals": "stl",
        "stl": "stl",
        "turnovers": "turnover",
        "turnover": "turnover",
    }
    return mapping.get(prop_type.lower().strip(), "pts")


def prop_type_is_threes(prop_type: str) -> bool:
    return prop_type.lower().strip() in (
        "threes", "three_pointers_made", "fg3m", "3pt_made",
        "three_points_made", "three_point_field_goals_made"
    )


def median_stat(games, n=None) -> float:
    """
    Median is better than mean for volatile players.
    A player with games [19,20,21,42] has mean=25.5 but median=20.5.
    The median is the true floor of consistent performance.
    Use this for ladder bet floor estimates and for filtering out fluky averages.
    """
    if not games:
        return 0.0
    sl = games[-min(len(games), n):] if n else games
    vals = sorted(x[1] for x in sl)
    mid = len(vals) // 2
    if len(vals) % 2 == 1:
        return float(vals[mid])
    return float((vals[mid-1] + vals[mid]) / 2.0)


def floor_ceiling(games, n=10) -> tuple[float, float]:
    """
    Floor = 25th percentile of recent scoring (what player almost always hits)
    Ceiling = 75th percentile (what player often reaches on good nights)
    Used for ladder bets -- floor tells you the safe rung, ceiling the aggressive rung.
    """
    if not games:
        return 0.0, 0.0
    sl = sorted(x[1] for x in games[-min(len(games), n):])
    n_sl = len(sl)
    if n_sl < 4:
        return float(sl[0]), float(sl[-1])
    floor_idx = max(0, int(n_sl * 0.25))
    ceil_idx = min(n_sl-1, int(n_sl * 0.75))
    return float(sl[floor_idx]), float(sl[ceil_idx])


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
        return {"players": {}, "sent_bets": {}, "market": {}}
    try:
        with open(STATE_FILE, "r") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return {"players": {}, "sent_bets": {}, "market": {}}
        raw.setdefault("players", {})
        raw.setdefault("sent_bets", {})
        raw.setdefault("market", {})
        return raw
    except Exception:
        return {"players": {}, "sent_bets": {}, "market": {}}


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


def _chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def _safe_rate(stat_avg: float, min_avg: float) -> float:
    return float(stat_avg) / max(float(min_avg), 1e-6)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# -------------------- NEW: OPPONENT DEFENSIVE ADJUSTMENT --------------------
def kelly_bet_size(prob_over: float, over_odds: float) -> float:
    """
    Quarter Kelly criterion bet sizing.

    Kelly formula: f = (bp - q) / b
      b = decimal odds payout (e.g. +100 = 1.0, -110 = 0.909)
      p = your estimated probability of winning
      q = 1 - p

    Quarter Kelly = full Kelly x 0.25 for safety.
    Capped between MIN_BET and MAX_BET.

    Examples at $2000 bankroll:
      +100 odds, 73% prob -> Kelly = (1.0*0.73 - 0.27)/1.0 = 0.46 -> 1/4 = 11.5% -> $230
      -105 odds, 65% prob -> Kelly = (0.952*0.65 - 0.35)/0.952 = 0.28 -> 1/4 = 7% -> $140
      -115 odds, 60% prob -> Kelly = (0.87*0.60 - 0.40)/0.87 = 0.14 -> 1/4 = 3.5% -> $70
    """
    if not ENABLE_BET_SIZING:
        return 0.0
    try:
        b = american_to_payout(over_odds)
        p = float(prob_over)
        q = 1.0 - p
        if b <= 0 or p <= 0:
            return MIN_BET
        kelly_full = (b * p - q) / b
        if kelly_full <= 0:
            return 0.0  # negative Kelly = do not bet
        kelly_quarter = kelly_full * KELLY_FRACTION
        raw_bet = kelly_quarter * BANKROLL
        return _clamp(round(raw_bet / 5) * 5, MIN_BET, MAX_BET)  # round to $5
    except Exception:
        return MIN_BET


def passes_juice_filter(over_odds: float) -> bool:
    """
    Filter out plays where the juice is too high.
    MAX_JUICE=-105 means we only accept -105 or better (less juice).
    Accepts: +200, +100, -100, -103, -105
    Rejects: -108, -110, -115, -120
    """
    try:
        odds = float(over_odds)
        # Plus odds always pass
        if odds >= 0:
            return True
        # Negative odds: accept if less negative than MAX_JUICE
        # e.g. -103 > -105 (less negative) = passes
        return odds >= float(MAX_JUICE)
    except Exception:
        return True


def find_sgp_opportunities(plays: list, games_map: dict) -> list:
    """
    Same-Game Parlay detector.

    Finds 2-3 player overs from the SAME game that are positively correlated:
    - Players on the same team in a high-total game (both benefit from pace)
    - Or both teams' leading scorers in a shootout game

    Correlation logic:
    - Two players on the same team = positive correlation (+15% combined prob boost)
    - High game total (228+) = additional boost to both overs
    - Avoid: players on opposite teams (negatively correlated -- one team wins = other loses)

    SGP payout estimated as: (p1 * p2 * correlation_factor)
    True SGP odds vary by book but we estimate based on individual odds.
    """
    if not ENABLE_SGP or len(plays) < SGP_MIN_PLAYS:
        print(f"[SGP] Skipped: ENABLE_SGP={ENABLE_SGP} plays={len(plays)}")
        return []

    # Group plays by game
    by_game = {}
    for p in plays:
        gid = p.get("gid", 0)
        if gid:
            by_game.setdefault(gid, []).append(p)

    print(f"[SGP] Checking {len(by_game)} games for same-game opportunities")
    for gid, gplays in by_game.items():
        print(f"[SGP]   gid={gid}: {len(gplays)} plays -- {[p['player_name'] for p in gplays]}")

    sgps = []
    for gid, game_plays in by_game.items():
        if len(game_plays) < 2:
            continue

        game = games_map.get(int(gid), {})
        home_team = normalize_team_name(game.get("home", ""))
        away_team = normalize_team_name(game.get("away", ""))

        # Get game total from odds cache
        odds_data = GAME_ODDS_CACHE.get(int(gid), {})
        game_total = odds_data.get("total", 0.0)

        # Group by team
        home_plays = [p for p in game_plays if normalize_team_name(p.get("team", "")) == home_team]
        away_plays = [p for p in game_plays if normalize_team_name(p.get("team", "")) == away_team]

        for team_plays in [home_plays, away_plays]:
            if len(team_plays) < 2:
                continue

            team_name_sgp = team_plays[0].get("team", "")

            # IMPROVEMENT: Require high game total for same-team SGP
            # Two players on the same team both going over requires the whole
            # team to score big -- that needs a high-total game (lots of possessions)
            # Low total = slow pace = harder for both players to hit their numbers
            if game_total > 0:
                if game_total < 215.0:
                    # Very low total game -- skip SGP
                    continue
                elif game_total < 225.0:
                    # Medium total -- only do 2-leg SGPs with strong individual EVs
                    max_legs = 2
                    min_individual_ev = 0.25
                else:
                    # High total (228+) -- green light for 2-3 leg SGPs
                    max_legs = SGP_MAX_PLAYS
                    min_individual_ev = 0.15
            else:
                # No total data -- be conservative
                max_legs = 2
                min_individual_ev = 0.20

            # Filter to players with sufficient individual EV
            eligible = [p for p in team_plays if p.get("ev", 0) >= min_individual_ev]
            if len(eligible) < 2:
                continue

            # Take top plays by final_score
            team_plays_sorted = sorted(eligible, key=lambda x: x.get("final_score", 0), reverse=True)

            for n in range(2, min(max_legs + 1, len(team_plays_sorted) + 1)):
                combo = team_plays_sorted[:n]

                # Correlation factor based on game total
                # Higher total = stronger positive correlation between teammates
                if game_total >= 232:
                    corr_factor = 1.12  # very high total = strong correlation boost
                elif game_total >= 228:
                    corr_factor = 1.08  # high total = moderate boost
                elif game_total >= 222:
                    corr_factor = 1.04  # medium total = small boost
                else:
                    corr_factor = 1.00  # low total = no boost

                combined_prob = 1.0
                for p in combo:
                    combined_prob *= min(0.95, p["prob_over"] * corr_factor)

                combined_ev = sum(p["ev"] for p in combo)

                if combined_ev < SGP_MIN_COMBINED_EV:
                    continue

                # Parlay odds estimate
                # Higher correlation = less SGP tax (books take less on correlated legs)
                sgp_tax = 0.80 if corr_factor >= 1.08 else 0.85
                parlay_decimal = 1.0
                for p in combo:
                    parlay_decimal *= (1.0 + american_to_payout(p["over_odds"]))
                parlay_decimal *= sgp_tax

                if parlay_decimal >= 2.0:
                    sgp_american = int((parlay_decimal - 1.0) * 100)
                else:
                    sgp_american = int(-100 / max(parlay_decimal - 1.0, 0.01))

                sgp_kelly = kelly_bet_size(combined_prob, sgp_american)

                # Build total context note
                if game_total >= 228:
                    total_context = f"HIGH-TOTAL({game_total:.0f}) -- pace favors both overs"
                elif game_total >= 220:
                    total_context = f"MED-TOTAL({game_total:.0f})"
                elif game_total > 0:
                    total_context = f"LOW-TOTAL({game_total:.0f}) -- caution"
                else:
                    total_context = "total unknown"

                sgps.append({
                    "plays": combo,
                    "gid": gid,
                    "team": team_name_sgp,
                    "combined_prob": combined_prob,
                    "combined_ev": combined_ev,
                    "sgp_odds": sgp_american,
                    "sgp_kelly": sgp_kelly,
                    "game_total": game_total,
                    "label": " + ".join(
                        f"{p['player_name']} O{p['cons_line']:.1f}"
                        for p in combo
                    ),
                    "note": (
                        f"SGP {team_name_sgp} | {total_context} | "
                        f"P={combined_prob*100:.0f}% | EV={combined_ev:+.2f} | "
                        f"est.odds={sgp_american:+d} | corr={corr_factor:.2f}"
                    ),
                })

    sgps.sort(key=lambda x: x["combined_ev"], reverse=True)
    return sgps[:3]


def get_rest_days(games: list, today_str: str) -> int:
    """
    Calculate days of rest since last game.
    0 = played yesterday (B2B), 1 = one day rest, 2+ = well rested.
    Returns -1 if unknown.
    """
    if not games:
        return -1
    try:
        today = datetime.strptime(today_str, "%Y-%m-%d").date()
        for g in reversed(games):
            try:
                gdate = datetime.strptime(g[0][:10], "%Y-%m-%d").date()
                if gdate < today:
                    return (today - gdate).days - 1
            except Exception:
                continue
    except Exception:
        pass
    return -1


def rest_day_adjustment(games: list, today_str: str) -> tuple[float, str]:
    """
    Rest advantage/penalty based on days since last game.
    B2B (0 rest days) = already handled by B2B_PENALTY.
    1 day rest = neutral baseline.
    2 days rest = +0.5 pts.
    3+ days rest = +1.0 pts.
    """
    rest = get_rest_days(games, today_str)
    if rest < 0:
        return 0.0, ""
    if rest == 0:
        return 0.0, "b2b"  # handled separately
    if rest == 1:
        return 0.0, "1d-rest"
    extra = min(rest - 1, 2)  # cap at 2 extra days
    boost = extra * REST_BOOST_PER_DAY
    return boost, f"{rest}d-rest(+{boost:.1f})"


def game_total_adjustment(game_total: float) -> tuple[float, str]:
    """
    Scale projection based on game over/under total.
    High-total games = more possessions = more scoring opportunities.
    Low-total games = slower pace = fewer points for everyone.
    """
    if game_total <= 0:
        return 0.0, ""
    if game_total >= GAME_TOTAL_HIGH:
        adj = GAME_TOTAL_BOOST
        return adj, f"high-total({game_total:.0f},+{adj*100:.0f}%)"
    if game_total <= GAME_TOTAL_LOW:
        adj = -GAME_TOTAL_BOOST
        return adj, f"low-total({game_total:.0f},{adj*100:.0f}%)"
    return 0.0, f"avg-total({game_total:.0f})"


def blowout_risk_penalty(spread: float, is_favorite: bool) -> tuple[float, str]:
    """
    Penalize star players on heavy favorites.
    In blowouts, stars sit in the 4th quarter -- kills prop totals.
    spread = absolute value of the point spread (e.g. 14.5)
    is_favorite = True if this player is on the favored team
    """
    if not is_favorite or spread < BLOWOUT_SPREAD_MIN:
        return 0.0, ""
    penalty = BLOWOUT_PENALTY * min(1.0, (spread - BLOWOUT_SPREAD_MIN) / 6.0 + 0.5)
    return -penalty, f"blowout-risk(-{penalty:.1f})"


def usage_rate_adjustment(games: list, prop_type: str) -> tuple[float, str]:
    """
    Usage rate adjustment -- players with high usage score more per minute.
    Approximated from scoring rate vs team average in available data.

    High usage (>28%): boost rate by up to 8%
    Low usage (<18%): penalize rate by up to 8%

    We estimate usage from pts_per_min relative to a typical starter (0.55 pts/min).
    """
    if prop_type != "points" or not games or USAGE_RATE_WEIGHT <= 0:
        return 0.0, ""

    sl = _slice_last(games, LOOKBACK_GAMES)
    if len(sl) < 5:
        return 0.0, ""

    avg_pts, avg_min, _ = avg_stat_min_std(sl)
    if avg_min < 10:
        return 0.0, ""

    pts_per_min = avg_pts / max(avg_min, 1e-6)
    # Typical starter scores ~0.55 pts/min in ~30 min
    BASELINE_RATE = 0.55
    usage_proxy = pts_per_min / BASELINE_RATE  # >1 = high usage, <1 = low usage

    # Scale: 20% above baseline rate = +8% boost, capped at 12%
    adj = _clamp((usage_proxy - 1.0) * 0.40 * USAGE_RATE_WEIGHT, -0.10, 0.12)

    if adj > 0.03:
        return adj, f"high-usage(+{adj*100:.0f}%)"
    if adj < -0.03:
        return adj, f"low-usage({adj*100:.0f}%)"
    return 0.0, ""


def track_closing_line_value(play: dict, now_ts: int):
    """
    Closing Line Value (CLV) tracking.
    The most important metric for long-term edge verification.

    A model that consistently beats the closing line has PROVEN edge.
    We log the line at time of bet -- compare to closing line later.

    How to use: After tip-off, note what the final line was.
    If your model consistently had the over at a lower line than closing,
    that means you got better of the number -- that is real alpha.
    """
    if not ENABLE_CLV_TRACKING:
        return
    try:
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE, "r") as f:
                log = json.load(f)
        else:
            log = {}

        date_str = datetime.fromtimestamp(now_ts, tz=ET).strftime("%Y-%m-%d")
        key = f"{play['prop_type']}|{play['player_id']}|{play.get('cons_line', play['line']):.1f}|{date_str}"

        if key in log:
            log[key]["open_line"] = play.get("cons_line", play["line"])
            log[key]["open_odds"] = play.get("over_odds")
            log[key]["close_line"] = None   # fill in manually after tip
            log[key]["clv"] = None          # close_line - open_line (positive = beat the close)

        with open(RESULTS_FILE, "w") as f:
            json.dump(log, f, indent=2, sort_keys=True)
    except Exception:
        pass


def get_game_total_from_odds(gid: int, games_map: dict) -> float:
    """
    Fetch game over/under total from BallDontLie odds.
    Returns 0.0 if not available -- caller handles gracefully.
    """
    if not gid:
        return 0.0
    try:
        resp = _bdl_get("/nba/v2/odds", params={"game_ids[]": [int(gid)]})
        for market in (resp.get("data") or []):
            mtype = (market.get("type") or "").lower()
            if "total" in mtype or "over_under" in mtype:
                total = market.get("total") or market.get("line_value")
                if total:
                    return float(total)
    except Exception:
        pass
    return 0.0


def get_spread_from_odds(gid: int, player_team: str, games_map: dict) -> tuple[float, bool]:
    """
    Fetch point spread for the game.
    Returns (spread, is_favorite) -- spread is absolute value.
    is_favorite = True if player_team is favored.
    """
    if not gid:
        return 0.0, False
    try:
        resp = _bdl_get("/nba/v2/odds", params={"game_ids[]": [int(gid)]})
        for market in (resp.get("data") or []):
            mtype = (market.get("type") or "").lower()
            if "spread" in mtype or "point_spread" in mtype:
                home_spread = market.get("home_spread") or market.get("spread")
                if home_spread is not None:
                    spread = float(home_spread)
                    game = games_map.get(gid, {})
                    home_team = normalize_team_name(game.get("home", ""))
                    pteam = normalize_team_name(player_team)
                    if pteam == home_team:
                        is_fav = spread < 0
                        return abs(spread), is_fav
                    else:
                        is_fav = spread > 0
                        return abs(spread), is_fav
    except Exception:
        pass
    return 0.0, False


def get_player_position(player_name: str) -> str:
    """Look up player position from map. Returns pg/sg/sf/pf/c or empty string."""
    key = _clean_name(player_name)
    if key in PLAYER_POSITION_MAP:
        return PLAYER_POSITION_MAP[key]
    for k, v in PLAYER_POSITION_MAP.items():
        if k in key or key in k:
            return v
    return ""



    """Look up player position from map. Returns pg/sg/sf/pf/c or empty string."""
    key = _clean_name(player_name)
    # Try direct match first
    if key in PLAYER_POSITION_MAP:
        return PLAYER_POSITION_MAP[key]
    # Try partial match for names with suffixes
    for k, v in PLAYER_POSITION_MAP.items():
        if k in key or key in k:
            return v
    return ""


def opponent_defense_adjustment(opponent_team: str, prop_type: str, player_name: str = "") -> tuple[float, str]:
    """
    Position-aware opponent defense adjustment.

    Uses position-specific pts allowed (pg/sg/sf/pf/c) when player position
    is known -- much more accurate than team-level defense rating.

    Falls back to team-level if position unknown.
    """
    if not ENABLE_OPP_DEF_ADJ or prop_type not in ("points", "pts"):
        return 0.0, "neutral"

    opp = normalize_team_name(opponent_team)
    def_stats = TEAM_DEFENSE_RATINGS.get(opp)
    if not def_stats:
        return 0.0, "no-data"

    pace_diff = def_stats["pace"] - LEAGUE_AVG_PACE
    pace_factor = (pace_diff / 2.0) * 0.01 * PACE_FACTOR_WEIGHT

    # Try position-specific adjustment first
    pos = get_player_position(player_name) if player_name else ""
    pos_key = f"pts_vs_{pos}" if pos else None

    if pos_key and pos_key in def_stats and pos in LEAGUE_AVG_VS_POS:
        pos_allowed = def_stats[pos_key]
        league_avg_pos = LEAGUE_AVG_VS_POS[pos]
        pts_diff = pos_allowed - league_avg_pos
        # Position-specific: +3 pts above avg => +4% boost
        pts_factor = (pts_diff / 3.0) * 0.04
        note_base = f"{pos.upper()}-def"
    else:
        pts_diff = def_stats["pts_allowed_pg"] - LEAGUE_AVG_PTS_ALLOWED
        pts_factor = (pts_diff / 5.0) * 0.04
        note_base = "team-def"

    adj = _clamp(pts_factor + pace_factor, -0.15, 0.15)

    if adj > 0.04:
        note = f"weak-{note_base}({opp},{adj*100:+.1f}%)"
    elif adj < -0.04:
        note = f"strong-{note_base}({opp},{adj*100:+.1f}%)"
    else:
        note = f"neutral-{note_base}({opp})"

    return adj, note


def matchup_adjustment(team_name: str, player_name: str, prop_type: str) -> tuple[float, str]:
    """Delegates to position-aware opponent_defense_adjustment."""
    return opponent_defense_adjustment(team_name, prop_type, player_name)


# -------------------- NEW: HOME/AWAY & B2B CONTEXT --------------------
def get_game_context(player_team: str, games_map: dict, gid: int) -> dict:
    """
    IMPROVEMENT: Extract home/away status and back-to-back flags.

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


def is_back_to_back(games: list, today_str: str) -> bool:
    """
    IMPROVEMENT: Detect if player played yesterday (back-to-back).
    games is a list of (date_str, val, mins) tuples sorted ascending.
    """
    if not games:
        return False
    try:
        today = datetime.strptime(today_str, "%Y-%m-%d").date()
        yesterday = today - timedelta(days=1)
        for g in reversed(games):
            try:
                gdate = datetime.strptime(g[0][:10], "%Y-%m-%d").date()
                if gdate == yesterday:
                    return True
                if gdate < yesterday:
                    break
            except Exception:
                continue
    except Exception:
        pass
    return False


# -------------------- NEW: CONSISTENCY SCORE --------------------
def consistency_score(games, n=LOOKBACK_GAMES) -> float:
    """
    IMPROVEMENT: Measures how consistently a player hits their average.
    Returns 0.0-1.0. High consistency = more reliable projection.

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


def adaptive_sigma(games, base_sigma: float, cons_score: float) -> float:
    """
    IMPROVEMENT: Scale sigma down for consistent players.
    A player with 80% consistency within +/-20% of avg gets sigma reduced by up to 25%.
    This lets us have more confidence on consistent over-achievers vs the line.
    """
    reduction = (cons_score - 0.5) * 0.5  # 0.0 at 50% cons, 0.25 at 100% cons
    return max(STD_FLOOR, base_sigma * (1.0 - reduction))


# -------------------- NEW: RESULT TRACKING --------------------
def log_play_for_tracking(play: dict, now_ts: int):
    """
    IMPROVEMENT: Persist each sent play so you can later compare proj vs actual.
    Load results_log.json, check actual scores daily, measure hit rate over time.

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


def get_hit_rate_summary() -> str:
    """
    IMPROVEMENT: Compute running hit rate from results log.
    Returns a human-readable summary string for the WhatsApp message.
    """
    if not ENABLE_RESULT_TRACKING or not os.path.exists(RESULTS_FILE):
        return ""
    try:
        with open(RESULTS_FILE, "r") as f:
            log = json.load(f)

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


# -------------------- NEW: CONFIDENCE TIER --------------------
def confidence_tier(edge: float, prob_over: float, ev: float, value_edge: float) -> str:
    """
    IMPROVEMENT: Assign a human-readable confidence tier to each pick.
    Makes WhatsApp output scannable at a glance for fast bet decisions.

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


def projected_minutes(base_min, l10_min, l3_min, min_delta, injury_boost_min=0.0, le_score=0.0):
    proj = (0.15 * base_min) + (0.55 * l10_min) + (0.30 * l3_min)

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
    if prop_type == "threes":
        return "high"
    if l10_min < 18:
        return "high"
    if l10_min < 26:
        return "medium"
    if sigma is not None and sigma >= 9.0:
        return "medium"
    return "low"


def thresholds_for_bucket(bucket: str) -> dict:
    if bucket == "high":
        return {"min_edge": 2.2, "min_prob": 0.60, "min_ev": 0.01, "min_value_edge": 0.01}
    if bucket == "medium":
        return {"min_edge": 1.8, "min_prob": 0.57, "min_ev": 0.00, "min_value_edge": 0.00}
    return {"min_edge": 1.2, "min_prob": 0.54, "min_ev": 0.00, "min_value_edge": 0.00}


def minutes_stability_ok(l10_min: float, l3_min: float, le_score: float) -> bool:
    # Allow larger minute swings -- role expansion is a GOOD signal not a bad one
    if abs(l3_min - l10_min) <= 6.0:
        return True
    if le_score >= 0.5:
        return True
    # Also OK if player is getting MORE minutes (role expansion)
    if l3_min > l10_min:
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


# -------------------- TEAM ALIASES --------------------
TEAM_ALIAS_TO_FULL = {
    "atlanta hawks": "Hawks", "hawks": "Hawks",
    "boston celtics": "Celtics", "celtics": "Celtics",
    "brooklyn nets": "Nets", "nets": "Nets",
    "charlotte hornets": "Hornets", "hornets": "Hornets",
    "chicago bulls": "Bulls", "bulls": "Bulls",
    "cleveland cavaliers": "Cavaliers", "cavaliers": "Cavaliers",
    "dallas mavericks": "Mavericks", "mavericks": "Mavericks",
    "denver nuggets": "Nuggets", "nuggets": "Nuggets",
    "detroit pistons": "Pistons", "pistons": "Pistons",
    "golden state warriors": "Warriors", "warriors": "Warriors",
    "houston rockets": "Rockets", "rockets": "Rockets",
    "indiana pacers": "Pacers", "pacers": "Pacers",
    "la clippers": "Clippers", "los angeles clippers": "Clippers", "clippers": "Clippers",
    "la lakers": "Lakers", "los angeles lakers": "Lakers", "lakers": "Lakers",
    "memphis grizzlies": "Grizzlies", "grizzlies": "Grizzlies",
    "miami heat": "Heat", "heat": "Heat",
    "milwaukee bucks": "Bucks", "bucks": "Bucks",
    "minnesota timberwolves": "Timberwolves", "timberwolves": "Timberwolves",
    "new orleans pelicans": "Pelicans", "pelicans": "Pelicans",
    "new york knicks": "Knicks", "knicks": "Knicks",
    "oklahoma city thunder": "Thunder", "thunder": "Thunder",
    "orlando magic": "Magic", "magic": "Magic",
    "philadelphia 76ers": "76ers", "76ers": "76ers", "sixers": "76ers",
    "phoenix suns": "Suns", "suns": "Suns",
    "portland trail blazers": "Blazers", "trail blazers": "Blazers", "blazers": "Blazers",
    "sacramento kings": "Kings", "kings": "Kings",
    "san antonio spurs": "Spurs", "spurs": "Spurs",
    "toronto raptors": "Raptors", "raptors": "Raptors",
    "utah jazz": "Jazz", "jazz": "Jazz",
    "washington wizards": "Wizards", "wizards": "Wizards",
}


def normalize_team_name(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return ""
    sl = s.lower()
    if sl in TEAM_ALIAS_TO_FULL:
        return TEAM_ALIAS_TO_FULL[sl]
    for k, v in TEAM_ALIAS_TO_FULL.items():
        if sl == k or sl.endswith(k) or k in sl:
            return v
    return s


def extract_team_from_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    m = re.match(r"^([A-Za-z0-9 .&'-]+?)'s\s+", text, flags=re.I)
    if m:
        return normalize_team_name(m.group(1))
    patterns = [
        r"\b([A-Za-z0-9 .&'-]+?)'s\s+[A-Za-z .'-]+:",
        r"\b([A-Za-z0-9 .&'-]+?)\s+vs\.?\s+",
        r"\bagainst\s+the\s+([A-Za-z0-9 .&'-]+)\b",
        r"\bagainst\s+([A-Za-z0-9 .&'-]+)\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            team = normalize_team_name(m.group(1))
            if team:
                return team
    return ""


def extract_team_from_url(url: str) -> str:
    u = (url or "").lower()
    if not u:
        return ""
    for alias, team in TEAM_ALIAS_TO_FULL.items():
        slug = alias.replace(" ", "-")
        if f"/{slug}/" in u or f"/{alias.replace(' ', '')}/" in u or alias in u:
            return team
    return ""


# -------------------- SPORTRADAR --------------------
def fetch_sportradar_injuries():
    if not SPORTRADAR_KEY:
        raise RuntimeError("No SPORTRADAR_API_KEY provided")
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
        team_name = normalize_team_name(team_name)
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

BDL_MAX_RETRIES = int(os.environ.get("BDL_MAX_RETRIES", "3"))
BDL_RETRY_BASE_SEC = float(os.environ.get("BDL_RETRY_BASE_SEC", "0.8"))
BDL_PER_PAGE = int(os.environ.get("BDL_PER_PAGE", "100"))
BDL_MAX_PAGES = int(os.environ.get("BDL_MAX_PAGES", "5"))

TEAM_CACHE = None
PLAYER_NAME_CACHE = {}
PLAYER_TEAM_CACHE = {}
PROPS_CACHE = {}
ADV_STATS_CACHE = {}       # pid -> list of advanced stat dicts
LINEUP_CACHE = {}          # gid -> list of lineup entries
PLAYER_POS_CACHE = {}      # pid -> position string (from API)
GAME_ODDS_CACHE = {}       # gid -> {"total": float, "spread": float, "fav_team": str}


def _bdl_get(path: str, params=None, timeout: int = 8) -> dict:
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


def bdl_games_today(now_et: datetime):
    today = now_et.strftime("%Y-%m-%d")
    resp = _bdl_get("/v1/games", params={"dates[]": [today], "per_page": 100})
    games = resp.get("data") or []
    out = {}
    for g in games:
        try:
            gid = int(g["id"])
        except Exception:
            continue
        home = normalize_team_name(((g.get("home_team") or {}).get("name") or "").strip())
        away = normalize_team_name(((g.get("visitor_team") or {}).get("name") or "").strip())
        out[gid] = {"home": home, "away": away}
    return out


def bdl_team_name_to_id():
    global TEAM_CACHE
    if TEAM_CACHE is not None:
        return TEAM_CACHE
    data = _bdl_get("/v1/teams", params={"per_page": 100})
    m = {}
    for t in data.get("data", []):
        nm = normalize_team_name((t.get("name") or "").strip())
        if nm and t.get("id") is not None:
            m[nm] = int(t["id"])
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
        if normalize_team_name((team.get("name") or "").strip()) == team_short:
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


def bdl_last_n_games_threes(player_ids, season: int, n: int):
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

    if DEBUG_PROP_SAMPLE_TYPES and (prop_type in DEBUG_PROP_SAMPLE_TYPES.split(",")) and props:
        print(f"[DEBUG] SAMPLE PROP ROW ({prop_type}, vendor={vendor or 'NO_VENDOR'}): {json.dumps(props[0])[:2000]}")

    PROPS_CACHE[key] = props
    return props


# -------------------- BDL GOD TIER: ADVANCED STATS --------------------

def bdl_fetch_advanced_stats(player_ids: list, season: int) -> dict:
    """
    Fetch advanced stats for players using BDL v2 endpoint.
    Returns dict: pid -> list of per-game advanced stat dicts.

    Key fields used:
      - usage_percentage: % of team plays used by player while on floor
      - offensive_rating: points per 100 possessions
      - defensive_rating: points allowed per 100 possessions
      - net_rating: off_rating - def_rating
      - pie: player impact estimate (composite efficiency)
      - pace: game pace (possessions per 48 min)
      - effective_field_goal_percentage: FG quality
    """
    out = {int(pid): [] for pid in player_ids}
    if not player_ids:
        return out

    # Check cache first
    uncached = [pid for pid in player_ids if int(pid) not in ADV_STATS_CACHE]
    if not uncached:
        return {int(pid): ADV_STATS_CACHE.get(int(pid), []) for pid in player_ids}

    cursor = None
    pages = 0
    while pages < BDL_MAX_PAGES and not deadline_exceeded():
        params = {
            "per_page": 100,
            "seasons[]": [season],
            "player_ids[]": uncached,
        }
        if cursor:
            params["cursor"] = cursor
        try:
            resp = _bdl_get("/nba/v2/stats/advanced", params=params)
        except Exception as e:
            print(f"[WARN] advanced stats fetch failed: {e}")
            break

        rows = resp.get("data") or []
        for row in rows:
            p = row.get("player") or {}
            pid = p.get("id")
            if pid is None:
                continue
            pid = int(pid)
            if pid not in out:
                continue

            # Cache player position from API response
            pos = (p.get("position") or "").strip().lower()
            if pos and pid not in PLAYER_POS_CACHE:
                # Normalize position (G->pg/sg, F->sf/pf, C->c)
                if pos in ("g", "pg", "sg", "point guard", "shooting guard"):
                    PLAYER_POS_CACHE[pid] = "pg" if "point" in pos else "sg"
                elif pos in ("f", "sf", "pf", "small forward", "power forward"):
                    PLAYER_POS_CACHE[pid] = "sf" if "small" in pos else "pf"
                elif pos in ("c", "center"):
                    PLAYER_POS_CACHE[pid] = "c"
                elif pos == "g":
                    PLAYER_POS_CACHE[pid] = "pg"
                elif pos == "f":
                    PLAYER_POS_CACHE[pid] = "sf"
                else:
                    PLAYER_POS_CACHE[pid] = pos[:2]

            game = row.get("game") or {}
            date = game.get("date")
            if not date:
                continue

            out[pid].append({
                "date": date,
                "usage_pct": float(row.get("usage_percentage") or row.get("usg_pct") or 0),
                "off_rtg": float(row.get("offensive_rating") or row.get("off_rating") or 0),
                "def_rtg": float(row.get("defensive_rating") or row.get("def_rating") or 0),
                "net_rtg": float(row.get("net_rating") or 0),
                "pie": float(row.get("pie") or 0),
                "pace": float(row.get("pace") or 0),
                "efg_pct": float(row.get("effective_field_goal_percentage") or row.get("efg_pct") or 0),
                "ast_pct": float(row.get("assist_percentage") or 0),
            })

        cursor = (resp.get("meta") or {}).get("next_cursor")
        pages += 1
        if not cursor:
            break

    # Sort and cache
    for pid in list(out.keys()):
        out[pid].sort(key=lambda x: x["date"])
        ADV_STATS_CACHE[pid] = out[pid]

    # Fill in cached results for any already-cached pids
    for pid in player_ids:
        pid = int(pid)
        if pid not in out:
            out[pid] = ADV_STATS_CACHE.get(pid, [])

    return out


def bdl_fetch_lineups(gid: int) -> list:
    """
    Fetch confirmed lineups for a game from BDL.
    Tries multiple endpoint paths since lineup availability varies by tier.
    Lineups are typically posted 1-2 hours before tip-off.
    """
    if gid in LINEUP_CACHE:
        return LINEUP_CACHE[gid]

    entries = []
    # Try multiple possible paths
    paths = [
        ("/nba/v1/lineups", {"game_ids[]": [gid]}),
        ("/nba/v1/lineups", {"game_id": gid}),
        ("/v1/lineups", {"game_ids[]": [gid]}),
    ]
    for path, params in paths:
        try:
            resp = _bdl_get(path, params=params)
            entries = resp.get("data") or []
            if entries:
                print(f"[INFO] Lineup data found via {path} for gid={gid}: {len(entries)} entries")
                break
        except Exception:
            continue

    LINEUP_CACHE[gid] = entries

    # Cache positions from lineup data
    for e in entries:
        p = e.get("player") or {}
        pid = p.get("id")
        pos = (e.get("position") or p.get("position") or "").strip().lower()
        if pid and pos and int(pid) not in PLAYER_POS_CACHE:
            PLAYER_POS_CACHE[int(pid)] = pos[:2]

    return entries


def is_player_confirmed_starter(pid: int, gid: int) -> tuple[bool, bool]:
    """
    Check if player is confirmed in lineup and whether they are starting.
    Returns (is_in_lineup, is_starter).
    If lineup not yet posted, returns (False, False) -- don't filter on this.
    """
    entries = bdl_fetch_lineups(gid)
    if not entries:
        return False, False
    for e in entries:
        p = e.get("player") or {}
        if p.get("id") and int(p["id"]) == int(pid):
            return True, bool(e.get("starter", False))
    return False, False


def bdl_fetch_game_odds_full(gid: int) -> dict:
    """
    Fetch game odds (total + spread) from BDL God Tier.
    Correct endpoint: /nba/v2/odds with game_ids[] param.

    Response fields per row:
      game_id, vendor
      spread_home_value, spread_home_odds, spread_away_value, spread_away_odds
      moneyline_home_odds, moneyline_away_odds
      total_value, total_over_odds, total_under_odds

    Returns {"total": float, "spread": float, "fav_team": str}
    fav_team is "home" or "away" based on negative spread side.
    """
    if gid in GAME_ODDS_CACHE:
        return GAME_ODDS_CACHE[gid]

    result = {"total": 0.0, "spread": 0.0, "fav_team": ""}
    try:
        resp = _bdl_get("/nba/v2/odds", params={"game_ids[]": [int(gid)]})
        rows = resp.get("data") or []
        for row in rows:
            if int(row.get("game_id", 0)) != int(gid):
                continue
            # Total
            if result["total"] == 0.0:
                tv = row.get("total_value")
                if tv:
                    try:
                        result["total"] = float(tv)
                    except Exception:
                        pass
            # Spread
            if result["spread"] == 0.0:
                sv = row.get("spread_home_value")
                if sv:
                    try:
                        spread_val = float(sv)
                        result["spread"] = abs(spread_val)
                        result["fav_team"] = "home" if spread_val < 0 else "away"
                    except Exception:
                        pass
            if result["total"] > 0 and result["spread"] > 0:
                break
    except Exception as e:
        print(f"[WARN] game odds fetch failed gid={gid}: {e}")

    GAME_ODDS_CACHE[gid] = result
    return result


def get_player_usage_stats(adv_games: list, n: int = 10) -> dict:
    """
    Compute usage metrics from recent advanced stat games.
    Returns dict with avg_usage_pct, avg_off_rtg, avg_pie, usage_trend.
    """
    if not adv_games:
        return {"avg_usage_pct": 0.0, "avg_off_rtg": 0.0, "avg_pie": 0.0, "usage_trend": 0.0}

    recent = adv_games[-min(len(adv_games), n):]
    usages = [g["usage_pct"] for g in recent if g["usage_pct"] > 0]
    off_rtgs = [g["off_rtg"] for g in recent if g["off_rtg"] > 0]
    pies = [g["pie"] for g in recent if g["pie"] != 0]

    avg_usage = sum(usages) / len(usages) if usages else 0.0
    avg_off_rtg = sum(off_rtgs) / len(off_rtgs) if off_rtgs else 0.0
    avg_pie = sum(pies) / len(pies) if pies else 0.0

    # Usage trend: last 3 vs last 10
    l3_usage = [g["usage_pct"] for g in adv_games[-3:] if g["usage_pct"] > 0]
    avg_l3_usage = sum(l3_usage) / len(l3_usage) if l3_usage else avg_usage
    usage_trend = avg_l3_usage - avg_usage  # positive = usage increasing

    return {
        "avg_usage_pct": avg_usage,
        "avg_off_rtg": avg_off_rtg,
        "avg_pie": avg_pie,
        "usage_trend": usage_trend,
        "avg_l3_usage": avg_l3_usage,
    }


def advanced_usage_adjustment(usage_stats: dict, prop_type: str) -> tuple[float, str]:
    """
    Real usage rate adjustment using actual BDL advanced stats.
    Replaces the approximation in usage_rate_adjustment().

    Usage% thresholds:
      >32% = primary option = +10% projection boost
      28-32% = high usage = +5%
      22-28% = normal = 0%
      18-22% = role player = -5%
      <18% = low usage = -10%

    Also factors in usage trend (increasing/decreasing role).
    """
    if prop_type != "points":
        return 0.0, ""

    usage = usage_stats.get("avg_usage_pct", 0.0)
    trend = usage_stats.get("usage_trend", 0.0)
    pie = usage_stats.get("avg_pie", 0.0)

    if usage <= 0:
        return 0.0, ""

    # Base adjustment from usage %
    if usage >= 32:
        base = 0.10
    elif usage >= 28:
        base = 0.05
    elif usage >= 22:
        base = 0.0
    elif usage >= 18:
        base = -0.05
    else:
        base = -0.10

    # Trend bonus: if usage increasing over last 3 games, add up to +3%
    trend_bonus = _clamp(trend / 5.0 * 0.03, -0.03, 0.03)

    # PIE bonus: high impact players (PIE > 0.15) get small boost
    pie_bonus = 0.02 if pie > 0.15 else 0.0

    adj = _clamp(base + trend_bonus + pie_bonus, -0.12, 0.14)

    if adj > 0.03:
        label = f"usage={usage:.0f}%"
        if trend > 2:
            label += f"(+trending)"
        return adj, f"high-usage({label},+{adj*100:.0f}%)"
    elif adj < -0.03:
        return adj, f"low-usage({usage:.0f}%,{adj*100:.0f}%)"
    return 0.0, f"avg-usage({usage:.0f}%)"


def get_api_player_position(pid: int, player_name: str) -> str:
    """
    Get player position -- tries API cache first, then hardcoded map.
    This replaces the hardcoded PLAYER_POSITION_MAP lookup.
    """
    # API cache (populated from advanced stats or lineup fetch)
    if int(pid) in PLAYER_POS_CACHE:
        return PLAYER_POS_CACHE[int(pid)]
    # Fall back to hardcoded map
    return get_player_position(player_name)


# -------------------- PROPS --------------------
def build_today_props(now_et: datetime):
    games_map = bdl_games_today(now_et)
    game_ids = list(games_map.keys())
    lines_map = {pt: {} for pt in PROP_TYPES}

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

                # Filter: only accept over_under markets, skip milestones
                if mtype != "over_under":
                    continue

                try:
                    line = float(pp.get("line_value"))
                except Exception:
                    continue

                over_odds = market.get("over_odds")
                under_odds = market.get("under_odds")

                # Extra safety: milestone odds are extreme (-1000+), skip those too
                if isinstance(over_odds, (int, float)) and abs(float(over_odds)) > 500:
                    continue
                if not isinstance(over_odds, (int, float)) or not isinstance(under_odds, (int, float)):
                    continue

                # Always use vendor name from the row itself, not the fetch param
                row_vendor = str(pp.get("vendor") or "").strip().lower()
                if not row_vendor and v:
                    row_vendor = str(v).strip().lower()
                if not row_vendor:
                    row_vendor = "unknown"

                row = {
                    "pid": pid,
                    "gid": int(pp.get("game_id")) if pp.get("game_id") is not None else int(gid),
                    "vendor": row_vendor,
                    "prop_type": (pp.get("prop_type") or pt),
                    "line": float(line),
                    "over_odds": float(over_odds),
                    "under_odds": float(under_odds),
                    "updated_at": pp.get("updated_at"),
                }
                lines_map.setdefault(pt, {}).setdefault(pid, []).append(row)

    return lines_map, games_map


# -------------------- CONSENSUS --------------------
def _round_to_half(x: float) -> float:
    return round(float(x) * 2.0) / 2.0


def consensus_line(rows):
    """
    Compute consensus line from available books.
    Uses sharp books when available, falls back to any book.
    Returns (median_line, n_vendors, n_sharp_vendors).

    Note: different books use different player_ids in the BDL API,
    so a player may only have rows from 1-2 books. We accept 1 book
    as valid consensus (MIN_VENDORS_FOR_CONSENSUS=1) and just need
    the line to be reasonable.
    """
    if not rows:
        return None, 0, 0

    by_vendor_sharp = {}
    by_vendor_any = {}

    for r in rows:
        v = str(r.get("vendor") or "").strip().lower()
        if not v:
            continue
        try:
            line = float(r["line"])
        except Exception:
            continue

        if v not in by_vendor_any:
            by_vendor_any[v] = _round_to_half(line)
        if v in CONSENSUS_VENDORS and v not in by_vendor_sharp:
            by_vendor_sharp[v] = _round_to_half(line)

    # Prefer sharp vendors, fall back to any vendor
    by_vendor = by_vendor_sharp if by_vendor_sharp else by_vendor_any
    sharp_count = len(by_vendor_sharp)

    lines = sorted(by_vendor.values())
    n = len(lines)

    if n == 0:
        return None, 0, 0

    mid = n // 2
    if n % 2 == 1:
        return float(lines[mid]), n, sharp_count
    return float(0.5 * (lines[mid - 1] + lines[mid])), n, sharp_count


def best_offer_near_consensus(rows, cons_line: float):
    if not rows or cons_line is None:
        return None

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


# -------------------- PROJECTION CORE --------------------
def compute_projection_components_points(games_all, line):
    base_slice = _slice_last(games_all, BASELINE_GAMES)
    l10_slice = _slice_last(games_all, LOOKBACK_GAMES)
    l3_slice = _slice_last(games_all, SHORT_GAMES)

    base_avg, base_min, base_std = avg_stat_min_std(base_slice)
    l10_avg, l10_min, l10_std = avg_stat_min_std(l10_slice)
    l3_avg, l3_min, _ = avg_stat_min_std(l3_slice)

    rate_base = _safe_rate(base_avg, base_min)
    rate_l10 = _safe_rate(l10_avg, l10_min)
    rate_l3 = _safe_rate(l3_avg, l3_min)

    raw_sigma = l10_std if l10_std > 0 else base_std if base_std > 0 else STD_FLOOR

    # Use adaptive sigma based on consistency
    cons = consistency_score(games_all)
    sigma = adaptive_sigma(games_all, max(STD_FLOOR, raw_sigma), cons)

    # Median-blend: for volatile players, blend mean with median
    # This prevents one big game from inflating projections
    l10_median = median_stat(l10_slice)
    l3_median = median_stat(l3_slice)
    # If median is >15% below mean, player is volatile -- use blend
    if l10_avg > 5 and l10_median < l10_avg * 0.85:
        l10_avg_adj = (l10_avg * 0.60) + (l10_median * 0.40)  # 60% mean 40% median
        l3_avg_adj = (l3_avg * 0.60) + (l3_median * 0.40)
    else:
        l10_avg_adj = l10_avg
        l3_avg_adj = l3_avg

    # Recalculate rates with adjusted averages
    rate_l10 = _safe_rate(l10_avg_adj, l10_min)
    rate_l3 = _safe_rate(l3_avg_adj, l3_min)

    # Floor and ceiling for ladder hints
    p_floor, p_ceiling = floor_ceiling(l10_slice)

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
        "floor": p_floor if "p_floor" in dir() else 0.0,
        "ceiling": p_ceiling if "p_ceiling" in dir() else 0.0,
        "l10_median": l10_median if "l10_median" in dir() else 0.0,
        "situation_label": situation_label if "situation_label" in dir() else "",
        "minute_warn": minute_warn if "minute_warn" in dir() else "",
    }


def compute_proj_rate(rate_base, rate_l10, rate_l3, base_avg=0.0, l3_avg=0.0,
                      base_min=0.0, l3_min=0.0):
    """
    Smart recency weighting with breakout detection.

    Normal players: base 35% / L10 35% / L3 30%

    Breakout mode (L3 avg >= 25% above base avg):
    Heavily weight recent form -- this is a player on a hot streak
    or in an expanded role. The season average is misleading.
    Example: Jared McCain averaging 12pts on season but 22pts L3
    -> weight L3 at 65%, ignore season average almost entirely

    Role expansion mode (L3 minutes >= 4 more than L10 minutes):
    Player is getting significantly more playing time recently
    -> adjust for the minutes increase, not just the scoring rate
    """
    # Check for breakout
    if (base_avg > 3.0 and l3_avg > 0 and
            l3_avg >= base_avg * BREAKOUT_MIN_RATIO):
        # Breakout mode -- trust recent form much more
        w_base = BREAKOUT_BASE_WEIGHT
        w_l10 = BREAKOUT_L10_WEIGHT
        w_l3 = BREAKOUT_L3_WEIGHT
        total = w_base + w_l10 + w_l3
    else:
        total = PROJ_WEIGHT_BASE + PROJ_WEIGHT_L10 + PROJ_WEIGHT_L3
        w_base = PROJ_WEIGHT_BASE / total
        w_l10 = PROJ_WEIGHT_L10 / total
        w_l3 = PROJ_WEIGHT_L3 / total
        total = 1.0

    return (w_base * rate_base + w_l10 * rate_l10 + w_l3 * rate_l3) / total


def is_breakout_player(base_avg: float, l3_avg: float, l3_min: float, l10_min: float) -> tuple[bool, str]:
    """
    Detect if a player is in breakout/hot-streak mode.
    Returns (is_breakout, reason_string)

    This catches:
    - Jared McCain type: low season avg but recent explosion
    - OG Anunoby type: consistent player suddenly playing 40min
    - Injury beneficiary: role player absorbing star's minutes
    """
    reasons = []

    # Scoring breakout: L3 avg >= 25% above season avg
    if base_avg > 3.0 and l3_avg >= base_avg * BREAKOUT_MIN_RATIO:
        pct = ((l3_avg / base_avg) - 1.0) * 100
        reasons.append(f"scoring-breakout(+{pct:.0f}%)")

    # Minutes breakout: getting significantly more playing time
    if l3_min - l10_min >= ROLE_EXP_MIN_DELTA:
        reasons.append(f"role-expansion(+{l3_min - l10_min:.1f}min)")

    return bool(reasons), " | ".join(reasons)


# -------------------- THREES --------------------
def _betaln(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _log_choose(n: int, k: int) -> float:
    try:
        n = int(n)
        k = int(k)
    except Exception:
        return float("-inf")
    if n < 0 or k < 0 or k > n:
        return float("-inf")
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


# -------------------- STEAM --------------------
def steam_score(prev, cur):
    try:
        prev_line = float(prev.get("line"))
        cur_line = float(cur.get("line"))
        prev_over = float(prev.get("over_odds"))
        cur_over = float(cur.get("over_odds"))
    except Exception:
        return 0.0
    line_move = (prev_line - cur_line)
    odds_move = (cur_over - prev_over) / 50.0
    score = (line_move / 0.5) * 0.9 + odds_move * 0.6
    return float(score)


def market_key(prop_type: str, pid: int) -> str:
    return f"{prop_type}|{int(pid)}"


def remember_market(state, prop_type, pid, offer, cons_line, n_cons, now_ts):
    mk = market_key(prop_type, pid)
    state.setdefault("market", {})
    state["market"][mk] = {
        "ts": int(now_ts),
        "line": float(cons_line) if cons_line is not None else float(offer["line"]),
        "over_odds": float(offer["over_odds"]),
        "under_odds": float(offer["under_odds"]),
        "vendor": str(offer.get("vendor", "")),
        "n_cons": int(n_cons),
    }


def get_prev_market(state, prop_type, pid, now_ts):
    mk = market_key(prop_type, pid)
    prev = (state.get("market") or {}).get(mk)
    if not prev:
        return None
    age_min = (int(now_ts) - int(prev.get("ts", 0))) / 60.0
    if age_min > STEAM_MAX_AGE_MIN:
        return None
    return prev


# -------------------- LINEUPEXPERTS --------------------
def _try_parse_dt(s: str):
    if not s:
        return None
    s = str(s).strip()
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%b %d, %Y",
        "%b %d, %Y %I:%M %p",
        "%b %d, %Y %H:%M",
        "%b %d, %Y %I:%M%p",
    ):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


def find_team_for_player_from_cache(player_name: str) -> str:
    target = _clean_name(player_name)
    if not target:
        return ""
    if target in PLAYER_TEAM_OVERRIDES:
        return PLAYER_TEAM_OVERRIDES[target]
    for pid, nm in PLAYER_NAME_CACHE.items():
        if _clean_name(nm) == target:
            team = normalize_team_name(PLAYER_TEAM_CACHE.get(pid, ""))
            if team:
                return team
    return ""


def fetch_lineupexperts_news(now_et: datetime):
    if not LINEUPEXPERTS or not LINEUPEXPERTS_KEY:
        return []

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


def build_news_boost_map(news_items):
    boosts = {}

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


def build_news_score_map(news_items):
    out = {}

    pos_strong = re.compile(r"\b(will start|expected to start|starting lineup|draws start|named starter)\b", re.I)
    pos_med = re.compile(r"\b(available|cleared|returns?|good to go|active|probable|no injury designation)\b", re.I)
    pos_min = re.compile(r"\b(minutes restriction lifted|no minutes limit|minutes limit lifted|workload increased|bigger role|increased role|expanded role)\b", re.I)
    pos_hot = re.compile(r"\b(career high|career-high|season high|season-high|hot streak|upgraded to available)\b", re.I)
    neg_status = re.compile(r"\b(questionable|doubtful|game-time decision|gtd)\b", re.I)
    neg_limit = re.compile(r"\b(minutes restriction|minutes monitored|limited to|on a minutes limit)\b", re.I)
    neg_bench = re.compile(r"\b(coming off the bench|bench role|returning to bench role|moved to bench)\b", re.I)
    neg_out = re.compile(r"\b(ruled out|will miss|out for|out vs|out tonight|inactive|won't play|will not play)\b", re.I)
    neg_load = re.compile(r"\b(load management|sitting out|load manage)\b", re.I)

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
        if pos_hot.search(text):
            score += 0.7
            why_bits.append("hot_streak")
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
        if neg_load.search(text):
            score -= 1.6
            why_bits.append("load_mgmt")

        if score == 0.0:
            continue
        push(player, score, f"{'|'.join(why_bits)}: {title}")

    return out


def apply_news_to_projection(proj: float, boost_rec: dict | None, cap: float = 0.30):
    if not boost_rec:
        return proj, 0.0, None
    b = float(boost_rec.get("boost", 0.0))
    c = float(boost_rec.get("confidence", 0.0))
    eff = max(-cap, min(cap, b * c))
    if abs(eff) <= 0:
        return proj, 0.0, None
    why = boost_rec.get("why") or ""
    return proj * (1.0 + eff), eff, why


def parse_le_injuries(news_items):
    out_pat = re.compile(r"\b(ruled out|will miss|out for|out vs|out tonight|inactive|won't play|will not play|dnp)\b", re.I)
    doubtful_pat = re.compile(r"\b(doubtful)\b", re.I)
    questionable_pat = re.compile(r"\b(questionable|game-time decision|gtd|listed as questionable)\b", re.I)
    load_pat = re.compile(r"\b(load management|rest|sitting out)\b", re.I)

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
        elif load_pat.search(title):
            status = "Out"  # treat load management same as out for prop purposes
        elif doubtful_pat.search(title):
            status = "Doubtful"
        elif questionable_pat.search(title):
            status = "Questionable"

        if not status:
            continue

        k = _clean_name(player)

        # Estimate avg pts from cache if available
        avg_pts = 0.0
        for pid, nm in PLAYER_NAME_CACHE.items():
            if _clean_name(nm) == k:
                # Rough estimate from recent games
                avg_pts = 15.0  # default, gets refined in build_injury_edges
                break

        injuries[k] = {
            "name": player,
            "team": team_hint,
            "status": status,
            "detail": title,
            "avg_pts": avg_pts,
            "is_star": avg_pts >= 20.0,
        }

    return injuries


# -------------------- ENGINE HELPERS --------------------
def should_bad_role_filter(min_delta: float, rate_delta: float, le_score: float) -> bool:
    if min_delta <= -ROLE_DROP_MIN and rate_delta <= -ROLE_DROP_RATE:
        return True
    if min_delta <= -(ROLE_DROP_MIN + 2.0) and le_score < -0.5:
        return True
    return False


# -------------------- UNIFIED PROJECTION ENGINE --------------------
def build_player_projection(
    games,
    line: float,
    prop_type: str,
    injury_boost_stat: float = 0.0,
    injury_boost_min: float = 0.0,
    le_score: float = 0.0,
    news_boosts: dict = None,
    player_name: str = "",
    player_team: str = "",
    opponent_team: str = "",
    games_map: dict = None,
    gid: int = 0,
    today_str: str = "",
    adv_games: list = None,
    player_id: int = 0,
) -> dict:
    """
    IMPROVEMENT: Unified projection engine replaces duplicated logic across
    build_injury_edges, slate_scan_edges, lineup_news_edges, plus_odds_hunt_edges.

    Now includes:
    - Configurable recency weights (PROJ_WEIGHT_BASE/L10/L3)
    - Opponent defensive adjustment (replaces stub matchup_adjustment)
    - Home/away boost
    - Back-to-back penalty
    - Adaptive sigma based on consistency
    - Confidence tier label
    """
    if prop_type_is_threes(prop_type):
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

    # Apply minute cap if set
    if "minute_cap_signal" in dir() and minute_cap_signal != 0.0:
        proj, minute_warn = apply_minute_cap(proj, proj_min, l10_min, minute_cap_signal)

    min_conf = minutes_confidence(proj_min, l10_min, l3_min)

    # Use breakout-aware projection rate
    proj_rate = compute_proj_rate(
        rate_base, rate_l10, rate_l3,
        base_avg=base_avg, l3_avg=l3_avg,
        base_min=base_min, l3_min=l3_min
    )

    # Detect breakout for display
    is_breakout, breakout_reason = is_breakout_player(base_avg, l3_avg, l3_min, l10_min)

    if prop_type_is_threes(prop_type) and THREES_BETA_BINOM:
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

    # B2B penalty
    b2b_note = ""
    if today_str and is_back_to_back(games, today_str):
        proj -= B2B_PENALTY
        b2b_note = f"b2b(-{B2B_PENALTY:.1f}pts)"
        print(f"[INFO] B2B penalty applied: {player_name} -{B2B_PENALTY} pts")

    # Rest day adjustment (bonus for well-rested players)
    rest_boost, rest_note = rest_day_adjustment(games, today_str) if today_str else (0.0, "")
    if rest_boost != 0.0:
        proj += rest_boost

    # Game total adjustment (high-total games = more scoring)
    game_total = get_game_total_from_odds(gid, games_map) if gid and games_map else 0.0
    total_adj, total_note = game_total_adjustment(game_total)
    if total_adj != 0.0:
        proj *= (1.0 + total_adj)

    # Blowout risk penalty (stars sit in blowouts)
    spread, is_fav = get_spread_from_odds(gid, player_team, games_map) if gid and games_map and player_team else (0.0, False)
    blowout_adj, blowout_note = blowout_risk_penalty(spread, is_fav)
    if blowout_adj != 0.0:
        proj += blowout_adj

    # Usage rate adjustment -- use real advanced stats if available
    if adv_games:
        usage_stats = get_player_usage_stats(adv_games)
        usage_adj, usage_note = advanced_usage_adjustment(usage_stats, prop_type)
    else:
        usage_stats = {}
        usage_adj, usage_note = usage_rate_adjustment(games, prop_type)
    if usage_adj != 0.0:
        proj *= (1.0 + usage_adj)

    # ---- MINUTE CAP CHECK ----
    # Check manual caps first, then LE news
    minute_cap_signal = 0.0
    minute_warn = ""
    if games:
        name_key_mc = _clean_name(player_name)
        manual_caps = get_manual_minute_caps()
        if name_key_mc in manual_caps:
            minute_cap_signal = manual_caps[name_key_mc]
        # Will apply after proj_min is calculated

    # Situational edge analysis
    if ENABLE_SITUATION_ENGINE and games and games_map and gid:
        game_info = (games_map or {}).get(int(gid), {})
        home_team = normalize_team_name(game_info.get("home", ""))
        away_team = normalize_team_name(game_info.get("away", ""))
        # Figure out opponent
        norm_player_team = normalize_team_name(player_team or "")
        if norm_player_team and home_team and norm_player_team.lower() in home_team.lower():
            opponent = away_team
        else:
            opponent = home_team
        sit = analyze_situation(
            player_name=player_name,
            player_team=player_team or "",
            opponent_team=opponent,
            games=games,
            now_et=datetime.now(ET),
        )
        if sit["situation_boost"] != 0:
            proj += sit["situation_boost"]
            situation_label = sit["situation_label"]
        else:
            situation_label = ""
    else:
        situation_label = ""

    # Starter confirmation from BDL lineups
    starter_note = ""
    if gid and player_id:
        in_lineup, is_starter = is_player_confirmed_starter(player_id, gid)
        if in_lineup:
            if is_starter:
                starter_note = "confirmed-starter"
            else:
                # Bench player -- penalize projection slightly
                proj *= 0.92
                starter_note = "confirmed-bench(-8%)"

    # Get real game odds (total + spread) from BDL
    if gid and not game_total:
        odds_data = bdl_fetch_game_odds_full(gid)
        if odds_data["total"] > 0:
            game_total = odds_data["total"]
            total_adj2, total_note2 = game_total_adjustment(game_total)
            if total_adj2 != 0.0 and total_note == "":
                proj *= (1.0 + total_adj2)
                total_note = total_note2
        if odds_data["spread"] > 0 and spread == 0.0:
            spread = odds_data["spread"]
            fav = normalize_team_name(odds_data.get("fav_team", ""))
            is_fav = (fav == normalize_team_name(player_team)) if fav else False
            blowout_adj2, blowout_note2 = blowout_risk_penalty(spread, is_fav)
            if blowout_adj2 != 0.0 and blowout_note == "":
                proj += blowout_adj2
                blowout_note = blowout_note2

    # Position from API (more accurate than hardcoded map)
    if player_id and int(player_id) in PLAYER_POS_CACHE:
        api_pos = PLAYER_POS_CACHE[int(player_id)]
        if api_pos and not opponent_team:
            pass  # position cached for future use

    # Compute edge and probability
    if prop_type_is_threes(prop_type) and THREES_BETA_BINOM:
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
        "rest_note": rest_note,
        "total_note": total_note,
        "blowout_note": blowout_note,
        "usage_note": usage_note,
        "game_total": game_total,
        "spread": spread,
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
        "starter_note": starter_note if "starter_note" in dir() else "",
        "adv_usage_pct": usage_stats.get("avg_usage_pct", 0.0) if "usage_stats" in dir() else 0.0,
        "adv_pie": usage_stats.get("avg_pie", 0.0) if "usage_stats" in dir() else 0.0,
        "is_breakout": is_breakout if "is_breakout" in dir() else False,
        "breakout_reason": breakout_reason if "breakout_reason" in dir() else "",
    }


# -------------------- ENGINES --------------------
def build_injury_edges(
    team_short, injured_name, injured_status, exclude_names_lower,
    now_et, prop_type, lines_map_for_prop, state, now_ts,
    news_boosts, news_scores, games_map=None, adv_stats=None,
):
    if deadline_exceeded():
        return []

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

    if prop_type_is_threes(prop_type):
        inj_games = bdl_last_n_games_threes([injured_pid], season, BASELINE_GAMES).get(injured_pid, [])
        ip10 = sum(float(x[1]) for x in _slice_last(inj_games, LOOKBACK_GAMES)) / max(1, len(_slice_last(inj_games, LOOKBACK_GAMES)))
        im10 = sum(float(x[3]) for x in _slice_last(inj_games, LOOKBACK_GAMES)) / max(1, len(_slice_last(inj_games, LOOKBACK_GAMES)))
    else:
        inj_stat_key = prop_type_to_stat_key(prop_type)
        inj_games = bdl_last_n_games_stats([injured_pid], season, BASELINE_GAMES, inj_stat_key).get(injured_pid, [])
        ip10, im10, _ = avg_stat_min_std(_slice_last(inj_games, LOOKBACK_GAMES))

    if len(inj_games) < 3:
        return []

    status = (injured_status or "").lower()
    status_mult = {"out": 1.0, "doubtful": 0.55, "questionable": 0.25}.get(status, 0.35)

    vac_stat = ip10 * status_mult
    vac_min = im10 * status_mult

    # Star player multiplier -- losing a 25+ ppg player is worth much more
    # than losing a 12 ppg role player. Books are slow to adjust backup lines.
    is_star = ip10 >= STAR_PLAYER_MIN_AVG
    if is_star:
        vac_stat *= STAR_VACANCY_MULT
        vac_min *= 1.4
        print(f"[INFO] STAR OUT: {injured_name} avg={ip10:.1f}pts -- boosting vacancy by {STAR_VACANCY_MULT}x")

    if not ((vac_min >= MIN_VAC_MIN) or (vac_stat >= MIN_VAC_STAT)):
        return []

    trigger_strength = min(100.0, (vac_min * 1.2 + vac_stat * 1.5))
    if is_star:
        trigger_strength = min(100.0, trigger_strength * 1.5)
    cand_ids = [pid for pid, _ in roster_tuples]

    stats_all = {}
    if prop_type_is_threes(prop_type):
        for chunk_ids in _chunk(cand_ids, STAT_BATCH_SIZE):
            if deadline_exceeded():
                break
            stats_all.update(bdl_last_n_games_threes(chunk_ids, season, BASELINE_GAMES))
    else:
        inj_stat_key = prop_type_to_stat_key(prop_type)
        for chunk_ids in _chunk(cand_ids, STAT_BATCH_SIZE):
            if deadline_exceeded():
                break
            stats_all.update(bdl_last_n_games_stats(chunk_ids, season, BASELINE_GAMES, inj_stat_key))

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
            "is_breakout": p.get("is_breakout", False),
            "breakout_reason": p.get("breakout_reason", ""),
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


def slate_scan_edges(now_et, prop_type, lines_map_for_prop, state, now_ts, news_boosts, news_scores, games_map=None, adv_stats=None):
    if not ENABLE_SLATE_SCAN or deadline_exceeded():
        return []

    season = _season_year(now_et)
    today_str = now_et.strftime("%Y-%m-%d")
    pids = list((lines_map_for_prop or {}).keys())
    if not pids:
        return []

    stats_all = {}
    if prop_type_is_threes(prop_type):
        for chunk_ids in _chunk(pids, STAT_BATCH_SIZE):
            if deadline_exceeded():
                break
            stats_all.update(bdl_last_n_games_threes(chunk_ids, season, BASELINE_GAMES))
    else:
        stat_key = prop_type_to_stat_key(prop_type)
        for chunk_ids in _chunk(pids, STAT_BATCH_SIZE):
            if deadline_exceeded():
                break
            stats_all.update(bdl_last_n_games_stats(chunk_ids, season, BASELINE_GAMES, stat_key))

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
        
            adv_games=(adv_stats or {}).get(int(pid), []),
            player_id=int(pid),
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

        # For breakout players, use softer thresholds
        if p.get("is_breakout"):
            slate_min_edge = max(thr["min_edge"] * 0.7, 1.2)
            slate_min_prob = max(thr["min_prob"] * 0.92, 0.55)
            slate_min_ev = 0.00
            slate_min_value = 0.00
        else:
            slate_min_edge = max(thr["min_edge"], 1.8)
            slate_min_prob = max(thr["min_prob"], 0.57)
            slate_min_ev = max(thr["min_ev"], 0.00)
            slate_min_value = max(thr["min_value_edge"], 0.00)

        if p["edge"] < slate_min_edge or p["prob_over"] < slate_min_prob:
            continue
        if value_edge < slate_min_value:
            continue
        if ev < slate_min_ev:
            continue

        # Juice filter -- skip plays with too much vig
        if not passes_juice_filter(float(offer["over_odds"])):
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
        # Boost breakout players to surface them higher in the rankings
        if p.get("is_breakout"):
            final_score += 15.0
        # Boost situational plays -- bounce back, revenge, return
        sit = p.get("situation_label", "")
        if "BOUNCE-BACK" in sit or "REVENGE" in sit or "RETURN FROM INJURY" in sit:
            final_score += 20.0
        elif "NATIONAL TV" in sit or "REST-SPOT" in sit:
            final_score += 8.0
        tier = confidence_tier(p["edge"], p["prob_over"], ev, value_edge)
        context_notes = " | ".join(filter(None, [p["home_away_note"], p["b2b_note"]]))
        breakout_tag = f" [BREAKOUT: {p['breakout_reason']}]" if p.get("is_breakout") else ""

        why = (
            f"[{tier}]{breakout_tag} Slate. base {p['base_avg']:.1f} | L10 {p['l10_avg']:.1f} | L3 {p['l3_avg']:.1f} "
            f"(minL10 {p['l10_min']:.1f}, projMin {p['proj_min']:.1f}, rate {p['proj_rate']:.3f}, cons {p['consistency']:.2f}). "
            f"Deltamin={p['min_delta']:+.1f} Deltarate={p['rate_delta']:+.2f}. "
            f"Proj {p['proj']:.1f} vs line {line:.1f} | "
            f"{offer['vendor']} ({int(offer['over_odds']):+d}) | "
            f"edge +{p['edge']:.1f} | P={p['prob_over']*100:.0f}% (mkt={p_market*100:.0f}%, val={value_edge:+.2f}) | "
            f"EV={ev:+.2f} | matchup={p['matchup_note']}{(' | ' + context_notes) if context_notes else ''}"
        )

        kelly = kelly_bet_size(p["prob_over"], float(offer["over_odds"]))
        ideas.append({
            "section": "slate",
            "is_breakout": p.get("is_breakout", False),
            "breakout_reason": p.get("breakout_reason", ""),
            "situation_label": p.get("situation_label", ""),
            "minute_warn": p.get("minute_warn", ""),
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


def lineup_news_edges(now_et, prop_type, lines_map_for_prop, state, now_ts, news_boosts, news_scores, games_map=None, adv_stats=None):
    if not LE_NEWS_ENGINE_ENABLED or deadline_exceeded():
        return []

    season = _season_year(now_et)
    today_str = now_et.strftime("%Y-%m-%d")
    pids = list((lines_map_for_prop or {}).keys())
    if not pids:
        return []

    stats_all = {}
    if prop_type_is_threes(prop_type):
        for chunk_ids in _chunk(pids, STAT_BATCH_SIZE):
            if deadline_exceeded():
                break
            stats_all.update(bdl_last_n_games_threes(chunk_ids, season, BASELINE_GAMES))
    else:
        stat_key = prop_type_to_stat_key(prop_type)
        for chunk_ids in _chunk(pids, STAT_BATCH_SIZE):
            if deadline_exceeded():
                break
            stats_all.update(bdl_last_n_games_stats(chunk_ids, season, BASELINE_GAMES, stat_key))

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
        
                    adv_games=(adv_stats or {}).get(int(pid), []),
                    player_id=int(pid),
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

        le_min_edge = max(1.2, thr["min_edge"] - 0.8)
        le_min_prob = max(0.54, thr["min_prob"] - 0.06)
        le_min_value = 0.00
        le_min_ev = 0.00

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


def plus_odds_hunt_edges(now_et, prop_type, lines_map_for_prop, state, now_ts, news_boosts, news_scores, games_map=None, adv_stats=None):
    if not PLUS_HUNT_ENABLED or deadline_exceeded():
        return []

    season = _season_year(now_et)
    today_str = now_et.strftime("%Y-%m-%d")
    pids = list((lines_map_for_prop or {}).keys())
    if not pids:
        return []

    stats_all = {}
    if prop_type_is_threes(prop_type):
        for chunk_ids in _chunk(pids, STAT_BATCH_SIZE):
            if deadline_exceeded():
                break
            stats_all.update(bdl_last_n_games_threes(chunk_ids, season, BASELINE_GAMES))
    else:
        stat_key = prop_type_to_stat_key(prop_type)
        for chunk_ids in _chunk(pids, STAT_BATCH_SIZE):
            if deadline_exceeded():
                break
            stats_all.update(bdl_last_n_games_stats(chunk_ids, season, BASELINE_GAMES, stat_key))

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
        
      adv_games=(adv_stats or {}).get(int(pid), []),
      player_id=int(pid),
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


# -------------------- SITUATIONAL EDGE ENGINE --------------------
# These factors beat books because they are behavioral and narrative-based
# Books model statistics. They dont model human nature as well.

# Known revenge situations -- player vs team that traded them
# Updated manually via REVENGE_MATCHUPS env var or hardcoded here
# Format: "player_name:opponent_team"
REVENGE_MATCHUPS_RAW = os.environ.get("REVENGE_MATCHUPS", "").strip()

# Known return-from-injury players (set in env when you see news)
# Format: "player_name,player_name2"
RETURNING_PLAYERS_RAW = os.environ.get("RETURNING_PLAYERS", "").strip()

# Manual minute caps -- set this when you see restriction news
# Format: "player_name:minutes;player_name2:minutes"
# Example: "Kawhi Leonard:22;LeBron James:28"
MINUTE_CAPS_RAW = os.environ.get("MINUTE_CAPS", "").strip()

# How aggressively to apply minute restrictions from LE news
# When we detect "minutes restriction" in news, cap minutes to this fraction
# of their normal average (0.65 = cap at 65% of usual minutes)
MINUTES_RESTRICT_FRACTION = float(os.environ.get("MINUTES_RESTRICT_FRACTION", "0.65"))
MINUTES_LOAD_MANAGE_FRACTION = float(os.environ.get("MINUTES_LOAD_MANAGE_FRACTION", "0.55"))


def get_manual_minute_caps() -> dict:
    """Manual minute caps set via MINUTE_CAPS env var."""
    out = {}
    if not MINUTE_CAPS_RAW:
        return out
    for pair in MINUTE_CAPS_RAW.split(";"):
        pair = pair.strip()
        if ":" not in pair:
            continue
        player, mins = pair.split(":", 1)
        try:
            out[_clean_name(player.strip())] = float(mins.strip())
        except Exception:
            pass
    return out


def extract_minute_cap_from_news(news_items: list, player_name: str) -> float:
    """
    Scan LE news for minute restrictions for a specific player.
    Returns: specific minutes (e.g. 22.0), -1.0 (general restriction),
             -2.0 (load management), or 0.0 (no restriction found).
    """
    if not news_items:
        return 0.0

    name_key = _clean_name(player_name)
    number_pat = re.compile(r"([0-9]+)\s*[-]?\s*minute", re.I)
    general_pat = re.compile(
        r"minutes?\s*(restriction|limit|cap|monitored|managed)|"
        r"load\s*management|sitting\s*out|conditioning\s*limit",
        re.I
    )

    for item in news_items:
        player = str(item.get("player") or "").strip()
        if _clean_name(player) != name_key:
            continue

        title = str(item.get("title") or "").strip()
        body = str(item.get("news") or item.get("description") or "").strip()
        text = f"{title} {body}"

        # Try to find specific minute number
        m = number_pat.search(text)
        if m:
            try:
                mins = float(m.group(1))
                if 10 <= mins <= 38:
                    return mins
            except Exception:
                pass

        # General restriction
        if general_pat.search(text):
            if re.search(r"load\s*management|sitting\s*out", text, re.I):
                return -2.0
            return -1.0

    return 0.0


def apply_minute_cap(proj: float, proj_min: float, typical_min: float,
                     cap_signal: float) -> tuple:
    """
    Apply minute restriction to projection.
    Returns (adjusted_proj, warning_label)

    cap_signal > 0: specific cap in minutes
    cap_signal == -1.0: general restriction (65% of normal)
    cap_signal == -2.0: load management (55% of normal)
    cap_signal == 0.0: no restriction
    """
    if cap_signal == 0.0 or typical_min <= 0:
        return proj, ""

    if cap_signal > 0:
        capped_min = cap_signal
        cap_label = f"MINS CAP {cap_signal:.0f}min"
    elif cap_signal == -2.0:
        capped_min = typical_min * MINUTES_LOAD_MANAGE_FRACTION
        cap_label = f"LOAD MGMT (~{capped_min:.0f}min)"
    else:
        capped_min = typical_min * MINUTES_RESTRICT_FRACTION
        cap_label = f"MIN RESTRICT (~{capped_min:.0f}min)"

    if capped_min >= proj_min:
        return proj, ""

    min_ratio = capped_min / max(proj_min, 1.0)
    adjusted = proj * min_ratio
    reduction = proj - adjusted
    return adjusted, f"FADE -- {cap_label} proj -{reduction:.1f}pts"


def get_revenge_matchups() -> dict:
    """Parse revenge matchup env var into lookup dict."""
    out = {}
    if not REVENGE_MATCHUPS_RAW:
        return out
    for pair in REVENGE_MATCHUPS_RAW.split(";"):
        pair = pair.strip()
        if ":" not in pair:
            continue
        player, team = pair.split(":", 1)
        out[_clean_name(player.strip())] = normalize_team_name(team.strip())
    return out


def get_returning_players() -> set:
    """Parse returning-from-injury players."""
    if not RETURNING_PLAYERS_RAW:
        return set()
    return {_clean_name(p.strip()) for p in RETURNING_PLAYERS_RAW.split(",")}


def analyze_situation(player_name: str, player_team: str, opponent_team: str,
                       games: list, now_et: datetime,
                       news_items: list = None) -> dict:
    """
    The educated guess engine.

    Looks at real basketball situations that create edge:

    1. BOUNCE BACK -- player scored under 10 last game
       Books keep the line the same. Player is pissed and motivated.
       Historical hit rate on bounce-back games is measurably higher.

    2. REVENGE GAME -- player vs team that traded/waived them
       Set REVENGE_MATCHUPS=player_name:team_name in Render env vars
       when you see trade news. Dont need this automated -- you know
       when Kyrie plays Dallas or when KD plays Brooklyn.

    3. RETURN FROM INJURY -- first 3 games back
       Books set conservative lines based on pre-injury averages.
       Modern players come back in shape and often go nuclear
       in their return to prove a point.

    4. FATIGUE -- 4th game in 6 nights
       Easy to calculate from schedule. Performance drops ~8%.
       Books adjust game spreads but not enough on individual props.

    5. BOUNCE-BACK SPOT -- lost by 20+ last game
       Team was embarrassed. Stars get extra minutes and shots.
       Lines dont adjust for motivation.

    Returns dict with:
      - situation_boost: pts to add/subtract from projection
      - situation_label: plain English description
      - situation_confidence: how confident we are (0-1)
    """
    result = {
        "situation_boost": 0.0,
        "situation_label": "",
        "situation_confidence": 0.0,
        "flags": [],
    }

    if not games:
        return result

    name_key = _clean_name(player_name)
    boosts = []
    flags = []

    # ---- 1. BOUNCE BACK ----
    # Last game was bad. Player historically responds.
    last_game = games[-1] if games else None
    if last_game:
        last_pts = float(last_game[1])
        recent_avg = sum(float(g[1]) for g in games[-5:]) / max(1, len(games[-5:]))
        if last_pts < 10 and recent_avg > 14:
            boosts.append(BOUNCE_BACK_BOOST)
            flags.append(f"BOUNCE-BACK (scored {last_pts:.0f} last game, avg {recent_avg:.0f})")

        # Big blowout loss -- star gets extra usage in next game
        # We detect this by checking if mins were very low last game
        last_mins = float(last_game[2]) if len(last_game) > 2 else 0
        typical_mins = sum(float(g[2]) for g in games[-10:] if len(g) > 2) / max(1, len(games[-10:]))
        if last_mins < typical_mins * 0.70 and typical_mins > 25:
            boosts.append(2.0)
            flags.append(f"REST-SPOT (only {last_mins:.0f} min last game)")

    # ---- 2. REVENGE GAME ----
    revenge_map = get_revenge_matchups()
    if name_key in revenge_map:
        revenge_opp = revenge_map[name_key]
        if opponent_team and revenge_opp and (
            revenge_opp in normalize_team_name(opponent_team).lower() or
            normalize_team_name(opponent_team).lower() in revenge_opp
        ):
            boosts.append(REVENGE_BOOST)
            flags.append(f"REVENGE GAME vs {opponent_team}")

    # ---- 3. RETURN FROM INJURY ----
    returning = get_returning_players()
    if name_key in returning:
        boosts.append(RETURN_BOOST)
        flags.append("RETURN FROM INJURY (books conservative)")

    # ---- 4. FATIGUE CHECK ----
    # Check if player has played 3+ games in last 5 days
    # Use game dates from stats history
    if len(games) >= 3:
        recent_dates = sorted([g[0] for g in games[-4:]], reverse=True)
        if len(recent_dates) >= 3:
            try:
                d1 = datetime.strptime(recent_dates[0], "%Y-%m-%d")
                d3 = datetime.strptime(recent_dates[2], "%Y-%m-%d")
                days_span = (d1 - d3).days
                if days_span <= 4:  # 3 games in 4 days
                    boosts.append(-FATIGUE_PENALTY)
                    flags.append(f"FATIGUE (3 games in {days_span} days)")
            except Exception:
                pass

    # ---- 5. NATIONAL TV SPOTLIGHT ----
    tv_teams = {t.strip() for t in NATIONAL_TV_TEAMS.split(",") if t.strip()}
    if tv_teams and player_team:
        norm_team = normalize_team_name(player_team).lower()
        if any(t in norm_team or norm_team in t for t in tv_teams):
            boosts.append(SPOTLIGHT_BOOST)
            flags.append("NATIONAL TV (stars perform)")

    # Combine
    if not boosts:
        return result

    total_boost = sum(boosts)
    confidence = min(0.85, 0.40 + len([b for b in boosts if b > 0]) * 0.15)

    result["situation_boost"] = total_boost
    result["situation_label"] = " + ".join(flags)
    result["situation_confidence"] = confidence
    result["flags"] = flags

    return result


# -------------------- CORRELATED PARLAY ENGINE --------------------

# Correlation coefficients between prop types for the same player
# Based on NBA statistical research:
# pts-reb: 0.45 (big games = more rebounds from extra possessions)
# pts-ast: 0.35 (high scorers also tend to create)
# pts-pts2 (two lines same player): 0.95 (if scoring 25, definitely scoring 20)
# reb-ast: 0.20 (low correlation)
PROP_CORRELATIONS = {
    ("points", "rebounds"): 0.45,
    ("rebounds", "points"): 0.45,
    ("points", "assists"): 0.35,
    ("assists", "points"): 0.35,
    ("points", "threes"): 0.40,
    ("threes", "points"): 0.40,
    ("rebounds", "assists"): 0.20,
    ("assists", "rebounds"): 0.20,
}

def correlated_prob(p1: float, p2: float, correlation: float) -> float:
    """
    Estimate joint probability accounting for positive correlation.
    Independent: p1 * p2
    Correlated: p1 * p2 + correlation * sqrt(p1*(1-p1)) * sqrt(p2*(1-p2))

    Example:
    p1=0.70, p2=0.65, corr=0.45
    Independent: 0.455
    Correlated:  0.455 + 0.45 * 0.458 * 0.477 = 0.455 + 0.098 = 0.553

    That 10 percentage point difference = much better than books price it
    """
    independent = p1 * p2
    bonus = correlation * math.sqrt(p1 * (1 - p1)) * math.sqrt(p2 * (1 - p2))
    return min(0.92, independent + bonus)


def estimate_parlay_odds(odds_list: list) -> int:
    """Convert list of american odds to combined parlay american odds."""
    decimal = 1.0
    for o in odds_list:
        decimal *= (1.0 + american_to_payout(float(o)))
    # Apply SGP tax (books take 10-15% on correlated legs)
    decimal *= 0.88
    if decimal >= 2.0:
        return int((decimal - 1.0) * 100)
    return int(-100 / max(decimal - 1.0, 0.01))


def find_correlated_parlays(final_out: list, lines_map: dict,
                             adv_stats: dict, now_et, state,
                             now_ts: int) -> list:
    """
    Find the best correlated 2-3 leg parlays.

    Priority order:
    1. Same player, different props (pts+reb, pts+ast) -- strongest correlation
    2. Same player, ladder rungs (15+ AND 20+) -- near perfect correlation
    3. Two players same team, high-total game -- moderate correlation

    The goal: find parlays paying +150 to +300 where true probability
    is 55%+ so EV is strongly positive. This is how you turn a $200 bet
    into a $500+ winner.

    Example output:
    Sengun PTS O17.5 (-110) + REB O8.5 (-115)
    True prob: 58% (books price at 45%)
    Est. odds: +210
    Bet $150 -> win $315 on a 58% shot
    """
    if not ENABLE_CORR_PARLAY:
        return []

    season = _season_year(now_et)
    today_str = now_et.strftime("%Y-%m-%d")
    parlays = []

    # ---- TYPE 1: Same player, different prop types ----
    # Get all players who appear in final_out
    player_plays = {}
    for play in final_out:
        pid = int(play["player_id"])
        if pid not in player_plays:
            player_plays[pid] = []
        player_plays[pid].append(play)

    # For players with only one prop type in final_out,
    # look up their other prop lines and project them
    for pid, plays in player_plays.items():
        if deadline_exceeded():
            break

        play1 = plays[0]  # primary play already found
        name = play1["player_name"]
        team = play1.get("team", "")
        gid = play1.get("gid", 0)

        for pt2 in ["rebounds", "assists", "threes"]:
            if deadline_exceeded():
                break
            if pt2 == play1["prop_type"]:
                continue

            # Check if we have lines for this player in this prop type
            rows2 = lines_map.get(pt2, {}).get(pid, [])
            if not rows2:
                continue

            cons2, n_cons2, _ = consensus_line(rows2)
            if cons2 is None:
                continue

            offer2 = best_offer_near_consensus(rows2, cons2)
            if not offer2:
                continue

            # Fetch stat history for pt2
            stat_key2 = prop_type_to_stat_key(pt2)
            try:
                games2 = bdl_last_n_games_stats(
                    [pid], season, BASELINE_GAMES, stat_key2
                ).get(pid, [])
            except Exception:
                continue

            if len(games2) < 5:
                continue

            # Project for pt2
            news_scores_empty = {}
            news_boosts_empty = {}
            p2 = build_player_projection(
                games=games2,
                line=float(cons2),
                prop_type=pt2,
                le_score=0.0,
                news_boosts=news_boosts_empty,
                player_name=name,
                player_team=team,
                games_map=None,
                gid=gid,
                today_str=today_str,
                adv_games=adv_stats.get(pid, []),
                player_id=pid,
            )

            # Get correlation between the two prop types
            corr = PROP_CORRELATIONS.get(
                (play1["prop_type"], pt2),
                PROP_CORRELATIONS.get((pt2, play1["prop_type"]), 0.15)
            )

            # Calculate correlated joint probability
            joint_prob = correlated_prob(
                play1["prob_over"], p2["prob_over"], corr
            )

            if joint_prob < CORR_PARLAY_MIN_PROB:
                continue

            # Estimate parlay odds
            parlay_odds = estimate_parlay_odds([
                play1["over_odds"],
                float(offer2["over_odds"])
            ])

            if parlay_odds < CORR_PARLAY_MIN_ODDS:
                continue

            # EV of the parlay
            parlay_ev = ev_per_dollar(joint_prob, parlay_odds)
            if parlay_ev < CORR_PARLAY_MIN_EV:
                continue

            # Win amount on standard bet
            win_amt = CORR_PARLAY_BET * american_to_payout(parlay_odds)

            prop_label = {
                "rebounds": "REB",
                "assists": "AST",
                "threes": "3PT",
                "points": "PTS",
            }.get(pt2, pt2.upper())

            parlays.append({
                "type": "same_player",
                "player_name": name,
                "team": team,
                "gid": gid,
                "leg1": f"PTS O{play1['cons_line']:.1f} "
                        f"{play1['vendor'].upper()} {int(play1['over_odds']):+d}",
                "leg2": f"{prop_label} O{cons2:.1f} "
                        f"{str(offer2['vendor']).upper()} {int(offer2['over_odds']):+d}",
                "joint_prob": joint_prob,
                "parlay_odds": parlay_odds,
                "parlay_ev": parlay_ev,
                "correlation": corr,
                "win_amt": win_amt,
                "bet": CORR_PARLAY_BET,
                "proj1": play1["proj"],
                "proj2": p2["proj"],
                "label": (
                    f"{name} PTS O{play1['cons_line']:.1f} "
                    f"+ {prop_label} O{cons2:.1f}"
                ),
                "note": (
                    f"corr={corr:.2f} | P={joint_prob*100:.0f}% | "
                    f"est. {parlay_odds:+d} | "
                    f"BET ${CORR_PARLAY_BET:.0f} -> WIN ${win_amt:.0f} | "
                    f"EV={parlay_ev:+.2f}"
                ),
            })

    # ---- TYPE 2: Same player ladder rungs (15+ AND 20+) ----
    # Already handled in ladder section -- skip here

    # Sort by EV and win amount
    parlays.sort(key=lambda x: (x["parlay_ev"], x["win_amt"]), reverse=True)
    return parlays[:CORR_PARLAY_TOPN]


# -------------------- LADDER SCAN --------------------

def scan_ladder_plays(now_et, lines_map_for_prop, state, now_ts,
                       news_boosts, news_scores, games_map=None, adv_stats=None):
    """
    Ladder bet scanner.

    How ladders work on FanDuel:
    - You pick a player to score 10+, 20+, 30+ points
    - Each leg is an independent over bet at escalating lines
    - The payout compounds as you add legs
    - Best played as individual bets on the mispriced middle leg
      OR as a 2-leg same-game parlay (10+ and 20+)

    Strategy:
    1. Find players projecting 22+ points (high floor)
    2. Check each ladder rung for value
    3. The 20+ rung is most often mispriced -- books use round numbers
       and a player averaging 23pts often gets 20+ at only +120 when
       true probability is 68%+
    4. Flag the best single leg AND suggest the 2-leg combo

    Key insight: Consistent high scorers (cons >= 60%) are gold for
    ladders because variance is low -- they reliably hit the middle rung.
    """
    if not ENABLE_LADDER_SCAN or deadline_exceeded():
        return []

    season = _season_year(now_et)
    today_str = now_et.strftime("%Y-%m-%d")
    prop_type = "points"

    pids = list((lines_map_for_prop or {}).keys())
    if not pids:
        return []

    # Load stats
    stats_all = {}
    stats_deadline = time.time() + 30
    for chunk_ids in _chunk(pids, STAT_BATCH_SIZE):
        if deadline_exceeded() or time.time() > stats_deadline:
            break
        try:
            stats_all.update(bdl_last_n_games_stats(chunk_ids, season, BASELINE_GAMES, "pts"))
        except Exception:
            break

    ladder_plays = []

    for pid in pids:
        if deadline_exceeded():
            break

        games = stats_all.get(int(pid), [])
        if len(games) < 5:
            continue

        name = PLAYER_NAME_CACHE.get(int(pid), f"Player {pid}")
        player_team = PLAYER_TEAM_CACHE.get(int(pid), "")
        le_score = float(news_scores.get(_clean_name(name), {"score": 0.0}).get("score", 0.0))

        # Get a projection first -- only chase ladders for real scorers
        all_rows = (lines_map_for_prop or {}).get(int(pid), [])
        if not all_rows:
            continue

        # Use consensus line as proxy for player quality
        cons, n_cons, _ = consensus_line(all_rows)
        if cons is None or cons < 12.0:
            continue

        # Build full projection
        gid = int((all_rows[0].get("gid") or all_rows[0].get("game_id") or 0))
        p = build_player_projection(
            games=games,
            line=float(cons),
            prop_type=prop_type,
            le_score=le_score,
            news_boosts=news_boosts,
            player_name=name,
            player_team=player_team,
            games_map=games_map,
            gid=gid,
            today_str=today_str,
            adv_games=(adv_stats or {}).get(int(pid), []),
            player_id=int(pid),
        )

        proj = p["proj"]
        sigma = p["sigma"] or 5.0
        consistency = p.get("consistency", 0.5)

        # Only chase ladders for players projecting well
        if proj < LADDER_MIN_PROJ:
            continue

        # Get floor/ceiling for smarter rung selection
        p_floor_val, p_ceiling_val = floor_ceiling(games)
        l10_med = median_stat(games, n=10)

        # Flag if mean >> median (volatile player -- risky for ladders)
        is_volatile = (p["base_avg"] > 5 and
                       l10_med < p.get("l10_avg", 0) * 0.80)
        if is_volatile:
            # For volatile players, anchor projection to median not mean
            proj = max(proj * 0.88, l10_med * 1.05)

        # Find all ladder-relevant rows (milestone-style lines at round numbers)
        # FanDuel posts these as regular over_under at 10.5, 15.5, 20.5, 25.5
        ladder_rows = {}
        for r in all_rows:
            try:
                line_val = float(r["line"])
                odds_val = float(r.get("over_odds", -9999))
                vendor = str(r.get("vendor", "")).lower()
            except Exception:
                continue

            # Match ladder rungs
            for rung in LADDER_LEGS:
                if abs(line_val - rung) < 0.6:
                    # Keep best odds for this rung
                    if rung not in ladder_rows or odds_val > float(ladder_rows[rung].get("over_odds", -9999)):
                        ladder_rows[rung] = r
                    break

        if not ladder_rows:
            continue

        # Evaluate each ladder rung
        best_leg = None
        best_leg_ev = -999
        all_legs = []

        for rung, row in sorted(ladder_rows.items()):
            over_odds = float(row.get("over_odds", -9999))
            if over_odds < -300:  # too much juice
                continue

            # Probability of scoring over this rung
            z = (proj - rung) / max(sigma, 1e-6)
            prob = _norm_cdf(z)

            # Adjust for consistency -- consistent players hit rungs more reliably
            prob_adj = _clamp(prob * (0.85 + consistency * 0.30), 0.0, 0.97)

            if prob_adj < LADDER_MIN_PROB:
                continue

            ev = ev_per_dollar(prob_adj, over_odds)
            if ev < LADDER_MIN_EV:
                continue

            mkt_prob = american_to_prob(over_odds)
            value_edge = prob_adj - mkt_prob
            kelly = kelly_bet_size(prob_adj, over_odds)

            leg = {
                "rung": rung,
                "prob": prob_adj,
                "ev": ev,
                "value_edge": value_edge,
                "over_odds": over_odds,
                "vendor": str(row.get("vendor", "")),
                "kelly": kelly,
                "row": row,
            }
            all_legs.append(leg)

            if ev > best_leg_ev:
                best_leg_ev = ev
                best_leg = leg

        if not best_leg:
            continue

        # Build 2-leg combo suggestion if we have 2+ valid legs
        combo_note = ""
        if len(all_legs) >= 2:
            # Best 2 legs by EV
            top2 = sorted(all_legs, key=lambda x: x["ev"], reverse=True)[:2]
            # Combined prob (correlated -- same player scoring both)
            # Use 0.92 correlation factor (hitting 10+ makes 20+ more likely)
            corr = 0.92
            combo_prob = top2[0]["prob"] * top2[1]["prob"] * (1.0 / corr)
            combo_prob = min(combo_prob, 0.92)
            # Estimated parlay odds
            d1 = 1.0 + american_to_payout(top2[0]["over_odds"])
            d2 = 1.0 + american_to_payout(top2[1]["over_odds"])
            combo_dec = d1 * d2 * 0.88  # SGP tax
            if combo_dec >= 2.0:
                combo_am = int((combo_dec - 1.0) * 100)
            else:
                combo_am = int(-100 / max(combo_dec - 1.0, 0.01))
            combo_ev = ev_per_dollar(combo_prob, combo_am)
            combo_kelly = kelly_bet_size(combo_prob, combo_am)
            combo_note = (
                f"2-LEG COMBO: {top2[0]['rung']:.0f}+ & {top2[1]['rung']:.0f}+ | "
                f"est. {combo_am:+d} | P={combo_prob*100:.0f}% | EV={combo_ev:+.2f} | BET ${combo_kelly:.0f}"
            )

        # Format all valid legs
        legs_str = " | ".join(
            f"{l['rung']:.0f}+ {int(l['over_odds']):+d} P={l['prob']*100:.0f}% EV={l['ev']:+.2f}"
            for l in sorted(all_legs, key=lambda x: x["rung"])
        )

        floor_str = f"floor {p_floor_val:.1f}" if "p_floor_val" in dir() else ""
        ceil_str = f"ceil {p_ceiling_val:.1f}" if "p_ceiling_val" in dir() else ""
        med_str = f"median {l10_med:.1f}" if "l10_med" in dir() else ""
        volatile_str = " VOLATILE-use-median" if ("is_volatile" in dir() and is_volatile) else ""

        why = (
            f"LADDER. Proj {proj:.1f} | {floor_str} | {ceil_str} | {med_str} | "
            f"sigma {sigma:.1f} | cons {consistency:.0%}{volatile_str} | "
            f"legs: {legs_str}"
        )
        if combo_note:
            why += f" | {combo_note}"

        is_breakout, breakout_reason = is_breakout_player(
            p.get("base_avg", 0), p.get("l3_avg", 0),
            p.get("l3_min", 0), p.get("l10_min", 0)
        )

        final_score = (best_leg_ev * 40.0) + (best_leg["value_edge"] * 80.0) + (proj * 0.5)
        if is_breakout:
            final_score += 12.0

        ladder_plays.append({
            "section": "ladder",
            "prop_type": "points",
            "player_name": name,
            "player_id": int(pid),
            "team": player_team,
            "gid": gid,
            "cons_line": float(cons),
            "line": float(best_leg["rung"]),
            "proj": float(proj),
            "edge": float(proj - best_leg["rung"]),
            "prob_over": float(best_leg["prob"]),
            "market_prob": float(american_to_prob(best_leg["over_odds"])),
            "value_edge": float(best_leg["value_edge"]),
            "ev": float(best_leg["ev"]),
            "vendor": str(best_leg["vendor"]),
            "over_odds": float(best_leg["over_odds"]),
            "under_odds": -999.0,
            "n_cons": int(n_cons),
            "n_sharp": 1,
            "steam": 0.0,
            "trigger_strength": 0.0,
            "trigger": "Ladder scan",
            "why": why,
            "le_score": float(le_score),
            "min_conf": float(p.get("min_conf", 0.5)),
            "stability_score": float(stability_score(proj - best_leg["rung"], sigma)),
            "final_score": float(final_score),
            "tier": confidence_tier(proj - best_leg["rung"], best_leg["prob"],
                                     best_leg["ev"], best_leg["value_edge"]),
            "consistency": float(consistency),
            "kelly_bet": float(best_leg["kelly"]),
            "is_breakout": is_breakout,
            "breakout_reason": breakout_reason,
            "all_legs": all_legs,
            "combo_note": combo_note,
        })

    ladder_plays.sort(key=lambda x: x["final_score"], reverse=True)
    return ladder_plays[:LADDER_TOPN]


# -------------------- HIGH ODDS HUNTER (+250) --------------------

def high_odds_hunt_edges(now_et, prop_type, lines_map_for_prop, state, now_ts,
                          news_boosts, news_scores, games_map=None, adv_stats=None,
                          injury_vacancies=None):
    """
    Dedicated +250 and above odds hunter on FanDuel.

    Strategy: beat the books at high odds by finding plays where:
    1. Injury news has boosted a player (star teammate is out)
    2. Player is on a hot streak (L3 avg well above season avg)
    3. FanDuel is offering +250 or better on an alternate line
    4. Model probability says the line is beatable

    Why this works: When a star goes out, books post conservative alternate
    lines for the backup player at high odds because they don't have good
    data on how the backup performs as a primary option. We do.

    Examples:
    - Luka out -> Kyrie gets +300 for OVER 28.5 pts (he averages 24 but now primary)
    - Embiid out -> Maxey gets +275 for OVER 32.5 pts (huge usage spike incoming)
    """
    if not HIGH_ODDS_HUNT_ENABLED or deadline_exceeded():
        return []

    season = _season_year(now_et)
    today_str = now_et.strftime("%Y-%m-%d")
    pids = list((lines_map_for_prop or {}).keys())
    if not pids:
        return []

    stats_all = {}
    if prop_type_is_threes(prop_type):
        for chunk_ids in _chunk(pids, STAT_BATCH_SIZE):
            if deadline_exceeded():
                break
            stats_all.update(bdl_last_n_games_threes(chunk_ids, season, BASELINE_GAMES))
    else:
        stat_key = prop_type_to_stat_key(prop_type)
        for chunk_ids in _chunk(pids, STAT_BATCH_SIZE):
            if deadline_exceeded():
                break
            stats_all.update(bdl_last_n_games_stats(chunk_ids, season, BASELINE_GAMES, stat_key))

    ideas = []
    for pid in pids:
        if deadline_exceeded():
            break

        games = stats_all.get(int(pid), [])
        if len(games) < 5:
            continue

        # Only look at rows from the target high-odds vendor
        all_rows = (lines_map_for_prop or {}).get(int(pid), [])
        high_rows = [r for r in all_rows
                     if str(r.get("vendor", "")).lower() == HIGH_ODDS_VENDOR
                     and isinstance(r.get("over_odds"), (int, float))
                     and float(r["over_odds"]) >= HIGH_ODDS_MIN]

        if not high_rows:
            continue

        # Take the best odds (highest payout) row
        high_rows.sort(key=lambda r: float(r["over_odds"]), reverse=True)
        offer = high_rows[0]
        line = float(offer["line"])
        over_odds = float(offer["over_odds"])

        name = PLAYER_NAME_CACHE.get(int(pid), f"Player {pid}")
        le_score = float(news_scores.get(_clean_name(name), {"score": 0.0}).get("score", 0.0))
        player_team = PLAYER_TEAM_CACHE.get(int(pid), "")

        # Check if this player has injury news boost (teammate is out)
        boost_rec = (news_boosts or {}).get(_clean_name(name))
        has_injury_boost = boost_rec is not None and float(boost_rec.get("boost", 0)) > 0

        # Check if player is in injury_vacancies (they are the BENEFICIARY)
        is_beneficiary = False
        if injury_vacancies:
            for vac in injury_vacancies:
                if _clean_name(name) in [_clean_name(n) for n in vac.get("beneficiaries", [])]:
                    is_beneficiary = True
                    break

        # Build projection
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
            adv_games=(adv_stats or {}).get(int(pid), []),
            player_id=int(pid),
        )

        # Apply injury boost to projection if applicable
        if has_injury_boost or is_beneficiary:
            p_proj = p["proj"] * (1.0 + INJURY_HIGH_ODDS_BOOST)
            p_edge = p_proj - line
            # Recalculate prob with boosted projection
            if p["sigma"] and p["sigma"] > 0:
                z = p_edge / max(p["sigma"], 1e-6)
                p_prob = _norm_cdf(z)
            else:
                p_prob = p["prob_over"]
            injury_tag = "INJURY-BOOST"
        else:
            p_proj = p["proj"]
            p_edge = p["edge"]
            p_prob = p["prob_over"]
            injury_tag = ""

        # Minimum probability check
        if p_prob < HIGH_ODDS_MIN_PROB:
            continue

        # EV check
        ev = ev_per_dollar(p_prob, over_odds)
        if ev < HIGH_ODDS_MIN_EV:
            continue

        # Market prob
        p_over_mkt = american_to_prob(over_odds)
        value_edge = p_prob - p_over_mkt

        # Hot streak detection
        l3_avg = p.get("l3_avg", 0)
        base_avg = p.get("base_avg", 0)
        is_hot = l3_avg > base_avg * 1.15 if base_avg > 0 else False
        hot_tag = "HOT-STREAK" if is_hot else ""

        # Kelly size -- high odds = smaller Kelly
        kelly = kelly_bet_size(p_prob, over_odds)

        tags = " | ".join(filter(None, [injury_tag, hot_tag,
                                         f"le={le_score:+.1f}" if le_score != 0 else ""]))

        why = (
            f"HIGH-ODDS HUNT {HIGH_ODDS_VENDOR.upper()} {int(over_odds):+d} | "
            f"Proj {p_proj:.1f} vs line {line:.1f} | "
            f"P={p_prob*100:.0f}% | EV={ev:+.2f} | edge +{p_edge:.1f} | "
            f"L3={l3_avg:.1f} base={base_avg:.1f} | {tags}"
        )

        gid = int(offer.get("gid") or offer.get("game_id") or 0)
        final_score = (ev * 50.0) + (value_edge * 80.0) + (p_edge * 1.5)
        if has_injury_boost or is_beneficiary:
            final_score += 20.0
        if is_hot:
            final_score += 10.0

        ideas.append({
            "section": "high_odds",
            "prop_type": prop_type,
            "player_name": name,
            "player_id": int(pid),
            "team": player_team,
            "gid": gid,
            "cons_line": float(line),
            "line": float(line),
            "proj": float(p_proj),
            "edge": float(p_edge),
            "prob_over": float(p_prob),
            "market_prob": float(p_over_mkt),
            "value_edge": float(value_edge),
            "ev": float(ev),
            "vendor": str(offer["vendor"]),
            "over_odds": float(over_odds),
            "under_odds": float(offer.get("under_odds", -999)),
            "n_cons": 1,
            "n_sharp": 1,
            "steam": 0.0,
            "trigger_strength": 20.0 if (has_injury_boost or is_beneficiary) else 0.0,
            "trigger": f"High-odds hunt {HIGH_ODDS_VENDOR} {int(over_odds):+d}",
            "why": why,
            "le_score": float(le_score),
            "min_conf": float(p.get("min_conf", 0.5)),
            "stability_score": 0.0,
            "final_score": float(final_score),
            "tier": confidence_tier(p_edge, p_prob, ev, value_edge),
            "consistency": float(p.get("consistency", 0.5)),
            "kelly_bet": float(kelly),
            "is_high_odds": True,
            "injury_boost": has_injury_boost or is_beneficiary,
        })

        remember_market(state, prop_type, int(pid), offer, line, 1, now_ts)

    ideas.sort(key=lambda x: (x["final_score"], x["ev"]), reverse=True)
    return ideas[:HIGH_ODDS_TOPN]


# -------------------- COOLDOWN / EXPOSURE --------------------
def apply_cooldown(state, ideas, now_ts: int):
    sent = state.get("sent_bets", {}) or {}
    cooldown_sec = BET_COOLDOWN_MIN * 60
    kept = []
    for i in ideas:
        key = f"{i['prop_type']}|{i['section']}|{int(i['player_id'])}|{float(i.get('cons_line', i.get('line', 0.0))):.1f}"
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
        key = f"{i['prop_type']}|{i['section']}|{int(i['player_id'])}|{float(i.get('cons_line', i.get('line', 0.0))):.1f}"
        sent[key] = {"ts": now_ts, "edge": float(i.get("edge", 0.0))}
        if ENABLE_CLV_TRACKING:
            track_closing_line_value(i, now_ts)
    state["sent_bets"] = sent


def apply_exposure_caps(ideas):
    if not ideas:
        return ideas
    team_ct = {}
    game_ct = {}
    out = []
    for it in ideas:
        team = (it.get("team") or "").strip().lower()
        gid = int(it.get("gid") or 0)
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


# -------------------- WHATSAPP CARD FORMATTER --------------------
def format_play_card(play: dict, idx: int) -> str:
    """
    Clean actionable WhatsApp card.
    Designed to be read in under 3 seconds and acted on immediately.
    """
    tier = play.get("tier", "")
    name = play["player_name"]
    line = play["cons_line"]
    proj = play["proj"]
    edge = play["edge"]
    prob = play["prob_over"] * 100
    ev = play["ev"]
    over_odds = int(play["over_odds"])
    vendor = play["vendor"].upper()
    team = play.get("team", "")

    # Extract key context from why string -- keep it short
    matchup_hint = ""
    why = play.get("why", "")
    m = re.search(r"matchup=([^\s|]+)", why)
    if m:
        matchup_hint = m.group(1)

    b2b = "[B2B]" if "b2b(" in why else ""
    rest = ""
    if play.get("rest_note", "") and "d-rest" in play.get("rest_note", ""):
        rest = f"[{play['rest_note']}]"
    le_note = ""
    if play.get("le_score", 0) > 0.5:
        le_note = "[NEWS+]"
    elif play.get("le_score", 0) < -0.5:
        le_note = "[NEWS-]"
    blowout = "[BLOWOUT-RISK]" if play.get("blowout_note", "") else ""
    usage = play.get("usage_note", "")
    usage_tag = f"[{usage}]" if usage else ""
    total_note = play.get("total_note", "")
    total_tag = f"[{total_note}]" if total_note and "avg" not in total_note else ""

    consistency = play.get("consistency", 0)
    cons_note = f"cons:{consistency:.0%}" if consistency > 0 else ""

    # Starter/bench status from confirmed lineups
    starter_tag = ""
    s_note = play.get("starter_note", "")
    if s_note == "confirmed-starter":
        starter_tag = "[STARTING]"
    elif "bench" in s_note:
        starter_tag = "[BENCH-8%]"

    # Real usage % from advanced stats
    usage_pct = play.get("adv_usage_pct", 0.0)
    usage_display = f"usg:{usage_pct:.0f}%" if usage_pct > 0 else ""

    context = " ".join(filter(None, [b2b, rest, le_note, blowout, usage_tag, total_tag, starter_tag]))

    # Alternate line detector -- flag when line is suspiciously far below projection
    # Real main lines are usually within 4pts of projection
    # Anything more = likely an alternate line at better odds
    alt_line_flag = ""
    if edge > 6.0 and prob > 0.88:
        alt_line_flag = "[ALT-LINE? verify on app]"

    # Bet size
    kelly = play.get("kelly_bet", 0)
    bet_str = f"  BET ${kelly:.0f}" if ENABLE_BET_SIZING and kelly > 0 else ""

    # News signal
    le = play.get("le_score", 0)
    news_flag = " [NEWS+]" if le > 0.8 else " [NEWS-]" if le < -0.5 else ""

    # Load management warning
    load_warn = " [LOAD-MGMT]" if "load_mgmt" in play.get("why", "") else ""

    # Alt line warning
    alt = " [ALT-LINE-verify]" if alt_line_flag else ""

    # Breakout tag
    breakout_flag = ""
    if play.get("is_breakout"):
        breakout_flag = f" [BREAKOUT: {play.get('breakout_reason','')}]"

    card = [
        f"{idx}. {tier} -- {name} ({team}){breakout_flag}",
        f"   OVER {line:.1f} @ {vendor} {over_odds:+d}{bet_str}",
        f"   Proj {proj:.1f} | P={prob:.0f}% | EV={ev:+.2f} | edge +{edge:.1f}",
        f"   {matchup_hint} {cons_note}{news_flag}{load_warn}{alt}".strip(),
    ]
    return "\n".join(card)


# -------------------- MAIN --------------------
def run():
    now_et = _now_et()
    ts_et = now_et.strftime("%Y-%m-%d %I:%M %p ET")
    now_ts = int(now_et.timestamp())
    today_str = now_et.strftime("%Y-%m-%d")

    print(
        f"[BOOT] v3.0 ts={ts_et} TEST_MODE={int(TEST_MODE)} "
        f"PROP_TYPES={','.join(PROP_TYPES)} STD_FLOOR={STD_FLOOR} "
        f"PROJ_WEIGHTS=base{PROJ_WEIGHT_BASE}/l10{PROJ_WEIGHT_L10}/l3{PROJ_WEIGHT_L3} "
        f"HOME_BOOST={HOME_COURT_BOOST} B2B_PENALTY={B2B_PENALTY} OPP_DEF={int(ENABLE_OPP_DEF_ADJ)}"
    )

    if TEST_MODE:
        send_one(f"[STRONG] NBA betting agent v2 test OK ({ts_et})")
        return



    state = load_state()
    old_players = state.get("players", {})

    # ---- RUN WINDOW CHECK ----
    hour = now_et.hour
    # After 11:30pm ET -- late games over
    if hour >= 23 and minute >= 30:
        print(f"[INFO] After 11:30pm ET -- all games over, skipping")
        save_state(state)
        return
    if hour == 23 and minute < 30:
        pass  # Still running, 9:30pm games still live
    elif hour > 23:
        save_state(state)
        return
    # Before 7am ET -- no lines posted
    if hour < 7:
        print(f"[INFO] Before 7am ET -- no lines yet, skipping")
        save_state(state)
        return

    # ---- MORNING RESULTS REMINDER ----
    if 9 <= hour < 10:
        today_key = f"results_reminder_{now_et.strftime('%Y-%m-%d')}"
        reminder_state = state.get("reminders", {})
        if today_key not in reminder_state:
            try:
                pending = get_pending_plays_for_reminder()
                if pending:
                    reminder_state[today_key] = int(now_et.timestamp())
                    state["reminders"] = reminder_state
                    save_state(state)
            except Exception as e:
                print(f"[WARN] results reminder failed: {e}")

    lines_map, games_map = build_today_props(now_et)

    season = _season_year(now_et)
    all_prop_pids = []
    for pt in PROP_TYPES:
        all_prop_pids.extend(list(lines_map.get(pt, {}).keys()))
    all_prop_pids = list({int(x) for x in all_prop_pids})

    if all_prop_pids:
        # Warmup stats for each prop type separately
        stats_deadline = time.time() + 50
        warmed_keys = set()
        for pt in PROP_TYPES:
            if deadline_exceeded() or time.time() > stats_deadline:
                break
            if prop_type_is_threes(pt):
                continue  # threes handled separately below
            sk = prop_type_to_stat_key(pt)
            if sk in warmed_keys:
                continue
            warmed_keys.add(sk)
            for chunk_ids in _chunk(all_prop_pids, STAT_BATCH_SIZE):
                if deadline_exceeded() or time.time() > stats_deadline:
                    print(f"[INFO] Stats warmup time budget reached ({sk})")
                    break
                try:
                    bdl_last_n_games_stats(chunk_ids, season, max(LOOKBACK_GAMES, 8), sk)
                except Exception as e:
                    print(f"[WARN] warmup stats failed ({sk}): {e}")
                    break

        if "threes" in PROP_TYPES and THREES_BETA_BINOM:
            threes_deadline = time.time() + 30  # hard 30s budget
            for chunk_ids in _chunk(all_prop_pids, STAT_BATCH_SIZE):
                if deadline_exceeded() or time.time() > threes_deadline:
                    print("[INFO] Threes warmup time budget reached")
                    break
                try:
                    bdl_last_n_games_threes(chunk_ids, season, max(LOOKBACK_GAMES, 8))
                except Exception as e:
                    print(f"[WARN] warmup threes failed: {e}")
                    break

        # God Tier: fetch advanced stats -- time-budgeted
        adv_stats_all = {}
        adv_deadline = time.time() + 40  # max 40s for advanced stats
        print(f"[INFO] Fetching advanced stats for {len(all_prop_pids)} players...")
        for chunk_ids in _chunk(all_prop_pids, STAT_BATCH_SIZE):
            if deadline_exceeded() or time.time() > adv_deadline:
                print(f"[INFO] Advanced stats time budget reached, stopping early")
                break
            try:
                chunk_adv = bdl_fetch_advanced_stats(chunk_ids, season)
                adv_stats_all.update(chunk_adv)
            except Exception as e:
                print(f"[WARN] advanced stats warmup failed: {e}")
                break
        print(f"[INFO] Advanced stats loaded for {len(adv_stats_all)} players | positions cached: {len(PLAYER_POS_CACHE)}")

        # Pre-fetch lineups -- 10s budget, skip if early in day
        now_hour = now_et.hour
        if now_hour >= 15:  # only fetch lineups after 3pm ET
            print(f"[INFO] Fetching lineups for {len(games_map)} games...")
            lineup_deadline = time.time() + 10
            for gid_lineup in list(games_map.keys()):
                if deadline_exceeded() or time.time() > lineup_deadline:
                    break
                try:
                    bdl_fetch_lineups(gid_lineup)
                except Exception as e:
                    pass
            starters = sum(1 for e in LINEUP_CACHE.values() for x in e if x.get("starter"))
            print(f"[INFO] Lineups loaded, confirmed starters: {starters}")
        else:
            print(f"[INFO] Skipping lineup fetch before 3pm ET (lineups not posted yet)")

        # Pre-fetch game odds -- 15s budget
        odds_deadline = time.time() + 15
        for gid_odds in list(games_map.keys()):
            if deadline_exceeded() or time.time() > odds_deadline:
                break
            try:
                bdl_fetch_game_odds_full(gid_odds)
            except Exception as e:
                pass
    else:
        adv_stats_all = {}

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
                                 news_boosts=news_boosts, news_scores=news_scores, games_map=games_map, adv_stats=adv_stats_all)
            )

    # ---- Lineup news edges ----
    lineup_news_ideas_all = []
    if LE_NEWS_ENGINE_ENABLED and (not deadline_exceeded()):
        for pt in PROP_TYPES:
            if deadline_exceeded():
                break
            lineup_news_ideas_all.extend(
                lineup_news_edges(now_et, pt, lines_map.get(pt, {}), state=state, now_ts=now_ts,
                                  news_boosts=news_boosts, news_scores=news_scores, games_map=games_map, adv_stats=adv_stats_all)
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
                                     news_boosts=news_boosts, news_scores=news_scores, games_map=games_map, adv_stats=adv_stats_all)
            )
    plus_ideas_all = plus_ideas_all[:PLUS_HUNT_TOPN]

    # ---- Ladder scan ----
    ladder_all = []
    if ENABLE_LADDER_SCAN and (not deadline_exceeded()):
        ladder_all = scan_ladder_plays(
            now_et, lines_map.get("points", {}), state=state, now_ts=now_ts,
            news_boosts=news_boosts, news_scores=news_scores,
            games_map=games_map, adv_stats=adv_stats_all
        )
        print(f"[INFO] Ladder scan: {len(ladder_all)} plays found")

    # ---- High odds hunt (+250 and above on FanDuel) ----
    high_odds_all = []
    if HIGH_ODDS_HUNT_ENABLED and (not deadline_exceeded()):
        for pt in PROP_TYPES:
            if deadline_exceeded():
                break
            high_odds_all.extend(
                high_odds_hunt_edges(now_et, pt, lines_map.get(pt, {}), state=state, now_ts=now_ts,
                                     news_boosts=news_boosts, news_scores=news_scores,
                                     games_map=games_map, adv_stats=adv_stats_all)
            )
        high_odds_all = sorted(high_odds_all, key=lambda x: x["final_score"], reverse=True)[:HIGH_ODDS_TOPN]
        print(f"[INFO] High-odds hunt: {len(high_odds_all)} plays at +{HIGH_ODDS_MIN:.0f} or better")

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

        # ---- SECTION 1: HEADER ----
        # One line. Time + hit rate if we have it.
        header = f"NBA {ts_et}"
        if hit_rate_str:
            header += f" | {hit_rate_str}"
        msg.append(header)

        # Injury flags on one line at top
        if triggers:
            star_outs = [t for t in triggers if any(
                n in t.lower() for n in ["luka","embiid","giannis","curry",
                "lebron","durant","tatum","jokic","gilgeous","mitchell","sga"]
            )]
            if star_outs:
                msg.append("STAR OUT: " + " | ".join(star_outs[:3]))
            elif triggers:
                msg.append("INJ: " + " | ".join(triggers[:2]))
        msg.append("")

        # ---- SECTION 2: STRAIGHT BETS ----
        # Each play is exactly 3 lines:
        # Line 1: Player + market + line + book + odds
        # Line 2: Proj / probability / EV / kelly bet
        # Line 3: Why in plain English
        msg.append("-- STRAIGHT BETS --")
        for i, play in enumerate(final_out, 1):
            tier = play.get("tier", "")
            name = play["player_name"]
            team = play.get("team", "")
            line = play["cons_line"]
            book = play["vendor"].upper()
            odds = int(play["over_odds"])
            proj = play["proj"]
            prob = play["prob_over"] * 100
            ev = play["ev"]
            kelly = play.get("kelly_bet", 0)
            prop = play.get("prop_type", "pts").upper()[:3]
            breakout = " BREAKOUT" if play.get("is_breakout") else ""
            inj = " INJ" if play.get("section") == "injury" else ""
            bet_str = f"BET ${kelly:.0f}" if kelly > 0 else ""

            # Plain English why
            base = play.get("base_avg", 0) or play.get("base_avg", 0)
            l3 = play.get("l3_avg", proj)
            why_plain = ""
            if play.get("is_breakout") and play.get("breakout_reason"):
                why_plain = play["breakout_reason"]
            elif play.get("section") == "injury":
                why_plain = play.get("trigger", "injury trigger")[:60]
            elif l3 > 0 and base > 0:
                trend = "trending up" if l3 > base * 1.1 else "trending down" if l3 < base * 0.9 else "consistent"
                why_plain = f"L3 avg {l3:.1f} vs season {base:.1f} -- {trend}"

            msg.append(
                f"{i}. [{tier}]{breakout}{inj} {name} ({team})"
            )
            msg.append(
                f"   {prop} OVER {line:.1f} | {book} {odds:+d} | {bet_str}"
            )
            msg.append(
                f"   Proj {proj:.1f} | P={prob:.0f}% | EV={ev:+.2f}"
            )
            # Situation label -- the educated guess factor
            minute_warn = play.get("minute_warn", "")
            sit_label = play.get("situation_label", "")
            if minute_warn:
                msg.append(f"   {minute_warn}")
            elif sit_label:
                msg.append(f"   SPOT: {sit_label}")
            elif why_plain:
                msg.append(f"   {why_plain}")
            msg.append("")

        # ---- SECTION 3: PLUS ODDS (value bets at good prices) ----
        plus_bucket = [x for x in final_out if x.get("over_odds", -999) >= 100]
        extra_plus = [x for x in plus_ideas_all if x["player_id"] not in
                      {p["player_id"] for p in final_out}]
        all_plus = sorted(plus_bucket + extra_plus,
                          key=lambda x: x["ev"], reverse=True)[:PLUS_ODDS_TOPN]

        if all_plus:
            msg.append("-- PLUS ODDS --")
            for i in all_plus:
                odds = int(i["over_odds"])
                kelly = i.get("kelly_bet", 0)
                bet_str = f" BET ${kelly:.0f}" if kelly > 0 else ""
                msg.append(
                    f"{i['player_name']} ({i.get('team','')}) "
                    f"OVER {i['cons_line']:.1f} {i['vendor'].upper()} {odds:+d}{bet_str}"
                )
                msg.append(
                    f"  P={i['prob_over']*100:.0f}% EV={i['ev']:+.2f} "
                    f"Proj {i['proj']:.1f}"
                )
            msg.append("")

        # High odds (+250 and above)
        if high_odds_all:
            msg.append("-- HIGH ODDS +250 --")
            for i in high_odds_all:
                odds = int(i["over_odds"])
                inj = " INJ" if i.get("injury_boost") else ""
                kelly = i.get("kelly_bet", 0)
                bet_str = f" BET ${kelly:.0f}" if kelly > 0 else ""
                msg.append(
                    f"{i['player_name']} ({i['team']}) "
                    f"OVER {i['cons_line']:.1f} FD {odds:+d}{inj}{bet_str}"
                )
                msg.append(
                    f"  P={i['prob_over']*100:.0f}% EV={i['ev']:+.2f} "
                    f"Proj {i['proj']:.1f}"
                )
            msg.append("")

        # Ladders
        if ladder_all:
            msg.append("-- LADDERS --")
            for i in ladder_all:
                breakout = " BREAKOUT" if i.get("is_breakout") else ""
                # Best single leg
                odds = int(i["over_odds"])
                kelly = i.get("kelly_bet", 0)
                bet_str = f" BET ${kelly:.0f}" if kelly > 0 else ""
                msg.append(
                    f"{i['player_name']} ({i['team']}){breakout}"
                )
                msg.append(
                    f"  BEST: {i['cons_line']:.0f}+ "
                    f"{i['vendor'].upper()} {odds:+d}{bet_str} "
                    f"P={i['prob_over']*100:.0f}% EV={i['ev']:+.2f}"
                )
                # All rungs on one line
                if i.get("all_legs"):
                    rungs = " | ".join(
                        f"{l['rung']:.0f}+ {int(l['over_odds']):+d}"
                        for l in sorted(i["all_legs"], key=lambda x: x["rung"])
                    )
                    msg.append(f"  Rungs: {rungs}")
                # 2-leg combo
                if i.get("combo_note"):
                    msg.append(f"  COMBO: {i['combo_note']}")
                msg.append("")
            msg.append("")

        # Correlated parlays
        if ENABLE_CORR_PARLAY:
            corr_parlays = find_correlated_parlays(
                final_out, lines_map, adv_stats_all, now_et, state, now_ts
            )
            if corr_parlays:
                msg.append("-- CORRELATED PARLAYS --")
                msg.append("These are your big-win plays. Both legs move together.")
                msg.append("")
                for p in corr_parlays:
                    msg.append(f"{p['player_name']} ({p['team']})")
                    msg.append(f"  {p['leg1']}")
                    msg.append(f"  {p['leg2']}")
                    msg.append(
                        f"  {p['note']}"
                    )
                    msg.append("")
                msg.append("")

        # SGP
        if ENABLE_SGP:
            sgp_opps = find_sgp_opportunities(final_out, games_map)
            if sgp_opps:
                msg.append("-- SAME-GAME PARLAYS --")
                for sgp in sgp_opps:
                    kelly = sgp.get("sgp_kelly", 0)
                    bet_str = f" BET ${kelly:.0f}" if kelly > 0 else ""
                    msg.append(
                        f"{sgp['label']}"
                    )
                    msg.append(
                        f"  est. {sgp['sgp_odds']:+d}{bet_str} "
                        f"P={sgp['combined_prob']*100:.0f}% "
                        f"EV={sgp['combined_ev']:+.2f}"
                    )
                    msg.append("")

        # Footer -- total action summary
        total_kelly = sum(p.get("kelly_bet", 0) for p in final_out)
        total_ev = sum(p.get("ev", 0) * p.get("kelly_bet", 0) for p in final_out)
        if total_kelly > 0:
            msg.append(
                f"ACTION: ${total_kelly:.0f} | EXPECTED: +${total_ev:.0f}"
            )
        send_chunked("\n".join(msg).strip())

        record_sent(state, final_out, now_ts)
    else:
        # Still show high odds even if no main plays
        if high_odds_all:
            msg = [f"NBA PROPS {ts_et}", "No main plays -- but found high odds:",""]
            for i in high_odds_all:
                odds = int(i["over_odds"])
                inj_flag = " [INJ]" if i.get("injury_boost") else ""
                msg.append(f"- {i['player_name']} OVER {i['cons_line']:.1f} FANDUEL {odds:+d}{inj_flag}")
                msg.append(f"  P={i['prob_over']*100:.0f}% EV={i['ev']:+.2f} Proj {i['proj']:.1f}")
                msg.append("")
            send_chunked("\n".join(msg).strip())
            record_sent(state, high_odds_all, now_ts)
        elif ladder_all:
            # No main plays but found ladder opportunities
            msg2 = [f"NBA PROPS {ts_et}", "No main plays -- but ladder opps found:", ""]
            for i in ladder_all[:3]:
                odds = int(i["over_odds"])
                msg2.append(f"- {i['player_name']} ({i['team']}) {i['cons_line']:.0f}+ pts {odds:+d}")
                msg2.append(f"  Proj {i['proj']:.1f} | P={i['prob_over']*100:.0f}% | EV={i['ev']:+.2f}")
                if i.get("combo_note"):
                    msg2.append(f"  {i['combo_note']}")
                msg2.append("")
            send_chunked("\n".join(msg2).strip())
        else:
            print("[INFO] No plays cleared thresholds this run.")


    state["players"] = new_players
    save_state(state)


if __name__ == "__main__":
    run()
