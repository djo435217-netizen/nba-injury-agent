import os
import re
import time
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Tuple, Any

import requests
from twilio.rest import Client

ET = ZoneInfo("America/New_York")

# ----------------------------
# ENV
# ----------------------------
TWILIO_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
FROM_WHATSAPP = os.environ.get("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
TO_WHATSAPP = f"whatsapp:{os.environ['MY_WHATSAPP_NUMBER']}"

BALLDONTLIE_API_KEY = os.environ.get("BALLDONTLIE_API_KEY", "").strip()
SPORTRADAR_KEY = os.environ.get("SPORTRADAR_KEY", "").strip()  # optional for now

TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
SEND_NO_EDGE_PING = os.environ.get("SEND_NO_EDGE_PING", "1") in ("1", "true", "True")
ODDS_ONLY_IN_BURST = os.environ.get("ODDS_ONLY_IN_BURST", "1") in ("1", "true", "True")
HEARTBEAT_PING = os.environ.get("HEARTBEAT_PING", "0") in ("1", "true", "True")

BURST_START_ET = os.environ.get("BURST_START_ET", "17:00").strip()
BURST_END_ET = os.environ.get("BURST_END_ET", "23:45").strip()

EDGE_THRESHOLD = float(os.environ.get("EDGE_THRESHOLD", "2.0"))
MAX_BET_IDEAS = int(os.environ.get("MAX_BET_IDEAS", "4"))

SPORTSBOOK = os.environ.get("SPORTSBOOK", "fanduel").strip().lower()
MARKET = os.environ.get("MARKET", "player_points").strip().lower()

# Example: 2025 means 2025-26 season
SEASON = int(os.environ.get("SEASON", str(datetime.now(ET).year - 1)))

# Debug: print sample odds rows structure
DEBUG_SAMPLE = os.environ.get("DEBUG_SAMPLE", "1") in ("1", "true", "True")

twilio = Client(TWILIO_SID, TWILIO_TOKEN)
BDL_HEADERS = {"Authorization": BALLDONTLIE_API_KEY} if BALLDONTLIE_API_KEY else {}

# ----------------------------
# Helpers
# ----------------------------
def _now_et() -> datetime:
    return datetime.now(ET)

def _parse_hhmm(hhmm: str) -> Tuple[int, int]:
    m = re.match(r"^(\d{1,2}):(\d{2})$", hhmm)
    if not m:
        raise ValueError(f"Bad time format (expected HH:MM): {hhmm}")
    return int(m.group(1)), int(m.group(2))

def _in_burst_window(now_et: datetime) -> bool:
    sh, sm = _parse_hhmm(BURST_START_ET)
    eh, em = _parse_hhmm(BURST_END_ET)
    start = now_et.replace(hour=sh, minute=sm, second=0, microsecond=0)
    end = now_et.replace(hour=eh, minute=em, second=0, microsecond=0)
    return start <= now_et <= end

def send_one(body: str) -> None:
    msg = twilio.messages.create(from_=FROM_WHATSAPP, to=TO_WHATSAPP, body=body[:1500])
    print(f"[TWILIO] sent sid={msg.sid} status={msg.status}")

def send_chunked(body: str, chunk_size: int = 1400) -> None:
    if len(body) <= chunk_size:
        send_one(body)
        return
    lines = body.splitlines()
    parts: List[str] = []
    cur = ""
    for ln in lines:
        if len(cur) + len(ln) + 1 > chunk_size:
            parts.append(cur)
            cur = ln
        else:
            cur = cur + ("\n" if cur else "") + ln
    if cur:
        parts.append(cur)

    total = len(parts)
    for i, p in enumerate(parts, start=1):
        send_one(f"(Part {i}/{total})\n{p}")

def _pick(d: dict, keys: List[str]):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

def _as_int(x) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None

def _as_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def _safe_sample(row: Any, max_chars: int = 1200) -> str:
    """
    Print a sanitized/minified JSON sample to logs.
    """
    try:
        s = json.dumps(row, ensure_ascii=False)[:max_chars]
        return s
    except Exception:
        return str(row)[:max_chars]

# ----------------------------
# BallDontLie client (retry/backoff)
# ----------------------------
def bdl_get(path: str, params: Optional[dict] = None, timeout: int = 25) -> dict:
    bases = ["https://api.balldontlie.io/nba", "https://api.balldontlie.io"]
    last_err = None

    for base in bases:
        url = base + path
        for attempt in range(1, 6):
            try:
                r = requests.get(url, headers=BDL_HEADERS, params=params or {}, timeout=timeout)

                if r.status_code == 404:
                    last_err = f"404 {url}"
                    break

                if r.status_code in (429, 500, 502, 503, 504):
                    wait = min(2 ** attempt, 20)
                    print(f"[BDL] {r.status_code} on {path}; retry in {wait}s...")
                    time.sleep(wait)
                    last_err = f"{r.status_code}: {r.text[:160]}"
                    continue

                if r.status_code != 200:
                    raise RuntimeError(f"{r.status_code}: {r.text[:300]}")

                return r.json()

            except Exception as e:
                last_err = str(e)
                time.sleep(1.0)

    raise RuntimeError(f"BallDontLie request failed for {path}. Last error: {last_err}")

def bdl_games_today(now_et: datetime) -> List[dict]:
    d = now_et.date().isoformat()
    resp = bdl_get("/v1/games", params={"dates[]": [d], "per_page": 100})
    return resp.get("data") or []

def bdl_player_points_props_for_game(game_id: int) -> List[dict]:
    params = {
        "sportsbook": SPORTSBOOK,
        "market": MARKET,
        "game_id": int(game_id),
        "per_page": 200
    }
    resp = bdl_get("/v2/odds/player_props", params=params)
    return resp.get("data") or []

def bdl_player_points_props_for_today(now_et: datetime) -> List[dict]:
    games = bdl_games_today(now_et)
    print(f"[INFO] BDL games today: {len(games)}")
    all_rows: List[dict] = []
    for g in games:
        gid = g.get("id")
        if gid is None:
            continue
        try:
            rows = bdl_player_points_props_for_game(int(gid))
            all_rows.extend(rows)
            time.sleep(0.35)
        except Exception as e:
            print(f"[WARN] props fetch failed for game_id={gid}: {e}")
            continue
    print(f"[INFO] BDL props rows total: {len(all_rows)}")
    if DEBUG_SAMPLE and all_rows:
        print("[DEBUG] Sample row 1:", _safe_sample(all_rows[0]))
        if len(all_rows) > 1:
            print("[DEBUG] Sample row 2:", _safe_sample(all_rows[1]))
    return all_rows

def bdl_season_averages(player_ids: List[int]) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    if not player_ids:
        return out

    chunk_size = 40
    for i in range(0, len(player_ids), chunk_size):
        chunk = player_ids[i:i+chunk_size]
        params = {"season": SEASON, "per_page": 100}
        for pid in chunk:
            params.setdefault("player_ids[]", []).append(pid)

        resp = bdl_get("/v1/season_averages", params=params)
        rows = resp.get("data") or []
        for row in rows:
            pid = row.get("player_id")
            if pid is not None:
                out[int(pid)] = row

        time.sleep(0.35)
    return out

# ----------------------------
# Odds row parsing (robust)
# ----------------------------
def extract_prop_fields(row: dict) -> Optional[dict]:
    """
    Try multiple possible BDL v2 prop row shapes.
    We need: player_id, player_name, game_id, line.
    """
    # Common direct keys
    line = _as_float(_pick(row, ["line", "value", "stat_value", "prop_line"]))
    game_id = _as_int(_pick(row, ["game_id"]))
    player_id = _as_int(_pick(row, ["player_id"]))

    # Nested player
    player = row.get("player") or row.get("athlete") or row.get("participant") or row.get("entity") or {}
    if player_id is None:
        player_id = _as_int(_pick(player, ["id", "player_id"]))
    player_name = _pick(player, ["full_name", "name", "display_name"])

    # Nested game
    game = row.get("game") or row.get("event") or {}
    if game_id is None:
        game_id = _as_int(_pick(game, ["id", "game_id", "event_id"]))

    # Sometimes line is nested
    if line is None:
        odds = row.get("odds") or row.get("prices") or row.get("outcomes") or None
        # If outcomes array exists, some providers place line there
        if isinstance(odds, list) and odds:
            line = _as_float(_pick(odds[0], ["line", "value", "stat_value"]))
        elif isinstance(odds, dict):
            line = _as_float(_pick(odds, ["line", "value", "stat_value"]))

    if player_id is None or game_id is None or line is None:
        return None

    if not player_name:
        # fallback: direct keys
        player_name = _pick(row, ["player_name", "name"])

    if not player_name:
        player_name = f"player_id {player_id}"

    return {
        "player_id": int(player_id),
        "player_name": str(player_name),
        "game_id": int(game_id),
        "line": float(line),
    }

def reduce_props_rows(props_rows: List[dict]) -> List[dict]:
    """
    Reduce to 1 row per (player_id, game_id) using extracted fields.
    """
    best: Dict[Tuple[int, int], dict] = {}
    for row in props_rows:
        ex = extract_prop_fields(row)
        if not ex:
            continue
        k = (ex["player_id"], ex["game_id"])
        if k not in best:
            best[k] = ex
    return list(best.values())

# ----------------------------
# Edges (slow start): Season PPG vs line
# ----------------------------
def tier_and_reco(edge: float) -> Tuple[str, str]:
    if edge >= 3.0:
        return "A+", "45–50%"
    if edge >= EDGE_THRESHOLD:
        return "A", "30–40%"
    return "", ""

def build_edges(reduced_rows: List[dict]) -> Tuple[List[dict], int]:
    player_ids = sorted(list({r["player_id"] for r in reduced_rows}))
    print(f"[INFO] Unique players w/ lines: {len(player_ids)}")
    avgs = bdl_season_averages(player_ids)

    ideas: List[dict] = []
    for r in reduced_rows:
        pid = r["player_id"]
        avg = avgs.get(pid) or {}
        ppg = avg.get("pts")
        try:
            proj = float(ppg) if ppg is not None else 0.0
        except Exception:
            proj = 0.0

        edge = proj - r["line"]
        if edge < EDGE_THRESHOLD:
            continue

        tier, reco = tier_and_reco(edge)
        if not tier:
            continue

        ideas.append({
            "tier": tier,
            "reco": reco,
            "player_id": pid,
            "player_name": r["player_name"],
            "game_id": r["game_id"],
            "line": r["line"],
            "proj": proj,
            "edge": edge,
            "why": f"Season PPG {proj:.1f} vs line {r['line']:.1f}",
        })

    # Best per player
    best: Dict[int, dict] = {}
    for x in ideas:
        pid = x["player_id"]
        if pid not in best or x["edge"] > best[pid]["edge"]:
            best[pid] = x

    ideas = sorted(best.values(), key=lambda x: x["edge"], reverse=True)[:MAX_BET_IDEAS]
    return ideas, len(reduced_rows)

# ----------------------------
# MAIN
# ----------------------------
def run():
    now_et = _now_et()
    ts_et = now_et.strftime("%Y-%m-%d %I:%M %p ET")

    print(
        f"[BOOT] ts={ts_et} "
        f"TEST_MODE={os.environ.get('TEST_MODE')} "
        f"SEND_NO_EDGE_PING={os.environ.get('SEND_NO_EDGE_PING')} "
        f"ODDS_ONLY_IN_BURST={os.environ.get('ODDS_ONLY_IN_BURST')} "
        f"BURST_START_ET={os.environ.get('BURST_START_ET')} "
        f"BURST_END_ET={os.environ.get('BURST_END_ET')} "
        f"EDGE_THRESHOLD={EDGE_THRESHOLD} "
        f"SEASON={SEASON} "
        f"SPORTSBOOK={SPORTSBOOK} MARKET={MARKET}"
    )

    if TEST_MODE:
        send_one(f"✅ TEST_MODE ping ({ts_et})")
        return

    in_burst = _in_burst_window(now_et)
    if ODDS_ONLY_IN_BURST and not in_burst:
        print("[INFO] Outside burst window; skipping scan.")
        return

    if not BALLDONTLIE_API_KEY:
        send_one(f"⚠️ Missing BALLDONTLIE_API_KEY. ({ts_et})")
        return

    # Fetch props
    try:
        props_rows = bdl_player_points_props_for_today(now_et)
    except Exception as e:
        print(f"[ERROR] props fetch failed: {e}")
        if SEND_NO_EDGE_PING and in_burst:
            send_one(f"⚠️ Agent ran ({ts_et}) but props fetch failed.\n{str(e)[:500]}")
        return

    reduced = reduce_props_rows(props_rows)
    print(f"[INFO] BDL props rows reduced: {len(reduced)}")

    # If still zero, we need the sample logs to adjust extraction
    if not reduced:
        if SEND_NO_EDGE_PING and in_burst:
            send_one(f"⚠️ Agent ran ({ts_et}) but couldn't parse any props rows yet.\nCheck Render logs for [DEBUG] Sample row output.")
        return

    bet_ideas, reduced_count = build_edges(reduced)

    if HEARTBEAT_PING and in_burst:
        send_one(f"🟣 Heartbeat ({ts_et})\nParsed props: {reduced_count}\nEdges: {len(bet_ideas)}")

    if bet_ideas:
        lines: List[str] = []
        lines.append(f"📈 FanDuel Points Edges — {ts_et}")
        lines.append(f"A-only | threshold +{EDGE_THRESHOLD:.1f} | plays {len(bet_ideas)}")
        lines.append("")
        for b in bet_ideas:
            lines.append(f"🔥 {b['tier']} TIER — {b['reco']} bankroll")
            lines.append(f"{b['player_name']} OVER {b['line']:.1f} (edge +{b['edge']:.1f})")
            lines.append(f"Why: {b['why']}")
            lines.append(
                f"#BET pid={b['player_id']} gid={b['game_id']} "
                f"line={b['line']:.1f} proj={b['proj']:.1f} edge={b['edge']:.1f} tier={b['tier']}"
            )
            lines.append("")
        send_chunked("\n".join(lines))
        return

    if SEND_NO_EDGE_PING and in_burst:
        send_one(
            f"✅ Agent scan complete ({ts_et})\n"
            f"No A-tier edges ≥ {EDGE_THRESHOLD:.1f}.\n"
            f"Parsed props: {reduced_count}"
        )

if __name__ == "__main__":
    run()
