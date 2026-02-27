import os
import re
import time
import json
from dataclasses import dataclass
from datetime import datetime, date
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Tuple

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

SPORTRADAR_KEY = os.environ.get("SPORTRADAR_KEY", "").strip()
BALLDONTLIE_API_KEY = os.environ.get("BALLDONTLIE_API_KEY", "").strip()

SEND_NO_EDGE_PING = os.environ.get("SEND_NO_EDGE_PING", "1") in ("1", "true", "True")
ODDS_ONLY_IN_BURST = os.environ.get("ODDS_ONLY_IN_BURST", "1") in ("1", "true", "True")

BURST_START_ET = os.environ.get("BURST_START_ET", "17:00").strip()  # HH:MM
BURST_END_ET = os.environ.get("BURST_END_ET", "23:45").strip()      # HH:MM

IMPACT_STATUSES = [s.strip().lower() for s in os.environ.get(
    "IMPACT_STATUSES", "out,doubtful,questionable"
).split(",") if s.strip()]

EDGE_THRESHOLD = float(os.environ.get("EDGE_THRESHOLD", "2.0"))
MAX_BET_IDEAS = int(os.environ.get("MAX_BET_IDEAS", "4"))

# When we include questionable, we dampen the vacated redistribution (less certain)
Q_DAMPEN = float(os.environ.get("QUESTIONABLE_DAMPEN", "0.6"))  # 0.0–1.0

# Optional heartbeat so you *always* see activity (only inside burst window)
HEARTBEAT_PING = os.environ.get("HEARTBEAT_PING", "0") in ("1", "true", "True")

# BallDontLie odds settings (FanDuel points props)
SPORTSBOOK = os.environ.get("SPORTSBOOK", "fanduel").lower().strip()
MARKET = os.environ.get("MARKET", "player_points").lower().strip()

# Season for season averages (adjust if needed)
SEASON = int(os.environ.get("SEASON", str(datetime.now(ET).year - 1)))  # e.g., 2025 for 2025-26 season

# ----------------------------
# Twilio client
# ----------------------------
twilio = Client(TWILIO_SID, TWILIO_TOKEN)

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
    # Assume same-day window (works for your 17:00–23:45)
    return start <= now_et <= end

def _norm_name(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def send_one(body: str) -> None:
    # Twilio WhatsApp: from_ must be "whatsapp:+1..." and to must be "whatsapp:+1..."
    msg = twilio.messages.create(from_=FROM_WHATSAPP, to=TO_WHATSAPP, body=body[:1500])
    print(f"[TWILIO] sent sid={msg.sid} status={msg.status}")

def send_chunked(body: str, chunk_size: int = 1400) -> None:
    # WhatsApp message length can be finicky; chunk to be safe
    if len(body) <= chunk_size:
        send_one(body)
        return
    parts = []
    cur = ""
    for line in body.splitlines():
        if len(cur) + len(line) + 1 > chunk_size:
            parts.append(cur)
            cur = line
        else:
            cur = cur + ("\n" if cur else "") + line
    if cur:
        parts.append(cur)
    for idx, p in enumerate(parts, start=1):
        send_one(f"(Part {idx}/{len(parts)})\n{p}")

# ----------------------------
# Sportradar Injuries
# ----------------------------
@dataclass
class InjuryItem:
    team: str
    player: str
    status: str

def fetch_sportradar_injuries() -> List[InjuryItem]:
    if not SPORTRADAR_KEY:
        print("[WARN] SPORTRADAR_KEY not set; injuries list will be empty.")
        return []

    url = "https://api.sportradar.com/nba/trial/v8/en/league/injuries.json"
    r = requests.get(url, params={"api_key": SPORTRADAR_KEY}, timeout=25)

    if r.status_code != 200:
        raise RuntimeError(f"Sportradar error {r.status_code}: {r.text[:300]}")

    ct = (r.headers.get("Content-Type") or "").lower()
    if "json" not in ct:
        raise RuntimeError(f"Sportradar unexpected content-type: {ct}. Body: {r.text[:300]}")

    data = r.json()

    out: List[InjuryItem] = []

    # Sportradar JSON structure can vary; we try common patterns defensively
    league = data.get("league") or {}
    teams = league.get("teams") or data.get("teams") or []
    for t in teams:
        team_name = t.get("name") or t.get("market") or t.get("alias") or "Unknown Team"
        players = t.get("players") or []
        for p in players:
            # Injury might live under injuries[] or injury
            injuries = p.get("injuries") or []
            if isinstance(injuries, dict):
                injuries = [injuries]
            for inj in injuries:
                status = (inj.get("status") or inj.get("injury_status") or "").lower().strip()
                if not status:
                    continue
                # Sportradar statuses are usually: out, doubtful, questionable, probable
                full_name = p.get("full_name") or f"{p.get('first_name','')} {p.get('last_name','')}".strip()
                if not full_name.strip():
                    full_name = p.get("name") or "Unknown Player"
                out.append(InjuryItem(team=team_name, player=full_name, status=status))

    print(f"[INFO] Sportradar injuries parsed: {len(out)}")
    return out

# ----------------------------
# BallDontLie (Odds + Season Averages)
# ----------------------------
BDL_HEADERS = {"Authorization": BALLDONTLIE_API_KEY} if BALLDONTLIE_API_KEY else {}

def bdl_get(path: str, params: Optional[dict] = None, timeout: int = 25) -> dict:
    """
    BallDontLie can rate limit (429). We retry with backoff.
    We try two base variants: /nba and root (some accounts use /nba namespace).
    """
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
                    print(f"[BDL] {r.status_code} on {path}; retrying in {wait}s...")
                    time.sleep(wait)
                    last_err = f"{r.status_code}: {r.text[:120]}"
                    continue
                if r.status_code != 200:
                    raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
                return r.json()
            except Exception as e:
                last_err = str(e)
                time.sleep(1.0)

    raise RuntimeError(f"BallDontLie request failed for {path}. Last error: {last_err}")

def bdl_player_points_props_for_today(now_et: datetime) -> List[dict]:
    """
    Odds endpoint (v2) used earlier in your logs.
    We pull today's player_points for FanDuel.
    """
    # Use ET date for slate
    d = now_et.date().isoformat()

    # v2 odds endpoint
    params = {
        "sportsbook": SPORTSBOOK,     # "fanduel"
        "market": MARKET,             # "player_points"
        "date": d,                    # YYYY-MM-DD
        "per_page": 200
    }

    resp = bdl_get("/v2/odds/player_props", params=params)
    data = resp.get("data") or []
    print(f"[INFO] BDL props rows: {len(data)} for {d}")
    return data

def bdl_season_averages(player_ids: List[int]) -> Dict[int, dict]:
    """
    Pull season averages in chunks.
    Returns {player_id: averages_row}
    """
    out: Dict[int, dict] = {}
    if not player_ids:
        return out

    # Chunk to reduce 429 risk
    chunk_size = 50
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

        # gentle pause
        time.sleep(0.4)

    return out

# ----------------------------
# Betting model (simple, stable)
# ----------------------------
def tier_and_reco(edge: float, vac_pts: float) -> Tuple[str, str]:
    # A+ if huge edge OR strong edge + big vacuum
    if edge >= 3.0 or (edge >= 2.5 and vac_pts >= 20.0):
        return "A+", "45–50%"
    if edge >= EDGE_THRESHOLD:
        return "A", "30–40%"
    return "", ""

def build_edges(
    injuries: List[InjuryItem],
    props_rows: List[dict],
) -> Tuple[List[dict], int, int]:
    """
    Returns (bet_ideas, triggers_count, props_with_lines_count)
    """
    # Index props by normalized player name for mapping injuries->props
    props_by_name: Dict[str, List[dict]] = {}
    prop_player_ids: List[int] = []
    prop_rows_clean: List[dict] = []

    for row in props_rows:
        # We expect something like:
        # row["player"]["id"], row["player"]["full_name"], row["line"], row["game"]["id"], etc.
        player = row.get("player") or {}
        pid = player.get("id")
        pname = player.get("full_name") or player.get("name") or ""
        line = row.get("line")

        if pid is None or not pname or line is None:
            continue

        try:
            line_f = float(line)
        except Exception:
            continue

        row["_pid"] = int(pid)
        row["_pname"] = pname
        row["_line"] = line_f

        prop_rows_clean.append(row)
        prop_player_ids.append(int(pid))
        props_by_name.setdefault(_norm_name(pname), []).append(row)

    # Dedup IDs
    prop_player_ids = sorted(list(set(prop_player_ids)))

    # Filter injuries to impact statuses
    triggers = [x for x in injuries if x.status.lower() in IMPACT_STATUSES]
    triggers_count = len(triggers)

    # Map injured players to BDL ids when possible (by name match)
    injured_pid_by_team: Dict[str, List[int]] = {}
    injured_name_by_team: Dict[str, List[str]] = {}
    status_by_inj_pid: Dict[int, str] = {}

    for inj in triggers:
        nn = _norm_name(inj.player)
        candidates = props_by_name.get(nn) or []
        if not candidates:
            # no prop row with same name; still keep name (for message only)
            injured_name_by_team.setdefault(inj.team, []).append(inj.player)
            continue
        # if multiple, take first (same player usually)
        pid = candidates[0]["_pid"]
        injured_pid_by_team.setdefault(inj.team, []).append(pid)
        injured_name_by_team.setdefault(inj.team, []).append(inj.player)
        status_by_inj_pid[pid] = inj.status.lower()

    # Season averages for all prop players
    avgs = bdl_season_averages(prop_player_ids)

    # Baseline projection = season PPG (fallback 0)
    baseline_ppg: Dict[int, float] = {}
    for pid in prop_player_ids:
        row = avgs.get(pid) or {}
        ppg = row.get("pts")
        try:
            baseline_ppg[pid] = float(ppg) if ppg is not None else 0.0
        except Exception:
            baseline_ppg[pid] = 0.0

    # Compute vacated points per team from injured players (using their season PPG when available)
    vacated_ppg_by_team: Dict[str, float] = {}
    for team, pids in injured_pid_by_team.items():
        vac = 0.0
        for pid in pids:
            vac += baseline_ppg.get(pid, 0.0)
        vacated_ppg_by_team[team] = vac

    # Build bet ideas by redistributing vacated points across remaining prop players on that team.
    # NOTE: We often don't know team_id reliably from odds row, but we do have team names sometimes.
    # We'll use whatever team label exists on the row (home_team/away_team name) and fall back to no-redistribution.
    bet_ideas: List[dict] = []
    props_with_lines = len(prop_rows_clean)

    for row in prop_rows_clean:
        pid = row["_pid"]
        pname = row["_pname"]
        line = row["_line"]

        game = row.get("game") or {}
        gid = game.get("id") or row.get("game_id")
        try:
            gid = int(gid) if gid is not None else None
        except Exception:
            gid = None

        # Attempt to identify team name from row
        # Some responses include player.team or game.home_team/away_team
        team_name = ""
        pteam = (row.get("player") or {}).get("team") or {}
        team_name = pteam.get("full_name") or pteam.get("name") or team_name

        if not team_name:
            home = (game.get("home_team") or {})
            away = (game.get("visitor_team") or game.get("away_team") or {})
            # can't know which side player is on without team field, so leave blank
            team_name = ""

        base = baseline_ppg.get(pid, 0.0)

        # redistribution
        vac = 0.0
        damp = 1.0
        # We only redistribute if we have a matching team name key from Sportradar
        # Team names differ across providers; try fuzzy match: if team key is substring
        matched_team_key = None
        if team_name:
            tn = team_name.lower()
            for k in vacated_ppg_by_team.keys():
                if k.lower() in tn or tn in k.lower():
                    matched_team_key = k
                    break

        if matched_team_key:
            vac = vacated_ppg_by_team.get(matched_team_key, 0.0)
            # If the injured group contains "questionable" we dampen because uncertain
            # (simple: if any injured pid for team is questionable)
            pids = injured_pid_by_team.get(matched_team_key, [])
            if any(status_by_inj_pid.get(x) == "questionable" for x in pids):
                damp = Q_DAMPEN

        # Weight by baseline ppg so high-usage players absorb more
        # We approximate team_sum_ppg by summing baselines of all players with props and same matched_team_key.
        team_sum = 0.0
        if matched_team_key:
            for r2 in prop_rows_clean:
                pid2 = r2["_pid"]
                pteam2 = (r2.get("player") or {}).get("team") or {}
                t2 = (pteam2.get("full_name") or pteam2.get("name") or "")
                if t2:
                    t2l = t2.lower()
                    if matched_team_key.lower() in t2l or t2l in matched_team_key.lower():
                        team_sum += baseline_ppg.get(pid2, 0.0)

        # Redistribution fraction (tuned to be conservative)
        # We do not add all vacated points; books adjust. Use 35% of vacated as "modelable" shift.
        redist_factor = 0.35

        add = 0.0
        if matched_team_key and team_sum > 0.0 and vac > 0.0:
            add = (base / team_sum) * (vac * redist_factor) * damp

        proj = base + add
        edge = proj - line

        if edge < EDGE_THRESHOLD:
            continue

        tier, reco = tier_and_reco(edge, vac)
        if not tier:
            continue

        trigger_txt = ""
        if matched_team_key and injured_name_by_team.get(matched_team_key):
            # only show top 2 triggers
            top_trigs = injured_name_by_team[matched_team_key][:2]
            trigger_txt = f"{matched_team_key}: " + ", ".join(top_trigs) + ("..." if len(injured_name_by_team[matched_team_key]) > 2 else "")
        else:
            trigger_txt = "Injury impact model"

        why = f"Base PPG {base:.1f}"
        if matched_team_key and vac > 0:
            why += f" | Vacated {vac:.1f} PPG | +{add:.1f} adj (x{redist_factor:.2f}, damp {damp:.2f})"
        else:
            why += " | No team-match redistribution"

        bet_ideas.append({
            "tier": tier,
            "reco": reco,
            "player_id": pid,
            "player_name": pname,
            "game_id": gid,
            "line": line,
            "proj": proj,
            "edge": edge,
            "trigger": trigger_txt,
            "why": why,
        })

    # Dedup by player id: keep best edge
    best: Dict[int, dict] = {}
    for b in bet_ideas:
        pid = b["player_id"]
        if (pid not in best) or (b["edge"] > best[pid]["edge"]):
            best[pid] = b

    bet_ideas = sorted(best.values(), key=lambda x: x["edge"], reverse=True)[:MAX_BET_IDEAS]
    return bet_ideas, triggers_count, props_with_lines

# ----------------------------
# MAIN RUN
# ----------------------------
def run():
    now_et = _now_et()
    ts_et = now_et.strftime("%Y-%m-%d %I:%M %p ET")

    # BOOT line (keep forever; it’s gold for debugging)
    print(
        f"[BOOT] ts={ts_et} "
        f"TEST_MODE={os.environ.get('TEST_MODE')} "
        f"SEND_NO_EDGE_PING={os.environ.get('SEND_NO_EDGE_PING')} "
        f"ODDS_ONLY_IN_BURST={os.environ.get('ODDS_ONLY_IN_BURST')} "
        f"BURST_START_ET={os.environ.get('BURST_START_ET')} "
        f"BURST_END_ET={os.environ.get('BURST_END_ET')} "
        f"EDGE_THRESHOLD={EDGE_THRESHOLD} "
        f"IMPACT_STATUSES={','.join(IMPACT_STATUSES)}"
    )

    # TEST_MODE always pings
    if os.environ.get("TEST_MODE", "0") == "1":
        print("[DEBUG] TEST_MODE triggered, sending WhatsApp ping.")
        send_one(f"✅ TEST_MODE ping ({ts_et})")
        return

    in_burst = _in_burst_window(now_et)

    # If odds only in burst, and we're outside, do nothing (quiet)
    if ODDS_ONLY_IN_BURST and not in_burst:
        print("[INFO] Outside burst window; skipping odds scan.")
        return

    # Fetch injuries + props
    injuries = fetch_sportradar_injuries()

    # Pull props (BDL)
    try:
        props_rows = bdl_player_points_props_for_today(now_et)
    except Exception as e:
        # If odds fail, optionally ping once (inside burst) so you know it's not silent
        err = f"[ERROR] Odds fetch failed: {type(e).__name__}: {e}"
        print(err)
        if in_burst and SEND_NO_EDGE_PING:
            send_one(f"⚠️ Agent ran ({ts_et}) but odds fetch failed.\n{str(e)[:500]}")
        return

    bet_ideas, triggers_count, props_count = build_edges(injuries, props_rows)

    # Heartbeat (optional)
    if in_burst and HEARTBEAT_PING:
        send_one(f"🟣 Heartbeat ({ts_et})\nTriggers: {triggers_count}\nProps: {props_count}\nEdges: {len(bet_ideas)}")

    # Send bets
    if bet_ideas:
        msg: List[str] = []
        msg.append(f"📈 NBA Props Edges (FanDuel) — {ts_et}")
        msg.append(f"Threshold: +{EDGE_THRESHOLD:.1f} | Plays: {len(bet_ideas)}")
        msg.append("")
        for b in bet_ideas:
            msg.append(f"🔥 {b['tier']} TIER — {b['reco']} bankroll")
            msg.append(f"{b['player_name']} OVER {b['line']:.1f} (edge +{b['edge']:.1f})")
            msg.append(f"Trigger: {b['trigger']}")
            msg.append(f"Why: {b['why']}")
            # Machine tag for future auto-grader (optional)
            msg.append(f"#BET pid={b['player_id']} gid={b['game_id'] or 0} line={b['line']:.1f} proj={b['proj']:.1f} edge={b['edge']:.1f} tier={b['tier']}")
            msg.append("")
        send_chunked("\n".join(msg))
        return

    # No edges ping
    if in_burst and SEND_NO_EDGE_PING:
        send_one(
            f"✅ Agent scan complete ({ts_et})\n"
            f"No A-tier edges ≥ {EDGE_THRESHOLD:.1f} found.\n"
            f"Triggers: {triggers_count} | Props: {props_count}"
        )

if __name__ == "__main__":
    run()
