import os
import json
from datetime import datetime, timezone
import requests
from twilio.rest import Client

STATE_FILE = "state.json"

# ---------- ENV / CONFIG ----------
TWILIO_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
SPORTRADAR_KEY = os.environ["SPORTRADAR_API_KEY"]

# Twilio WhatsApp sandbox sender (keep as-is for sandbox)
FROM_WHATSAPP = os.environ.get("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")

# Your phone number (E.164) e.g. +13479029930
TO_WHATSAPP = f"whatsapp:{os.environ['MY_WHATSAPP_NUMBER']}"

# Set TEST_MODE=1 in Render env vars to force a WhatsApp test message on the next run
TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"

# WhatsApp message safety limits
MAX_BODY_CHARS = 1500
MAX_PLAYERS_PER_TEAM = int(os.environ.get("MAX_PLAYERS_PER_TEAM", "50"))

twilio = Client(TWILIO_SID, TWILIO_TOKEN)


def load_state():
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(s):
    with open(STATE_FILE, "w") as f:
        json.dump(s, f, indent=2, sort_keys=True)


def send_one(body: str):
    twilio.messages.create(
        from_=FROM_WHATSAPP,
        to=TO_WHATSAPP,
        body=body[:MAX_BODY_CHARS],
    )


def send_chunked(full_text: str):
    """
    Splits long text into multiple WhatsApp messages safely.
    """
    if len(full_text) <= MAX_BODY_CHARS:
        send_one(full_text)
        return

    parts = []
    remaining = full_text

    while len(remaining) > MAX_BODY_CHARS:
        # Try to split on a newline near the limit for readability
        cut = remaining.rfind("\n", 0, MAX_BODY_CHARS)
        if cut < 200:  # if no good newline, hard cut
            cut = MAX_BODY_CHARS
        parts.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()

    if remaining:
        parts.append(remaining)

    total = len(parts)
    for i, p in enumerate(parts, start=1):
        header = f"(Part {i}/{total})\n"
        # Ensure header doesn't push us over limit
        if len(header) + len(p) > MAX_BODY_CHARS:
            p = p[: MAX_BODY_CHARS - len(header)]
        send_one(header + p)


def fetch():
    url = "https://api.sportradar.com/nba/trial/v8/en/league/injuries.json"
    r = requests.get(url, params={"api_key": SPORTRADAR_KEY}, timeout=20)

    if r.status_code != 200:
        raise RuntimeError(f"Sportradar error {r.status_code}: {r.text[:300]}")

    content_type = (r.headers.get("Content-Type") or "").lower()
    if "json" not in content_type:
        raise RuntimeError(f"Unexpected content-type: {content_type}. Body: {r.text[:300]}")

    return r.json()


def parse_teams_and_injuries(data):
    """
    Returns:
      team_order: [team_name, ...] sorted alphabetically
      injuries_by_team: {team_name: [{"id": pid, "name":..., "status":..., "detail":...}, ...]}
      flat_by_player: {pid: {"name":..., "team":..., "status":..., "detail":...}}
    """
    injuries_by_team = {}
    flat_by_player = {}

    for team in data.get("teams", []):
        team_name = team.get("name") or team.get("market") or team.get("id", "TEAM")
        injuries_by_team.setdefault(team_name, [])

        for p in team.get("players", []):
            injuries = p.get("injuries") or []
            if not injuries:
                continue

            inj = injuries[-1]
            pid = p.get("id")
            if not pid:
                continue

            player_obj = {
                "id": pid,
                "name": p.get("full_name")
                or f"{p.get('first_name','')} {p.get('last_name','')}".strip(),
                "team": team_name,
                "status": (inj.get("status") or "Unknown").strip(),
                "detail": (inj.get("comment") or inj.get("description") or "").strip(),
            }

            injuries_by_team[team_name].append(player_obj)
            flat_by_player[pid] = {
                "name": player_obj["name"],
                "team": team_name,
                "status": player_obj["status"],
                "detail": player_obj["detail"],
            }

    # Sort teams and players for stable output
    team_order = sorted(injuries_by_team.keys())
    for t in team_order:
        injuries_by_team[t].sort(key=lambda x: (x["name"] or ""))

    return team_order, injuries_by_team, flat_by_player


def compute_changes(old, new):
    changes = []

    for pid, cur in new.items():
        prev = old.get(pid)
        if not prev:
            changes.append((None, cur))
        else:
            if (prev.get("status"), prev.get("detail")) != (cur.get("status"), cur.get("detail")):
                changes.append((prev, cur))

    for pid, prev in old.items():
        if pid not in new:
            changes.append((prev, None))

    return changes


def format_changes(changes):
    lines = ["🚨 Changes since last check:"]
    for prev, cur in changes[:30]:
        if prev is None and cur is not None:
            lines.append(
                f"➕ {cur['name']} ({cur['team']}): {cur['status']}"
                + (f" — {cur['detail']}" if cur.get("detail") else "")
            )
        elif prev is not None and cur is None:
            lines.append(f"✅ {prev['name']} ({prev['team']}): cleared (was {prev.get('status','')})")
        else:
            lines.append(
                f"🔄 {cur['name']} ({cur['team']}): {prev.get('status')} → {cur.get('status')}"
                + (f" — {cur.get('detail')}" if cur.get("detail") else "")
            )
    if len(changes) > 30:
        lines.append(f"…and {len(changes) - 30} more changes.")
    return "\n".join(lines)


def format_team_summary(team_order, injuries_by_team):
    """
    Always prints every team. If a team has no injuries, prints 'No reported injuries'.
    """
    lines = ["🏥 League-wide injury report (by team):"]
    for team in team_order:
        players = injuries_by_team.get(team, [])
        if not players:
            lines.append(f"\n{team}: No reported injuries")
            continue

        lines.append(f"\n{team}:")
        for p in players[:MAX_PLAYERS_PER_TEAM]:
            line = f"- {p['name']}: {p['status']}"
            if p.get("detail"):
                line += f" — {p['detail']}"
            lines.append(line)

        if len(players) > MAX_PLAYERS_PER_TEAM:
            lines.append(f"- …and {len(players) - MAX_PLAYERS_PER_TEAM} more")

    return "\n".join(lines)


def run():
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if TEST_MODE:
        send_one(f"✅ NBA Injury Agent test: WhatsApp delivery is working. ({ts})")
        return

    old = load_state()
    data = fetch()
    team_order, injuries_by_team, flat_new = parse_teams_and_injuries(data)

    # IMPORTANT: Some APIs may omit teams with truly no injuries from the payload.
    # We still print all teams that appear in the feed. (If you need all 30 always,
    # we can hardcode the full team list.)
    changes = compute_changes(old, flat_new)

    header = f"NBA Injury Update ({ts})"
    blocks = [header]

    if changes:
        blocks.append(format_changes(changes))
    else:
        blocks.append("✅ No changes since last check.")

    blocks.append(format_team_summary(team_order, injuries_by_team))

    full_message = "\n\n".join(blocks)
    send_chunked(full_message)

    save_state(flat_new)


if __name__ == "__main__":
    run()
