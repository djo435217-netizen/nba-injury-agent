import os, json, requests
from datetime import datetime, timezone
from twilio.rest import Client

STATE_FILE = "state.json"

TWILIO_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
FROM_WHATSAPP = "whatsapp:+14155238886"
TO_WHATSAPP = f"whatsapp:{os.environ['MY_WHATSAPP_NUMBER']}"

SPORTRADAR_KEY = os.environ["SPORTRADAR_API_KEY"]

twilio = Client(TWILIO_SID, TWILIO_TOKEN)

def load_state():
    if not os.path.exists(STATE_FILE):
        return {}
    return json.load(open(STATE_FILE))

def save_state(s):
    json.dump(s, open(STATE_FILE,"w"))

def send(msg):
    twilio.messages.create(
        from_=FROM_WHATSAPP,
        to=TO_WHATSAPP,
        body=msg[:1500]
    )

def fetch():
    url = "https://api.sportradar.com/nba/trial/v8/en/league/injuries.json"
    r = requests.get(url, params={"api_key": SPORTRADAR_KEY})
    data = r.json()
    out = {}
    for team in data.get("teams", []):
        for p in team.get("players", []):
            if not p.get("injuries"): continue
            inj = p["injuries"][-1]
            out[p["id"]] = {
                "name": p.get("full_name"),
                "team": team.get("name"),
                "status": inj.get("status"),
                "detail": inj.get("comment","")
            }
    return out

def run():
    old = load_state()
    new = fetch()
    changes = []

    for k,v in new.items():
        if k not in old:
            changes.append(f"➕ {v['name']} ({v['team']}): {v['status']}")
        elif old[k]["status"] != v["status"]:
            changes.append(
                f"🚨 {v['name']} ({v['team']}): {old[k]['status']} → {v['status']}"
            )

    for k,v in old.items():
        if k not in new:
            changes.append(
                f"✅ {v['name']} ({v['team']}): cleared from report"
            )

    if changes:
        msg = "🏥 NBA Injury Updates\n" + "\n".join(changes[:20])
        send(msg)

    save_state(new)

if __name__ == "__main__":
    run()
