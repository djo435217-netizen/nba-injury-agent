import os
from datetime import datetime
from zoneinfo import ZoneInfo
from twilio.rest import Client

ET = ZoneInfo("America/New_York")

TWILIO_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]

FROM_WHATSAPP = os.environ.get("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
TO_WHATSAPP = f"whatsapp:{os.environ['MY_WHATSAPP_NUMBER']}"

twilio = Client(TWILIO_SID, TWILIO_TOKEN)

def _now_et():
    return datetime.now(ET)

def send_one(body: str):
    twilio.messages.create(
        from_=FROM_WHATSAPP,
        to=TO_WHATSAPP,
        body=body[:1500]
    )

def run():
    now_et = _now_et()
    ts_et = now_et.strftime("%Y-%m-%d %I:%M %p ET")

    # ===== BOOT DEBUG PRINT =====
    print(
        f"[BOOT] ts={ts_et} "
        f"TEST_MODE={os.environ.get('TEST_MODE')} "
        f"SEND_NO_EDGE_PING={os.environ.get('SEND_NO_EDGE_PING')} "
        f"ODDS_ONLY_IN_BURST={os.environ.get('ODDS_ONLY_IN_BURST')} "
        f"BURST_START_ET={os.environ.get('BURST_START_ET')} "
        f"BURST_END_ET={os.environ.get('BURST_END_ET')}"
    )

    # ===== HARD TEST MODE =====
    if os.environ.get("TEST_MODE", "0") == "1":
        print("[DEBUG] TEST_MODE triggered, sending WhatsApp ping.")
        send_one(f"✅ TEST_MODE ping ({ts_et})")
        return

    # ===== NORMAL LOGIC PLACEHOLDER =====
    # This confirms cron is running but no bet logic executed yet.
    print("[DEBUG] TEST_MODE is OFF. No bet logic executed in this diagnostic version.")

if __name__ == "__main__":
    run()
