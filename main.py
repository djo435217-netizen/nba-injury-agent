def fetch():
    url = "https://api.sportradar.com/nba/trial/v8/en/league/injuries.json"
    r = requests.get(url, params={"api_key": SPORTRADAR_KEY}, timeout=20)

    # ✅ If it’s not JSON, print the real response
    if r.status_code != 200:
        raise RuntimeError(f"Sportradar error {r.status_code}: {r.text[:300]}")

    content_type = (r.headers.get("Content-Type") or "").lower()
    if "json" not in content_type:
        raise RuntimeError(f"Unexpected content-type: {content_type}. Body: {r.text[:300]}")

    return r.json()
