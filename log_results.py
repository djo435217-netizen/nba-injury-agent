"""
log_results.py -- Run this after games to record actual scores.

Usage:
  python log_results.py                  # show all pending (unresolved) plays
  python log_results.py "Player Name" 28 # log that player scored 28 pts
  python log_results.py --summary        # show hit rate summary

The results_log.json file is created automatically by main.py each time
a play is sent. This script just fills in the actual_pts and hit fields.
"""

import json
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

RESULTS_FILE = os.environ.get("RESULTS_FILE", "results_log.json")
ET = ZoneInfo("America/New_York")


def load_log():
    if not os.path.exists(RESULTS_FILE):
        print(f"No results file found at {RESULTS_FILE}")
        print("Run main.py first to generate some plays.")
        return {}
    with open(RESULTS_FILE, "r") as f:
        return json.load(f)


def save_log(log):
    with open(RESULTS_FILE, "w") as f:
        json.dump(log, f, indent=2, sort_keys=True)


def show_pending(log):
    pending = [(k, v) for k, v in log.items() if v.get("hit") is None]
    if not pending:
        print("No pending plays to resolve.")
        return
    pending.sort(key=lambda x: x[1].get("sent_ts", 0), reverse=True)
    print(f"\n{'='*55}")
    print(f"PENDING PLAYS ({len(pending)} unresolved)")
    print(f"{'='*55}")
    for k, v in pending:
        print(f"\n  {v['player_name']} OVER {v['line']} ({v['prop_type']})")
        print(f"  Date: {v['date']}  Proj: {v['proj']}  Edge: +{v['edge']}")
        print(f"  Key: {k}")
    print(f"\nTo resolve: python log_results.py \"Player Name\" ACTUAL_PTS")
    print(f"Example:    python log_results.py \"Donovan Mitchell\" 31\n")


def log_result(log, player_name, actual_pts):
    actual_pts = float(actual_pts)
    name_lower = player_name.lower().strip()

    matches = []
    for k, v in log.items():
        if v.get("hit") is not None:
            continue
        if name_lower in v.get("player_name", "").lower():
            matches.append((k, v))

    if not matches:
        print(f"No pending play found for '{player_name}'")
        print("Use --pending to see all unresolved plays.")
        return

    if len(matches) > 1:
        print(f"Multiple matches for '{player_name}':")
        for k, v in matches:
            print(f"  {v['player_name']} OVER {v['line']} on {v['date']} (key: {k})")
        print("Be more specific with the name.")
        return

    k, v = matches[0]
    line = float(v["line"])
    hit = actual_pts > line

    log[k]["actual_pts"] = actual_pts
    log[k]["hit"] = hit
    log[k]["resolved_ts"] = int(datetime.now(ET).timestamp())

    save_log(log)

    result_str = "HIT" if hit else "MISS"
    print(f"\n  {result_str}: {v['player_name']} scored {actual_pts} vs line {line}")
    print(f"  Proj was {v['proj']} | Edge was +{v['edge']}")
    print(f"  Saved to {RESULTS_FILE}\n")


def show_summary(log):
    resolved = [v for v in log.values() if v.get("hit") is not None]
    if not resolved:
        print("No resolved plays yet. Log some results first.")
        return

    total = len(resolved)
    hits = sum(1 for v in resolved if v["hit"])
    rate = hits / total

    # By section
    by_section = {}
    for v in resolved:
        s = v.get("section", "unknown")
        by_section.setdefault(s, {"hits": 0, "total": 0})
        by_section[s]["total"] += 1
        if v["hit"]:
            by_section[s]["hits"] += 1

    # Recent 20
    recent = sorted(resolved, key=lambda x: x.get("sent_ts", 0), reverse=True)[:20]
    recent_hits = sum(1 for v in recent if v["hit"])

    # By tier
    by_tier = {}
    for v in resolved:
        t = v.get("tier", "unknown")
        by_tier.setdefault(t, {"hits": 0, "total": 0})
        by_tier[t]["total"] += 1
        if v["hit"]:
            by_tier[t]["hits"] += 1

    print(f"\n{'='*55}")
    print(f"HIT RATE SUMMARY")
    print(f"{'='*55}")
    print(f"Overall:  {hits}/{total} ({rate*100:.1f}%)")
    print(f"Last 20:  {recent_hits}/{len(recent)} ({recent_hits/len(recent)*100:.1f}%)")
    print()
    print("By section:")
    for s, d in sorted(by_section.items()):
        r = d["hits"] / d["total"]
        print(f"  {s:<15} {d['hits']}/{d['total']} ({r*100:.1f}%)")
    print()
    print("By confidence tier:")
    for t, d in sorted(by_tier.items()):
        r = d["hits"] / d["total"]
        print(f"  {t:<20} {d['hits']}/{d['total']} ({r*100:.1f}%)")

    # Best and worst players
    by_player = {}
    for v in resolved:
        p = v["player_name"]
        by_player.setdefault(p, {"hits": 0, "total": 0})
        by_player[p]["total"] += 1
        if v["hit"]:
            by_player[p]["hits"] += 1

    qualified = [(p, d) for p, d in by_player.items() if d["total"] >= 3]
    if qualified:
        qualified.sort(key=lambda x: x[1]["hits"] / x[1]["total"], reverse=True)
        print()
        print("Best players (3+ plays):")
        for p, d in qualified[:5]:
            r = d["hits"] / d["total"]
            print(f"  {p:<25} {d['hits']}/{d['total']} ({r*100:.1f}%)")

    print(f"{'='*55}\n")


def main():
    args = sys.argv[1:]

    log = load_log()
    if not log:
        return

    if not args or args[0] == "--pending":
        show_pending(log)
    elif args[0] == "--summary":
        show_summary(log)
    elif len(args) == 2:
        log_result(log, args[0], args[1])
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
