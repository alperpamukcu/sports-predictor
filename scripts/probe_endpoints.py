# scripts/probe_endpoints.py
from datetime import date
from urllib.parse import urlparse
from src.data.sportsdata_client import SportsDataClient

def change_version(base_url: str, new_ver: str) -> str:
    parts = base_url.split("/")
    for i, p in enumerate(parts):
        if p.startswith("v") and len(p) <= 4:
            parts[i] = new_ver
            break
    return "/".join(parts)

def try_probes(base_url: str):
    today = date.today().isoformat()
    probes = [
        "Disciplines",
        "Leagues",
        "Competitions",
        "Tournaments",
        "Seasons",
        f"ScheduleByDate/{today}",
        f"MatchesByDate/{today}",
        f"Schedule/{today}",
        f"GamesByDate/{today}",
        "Rankings",
        "Players",
        "Venues",
        "Match/1",
    ]
    c = SportsDataClient(base_url=base_url)
    print(f"\n=== PROBE @ {base_url} ===")
    for ep in probes:
        try:
            data = c.get(ep)
            kind = type(data).__name__
            sample = str(data)[:120].replace("\n"," ")
            print(f"[OK  ] {ep:25s} -> {kind}: {sample}")
        except Exception as e:
            print(f"[FAIL] {ep:25s} -> {e}")

def main():
    from src.utils.config import settings
    try_probes(settings.base_url)
    v3 = change_version(settings.base_url, "v3")
    if v3 != settings.base_url:
        try_probes(v3)

if __name__ == "__main__":
    main()
