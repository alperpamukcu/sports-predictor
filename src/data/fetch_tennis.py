from datetime import date
from typing import List, Dict, Any
from .sportsdata_client import SportsDataClient

client = SportsDataClient()

def list_disciplines() -> List[Dict[str, Any]]:
    return client.get("Disciplines")

def list_competitions() -> List[Dict[str, Any]]:
    return client.get("Competitions")

def list_players() -> List[Dict[str, Any]]:
    return client.get("Players")

def list_venues() -> List[Dict[str, Any]]:
    return client.get("Venues")

def schedule_by_date(day: date) -> List[Dict[str, Any]]:
    candidates = [
        f"ScheduleByDate/{day.isoformat()}",
        f"MatchesByDate/{day.isoformat()}",
        f"GamesByDate/{day.isoformat()}",
        f"Schedule/{day.isoformat()}",
    ]
    last_err = None
    for ep in candidates:
        try:
            data = client.get(ep)
            if isinstance(data, list):
                return data
        except Exception as e:
            last_err = e

def match_details(match_id: int) -> Dict[str, Any]:
    return client.get(f"Match/{match_id}")
