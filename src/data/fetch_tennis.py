from typing import List, Dict, Any
from .sportsdata_client import SportsDataClient

client = SportsDataClient()

def list_competitions() -> List[Dict[str, Any]]:
    return client.get("Competitions")

def list_players() -> List[Dict[str, Any]]:
    return client.get("Players")

def list_venues() -> List[Dict[str, Any]]:
    return client.get("Venues")
