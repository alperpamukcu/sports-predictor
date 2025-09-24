from typing import List, Dict
from datetime import date
import pandas as pd
from .fetch_tennis import schedule_by_date

def build_today_feature_frame(day: date) -> pd.DataFrame:
    raw: List[Dict] = schedule_by_date(day)
    if not raw:
        return pd.DataFrame()
    rows = []
    for m in raw:
        rows.append({
            "day": m.get("Day"),
            "datetime_et": m.get("DateTime"),
            "comp": m.get("CompetitionName") or m.get("TournamentName"),
            "round": m.get("RoundName"),
            "a": m.get("ContestantA1Name"),
            "b": m.get("ContestantB1Name"),
            "status": m.get("Status"),
            "global_match_id": m.get("GlobalMatchId") or m.get("MatchId"),
        })
    return pd.DataFrame(rows)
