from fastapi import FastAPI, Query, HTTPException
from datetime import date
from src.data.fetch_tennis import (
    list_competitions, list_players, list_venues, list_disciplines, schedule_by_date
)

app = FastAPI(title="Tennis Predictor API (skeleton)")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/disciplines")
def disciplines():
    return list_disciplines()

@app.get("/competitions")
def competitions():
    return list_competitions()

@app.get("/players")
def players(q: str | None = None, limit: int = 50):
    """Basit arama: ?q=rafa gibi; yoksa tüm liste (limitli) döner."""
    data = list_players()
    if q:
        ql = q.lower()
        data = [
            p for p in data
            if any(
                (p.get("FirstName") or "").lower().startswith(ql) or
                (p.get("LastName") or "").lower().startswith(ql) or
                (p.get("CommonName") or "").lower().startswith(ql) or
                (p.get("ShortName") or "").lower().startswith(ql)
            )
        ]
    return data[:max(1, min(limit, 200))]

@app.get("/venues")
def venues():
    return list_venues()

@app.get("/schedule")
def schedule(day: date = Query(..., description="YYYY-MM-DD")):
    try:
        return schedule_by_date(day)
    except Exception as e:
        raise HTTPException(status_code=501, detail=str(e))
