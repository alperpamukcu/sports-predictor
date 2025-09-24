from fastapi import FastAPI, Query
from src.data.fetch_tennis import list_competitions, list_players, list_venues

app = FastAPI(title="Tennis Predictor API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/competitions")
def competitions():
    return list_competitions()

@app.get("/players")
def players(q: str | None = None, limit: int = 50):
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
