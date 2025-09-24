from datetime import date
from src.data.fetch_tennis import (
    list_disciplines, list_competitions, list_players, list_venues,
    schedule_by_date, match_details
)

def main():
    discs = list_disciplines()
    print(f"Disciplines: {len(discs)} örnek:", discs[:1])

    comps = list_competitions()
    print(f"Competitions: {len(comps)} örnek:", comps[:1])

    players = list_players()
    print(f"Players: {len(players)} örnek:", players[:1])

    venues = list_venues()
    print(f"Venues: {len(venues)} örnek:", venues[:1])

    today = date.today()
    try:
        sched = schedule_by_date(today)
        print(f"{today} için maç sayısı:", len(sched))
        if sched:
            m = sched[0]
            mid = m.get("GlobalMatchId") or m.get("MatchId") or m.get("GameId")
            print("ilk maç (özet):", {k: m.get(k) for k in [
                "DateTime","Status","GlobalMatchId","MatchId",
                "HomeTeamName","AwayTeamName","ContestantA1Name","ContestantB1Name"
            ]})
            if mid:
                md = match_details(int(mid))
                print("Detay örnek:", str({k: md.get(k) for k in ["Status","ScoreA","ScoreB","Periods"]})[:200])
    except Exception as e:
        print("ScheduleByDate denemesi:", str(e))

if __name__ == "__main__":
    main()
