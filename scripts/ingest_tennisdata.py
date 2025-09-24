# scripts/ingest_tennisdata.py
import os, sys, argparse, glob
import pandas as pd
import numpy as np
from datetime import datetime
from src.utils.names import normalize_name


def read_csv_safely(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise RuntimeError(f"CSV okunamadı: {path}")

def pick_first(df: pd.DataFrame, keys: list[str]) -> pd.Series:
    """İlk bulunan kolonu döndür (yoksa NaN). Örn: Pinnacle yoksa B365'e düş."""
    for k in keys:
        if k in df.columns:
            return df[k]
    return pd.Series([np.nan]*len(df))

def map_surface(x: str | float | None) -> str:
    s = str(x).strip().lower()
    if s.startswith("hard"): return "Hard"
    if s.startswith("clay"): return "Clay"
    if s.startswith("grass"): return "Grass"
    if s.startswith("carpet"): return "Carpet"
    return "Unknown"

def clean_round(x: str | float | None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)): return ""
    return str(x).strip()

def fair_probs_from_odds(o1: float | None, o2: float | None) -> tuple[float, float]:
    """Decimal odds -> implied prob -> fair (juice-free) normalize."""
    try:
        p1 = 1.0/float(o1) if o1 and float(o1)>0 else np.nan
        p2 = 1.0/float(o2) if o2 and float(o2)>0 else np.nan
        if np.isnan(p1) or np.isnan(p2):
            return (np.nan, np.nan)
        s = p1 + p2
        if s > 0:
            return (p1/s, p2/s)
        return (np.nan, np.nan)
    except Exception:
        return (np.nan, np.nan)

def detect_gender_from_path(path: str) -> str:
    p = path.lower()
    if "wta" in p or "women" in p or "female" in p:
        return "W"
    if "atp" in p or "men" in p or "male" in p:
        return "M"
    return "U" 

def is_bad_result(comment_val, score_val) -> bool:
    """Walkover/Retired gibi sonuçları ele (model eğitiminde gürültü)."""
    texts = []
    if isinstance(comment_val, str): texts.append(comment_val.lower())
    if isinstance(score_val, str):   texts.append(score_val.lower())
    t = " ".join(texts)
    bad_tokens = ["walkover", "w/o", "wo", "retired", "ret.", "abandoned", "cancelled"]
    return any(tok in t for tok in bad_tokens)

def normalize_single_file(path: str) -> pd.DataFrame:
    raw = read_csv_safely(path)
    raw_cols = {c: c.strip() for c in raw.columns}
    raw.rename(columns=raw_cols, inplace=True)

    date_col      = "Date" if "Date" in raw.columns else None
    tourn_col     = "Tournament" if "Tournament" in raw.columns else ("Series" if "Series" in raw.columns else None)
    surface_col   = "Surface" if "Surface" in raw.columns else None
    round_col     = "Round" if "Round" in raw.columns else None
    winner_col    = "Winner" if "Winner" in raw.columns else None
    loser_col     = "Loser" if "Loser" in raw.columns else None
    wrank_col     = "WRank" if "WRank" in raw.columns else None
    lrank_col     = "LRank" if "LRank" in raw.columns else None
    score_col     = "Score" if "Score" in raw.columns else None
    comment_col   = "Comment" if "Comment" in raw.columns else None

    odds_w = pick_first(raw, ["PSW","B365W","AvgW","IW","Sb"])
    odds_l = pick_first(raw, ["PSL","B365L","AvgL","IL","Sl"])

    if date_col:
        dt = pd.to_datetime(raw[date_col], errors="coerce")
    else:
        dt = pd.NaT

    df = pd.DataFrame({
        "date": dt,
        "tourney": raw[tourn_col] if tourn_col else "",
        "surface": raw[surface_col].map(map_surface) if surface_col else "Unknown",
        "round": raw[round_col].map(clean_round) if round_col else "",
        "playerA": raw[winner_col].fillna("") if winner_col else "",
        "playerB": raw[loser_col].fillna("") if loser_col else "",
        "rankA": raw[wrank_col] if wrank_col else np.nan,
        "rankB": raw[lrank_col] if lrank_col else np.nan,
        "oddsA": pd.to_numeric(odds_w, errors="coerce"),   # win
        "oddsB": pd.to_numeric(odds_l, errors="coerce"),   # lose
        "score": raw[score_col] if score_col else "",
        "comment": raw[comment_col] if comment_col else "",
        "source_file": os.path.basename(path),
        "gender": detect_gender_from_path(path),
        "winner": "A",  
    })

    df["playerA_norm"] = df["playerA"].map(normalize_name)
    df["playerB_norm"] = df["playerB"].map(normalize_name)

    bad_mask = df.apply(lambda r: is_bad_result(r["comment"], r["score"]), axis=1)
    df = df.loc[~bad_mask].copy()

    pA, pB = [], []
    for a, b in zip(df["oddsA"].tolist(), df["oddsB"].tolist()):
        pa, pb = fair_probs_from_odds(a, b)
        pA.append(pa); pB.append(pb)
    df["pA_implied_fair"] = pA
    df["pB_implied_fair"] = pB

    df = df.loc[~df["date"].isna()].copy()
    df["surface"] = df["surface"].map(map_surface)

    df["rankA"] = pd.to_numeric(df["rankA"], errors="coerce")
    df["rankB"] = pd.to_numeric(df["rankB"], errors="coerce")

    return df

def ingest_folder(root: str) -> pd.DataFrame:
    paths = glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)
    if not paths:
        raise SystemExit(f"Hiç CSV bulunamadı: {root}\nLütfen Tennis-Data sezon CSV'lerini bu klasöre koy.")
    print(f"Bulunan CSV sayısı: {len(paths)}")
    frames = []
    for i, p in enumerate(paths, 1):
        try:
            df = normalize_single_file(p)
            frames.append(df)
            if i % 10 == 0:
                print(f"...{i} dosya işlendi")
        except Exception as e:
            print(f"[WARN] {p} atlandı: {e}")
    if not frames:
        raise SystemExit("Hiçbir dosya işlenemedi.")
    out = pd.concat(frames, ignore_index=True)
    out.sort_values(["date","tourney"], inplace=True, ignore_index=True)
    return out

def main():
    ap = argparse.ArgumentParser(description="Ingest Tennis-Data CSVs into a single normalized matches table.")
    ap.add_argument("--src", default="data/raw/tennisdata", help="Kaynak klasör (CSV'ler)")
    ap.add_argument("--out_parquet", default="data/processed/matches.parquet", help="Parquet çıktı yolu")
    ap.add_argument("--out_csv", default="data/processed/matches.csv", help="CSV çıktı yolu")
    ap.add_argument("--sample_csv", default="data/processed/matches_sample.csv", help="İlk N satırlık örnek")
    ap.add_argument("--sample_n", type=int, default=1000, help="Örnek satır sayısı")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_parquet), exist_ok=True)

    df = ingest_folder(args.src)
    print(f"Toplam satır: {len(df)}")
    mind, maxd = df["date"].min(), df["date"].max()
    print(f"Tarih aralığı: {mind.date()} → {maxd.date()}")
    print("Sütunlar:", list(df.columns))

    try:
        df.to_parquet(args.out_parquet, index=False)
        print(f"Parquet yazıldı: {args.out_parquet}")
    except Exception as e:
        print(f"[WARN] Parquet yazılamadı ({e}); pyarrow kurulu mu?")

    df.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"CSV yazıldı: {args.out_csv}")

    # Örnek
    n = min(args.sample_n, len(df))
    df.head(n).to_csv(args.sample_csv, index=False, encoding="utf-8")
    print(f"Örnek yazıldı ({n} satır): {args.sample_csv}")

if __name__ == "__main__":
    main()
