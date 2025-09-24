import os, argparse, glob
import pandas as pd
import numpy as np
from src.utils.names import normalize_name

def read_csv_safely(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            continue
    raise RuntimeError(f"CSV okunamadı: {path}")

def pick_first(df: pd.DataFrame, keys: list[str]) -> pd.Series:
    for k in keys:
        if k in df.columns:
            return df[k]
    return pd.Series([np.nan]*len(df))

def map_surface(x):
    s = str(x).strip().lower()
    if s.startswith("hard"): return "Hard"
    if s.startswith("clay"): return "Clay"
    if s.startswith("grass"): return "Grass"
    if s.startswith("carpet"): return "Carpet"
    return "Unknown"

def clean_round(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return ""
    return str(x).strip()

def fair_probs_from_odds(o1, o2):
    try:
        p1 = 1.0/float(o1) if o1 and float(o1)>0 else np.nan
        p2 = 1.0/float(o2) if o2 and float(o2)>0 else np.nan
        s = p1 + p2
        return (p1/s, p2/s) if s>0 else (np.nan, np.nan)
    except Exception:
        return (np.nan, np.nan)

def detect_gender_from_path(path: str) -> str:
    p = path.lower()
    if "wta" in p or "women" in p or "female" in p: return "W"
    if "atp" in p or "men" in p or "male" in p:     return "M"
    return "U"

def is_bad_result(comment_val, score_val) -> bool:
    texts = []
    if isinstance(comment_val, str): texts.append(comment_val.lower())
    if isinstance(score_val, str):   texts.append(score_val.lower())
    t = " ".join(texts)
    for tok in ["walkover","w/o","wo","retired","ret.","abandoned","cancelled"]:
        if tok in t: return True
    return False

def normalize_single_file(path: str) -> pd.DataFrame:
    raw = read_csv_safely(path)
    raw.rename(columns={c: c.strip() for c in raw.columns}, inplace=True)

    get = lambda name, alt=None: name if name in raw.columns else (alt if alt in raw.columns else None)
    date_col    = get("Date")
    tourn_col   = get("Tournament", "Series")
    surface_col = get("Surface")
    round_col   = get("Round")
    winner_col  = get("Winner")
    loser_col   = get("Loser")
    wrank_col   = get("WRank")
    lrank_col   = get("LRank")
    score_col   = get("Score")
    comment_col = get("Comment")

    odds_w = pick_first(raw, ["PSW","B365W","AvgW","IW","Sb"])
    odds_l = pick_first(raw, ["PSL","B365L","AvgL","IL","Sl"])

    dt = pd.to_datetime(raw[date_col], errors="coerce") if date_col else pd.NaT

    df = pd.DataFrame({
        "date": dt,
        "tourney": raw[tourn_col] if tourn_col else "",
        "surface": raw[surface_col].map(map_surface) if surface_col else "Unknown",
        "round": raw[round_col].map(clean_round) if round_col else "",
        "playerA": raw[winner_col].fillna("") if winner_col else "",
        "playerB": raw[loser_col].fillna("") if loser_col else "",
        "rankA": raw[wrank_col] if wrank_col else np.nan,
        "rankB": raw[lrank_col] if lrank_col else np.nan,
        "oddsA": pd.to_numeric(odds_w, errors="coerce"),
        "oddsB": pd.to_numeric(odds_l, errors="coerce"),
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

    pa, pb = [], []
    for a, b in zip(df["oddsA"].tolist(), df["oddsB"].tolist()):
        x, y = fair_probs_from_odds(a, b); pa.append(x); pb.append(y)
    df["pA_implied_fair"] = pa; df["pB_implied_fair"] = pb

    df = df.loc[~df["date"].isna()].copy()
    df["surface"] = df["surface"].map(map_surface)
    df["rankA"] = pd.to_numeric(df["rankA"], errors="coerce")
    df["rankB"] = pd.to_numeric(df["rankB"], errors="coerce")
    return df

def ingest_folder(root: str) -> pd.DataFrame:
    paths = glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)
    if not paths:
        raise SystemExit(f"CSV bulunamadı: {root}")
    frames = []
    for p in paths:
        try:
            frames.append(normalize_single_file(p))
        except Exception:
            continue
    if not frames:
        raise SystemExit("Hiçbir dosya işlenemedi.")
    out = pd.concat(frames, ignore_index=True)
    out.sort_values(["date","tourney"], inplace=True, ignore_index=True)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw/tennisdata")
    ap.add_argument("--out_parquet", default="data/processed/matches.parquet")
    ap.add_argument("--out_csv", default="data/processed/matches.csv")
    ap.add_argument("--sample_csv", default="data/processed/matches_sample.csv")
    ap.add_argument("--sample_n", type=int, default=1000)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_parquet), exist_ok=True)

    df = ingest_folder(args.src)
    print(f"rows={len(df)}  range={df['date'].min().date()}→{df['date'].max().date()}")

    try:
        df.to_parquet(args.out_parquet, index=False)
        print("parquet ->", args.out_parquet)
    except Exception:
        pass

    df.to_csv(args.out_csv, index=False, encoding="utf-8")
    print("csv ->", args.out_csv)

    n = min(args.sample_n, len(df))
    df.head(n).to_csv(args.sample_csv, index=False, encoding="utf-8")
    print("sample ->", args.sample_csv)

if __name__ == "__main__":
    main()
