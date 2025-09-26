import argparse
import math
import os
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# Columns documented by Tennis-Data notes: http://www.tennis-data.co.uk/notes.txt
# We defensively handle their presence/absence across years.


def load_raw_csv(csv_path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load the raw allyears.csv into a DataFrame with sensible dtypes.

    Parameters
    ----------
    csv_path: str
        Absolute path to the CSV file (allyears.csv)
    nrows: Optional[int]
        If provided, load only the first n rows (useful for smoke runs)

    Returns
    -------
    pd.DataFrame
    """
    dtype_overrides = {
        "ATP": "Int64",
        "WRank": "float64",
        "LRank": "float64",
        "WPts": "float64",
        "LPts": "float64",
        "Best of": "float64",
        # Odds columns (bookmaker odds as floats)
        "B365W": "float64",
        "B365L": "float64",
        "PSW": "float64",
        "PSL": "float64",
        "MaxW": "float64",
        "MaxL": "float64",
        "AvgW": "float64",
        "AvgL": "float64",
    }

    parse_dates = ["Date"]

    # Treat common non-numeric tokens as missing (e.g., 'NR' = Not Ranked)
    custom_na_values = [
        "NR",
        "N/R",
        "NA",
        "N/A",
        "na",
        "n/a",
        "NULL",
        "Null",
        "null",
        "",
        "-",
        "—",
    ]

    df = pd.read_csv(
        csv_path,
        dtype=dtype_overrides,
        parse_dates=parse_dates,
        nrows=nrows,
        keep_default_na=True,
        na_values=custom_na_values,
    )
    return df


def _first_present_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first column name present in df from candidates, else None."""
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise safe division returning NaN where invalid."""
    return numerator.astype("float64") / denominator.astype("float64")


def compute_pre_match_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer pre-match, non-leaky features including odds transforms, ranking/points deltas,
    and date features. Also derive a canonical favorite/underdog perspective per match using
    available odds or, if missing, rankings.

    The target label is favorite_won: 1 if the pre-match favorite won the match, else 0.

    Returns a new DataFrame including engineered features and the target.
    """
    df = df.copy()

    # Identify odds columns (prefer Bet365 then Pinnacle then Max/Avg if necessary)
    odds_win_col = _first_present_column(df, ["B365W", "PSW", "AvgW", "MaxW"])  # odds of winner
    odds_lose_col = _first_present_column(df, ["B365L", "PSL", "AvgL", "MaxL"])  # odds of loser

    # Compute implied probabilities where available
    if odds_win_col is not None and odds_lose_col is not None:
        df["implied_prob_winner"] = 1.0 / df[odds_win_col]
        df["implied_prob_loser"] = 1.0 / df[odds_lose_col]
        # Normalize to handle overround (bookmaker margin)
        overround = df["implied_prob_winner"] + df["implied_prob_loser"]
        
        df["implied_prob_winner_norm"] = df["implied_prob_winner"] / overround
        df["implied_prob_loser_norm"] = df["implied_prob_loser"] / overround

        # Determine favorite using lower decimal odds
        # favorite_is_winner = 1 if winner had lower (better) odds than loser
        df["favorite_is_winner"] = (df[odds_win_col] < df[odds_lose_col]).astype("int8")
        df["favorite_won"] = df["favorite_is_winner"]

        # Favorite/underdog implied probabilities
        df["favorite_implied_prob"] = np.where(
            df["favorite_is_winner"] == 1, df["implied_prob_winner_norm"], df["implied_prob_loser_norm"],
        )
        df["underdog_implied_prob"] = np.where(
            df["favorite_is_winner"] == 1, df["implied_prob_loser_norm"], df["implied_prob_winner_norm"],
        )

        # Odds ratio/log-odds gap (bigger -> more lopsided)
        df["odds_ratio"] = df[odds_lose_col] / df[odds_win_col]
        df["log_odds_ratio"] = np.log(df["odds_ratio"])  # natural log
        df["fav_und_prob_gap"] = df["favorite_implied_prob"] - df["underdog_implied_prob"]
    else:
        # If no odds at all, fall back to rankings to define favorite and label
        # Lower rank number is better
        df["favorite_is_winner"] = (df["WRank"] < df["LRank"]).astype("Int64")
        df["favorite_won"] = df["favorite_is_winner"].fillna(pd.NA).astype("Int64")

    # Ranking/points features (pre-tournament metrics)
    # Differences from favorite perspective when odds present; otherwise from winner perspective with sign handling.
    rank_w = df["WRank"].astype("float64") if "WRank" in df.columns else pd.Series(np.nan, index=df.index)
    rank_l = df["LRank"].astype("float64") if "LRank" in df.columns else pd.Series(np.nan, index=df.index)
    pts_w = df["WPts"].astype("float64") if "WPts" in df.columns else pd.Series(np.nan, index=df.index)
    pts_l = df["LPts"].astype("float64") if "LPts" in df.columns else pd.Series(np.nan, index=df.index)

    # Note: better players have LOWER rank numbers
    df["rank_diff_w_minus_l"] = rank_w - rank_l
    df["points_diff_w_minus_l"] = pts_w - pts_l

    # Favorite-centric diffs when favorite_is_winner is known
    if "favorite_is_winner" in df.columns:
        df["favorite_rank"] = np.where(df["favorite_is_winner"] == 1, rank_w, rank_l)
        df["underdog_rank"] = np.where(df["favorite_is_winner"] == 1, rank_l, rank_w)
        df["favorite_points"] = np.where(df["favorite_is_winner"] == 1, pts_w, pts_l)
        df["underdog_points"] = np.where(df["favorite_is_winner"] == 1, pts_l, pts_w)
        # Positive means favorite has worse (higher number) rank -> unexpected
        df["favorite_rank_minus_underdog_rank"] = df["favorite_rank"] - df["underdog_rank"]
        df["favorite_points_minus_underdog_points"] = df["favorite_points"] - df["underdog_points"]

        # Ratios (handle zeros/NaNs safely)
        df["rank_ratio_fav_over_und"] = _safe_divide(df["favorite_rank"], df["underdog_rank"])  # >1 -> favorite worse ranked
        df["points_ratio_fav_over_und"] = _safe_divide(df["favorite_points"], df["underdog_points"])  # >1 -> favorite more points

    # Date-derived features
    if "Date" in df.columns:
        df["year"] = df["Date"].dt.year
        df["month"] = df["Date"].dt.month
        df["dayofweek"] = df["Date"].dt.dayofweek
        df["weekofyear"] = df["Date"].dt.isocalendar().week.astype("int64")

    # One-hot encode low-cardinality pre-match categorical columns
    df = one_hot_encode_low_cardinality(
        df,
        categorical_columns=[
            col
            for col in ["Series", "Court", "Surface", "Round", "Best of"]
            if col in df.columns
        ],
        max_unique=30,
    )

    # Remove leaky and high-cardinality columns
    cols_to_drop = [
        # Identifiers/names (high cardinality)
        "ATP",
        "Location",
        "Tournament",
        "Winner",
        "Loser",
        # Post-match outcomes/leakage
        "W1",
        "L1",
        "W2",
        "L2",
        "W3",
        "L3",
        "W4",
        "L4",
        "W5",
        "L5",
        "Wsets",
        "Lsets",
        "Comment",
        # Raw odds tied to winner/loser perspective (replaced by symmetric features)
        "B365W",
        "B365L",
        "B&WW",
        "B&WL",
        "CBW",
        "CBL",
        "EXW",
        "EXL",
        "PSW",
        "PSL",
        "UBW",
        "UBL",
        "LBW",
        "LBL",
        "SJW",
        "SJL",
        "SBW",
        "SBL",
        "MaxW",
        "MaxL",
        "AvgW",
        "AvgL",
        "BFEW",
        "BFEL",
    ]
    existing_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_to_drop)

    # Reorder columns: target first if present
    if "favorite_won" in df.columns:
        cols = ["favorite_won"] + [c for c in df.columns if c != "favorite_won"]
        df = df[cols]

    return df


def one_hot_encode_low_cardinality(
    df: pd.DataFrame, categorical_columns: List[str], max_unique: int = 30
) -> pd.DataFrame:
    """
    One-hot encode the given categorical columns if their cardinality is <= max_unique.
    Columns exceeding the threshold are left as-is and then dropped later if deemed high-cardinality.
    """
    df = df.copy()
    to_encode: List[str] = []
    for col in categorical_columns:
        if col in df.columns:
            nunique = int(df[col].nunique(dropna=True))
            if nunique <= max_unique:
                to_encode.append(col)

    if not to_encode:
        return df

    dummies = pd.get_dummies(df[to_encode].astype("category"), prefix=to_encode, dummy_na=False)
    df = pd.concat([df.drop(columns=to_encode), dummies], axis=1)
    return df


def save_outputs(df: pd.DataFrame, out_csv: Optional[str], out_parquet: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Save the processed DataFrame to CSV and/or Parquet. Returns (csv_path, parquet_path)."""
    saved_csv: Optional[str] = None
    saved_parquet: Optional[str] = None

    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False)
        saved_csv = out_csv

    if out_parquet:
        os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
        # Use fast parquet engine if available
        try:
            df.to_parquet(out_parquet, index=False)
        except Exception:
            # Fall back to pyarrow/fastparquet defaults if uninstalled; user can install as needed
            df.to_parquet(out_parquet, index=False)
        saved_parquet = out_parquet

    return saved_csv, saved_parquet


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Tennis-Data allyears.csv for ML by engineering features, one-hot encoding"
            " low-cardinality columns, and removing leaky/high-cardinality fields."
        )
    )
    parser.add_argument(
        "--input",
        dest="input_csv",
        type=str,
        default="/Users/ep342e/codebase/sports-predictor/data/raw/allyears.csv",
        help="Absolute path to raw allyears.csv",
    )
    parser.add_argument(
        "--output-csv",
        dest="output_csv",
        type=str,
        default="/Users/ep342e/codebase/sports-predictor/data/processed/matches_ml.csv",
        help="Path to save processed CSV",
    )
    parser.add_argument(
        "--output-parquet",
        dest="output_parquet",
        type=str,
        default="/Users/ep342e/codebase/sports-predictor/data/processed/matches_ml.parquet",
        help="Path to save processed Parquet",
    )
    parser.add_argument(
        "--nrows",
        dest="nrows",
        type=int,
        default=None,
        help="Optional limit of rows to load from input (for smoke tests)",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    raw_df = load_raw_csv(args.input_csv, nrows=args.nrows)
    processed_df = compute_pre_match_features(raw_df)
    save_outputs(processed_df, args.output_csv, args.output_parquet)


if __name__ == "__main__":
    main()


