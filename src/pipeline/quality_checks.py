"""Data quality gate between newly-ingested data and the training pipeline.

Detects missing weeks and outlier values, aligns a newly-ingested frame's schema to
match the existing raw CSV it's being merged into, and appends only the rows that
pass. This exists because the auto-retrain pipeline (src/pipeline/auto_retrain.py)
runs unattended: a scraping glitch or a bad PDF-table extraction must not be able to
silently corrupt data/raw/ and, through it, every downstream model. This is the
concern Jain et al. (2020) raise for manually-entered crop price data (see
research-papers/drafts/thesis/02_literature_review.md, Section 2.9) — the same
failure mode applies to automated ingestion, just with a different root cause.

Usage:
    python src/pipeline/quality_checks.py --audit          # read-only report on data/raw/*
    python src/pipeline/quality_checks.py --audit --source prices
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from src.utils.io import PROJECT_ROOT, load_config

# Week-over-week change beyond this fraction, or a z-score beyond this many standard
# deviations from the group's own history, is flagged as a likely data error rather
# than a real price move. These are deliberately loose: Sri Lankan vegetable prices
# do move fast (see the "genuine finding" in CLAUDE.md about SARIMAX's negative R² on
# the 2025 holdout) — the goal is catching entry/extraction errors, not real volatility.
DEFAULT_PCT_CHANGE_THRESHOLD = 1.5
DEFAULT_Z_THRESHOLD = 5.0


def find_missing_weeks(dates, freq: str = "W-MON") -> list:
    """Given week-start dates, return any expected date in the observed range that's absent."""
    dates = pd.to_datetime(pd.Series(dates)).dropna().sort_values().unique()
    if len(dates) < 2:
        return []
    full_range = pd.date_range(dates.min(), dates.max(), freq=freq)
    missing = sorted(set(full_range) - set(pd.DatetimeIndex(dates)))
    return [d.date().isoformat() for d in missing]


def detect_outliers(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    group_col: str | None = None,
    pct_change_threshold: float = DEFAULT_PCT_CHANGE_THRESHOLD,
    z_threshold: float = DEFAULT_Z_THRESHOLD,
    allow_zero: bool = False,
    use_pct_change: bool = True,
) -> pd.DataFrame:
    """Flags rows whose value looks like a data error: an implausibly large
    week-over-week jump, a value far outside the group's own historical
    distribution, or a non-positive value. Returns the flagged rows plus a
    'flag_reason' column; empty DataFrame if nothing is flagged.

    use_pct_change=False for naturally bursty, often-near-zero series (e.g. weekly
    rainfall): a jump from 0.3mm to 1.6mm is a 433% "change" but not a data error —
    percent-change simply isn't a meaningful anomaly signal for a zero-inflated
    variable, no threshold fixes that, so the check is disabled outright and the
    z-score (relative to the group's own historical distribution) carries the load.
    """
    groups = df.groupby(group_col) if group_col else [(None, df)]
    flagged_rows = []
    for _, group in groups:
        group = group.sort_values(date_col)
        mean, std = group[value_col].mean(), group[value_col].std()
        if use_pct_change:
            prior_value = group[value_col].shift(1)
            pct_change = (group[value_col] - prior_value).abs() / prior_value.replace(0, np.nan)
        else:
            pct_change = pd.Series(dtype=float)
        z = (group[value_col] - mean) / std if std and std > 0 else pd.Series(0.0, index=group.index)
        for idx in group.index:
            reasons = []
            value = group.loc[idx, value_col]
            if value < 0 or (value == 0 and not allow_zero):
                reasons.append("non-positive value")
            if idx in pct_change.index and pd.notna(pct_change[idx]) and pct_change[idx] > pct_change_threshold:
                reasons.append(f"week-over-week change {pct_change[idx]:.0%} > {pct_change_threshold:.0%}")
            if idx in z.index and abs(z[idx]) > z_threshold:
                reasons.append(f"z-score {z[idx]:.1f} beyond +/-{z_threshold}")
            if reasons:
                row = group.loc[idx].to_dict()
                row["flag_reason"] = "; ".join(reasons)
                flagged_rows.append(row)
    return pd.DataFrame(flagged_rows)


def align_price_schema(new_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Normalize a newly-ingested price frame to vegetable_prices.csv's schema:
    columns [vegetable, week_start, wholesale_price, retail_price], vegetable names
    matching the raw CSV's casing (BRINJALS, SNAKE GOURD, ...), week_start snapped
    to the Monday of that week (matching build_dataset.py's own alignment).
    """
    df = new_df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={"date": "week_start", "price": "retail_price"})

    valid_names = set(config["vegetable_name_map"].values())
    df["vegetable"] = df["vegetable"].astype(str).str.strip().str.upper()
    unknown = sorted(set(df["vegetable"]) - valid_names)
    if unknown:
        raise ValueError(f"Unrecognized vegetable name(s) in new data: {unknown}. Expected one of {sorted(valid_names)}")

    df["week_start"] = pd.to_datetime(df["week_start"])
    df["week_start"] = df["week_start"] - pd.to_timedelta(df["week_start"].dt.weekday, unit="D")

    for col in ["wholesale_price", "retail_price"]:
        if col not in df.columns:
            df[col] = np.nan

    return df[["vegetable", "week_start", "wholesale_price", "retail_price"]]


def validate_and_merge(existing_path: Path, new_df: pd.DataFrame, date_col: str, value_col: str, group_col: str | None = None) -> dict:
    """Validates new_df's rows against the existing dataset's own history, appends
    only the rows that pass, and writes the merged result back to existing_path.
    Rows already present (same date [+ group]) are updated, not duplicated.
    """
    existing = pd.read_csv(existing_path, parse_dates=[date_col])

    outliers = detect_outliers(pd.concat([existing, new_df], ignore_index=True), date_col, value_col, group_col)
    key_cols = [date_col] + ([group_col] if group_col else [])
    new_keys = set(map(tuple, new_df[key_cols].values))
    flagged_keys = set(map(tuple, outliers[key_cols].values)) if len(outliers) else set()
    reject_keys = new_keys & flagged_keys

    is_rejected = new_df[key_cols].apply(tuple, axis=1).isin(reject_keys)
    accepted, rejected = new_df[~is_rejected], new_df[is_rejected]

    merged = pd.concat([existing, accepted], ignore_index=True)
    merged = merged.drop_duplicates(subset=key_cols, keep="last").sort_values(key_cols).reset_index(drop=True)
    merged.to_csv(existing_path, index=False)

    return {
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "rejected_detail": rejected.assign(reason="flagged as outlier — see detect_outliers output").to_dict("records"),
        "missing_weeks_after": find_missing_weeks(merged[merged[group_col].isna()][date_col] if group_col and merged[group_col].isna().any() else merged[date_col]),
        "total_rows_after": len(merged),
    }


def audit_source(name: str, path: Path, date_col: str, value_cols: list, group_col: str | None = None, outlier_kwargs: dict | None = None) -> None:
    print(f"\n=== {name} ({path.relative_to(PROJECT_ROOT)}) ===")
    if not path.exists():
        print("  MISSING FILE")
        return
    df = pd.read_csv(path, parse_dates=[date_col])
    print(f"  {len(df)} rows, {df[date_col].min().date()} to {df[date_col].max().date()}")

    if group_col:
        missing_total = 0
        for g, grp in df.groupby(group_col):
            missing = find_missing_weeks(grp[date_col])
            missing_total += len(missing)
            if missing:
                print(f"  [{g}] {len(missing)} missing week(s): {missing[:5]}{' ...' if len(missing) > 5 else ''}")
        if missing_total == 0:
            print("  no missing weeks (per group)")
    else:
        missing = find_missing_weeks(df[date_col])
        print(f"  missing weeks: {missing if missing else 'none'}")

    outlier_kwargs = outlier_kwargs or {}
    for col in value_cols:
        outliers = detect_outliers(df, date_col, col, group_col, **outlier_kwargs.get(col, {}))
        if len(outliers):
            print(f"  {col}: {len(outliers)} flagged row(s):")
            for _, row in outliers.head(10).iterrows():
                label = f"[{row[group_col]}] " if group_col else ""
                print(f"    {label}{row[date_col].date() if hasattr(row[date_col], 'date') else row[date_col]}: {col}={row[col]} — {row['flag_reason']}")
        else:
            print(f"  {col}: no flagged rows")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true", help="Read-only quality report on data/raw/*")
    parser.add_argument("--source", choices=["prices", "fuel", "weather"], help="Limit audit to one source")
    args = parser.parse_args()

    if not args.audit:
        parser.print_help()
        return

    config = load_config()
    sources = {
        "prices": (
            PROJECT_ROOT / config["data"]["raw_prices_dir"] / "vegetable_prices.csv",
            "week_start",
            ["wholesale_price", "retail_price"],
            "vegetable",
            {},
        ),
        "fuel": (
            PROJECT_ROOT / config["data"]["raw_fuel_dir"] / "fuel_data_weekly.csv",
            "date",
            ["diesel_price"],
            None,
            {},
        ),
        "weather": (
            PROJECT_ROOT / config["data"]["raw_weather_dir"] / "weather.csv",
            "week_start_date",
            ["average_temperature", "average_rainfall"],
            "district",
            # rainfall is naturally bursty and often near-zero, so week-over-week
            # percent-change isn't a meaningful anomaly signal at any threshold —
            # disabled outright; z-score vs. the district's own history carries this.
            {"average_rainfall": {"allow_zero": True, "use_pct_change": False}},
        ),
    }

    for name, (path, date_col, value_cols, group_col, outlier_kwargs) in sources.items():
        if args.source and args.source != name:
            continue
        audit_source(name, path, date_col, value_cols, group_col, outlier_kwargs)


if __name__ == "__main__":
    main()
