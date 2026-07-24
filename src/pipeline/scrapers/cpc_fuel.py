"""Fetch and parse Ceylon Petroleum Corporation's (CEYPETCO) historical fuel price
table for diesel prices.

Source: ceypetco.gov.lk/historical-prices/ — a plain HTML table (no PDF involved),
listing every price *revision* since 1990 (irregular dates, one row per change, not
one row per week). Confirmed correct against a real news report: the 2026-05-31
revision's "LAD" (Lanka Auto Diesel) column reads 407, matching the reported
Rs.407/litre diesel price for that date exactly.

"LAD" = Lanka Auto Diesel, the column this project uses as diesel_price (the
transport-cost driver — see CLAUDE.md's resolved decisions). Since CPC only revises
prices periodically (not weekly), converting to this project's weekly cadence is a
forward-fill: for each Monday, use whatever price was in effect that day (the most
recent revision on or before it).

Usage:
    python src/pipeline/scrapers/cpc_fuel.py update            # strategic: only new weeks
    python src/pipeline/scrapers/cpc_fuel.py update --dry-run
"""

import argparse
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pandas as pd
import requests

CEYPETCO_URL = "https://ceypetco.gov.lk/historical-prices/"


def fetch_revision_table() -> pd.DataFrame:
    """Fetch and parse the price-revision table into a DataFrame with columns
    [date, diesel_price], one row per revision (not per week), sorted ascending.
    """
    resp = requests.get(CEYPETCO_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    resp.raise_for_status()
    html = resp.text

    tables = re.findall(r"<table.*?</table>", html, re.DOTALL)
    if not tables:
        raise ValueError(f"No <table> found on {CEYPETCO_URL} — page structure may have changed")

    rows = re.findall(r"<tr.*?</tr>", tables[0], re.DOTALL)
    header = [re.sub("<[^>]+>", "", c).strip() for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", rows[0], re.DOTALL)]
    if "Date" not in header or "LAD" not in header:
        raise ValueError(f"Expected 'Date' and 'LAD' columns, got {header!r} — page structure may have changed")
    date_idx, lad_idx = header.index("Date"), header.index("LAD")

    records = []
    for row in rows[1:]:
        cells = [re.sub("<[^>]+>", "", c).strip() for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.DOTALL)]
        if len(cells) <= max(date_idx, lad_idx) or not cells[date_idx]:
            continue
        try:
            revision_date = datetime.strptime(cells[date_idx], "%d.%m.%Y").date()
            price = float(cells[lad_idx])
        except ValueError:
            continue  # skip malformed/blank rows rather than fail the whole fetch
        records.append({"date": revision_date, "diesel_price": price})

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    if df.empty:
        raise ValueError("Parsed 0 usable rows from the CEYPETCO price table")
    return df


def to_weekly_series(revisions: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Forward-fill the irregular revision series onto this project's weekly (Monday)
    cadence: each Monday gets whatever price was in effect that day.
    """
    mondays = pd.date_range(start, end, freq="W-MON")
    revisions = revisions.sort_values("date")
    rows = []
    for monday in mondays:
        applicable = revisions[revisions["date"] <= monday.date()]
        if applicable.empty:
            continue  # before CPC's earliest recorded revision; nothing to forward-fill from
        rows.append({"date": monday.date().isoformat(), "diesel_price": applicable.iloc[-1]["diesel_price"]})
    return pd.DataFrame(rows)


def update_fuel_prices(config: dict, dry_run: bool = False) -> dict:
    """Strategic incremental update: only generates/merges weeks after the current
    max date in fuel_data_weekly.csv, not the full 1990-present history — mirrors
    the same philosophy as harti_prices.update_harti_prices (see its docstring).
    """
    from src.pipeline.quality_checks import validate_and_merge
    from src.utils.io import PROJECT_ROOT

    raw_path = PROJECT_ROOT / config["data"]["raw_fuel_dir"] / "fuel_data_weekly.csv"
    existing = pd.read_csv(raw_path, parse_dates=["date"])
    current_max = existing["date"].max().date()
    today = date.today()

    if current_max >= today - timedelta(days=7):
        return {"checked_weeks": 0, "accepted_rows": 0, "rejected_rows": 0, "message": "already up to date"}

    revisions = fetch_revision_table()
    new_start = current_max + timedelta(days=7)
    weekly = to_weekly_series(revisions, new_start, today)

    if dry_run:
        print(f"[dry run] would add {len(weekly)} week(s): {weekly['date'].tolist()}")
        return {"checked_weeks": len(weekly), "accepted_rows": 0, "rejected_rows": 0, "dry_run": True}

    if weekly.empty:
        return {"checked_weeks": 0, "accepted_rows": 0, "rejected_rows": 0, "message": "no new weeks to add"}

    weekly["date"] = pd.to_datetime(weekly["date"])
    report = validate_and_merge(raw_path, weekly, date_col="date", value_col="diesel_price", group_col=None)
    return {"checked_weeks": len(weekly), "accepted_rows": report["accepted_rows"], "rejected_rows": report["rejected_rows"]}


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    update = subparsers.add_parser("update", help="Fetch+merge only weeks newer than the current data")
    update.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.command == "update":
        from src.utils.io import load_config

        report = update_fuel_prices(load_config(), dry_run=args.dry_run)
        print(report)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
