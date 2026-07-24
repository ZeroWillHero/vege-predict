"""Fetch and parse HARTI's Weekly Food Commodities Bulletin PDF for vegetable
wholesale/retail prices.

Chosen over CBSL's daily report (src/pipeline/scrapers/cbsl_prices.py) because it
covers all 6 target vegetables including Leeks, which CBSL's report does not track
at all, and its cadence (weekly) matches this project's data granularity directly
instead of needing daily-to-weekly aggregation.

Bulletin structure (confirmed against Week 25, 2026 - 19-25 June 2026 issue; page
numbers are located dynamically per-PDF since they can shift issue to issue):
  - "Table 06: Selected Markets: Wholesale Price of Vegetables" (2 pages) — covers
    Carrot, Leeks, Cabbage, Brinjal. Does NOT include Pumpkin or Snake Gourd — those
    two vegetables' wholesale_price will be left as NaN from this source (retail_price,
    the actual forecast target, is unaffected).
  - "Table 07: Selected Markets: Retail Price of Vegetables" (Up Country + Low
    Country pages) — covers all 6 target vegetables.
  Each cell in these tables is a per-market price range ("MIN.MM− MAX.MM") or a
  lone "−" for no data that week/market. This module averages the range midpoints
  across all markets that reported data, to get one national weekly figure per
  vegetable, matching vegetable_prices.csv's schema.

Usage:
    python src/pipeline/scrapers/harti_prices.py fetch --week 25 --year 2026   # one specific bulletin
    python src/pipeline/scrapers/harti_prices.py update                        # strategic: only new bulletins
    python src/pipeline/scrapers/harti_prices.py update --dry-run              # preview without fetching
"""

import argparse
import json
import re
import sys
import time
from datetime import date, datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pandas as pd
import pdfplumber

from src.pipeline.scrapers.common import fetch_pdf

HARTI_BASE = "https://www.harti.gov.lk/"
# The bulletin index page (harti.gov.lk/weekly-price.php) lists the exact filename
# per week (spacing/capitalization is inconsistent issue to issue, e.g. "Final
# English Bulletin - week 25.pdf" vs "Final  English Bulletin - Week 24.pdf"), so a
# fixed URL template isn't reliable — find_bulletin_url() below scrapes the actual
# link for a given week/year rather than guessing the filename.

# HARTI column label -> this project's vegetable_prices.csv name
HARTI_NAME_MAP = {
    "Carrot": "CARROT",
    "Leeks": "LEEKS",
    "Cabbage": "CABBAGE",
    "Brinjal": "BRINJALS",
    "Pumpkin": "PUMPKIN",
    "Snake Gourd": "SNAKE GOURD",
}

CELL_RE = re.compile(r"(\d[\d,]*\.\d{2})−\s*(\d[\d,]*\.\d{2})|−")  # − = the en/minus dash HARTI uses, not a hyphen


def _fetch_index_html() -> str:
    import requests

    resp = requests.get(f"{HARTI_BASE}weekly-price.php", headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    resp.raise_for_status()
    return resp.text


# Links look like: assets/pdf/food_price/weekly/eng/<year>/Final English
# Bulletin - week <N>.pdf (spacing/case varies), inside an <a href="...">. Only
# matches this "week N" naming convention (confirmed used for 2026 bulletins) — see
# module docstring / CLAUDE.md for older years' different naming conventions, not
# yet supported here.
_LINK_RE_TEMPLATE = r'href="(assets/pdf/food_price/weekly/eng/{year}/[^"]*[Ww]eek\s*{week}\.pdf)"'


def list_available_weeks(year: int, html: str | None = None) -> list:
    """Return every bulletin week number available for `year` in the "week N"
    naming convention this module supports, sorted ascending. Fetches the index
    page once unless `html` is supplied (callers doing multiple lookups should
    fetch once and pass it in, rather than each calling this separately)."""
    html = html if html is not None else _fetch_index_html()
    pattern = re.compile(rf'href="assets/pdf/food_price/weekly/eng/{year}/[^"]*[Ww]eek\s*(\d+)\.pdf"', re.IGNORECASE)
    return sorted({int(w) for w in pattern.findall(html)})


def find_bulletin_url(week: int, year: int, html: str | None = None) -> str:
    """Scrape harti.gov.lk/weekly-price.php for the actual English bulletin link
    matching the given week/year, rather than guessing a filename pattern."""
    html = html if html is not None else _fetch_index_html()
    pattern = re.compile(_LINK_RE_TEMPLATE.format(year=year, week=week), re.IGNORECASE)
    match = pattern.search(html)
    if not match:
        raise ValueError(f"No English bulletin link found for week {week}, {year} on {HARTI_BASE}weekly-price.php")
    return HARTI_BASE + quote(match.group(1))


def _parse_row(line: str) -> tuple:
    """Parse one 'LOCATION cell cell cell ...' row into (location, [midpoint_or_None, ...])."""
    matches = list(CELL_RE.finditer(line))
    values = []
    for m in matches:
        if m.group(1) is not None:
            lo, hi = float(m.group(1).replace(",", "")), float(m.group(2).replace(",", ""))
            values.append((lo + hi) / 2)
        else:
            values.append(None)
    location = line[: matches[0].start()].strip() if matches else line.strip()
    return location, values


# Column layouts verified against the Week 25, 2026 bulletin (see module docstring).
# Crop names are given as complete strings (not derived by splitting the header on
# whitespace), since several are themselves multi-word ("Ladies Fingers", "Snake
# Gourd", "Knol Khol", ...) — naively splitting on spaces silently misaligns every
# column after the first multi-word name, which is exactly what broke the first
# version of this parser (confirmed: it produced a plausible-looking but wrong
# cabbage wholesale price by shifting the whole row over by one column).
TABLE06_BLOCKS = [
    ("Locations Butter Beans Green Beans Carrot Leeks Beetroot Knolkhol",
     ["Butter Beans", "Green Beans", "Carrot", "Leeks", "Beetroot", "Knolkhol"]),
    ("Locations Raddish Cabbage Tomato Ladies Fingers Brinjal Capsicum",
     ["Raddish", "Cabbage", "Tomato", "Ladies Fingers", "Brinjal", "Capsicum"]),
]
TABLE07_BLOCKS = [
    ("Butter Beans Green Beans Carrot Leeks Beetroot Knol Khol Raddish Cabbage Tomato",
     ["Butter Beans", "Green Beans", "Carrot", "Leeks", "Beetroot", "Knol Khol", "Raddish", "Cabbage", "Tomato"]),
    ("Ladies Fingers Brinjal Capsicum Pumpkin Cucumber Bitter Gourd Snake Gourd Drumstick Luffa Long Beans",
     ["Ladies Fingers", "Brinjal", "Capsicum", "Pumpkin", "Cucumber", "Bitter Gourd", "Snake Gourd", "Drumstick", "Luffa", "Long Beans"]),
]


def _parse_table_block(page_text: str, header_search: str, crop_columns: list) -> dict:
    """Parse one table block: header_search locates where the data rows begin;
    crop_columns (given explicitly, not derived from header_search — see module
    note above) gives the per-cell crop order. Returns {crop_name: [values]}.
    """
    lines = page_text.split("\n")
    header_idx = next((i for i, line in enumerate(lines) if line.strip() == header_search.strip()), None)
    if header_idx is None:
        return {}

    per_crop_values = {name: [] for name in crop_columns}
    for line in lines[header_idx + 1 :]:
        if not line.strip() or line.strip().startswith(("Table", "Locations", "LOCATIONS")):
            break
        location, values = _parse_row(line)
        if not location or location[0].isdigit():
            continue
        for crop, value in zip(crop_columns, values):
            if value is not None:
                per_crop_values[crop].append(value)

    return per_crop_values


# Matches one complete "<day><ordinal suffix> <Month> <Year>" date, tolerant of the
# suffix having a stray space before it ("23 th") and the month being abbreviated
# ("Mar") or spelled out ("March"). Deliberately does NOT try to match the *range*
# separator between start and end day: HARTI's own titles are inconsistent about it
# ("-" vs "–" en-dash, with or without surrounding spaces) and sometimes span two
# different months ("27th Mar- 02nd Apr 2026") — since only the end date is needed
# (see extract_week_end's docstring), taking the *last* full date match in the title
# sidesteps all of that formatting inconsistency instead of chasing each variant.
DATE_RE = re.compile(r"(\d{1,2})\s*(?:st|nd|rd|th)?\s+([A-Za-z]+)\s+(\d{4})")


def _month_number(name: str) -> int:
    import calendar

    name = name.strip().lower()
    full = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
    abbr = {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}
    if name in full:
        return full[name]
    if name in abbr:
        return abbr[name]
    raise ValueError(f"Unrecognized month name: {name!r}")


def extract_week_end(pdf_bytes: bytes) -> str:
    """Parse the bulletin's own stated date range (e.g. "19th - 25th June 2026" from
    its title page) and return that week's Monday, derived from the range's end date.

    HARTI's bulletin week runs Friday-Thursday (confirmed: Week 25, 2026 = Fri
    19 - Thu 25 June), not Monday-Sunday like this project's data — so HARTI's own
    week number can't be used to infer a Monday week_start via ISO calendar
    arithmetic (that was tried and produced the wrong date, off by 4 days). Using
    the range's end date instead works because Monday-Thursday of any HARTI week
    fall in the same Monday-Sunday week as our schema expects, and is the majority
    (4 of 7 days) of the bulletin's coverage.
    """
    from datetime import date, timedelta

    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        first_line = (pdf.pages[0].extract_text() or "").split("\n")[0]

    matches = list(DATE_RE.finditer(first_line))
    if not matches:
        raise ValueError(f"Could not parse a date from bulletin title: {first_line!r}")
    end_day, month_name, year = matches[-1].groups()
    end_date = date(int(year), _month_number(month_name), int(end_day))
    monday = end_date - timedelta(days=end_date.weekday())
    return monday.isoformat()


def parse_bulletin(pdf_bytes: bytes, week_start: str | None = None) -> pd.DataFrame:
    """Extract national-average wholesale/retail prices for the 6 target vegetables
    from a HARTI weekly bulletin PDF. week_start: ISO date string for the Monday of
    the bulletin's week; if omitted, parsed from the bulletin's own title text via
    extract_week_end() rather than guessed from a week number (see that function's
    docstring for why the guess was wrong).
    """
    if week_start is None:
        week_start = extract_week_end(pdf_bytes)
    wholesale, retail = {}, {}

    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            first_line = text.split("\n")[0] if text else ""

            if first_line.startswith("Table 06"):
                for header_search, crop_columns in TABLE06_BLOCKS:
                    wholesale.update(_parse_table_block(text, header_search, crop_columns))
            elif first_line.startswith("Table 07"):
                for header_search, crop_columns in TABLE07_BLOCKS:
                    retail.update(_parse_table_block(text, header_search, crop_columns))

    rows = []
    for harti_name, our_name in HARTI_NAME_MAP.items():
        w_values = wholesale.get(harti_name, [])
        r_values = retail.get(harti_name, [])
        rows.append(
            {
                "vegetable": our_name,
                "date": week_start,
                "wholesale_price": round(sum(w_values) / len(w_values), 2) if w_values else None,
                "retail_price": round(sum(r_values) / len(r_values), 2) if r_values else None,
            }
        )
    return pd.DataFrame(rows)


def fetch_and_parse(week: int, year: int, week_start: str | None = None) -> pd.DataFrame:
    url = find_bulletin_url(week, year)
    pdf_bytes = fetch_pdf(url)
    return parse_bulletin(pdf_bytes, week_start)


def _state_path(config: dict) -> Path:
    from src.utils.io import PROJECT_ROOT

    return PROJECT_ROOT / config["data"]["raw_prices_dir"] / ".harti_ingestion_state.json"


def _load_state(state_path: Path) -> dict:
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {"ingested": []}  # list of [year, week] pairs already fetched+merged


def update_harti_prices(config: dict, years: list | None = None, dry_run: bool = False) -> dict:
    """Strategic incremental update: only fetches bulletins not already recorded
    as ingested (state file next to vegetable_prices.csv), rather than re-checking
    or re-downloading everything on every run. Safe to run repeatedly/on a schedule:
    - Already-ingested (year, week) pairs are skipped without a network request.
    - Even if the state file is lost/reset, re-merging an already-present week is
      harmless — validate_and_merge dedupes by (date, vegetable), keeping the latest.
    - Every attempted week (year, week) is recorded once tried, whether or not it
      produced usable rows, so a genuinely-missing bulletin isn't retried forever.

    Returns a report dict; does not raise on a single bulletin's failure (a bad PDF
    for one week shouldn't block ingesting the rest) — failures are collected in
    report['errors'] instead.
    """
    from src.pipeline.quality_checks import align_price_schema, validate_and_merge
    from src.utils.io import PROJECT_ROOT

    raw_path = PROJECT_ROOT / config["data"]["raw_prices_dir"] / "vegetable_prices.csv"
    state_path = _state_path(config)
    state = _load_state(state_path)
    ingested = {tuple(pair) for pair in state["ingested"]}

    years = years or [date.today().year]
    html = _fetch_index_html()  # one fetch covers every year's links present on the single index page

    report = {"checked": 0, "fetched": 0, "accepted_rows": 0, "rejected_rows": 0, "skipped_empty": 0, "errors": []}
    for year in years:
        for week in list_available_weeks(year, html=html):
            if (year, week) in ingested:
                continue
            report["checked"] += 1
            if dry_run:
                print(f"[dry run] would fetch week {week}, {year}")
                continue
            try:
                url = find_bulletin_url(week, year, html=html)
                pdf_bytes = fetch_pdf(url)
                scraped = parse_bulletin(pdf_bytes).dropna(subset=["retail_price"])
                if scraped.empty:
                    print(f"week {week}, {year}: no usable rows")
                    report["skipped_empty"] += 1
                else:
                    aligned = align_price_schema(scraped, config)
                    merge_report = validate_and_merge(
                        raw_path, aligned, date_col="week_start", value_col="retail_price", group_col="vegetable"
                    )
                    print(
                        f"week {week}, {year}: week_start={aligned['week_start'].iloc[0].date()} "
                        f"accepted={merge_report['accepted_rows']} rejected={merge_report['rejected_rows']}"
                    )
                    report["accepted_rows"] += merge_report["accepted_rows"]
                    report["rejected_rows"] += merge_report["rejected_rows"]
                report["fetched"] += 1
                ingested.add((year, week))
            except Exception as e:
                print(f"week {week}, {year}: FAILED ({e})")
                report["errors"].append({"year": year, "week": week, "error": str(e)})
            time.sleep(2)  # be polite to HARTI's server

    if not dry_run:
        state["ingested"] = sorted(list(pair) for pair in ingested)
        state["last_run"] = datetime.now(timezone.utc).isoformat()
        state_path.write_text(json.dumps(state, indent=2))

    return report


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    single = subparsers.add_parser("fetch", help="Fetch one specific bulletin")
    single.add_argument("--week", type=int, required=True, help="HARTI's own bulletin week number (not an ISO week)")
    single.add_argument("--year", type=int, required=True)
    single.add_argument("--week-start", help="ISO date for this week's Monday; parsed from the bulletin's title if omitted")

    update = subparsers.add_parser("update", help="Fetch+merge only bulletins not already ingested (the strategic/incremental path)")
    update.add_argument("--year", type=int, action="append", help="Year(s) to check; defaults to the current year. Repeatable.")
    update.add_argument("--dry-run", action="store_true", help="List what would be fetched without downloading or merging")

    args = parser.parse_args()

    if args.command == "update":
        from src.utils.io import load_config

        report = update_harti_prices(load_config(), years=args.year, dry_run=args.dry_run)
        print()
        print(f"checked={report['checked']} fetched={report['fetched']} "
              f"accepted_rows={report['accepted_rows']} rejected_rows={report['rejected_rows']} "
              f"skipped_empty={report['skipped_empty']} errors={len(report['errors'])}")
        if report["errors"]:
            for err in report["errors"]:
                print(f"  {err['year']} week {err['week']}: {err['error']}")
    else:
        df = fetch_and_parse(args.week, args.year, args.week_start)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
