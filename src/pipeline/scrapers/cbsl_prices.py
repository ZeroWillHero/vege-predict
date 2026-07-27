"""Fetch and parse the Central Bank of Sri Lanka's Daily Price Report PDF for
vegetable wholesale/retail prices.

Covers Carrot, Cabbage, Brinjal, Pumpkin, Snake Gourd. CBSL's daily report does NOT
track Leeks (confirmed by inspecting the report's table on 2026-07-23) — this source
alone cannot fully replace vegetable_prices.csv; Leeks needs a different source
(HARTI's weekly bulletin, or another line item not yet checked).

Usage:
    python src/pipeline/scrapers/cbsl_prices.py --date 2026-07-23
"""

import argparse
import sys
from datetime import date, datetime
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pandas as pd
import pdfplumber

from src.pipeline.scrapers.common import extract_numbers, fetch_pdf

CBSL_URL_TEMPLATE = "https://www.cbsl.gov.lk/sites/default/files/cbslweb_documents/statistics/pricerpt/price_report_{date}_e.pdf"

# CBSL's row label -> this project's vegetable_prices.csv name (vegetable_name_map values)
CBSL_NAME_MAP = {
    "carrot": "CARROT",
    "cabbage": "CABBAGE",
    "brinjal": "BRINJALS",
    "pumpkin": "PUMPKIN",
    "snake gourd": "SNAKE GOURD",
}


def fetch_report(report_date: date) -> bytes:
    url = CBSL_URL_TEMPLATE.format(date=report_date.strftime("%Y%m%d"))
    return fetch_pdf(url)


def parse_report(pdf_bytes: bytes, report_date: date) -> pd.DataFrame:
    """Extract vegetable prices from the CBSL report's table page (page 2 of 2 in
    the format confirmed 2026-07-23; falls back to the last page if only one exists).

    Column order per row is [wholesale-Pettah-Yesterday, wholesale-Pettah-Today,
    wholesale-Dambulla-Yesterday, wholesale-Dambulla-Today, retail-Pettah-Yesterday,
    retail-Pettah-Today, retail-Dambulla-Yesterday, retail-Dambulla-Today,
    retail-Narahenpita-Yesterday, retail-Narahenpita-Today]. wholesale_price/
    retail_price are the mean of that day's ("Today") values across markets.
    """
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        table_page = pdf.pages[1] if len(pdf.pages) > 1 else pdf.pages[-1]
        text = table_page.extract_text() or ""

    rows = []
    for line in text.split("\n"):
        stripped = line.strip().lower()
        for cbsl_name, our_name in CBSL_NAME_MAP.items():
            if not stripped.startswith(cbsl_name + " "):
                continue
            numbers = extract_numbers(line)
            if len(numbers) < 8:
                break  # missing data for this item today; skip rather than guess
            wholesale_today = [float(numbers[1]), float(numbers[3])]
            retail_today = [float(numbers[5]), float(numbers[7])]
            if len(numbers) >= 10:
                retail_today.append(float(numbers[9]))
            rows.append(
                {
                    "vegetable": our_name,
                    "date": report_date.isoformat(),
                    "wholesale_price": round(sum(wholesale_today) / len(wholesale_today), 2),
                    "retail_price": round(sum(retail_today) / len(retail_today), 2),
                }
            )
            break

    missing = set(CBSL_NAME_MAP.values()) - set(r["vegetable"] for r in rows)
    if missing:
        print(f"  warning: no parseable row found for {sorted(missing)} on {report_date}")

    return pd.DataFrame(rows)


def fetch_and_parse(report_date: date) -> pd.DataFrame:
    pdf_bytes = fetch_report(report_date)
    return parse_report(pdf_bytes, report_date)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="Report date, YYYY-MM-DD")
    args = parser.parse_args()
    report_date = datetime.strptime(args.date, "%Y-%m-%d").date()

    df = fetch_and_parse(report_date)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
