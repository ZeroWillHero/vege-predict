"""Shared helpers for the price/fuel scrapers: PDF fetch with retry, and the
digit-kerning fix needed to parse numbers out of CBSL/HARTI PDF tables."""

import re
import time

import requests

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "application/pdf,*/*",
}

# Government PDF-hosting sites here reject direct requests without a Referer from
# their own domain (confirmed empirically for CBSL: identical request succeeds with
# this header and 404s without it, even though the file exists).
REFERER_BY_HOST = {
    "www.cbsl.gov.lk": "https://www.cbsl.gov.lk/en/statistics/economic-indicators/price-report",
    "www.harti.gov.lk": "https://www.harti.gov.lk/weekly-price.php",
}

# PDF text extraction (pdfplumber) renders numbers like "1,100.00" with a spurious
# space after the leading digit/comma group — e.g. "2 00.00", "1 ,100.00" — a font
# kerning artifact, not real whitespace in the source. This matches one full number
# token (including the artifact space) so it can be stripped back to "200.00".
NUMBER_TOKEN_RE = re.compile(r"\d(?:\s?[\d,])*\.\d{2}")


def fetch_pdf(url: str, retries: int = 3, timeout: int = 30) -> bytes:
    """Fetch a PDF, retrying with backoff, using the Referer the host needs (if any)."""
    from urllib.parse import urlparse

    host = urlparse(url).netloc
    headers = dict(DEFAULT_HEADERS)
    if host in REFERER_BY_HOST:
        headers["Referer"] = REFERER_BY_HOST[host]

    wait = 3
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            if not resp.content.startswith(b"%PDF"):
                raise ValueError(f"response from {url} is not a PDF (got {resp.content[:100]!r})")
            return resp.content
        except Exception as e:
            last_error = e
            if attempt < retries:
                time.sleep(wait)
                wait *= 2
    raise RuntimeError(f"failed to fetch PDF from {url} after {retries} attempts: {last_error}")


def extract_numbers(text: str) -> list:
    """Pull every number token out of a line of PDF-extracted text, fixing the
    kerning-artifact spaces along the way. 'n.a.' and similar are simply absent
    from the result (not zeros) — callers should treat a short match list as
    missing data for the trailing columns, not as zero prices.
    """
    return [tok.replace(" ", "").replace(",", "") for tok in NUMBER_TOKEN_RE.findall(text)]
