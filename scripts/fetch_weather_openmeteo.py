"""Fetch daily rainfall/temperature per Sri Lankan district from Open-Meteo (2014-present),
aggregate to weekly averages, and write data/raw/weather/weather.csv.

Usage:
    python scripts/fetch_weather_openmeteo.py
"""

import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
# Open-Meteo's separate forecast endpoint (distinct from the archive endpoint above) — free,
# no API key, real forecast skill for ~16 days out. Used by src/inference/future_forecast.py
# for the near term of a genuine future price forecast; beyond its horizon, that module falls
# back to climatology_estimate() below.
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
START_DATE = "2014-01-01"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "weather" / "weather.csv"

# Representative (district capital / main town) coordinates for all 25 administrative
# districts of Sri Lanka.
DISTRICTS = {
    "Colombo": (6.9271, 79.8612),
    "Gampaha": (7.0917, 80.0000),
    "Kalutara": (6.5854, 79.9607),
    "Kandy": (7.2906, 80.6337),
    "Matale": (7.4675, 80.6234),
    "Nuwara Eliya": (6.9497, 80.7891),
    "Galle": (6.0535, 80.2210),
    "Matara": (5.9549, 80.5550),
    "Hambantota": (6.1241, 81.1185),
    "Jaffna": (9.6615, 80.0255),
    "Kilinochchi": (9.3961, 80.4022),
    "Mannar": (8.9810, 79.9044),
    "Vavuniya": (8.7514, 80.4971),
    "Mullaitivu": (9.2670, 80.8142),
    "Batticaloa": (7.7170, 81.7000),
    "Ampara": (7.2975, 81.6747),
    "Trincomalee": (8.5874, 81.2152),
    "Kurunegala": (7.4863, 80.3623),
    "Puttalam": (8.0362, 79.8283),
    "Anuradhapura": (8.3114, 80.4037),
    "Polonnaruwa": (7.9403, 81.0188),
    "Badulla": (6.9934, 81.0550),
    "Monaragala": (6.8726, 81.3510),
    "Ratnapura": (6.6828, 80.4012),
    "Kegalle": (7.2513, 80.3464),
}


def fetch_district_daily(district: str, lat: float, lon: float, end_date: str, retries: int = 5) -> pd.DataFrame:
    """Fetch daily max/min temperature and total precipitation for one district."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "Asia/Colombo",
    }
    wait = 5
    resp = requests.get(ARCHIVE_URL, params=params, timeout=60)
    for attempt in range(1, retries + 1):
        if resp.status_code == 429 and attempt < retries:
            retry_after = int(resp.headers.get("Retry-After", wait))
            print(f"  rate limited, waiting {retry_after}s (attempt {attempt}/{retries})")
            time.sleep(retry_after)
            wait *= 2
            resp = requests.get(ARCHIVE_URL, params=params, timeout=60)
            continue
        resp.raise_for_status()
        break
    daily = resp.json()["daily"]
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(daily["time"]),
            "temperature_max": daily["temperature_2m_max"],
            "temperature_min": daily["temperature_2m_min"],
            # Open-Meteo's archive API only exposes a daily precipitation *total*
            # (no daily min/max), so the weekly average below is the mean of these
            # daily totals across the week, not derived from a min/max pair.
            "rainfall": daily["precipitation_sum"],
        }
    )
    df["district"] = district
    return df


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily records to Monday-start ISO weeks, per district."""
    df = df.copy()
    df["average_temperature"] = (df["temperature_max"] + df["temperature_min"]) / 2
    df["week_start_date"] = df["date"] - pd.to_timedelta(df["date"].dt.weekday, unit="D")
    weekly = (
        df.groupby(["district", "week_start_date"])
        .agg(average_temperature=("average_temperature", "mean"), average_rainfall=("rainfall", "mean"))
        .reset_index()
    )
    weekly["week_start_date"] = weekly["week_start_date"].dt.date
    return pd.DataFrame(weekly[["week_start_date", "district", "average_temperature", "average_rainfall"]])


def fetch_district_forecast(district: str, lat: float, lon: float, days: int = 16) -> pd.DataFrame:
    """Real Open-Meteo forecast (not historical) for the next `days` days, same shape as
    fetch_district_daily()'s output so it can be aggregated with the same to_weekly()."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "forecast_days": days,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "Asia/Colombo",
    }
    resp = requests.get(FORECAST_URL, params=params, timeout=60)
    resp.raise_for_status()
    daily = resp.json()["daily"]
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(daily["time"]),
            "temperature_max": daily["temperature_2m_max"],
            "temperature_min": daily["temperature_2m_min"],
            "rainfall": daily["precipitation_sum"],
        }
    )
    df["district"] = district
    return df


def climatology_estimate(district: str, iso_week: int, weather_csv_path: Path | None = None) -> tuple[float, float]:
    """Historical (average_temperature, average_rainfall) for a given ISO week-of-year in a
    district, averaged across every prior year in data/raw/weather/weather.csv. Fallback for
    future weeks beyond fetch_district_forecast()'s ~16-day real-forecast horizon."""
    path = weather_csv_path or OUTPUT_PATH
    df = pd.read_csv(path, parse_dates=["week_start_date"])
    df = df[df["district"] == district].copy()
    df["iso_week"] = df["week_start_date"].dt.isocalendar().week.astype(int)
    match = df[df["iso_week"] == iso_week]
    if match.empty:
        # No exact ISO-week match (rare, e.g. week 53) — fall back to the district's
        # all-time average rather than failing outright.
        match = df
    return float(match["average_temperature"].mean()), float(match["average_rainfall"].mean())


def main():
    end_date = (date.today() - timedelta(days=1)).isoformat()
    weekly_frames = []
    for i, (district, (lat, lon)) in enumerate(DISTRICTS.items(), start=1):
        print(f"[{i}/{len(DISTRICTS)}] Fetching {district} ({lat}, {lon}) -> {START_DATE}..{end_date}")
        daily = fetch_district_daily(district, lat, lon, end_date)
        weekly_frames.append(to_weekly(daily))
        time.sleep(2)  # be polite to the free API tier

    result = pd.concat(weekly_frames, ignore_index=True)
    result = result.round({"average_temperature": 2, "average_rainfall": 2})
    result = result.sort_values(["district", "week_start_date"]).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(result)} rows ({result['district'].nunique()} districts) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
