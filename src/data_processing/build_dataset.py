"""Merge raw vegetable price, weather, and fuel data into per-vegetable processed datasets."""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.io import PROJECT_ROOT, get_processed_path, load_config


def load_prices(vegetable: str, config: dict) -> pd.DataFrame:
    raw_name = config["vegetable_name_map"][vegetable]
    path = PROJECT_ROOT / config["data"]["raw_prices_dir"] / "vegetable_prices.csv"
    df = pd.read_csv(path, parse_dates=["week_start"])
    df = df[df["vegetable"] == raw_name].copy()
    df.rename(columns={"week_start": "date"}, inplace=True)
    return pd.DataFrame(df[["date", "wholesale_price", "retail_price"]])


def load_fuel(config: dict) -> pd.DataFrame:
    path = PROJECT_ROOT / config["data"]["raw_fuel_dir"] / "fuel_data_weekly.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return pd.DataFrame(df[["date", "diesel_price"]])


def load_weather(vegetable: str, config: dict) -> pd.DataFrame:
    district = config["weather_district_map"][vegetable]
    path = PROJECT_ROOT / config["data"]["raw_weather_dir"] / "weather.csv"
    df = pd.read_csv(path, parse_dates=["week_start_date"])
    df = df[df["district"] == district].copy()
    df.rename(columns={"week_start_date": "date"}, inplace=True)
    return pd.DataFrame(df[["date", "average_temperature", "average_rainfall"]])


def build_dataset(vegetable: str, config: dict) -> pd.DataFrame:
    prices = load_prices(vegetable, config)
    fuel = load_fuel(config)
    weather = load_weather(vegetable, config)

    merged = prices.merge(weather, on="date", how="inner").merge(fuel, on="date", how="inner")

    start = pd.Timestamp(config["data"]["aligned_start_date"])
    end = pd.Timestamp(config["data"]["aligned_end_date"])
    merged = merged[(merged["date"] >= start) & (merged["date"] <= end)]

    merged = merged.sort_values(by="date").reset_index(drop=True)
    columns = ["date", "retail_price", "wholesale_price", "average_temperature", "average_rainfall", "diesel_price"]
    return pd.DataFrame(merged[columns])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vegetable", help="Single vegetable to process; omit to process all")
    args = parser.parse_args()

    config = load_config()
    vegetables = [args.vegetable] if args.vegetable else config["vegetables"]

    for vegetable in vegetables:
        df = build_dataset(vegetable, config)
        out_path = get_processed_path(vegetable, config)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"{vegetable}: {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
