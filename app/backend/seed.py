"""ETL: load the research pipeline's CSV outputs into Postgres.

Reads data/processed/<vegetable>.csv (historical prices), results/metrics/
holdout_predictions.csv (forecasts), and results/metrics/all_results.csv (model
comparison metrics) — the same three artifacts scripts/train_all.py produces.

Idempotent: every insert is an upsert on the table's natural-key unique index, so
re-running this after a retrain updates existing rows rather than duplicating them.
Standalone for now; the natural place to call this is from auto_retrain.py's
promotion step once that exists (see CLAUDE.md's auto-retrain pipeline notes).

Usage:
    python seed.py
"""

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy.dialects.postgresql import insert

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.backend.constants import VEGETABLES  # noqa: E402
from app.backend.database import async_session_factory  # noqa: E402
from app.backend.models import Forecast, HistoricalPrice, ModelMetric  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


async def seed_historical_prices(session) -> int:
    total = 0
    for vegetable in VEGETABLES:
        path = PROJECT_ROOT / "data" / "processed" / f"{vegetable}.csv"
        df = pd.read_csv(path, parse_dates=["date"])
        rows = [
            {
                "vegetable": vegetable,
                "week_start": row["date"].date(),
                "wholesale_price": None if pd.isna(row["wholesale_price"]) else row["wholesale_price"],
                "retail_price": row["retail_price"],
                "average_temperature": row["average_temperature"],
                "average_rainfall": row["average_rainfall"],
                "diesel_price": row["diesel_price"],
            }
            for _, row in df.iterrows()
        ]
        stmt = insert(HistoricalPrice).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["vegetable", "week_start"],
            set_={c: stmt.excluded[c] for c in ["wholesale_price", "retail_price", "average_temperature", "average_rainfall", "diesel_price"]},
        )
        await session.execute(stmt)
        total += len(rows)
    return total


async def seed_forecasts(session) -> int:
    path = PROJECT_ROOT / "results" / "metrics" / "holdout_predictions.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    generated_at = datetime.now(timezone.utc)
    rows = [
        {
            "vegetable": row["vegetable"],
            "model_family": row["model"],
            "forecast_date": row["date"].date(),
            "predicted_price": row["predicted"],
            "actual_price": row["actual"],
            "generated_at": generated_at,
        }
        for _, row in df.iterrows()
    ]
    stmt = insert(Forecast).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["vegetable", "model_family", "forecast_date"],
        set_={c: stmt.excluded[c] for c in ["predicted_price", "actual_price", "generated_at"]},
    )
    await session.execute(stmt)
    return len(rows)


async def seed_model_metrics(session) -> int:
    path = PROJECT_ROOT / "results" / "metrics" / "all_results.csv"
    df = pd.read_csv(path)
    evaluated_at = datetime.now(timezone.utc)
    rows = [
        {
            "vegetable": row["vegetable"],
            "model_family": row["model"],
            "mae": row["mae"],
            "rmse": row["rmse"],
            "mape": row["mape"],
            "r2": row["r2"],
            "evaluated_at": evaluated_at,
        }
        for _, row in df.iterrows()
    ]
    stmt = insert(ModelMetric).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["vegetable", "model_family"],
        set_={c: stmt.excluded[c] for c in ["mae", "rmse", "mape", "r2", "evaluated_at"]},
    )
    await session.execute(stmt)
    return len(rows)


async def main():
    async with async_session_factory() as session:
        n_prices = await seed_historical_prices(session)
        n_forecasts = await seed_forecasts(session)
        n_metrics = await seed_model_metrics(session)
        await session.commit()
        print(f"historical_price: {n_prices} rows upserted")
        print(f"forecast: {n_forecasts} rows upserted")
        print(f"model_metric: {n_metrics} rows upserted")


if __name__ == "__main__":
    asyncio.run(main())
