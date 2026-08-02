"""ETL: load the research pipeline's CSV outputs into Postgres.

Reads data/processed/<vegetable>.csv (historical prices), results/metrics/
holdout_predictions.csv (backtest forecasts) plus results/metrics/future_predictions.csv
(genuine future forecasts, if present — written by scripts/predict_future.py, not
scripts/train_all.py), and results/metrics/all_results.csv (model comparison metrics).

Idempotent: every insert is an upsert on the table's natural-key unique index, so
re-running this after a retrain updates existing rows rather than duplicating them.
Also invalidates each vegetable's Redis cache entries after a successful commit, so a
re-seed is immediately visible through the API rather than serving stale cached JSON
for up to the 7-day TTL. Standalone for now; the natural place to *call* this script is
from auto_retrain.py's promotion step once that exists (see CLAUDE.md's auto-retrain
pipeline notes) — the invalidation itself already happens here, not deferred to that.

Usage:
    python seed.py
"""

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.backend.constants import VEGETABLES  # noqa: E402
from app.backend.database import async_session_factory  # noqa: E402
from app.backend.models import Forecast, HistoricalPrice, ModelMetric  # noqa: E402
from app.backend.services.cache_service import invalidate_all, invalidate_vegetable  # noqa: E402

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


def _forecast_rows_from_csv(path: Path, generated_at: datetime) -> list[dict]:
    df = pd.read_csv(path, parse_dates=["date"])
    return [
        {
            "vegetable": row["vegetable"],
            "model_family": row["model"],
            "forecast_date": row["date"].date(),
            "predicted_price": row["predicted"],
            "predicted_lower": None if pd.isna(row.get("predicted_lower")) else row["predicted_lower"],
            "predicted_upper": None if pd.isna(row.get("predicted_upper")) else row["predicted_upper"],
            # Genuinely future rows (results/metrics/future_predictions.csv) have no `actual`
            # column at all; holdout rows always do. Either way this must become SQL NULL, not
            # NaN — Postgres float8 accepts NaN as a valid (non-NULL) value, which would
            # silently break the "null for unresolved weeks" contract in schemas/forecast.py.
            "actual_price": None if pd.isna(row.get("actual")) else row["actual"],
            "generated_at": generated_at,
        }
        for _, row in df.iterrows()
    ]


async def seed_forecasts(session) -> int:
    generated_at = datetime.now(timezone.utc)
    rows = _forecast_rows_from_csv(PROJECT_ROOT / "results" / "metrics" / "holdout_predictions.csv", generated_at)

    future_path = PROJECT_ROOT / "results" / "metrics" / "future_predictions.csv"
    if future_path.exists():
        rows += _forecast_rows_from_csv(future_path, generated_at)

    stmt = insert(Forecast).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["vegetable", "model_family", "forecast_date"],
        set_={
            c: stmt.excluded[c]
            for c in ["predicted_price", "predicted_lower", "predicted_upper", "actual_price", "generated_at"]
        },
    )
    await session.execute(stmt)
    return len(rows)


async def backfill_actual_prices(session) -> int:
    """Once a genuinely-future forecast's target week has actually passed and real price data
    for it lands in historical_price (via a later seed_historical_prices() call), resolve that
    forecast row's actual_price from it — otherwise a forecast row seeded with actual_price=NULL
    when it was still in the future would stay NULL forever, even after the real price for that
    week becomes known. Only fills rows still NULL; never overwrites an already-resolved actual
    (e.g. one already set from a retrain's holdout_predictions.csv)."""
    stmt = text(
        """
        UPDATE forecast
        SET actual_price = historical_price.retail_price
        FROM historical_price
        WHERE forecast.vegetable = historical_price.vegetable
          AND forecast.forecast_date = historical_price.week_start
          AND forecast.actual_price IS NULL
          AND historical_price.retail_price IS NOT NULL
        """
    )
    result = await session.execute(stmt)
    return result.rowcount


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
            "interval_confidence": None if pd.isna(row.get("interval_confidence")) else row["interval_confidence"],
            "interval_coverage": None if pd.isna(row.get("interval_coverage")) else row["interval_coverage"],
            "evaluated_at": evaluated_at,
        }
        for _, row in df.iterrows()
    ]
    stmt = insert(ModelMetric).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["vegetable", "model_family"],
        set_={
            c: stmt.excluded[c]
            for c in ["mae", "rmse", "mape", "r2", "interval_confidence", "interval_coverage", "evaluated_at"]
        },
    )
    await session.execute(stmt)
    return len(rows)


async def main():
    async with async_session_factory() as session:
        n_prices = await seed_historical_prices(session)
        n_forecasts = await seed_forecasts(session)
        n_backfilled = await backfill_actual_prices(session)
        n_metrics = await seed_model_metrics(session)
        await session.commit()

        for vegetable in VEGETABLES:
            await invalidate_vegetable(vegetable)
        await invalidate_all()
        print(f"Invalidated Redis cache for {len(VEGETABLES)} vegetables (plus unfiltered list caches)")

        print(f"historical_price: {n_prices} rows upserted")
        print(f"forecast: {n_forecasts} rows upserted")
        print(f"forecast: {n_backfilled} rows backfilled with actual_price from historical_price")
        print(f"model_metric: {n_metrics} rows upserted")


if __name__ == "__main__":
    asyncio.run(main())
