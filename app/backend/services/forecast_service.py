"""Forecast/model-metric queries. "Best model per vegetable" is always derived by
querying model_metric for the lowest RMSE, never a denormalized flag — avoids
staleness when metrics are updated by a retrain."""

from sqlalchemy import extract, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.backend.models import Forecast, ModelMetric


async def best_model_family(session: AsyncSession, vegetable: str) -> str | None:
    stmt = (
        select(ModelMetric.model_family)
        .where(ModelMetric.vegetable == vegetable)
        .order_by(ModelMetric.rmse.asc())
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def list_model_metrics(session: AsyncSession, vegetable: str | None = None) -> list[ModelMetric]:
    stmt = select(ModelMetric)
    if vegetable is not None:
        stmt = stmt.where(ModelMetric.vegetable == vegetable)
    stmt = stmt.order_by(ModelMetric.vegetable, ModelMetric.rmse.asc())
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def latest_prediction(session: AsyncSession, vegetable: str, model_family: str) -> Forecast | None:
    stmt = (
        select(Forecast)
        .where(Forecast.vegetable == vegetable, Forecast.model_family == model_family)
        .order_by(Forecast.forecast_date.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def future_predictions(session: AsyncSession, vegetable: str, model_family: str) -> list[Forecast]:
    """Genuinely future, not-yet-resolved weeks only (actual_price IS NULL) — see
    scripts/predict_future.py. Chronological (oldest-first), since these are upcoming
    weeks a caller wants to read in order, unlike list_predictions()'s newest-first history."""
    stmt = (
        select(Forecast)
        .where(
            Forecast.vegetable == vegetable,
            Forecast.model_family == model_family,
            Forecast.actual_price.is_(None),
        )
        .order_by(Forecast.forecast_date.asc())
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def list_predictions(
    session: AsyncSession,
    vegetable: str | None = None,
    model_family: str | None = None,
    year: int | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Forecast]:
    stmt = select(Forecast)
    if vegetable is not None:
        stmt = stmt.where(Forecast.vegetable == vegetable)
    if model_family is not None:
        stmt = stmt.where(Forecast.model_family == model_family)
    if year is not None:
        stmt = stmt.where(extract("year", Forecast.forecast_date) == year)
    stmt = stmt.order_by(Forecast.forecast_date.desc()).limit(limit).offset(offset)
    result = await session.execute(stmt)
    return list(result.scalars().all())
