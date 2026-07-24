from datetime import date

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.backend.models import HistoricalPrice


async def list_prices(
    session: AsyncSession,
    vegetable: str,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[HistoricalPrice]:
    stmt = select(HistoricalPrice).where(HistoricalPrice.vegetable == vegetable)
    if start_date is not None:
        stmt = stmt.where(HistoricalPrice.week_start >= start_date)
    if end_date is not None:
        stmt = stmt.where(HistoricalPrice.week_start <= end_date)
    stmt = stmt.order_by(HistoricalPrice.week_start.asc()).limit(limit).offset(offset)
    result = await session.execute(stmt)
    return list(result.scalars().all())
