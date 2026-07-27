from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.backend.cache import get_cached, set_cached
from app.backend.constants import VEGETABLES
from app.backend.database import get_db
from app.backend.schemas import HistoricalPriceOut
from app.backend.services import price_service
from app.backend.services.cache_service import prices_key

router = APIRouter()


@router.get(
    "/prices/{vegetable}/history",
    response_model=list[HistoricalPriceOut],
    summary="Historical weekly prices",
    description=(
        "Returns weekly historical retail/wholesale prices plus the weather and diesel-price "
        "features used to train the forecasting models, ordered by week ascending (oldest first). "
        "An empty `start_date`/`end_date` range (start after end) or an `offset` past the end of "
        "the data returns an empty list rather than an error."
    ),
    response_description="Weekly price/weather/fuel rows for the vegetable, oldest first.",
    responses={404: {"description": "`vegetable` is not one of the tracked vegetables."}},
)
async def price_history(
    vegetable: str = Path(
        description="Vegetable identifier — see `GET /vegetables` for valid values.", examples=["carrot"]
    ),
    start_date: date | None = Query(
        default=None, description="Only return weeks on or after this date (inclusive).", examples=["2020-01-01"]
    ),
    end_date: date | None = Query(
        default=None, description="Only return weeks on or before this date (inclusive).", examples=["2020-12-31"]
    ),
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum rows to return (1–1000)."),
    offset: int = Query(default=0, ge=0, description="Rows to skip, for pagination."),
    session: AsyncSession = Depends(get_db),
):
    if vegetable not in VEGETABLES:
        raise HTTPException(status_code=404, detail=f"Unknown vegetable: {vegetable}")

    key = prices_key(vegetable, start_date, end_date, limit, offset)
    cached = await get_cached(key)
    if cached is not None:
        return cached

    prices = await price_service.list_prices(session, vegetable, start_date, end_date, limit, offset)
    result = [HistoricalPriceOut.model_validate(p).model_dump(mode="json") for p in prices]
    await set_cached(key, result)
    return result
