from fastapi import APIRouter, Depends, HTTPException, Path, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.backend.cache import get_cached, set_cached
from app.backend.constants import VEGETABLES
from app.backend.database import get_db
from app.backend.schemas import ForecastOut
from app.backend.services import forecast_service
from app.backend.services.cache_service import prediction_key, predictions_list_key

router = APIRouter()


@router.get(
    "/predictions/{vegetable}",
    response_model=ForecastOut,
    summary="Latest forecast for a vegetable",
    description=(
        "Returns the single most recent forecast for a vegetable. `model=best` (the default) "
        "resolves to whichever model family currently has the lowest holdout RMSE for that "
        "vegetable (see `GET /models`); pass an explicit model family (e.g. `sarimax`, "
        "`random_forest`, `hybrid_xgboost_sarimax`) to pin to it instead. Alongside the point "
        "forecast (`predicted_price`), the response includes a `predicted_lower`/`predicted_upper` "
        "prediction interval — see those fields' descriptions for how it's derived."
    ),
    response_description="The latest forecast row for the resolved model family.",
    responses={
        404: {
            "description": "`vegetable` is unknown, or no forecast exists for the "
            "vegetable/model combination (e.g. a misspelled model family name)."
        }
    },
)
async def latest_prediction(
    vegetable: str = Path(
        description="Vegetable identifier — see `GET /vegetables` for valid values.", examples=["carrot"]
    ),
    model: str = Query(
        default="best",
        description="A model family name, or `best` to auto-resolve to the lowest-RMSE model for "
        "this vegetable.",
        examples=["best"],
    ),
    session: AsyncSession = Depends(get_db),
):
    if vegetable not in VEGETABLES:
        raise HTTPException(status_code=404, detail=f"Unknown vegetable: {vegetable}")

    key = prediction_key(vegetable, model)
    cached = await get_cached(key)
    if cached is not None:
        return cached

    model_family = model
    if model == "best":
        model_family = await forecast_service.best_model_family(session, vegetable)
        if model_family is None:
            raise HTTPException(status_code=404, detail=f"No model metrics for {vegetable}")

    forecast = await forecast_service.latest_prediction(session, vegetable, model_family)
    if forecast is None:
        raise HTTPException(status_code=404, detail=f"No forecast for {vegetable}/{model_family}")

    result = ForecastOut.model_validate(forecast).model_dump(mode="json")
    await set_cached(key, result)
    return result


@router.get(
    "/predictions",
    response_model=list[ForecastOut],
    summary="Search/filter forecast history",
    description=(
        "Returns forecast rows filtered by any combination of vegetable, model family, and year, "
        "newest first, paginated with `limit`/`offset`. Unlike `GET /predictions/{vegetable}`, an "
        "unknown `model` value here is not an error — it simply matches zero rows, so this endpoint "
        "can also be used to check whether a model family produced any forecasts at all."
    ),
    response_description="Matching forecast rows, most recent first.",
    responses={404: {"description": "`vegetable` was given but is not one of the tracked vegetables."}},
)
async def list_predictions(
    vegetable: str | None = Query(
        default=None, description="Filter to one vegetable. Omit to search across all vegetables.", examples=["carrot"]
    ),
    model: str | None = Query(
        default=None,
        description="Filter to one model family (exact match, e.g. `xgboost`). Omit to include all "
        "families. Unknown values return an empty list, not an error.",
        examples=["xgboost"],
    ),
    year: int | None = Query(
        default=None, description="Filter to forecasts whose target week falls in this calendar year.", examples=[2026]
    ),
    limit: int = Query(default=50, ge=1, le=1000, description="Maximum rows to return (1–1000)."),
    offset: int = Query(default=0, ge=0, description="Rows to skip, for pagination."),
    session: AsyncSession = Depends(get_db),
):
    if vegetable is not None and vegetable not in VEGETABLES:
        raise HTTPException(status_code=404, detail=f"Unknown vegetable: {vegetable}")

    key = predictions_list_key(vegetable, model, year, limit, offset)
    cached = await get_cached(key)
    if cached is not None:
        return cached

    forecasts = await forecast_service.list_predictions(session, vegetable, model, year, limit, offset)
    result = [ForecastOut.model_validate(f).model_dump(mode="json") for f in forecasts]
    await set_cached(key, result)
    return result
