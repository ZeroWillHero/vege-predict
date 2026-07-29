from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.backend.cache import get_cached, set_cached
from app.backend.constants import VEGETABLES
from app.backend.database import get_db
from app.backend.schemas import ModelMetricOut
from app.backend.services import forecast_service
from app.backend.services.cache_service import models_key

router = APIRouter()


@router.get(
    "/models",
    response_model=list[ModelMetricOut],
    summary="Compare model performance",
    description=(
        "Returns holdout evaluation metrics (MAE, RMSE, MAPE, R²) for every trained model family, "
        "optionally filtered to one vegetable. Rows are sorted by RMSE ascending, so the first row "
        "for a given vegetable is its best-performing model family — the same one `model=best` "
        "resolves to on the `/predictions` endpoints."
    ),
    response_description="Model comparison rows, sorted by RMSE ascending within each vegetable.",
    responses={404: {"description": "`vegetable` was given but is not one of the tracked vegetables."}},
)
async def list_models(
    vegetable: str | None = Query(
        default=None,
        description="Filter to a single vegetable (see `GET /vegetables` for valid values). Omit "
        "to return all 54 rows (6 vegetables × 9 model families).",
        examples=["carrot"],
    ),
    session: AsyncSession = Depends(get_db),
):
    if vegetable is not None and vegetable not in VEGETABLES:
        raise HTTPException(status_code=404, detail=f"Unknown vegetable: {vegetable}")

    key = models_key(vegetable)
    cached = await get_cached(key)
    if cached is not None:
        return cached

    metrics = await forecast_service.list_model_metrics(session, vegetable)
    result = [ModelMetricOut.model_validate(m).model_dump(mode="json") for m in metrics]
    await set_cached(key, result)
    return result
