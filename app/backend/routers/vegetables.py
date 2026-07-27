from fastapi import APIRouter

from app.backend.cache import get_cached, set_cached
from app.backend.constants import VEGETABLES
from app.backend.services.cache_service import vegetables_key

router = APIRouter()


@router.get(
    "/vegetables",
    response_model=list[str],
    summary="List tracked vegetables",
    description="Returns the fixed list of vegetables this system forecasts prices for. Use these "
    "identifiers (lowercase, underscore-separated) as the `vegetable` value on every other endpoint.",
    response_description="Vegetable identifiers.",
)
async def list_vegetables() -> list[str]:
    key = vegetables_key()
    cached = await get_cached(key)
    if cached is not None:
        return cached
    await set_cached(key, VEGETABLES)
    return VEGETABLES
