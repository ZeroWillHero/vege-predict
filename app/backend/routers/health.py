from fastapi import APIRouter
from sqlalchemy import text

from app.backend.cache import ping as redis_ping
from app.backend.database import engine
from app.backend.schemas import HealthOut

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthOut,
    summary="Service health check",
    description=(
        "Checks connectivity to Postgres (`SELECT 1`) and Redis (`PING`) independently. "
        "Status is `ok` only if both succeed, `degraded` otherwise — the response body always "
        "reports which dependency failed."
    ),
    response_description="Health status of the API and each of its dependencies.",
)
async def health() -> HealthOut:
    db_ok, redis_ok = False, False
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    try:
        redis_ok = await redis_ping()
    except Exception:
        pass

    status = "ok" if (db_ok and redis_ok) else "degraded"
    return HealthOut(status=status, database=db_ok, redis=redis_ok)
