from typing import Literal

from pydantic import BaseModel, Field


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"] = Field(
        description="'ok' only when both database and redis are reachable, otherwise 'degraded'."
    )
    database: bool = Field(description="Whether a Postgres connection + SELECT 1 succeeded.")
    redis: bool = Field(description="Whether a Redis PING succeeded.")
