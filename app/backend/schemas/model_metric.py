from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelMetricOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    vegetable: str = Field(description="Vegetable identifier.", examples=["carrot"])
    model_family: str = Field(
        description="Model family name — one of the 7 trained families.",
        examples=["random_forest"],
    )
    mae: float = Field(description="Mean Absolute Error on the holdout set (LKR).")
    rmse: float = Field(
        description="Root Mean Squared Error on the holdout set (LKR) — the metric used to pick "
        "the best model family per vegetable (lowest RMSE wins)."
    )
    mape: float = Field(description="Mean Absolute Percentage Error on the holdout set (%).")
    r2: float = Field(
        description="R² on the holdout set. Can be negative for a poor fit — e.g. SARIMAX-based "
        "models on a volatile holdout year."
    )
    evaluated_at: datetime = Field(description="When this metric was computed/seeded (UTC).")
