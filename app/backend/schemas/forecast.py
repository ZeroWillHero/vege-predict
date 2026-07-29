from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field


class ForecastOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    vegetable: str = Field(description="Vegetable identifier.", examples=["carrot"])
    model_family: str = Field(
        description="Model family that produced this forecast.", examples=["random_forest"]
    )
    forecast_date: date = Field(description="Week-start date this forecast targets.")
    predicted_price: float = Field(description="Predicted retail price (LKR/kg) — the point forecast.")
    predicted_lower: float | None = Field(
        default=None,
        description="Lower bound of the prediction interval (LKR/kg), at the model's configured "
        "confidence level (80% by default — see `interval_confidence` in configs/config.yaml). "
        "For backtest rows, derived from the walk-forward CV folds' out-of-sample residuals, not "
        "the holdout itself; for genuinely future rows (see `actual_price`), derived from the "
        "holdout period's own residuals instead, since CV-fold residuals aren't persisted.",
    )
    predicted_upper: float | None = Field(
        default=None, description="Upper bound of the prediction interval (LKR/kg), same basis as `predicted_lower`."
    )
    actual_price: float | None = Field(
        default=None,
        description="Actual observed retail price for that week. Null for a genuinely future, "
        "not-yet-resolved week (see `scripts/predict_future.py`) — this is expected, not missing "
        "data.",
    )
    generated_at: datetime = Field(description="When this forecast was generated (UTC).")
