from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelMetricOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    vegetable: str = Field(description="Vegetable identifier.", examples=["carrot"])
    model_family: str = Field(
        description="Model family name — one of the 9 trained families.",
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
    interval_confidence: float | None = Field(
        default=None,
        description="Nominal confidence level the prediction interval was built at (e.g. 0.8 for "
        "an 80% interval). See `predicted_lower`/`predicted_upper` on the forecast endpoints.",
    )
    interval_coverage: float | None = Field(
        default=None,
        description="Empirically observed coverage on the holdout set — the fraction of actual "
        "prices that fell within [predicted_lower, predicted_upper]. Compare against "
        "`interval_confidence` to see how well-calibrated the interval actually is; well below "
        "the nominal level means the band is too narrow, well above means it's wider than needed.",
    )
    evaluated_at: datetime = Field(description="When this metric was computed/seeded (UTC).")
