from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field


class ForecastOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    vegetable: str = Field(description="Vegetable identifier.", examples=["carrot"])
    model_family: str = Field(
        description="Model family that produced this forecast.", examples=["random_forest"]
    )
    forecast_date: date = Field(description="Week-start date this forecast targets.")
    predicted_price: float = Field(description="Predicted retail price (LKR/kg).")
    actual_price: float | None = Field(
        default=None,
        description="Actual observed retail price for that week, if known. Null for weeks not "
        "yet resolved.",
    )
    generated_at: datetime = Field(description="When this forecast was generated (UTC).")
