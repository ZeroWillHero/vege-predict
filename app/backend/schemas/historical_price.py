from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class HistoricalPriceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    vegetable: str = Field(description="Vegetable identifier.", examples=["carrot"])
    week_start: date = Field(description="Start date (Monday) of the price week.")
    wholesale_price: float | None = Field(default=None, description="Wholesale price (LKR/kg), if available.")
    retail_price: float = Field(description="Retail price (LKR/kg) — the forecasting target.")
    average_temperature: float | None = Field(
        default=None,
        description="Average temperature (°C) for that week, from this vegetable's mapped "
        "growing district.",
    )
    average_rainfall: float | None = Field(
        default=None,
        description="Average rainfall (mm) for that week, from this vegetable's mapped growing "
        "district.",
    )
    diesel_price: float | None = Field(
        default=None, description="Diesel price (LKR/L) that week — a proxy for produce transport cost."
    )
