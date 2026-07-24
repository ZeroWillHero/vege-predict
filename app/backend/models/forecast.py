from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from app.backend.database import Base


class Forecast(Base):
    __tablename__ = "forecast"
    __table_args__ = (
        Index("ix_forecast_vegetable_model_date", "vegetable", "model_family", "forecast_date", unique=True),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    vegetable: Mapped[str] = mapped_column(String(32), index=True)
    model_family: Mapped[str] = mapped_column(String(64), index=True)
    forecast_date: Mapped[date] = mapped_column(Date, index=True)
    predicted_price: Mapped[float] = mapped_column(Float)
    actual_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
