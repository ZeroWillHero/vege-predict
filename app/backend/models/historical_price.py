from datetime import date

from sqlalchemy import Date, Float, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from app.backend.database import Base


class HistoricalPrice(Base):
    __tablename__ = "historical_price"
    __table_args__ = (Index("ix_historical_price_vegetable_week", "vegetable", "week_start", unique=True),)

    id: Mapped[int] = mapped_column(primary_key=True)
    vegetable: Mapped[str] = mapped_column(String(32), index=True)
    week_start: Mapped[date] = mapped_column(Date, index=True)
    wholesale_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    retail_price: Mapped[float] = mapped_column(Float)
    average_temperature: Mapped[float | None] = mapped_column(Float, nullable=True)
    average_rainfall: Mapped[float | None] = mapped_column(Float, nullable=True)
    diesel_price: Mapped[float | None] = mapped_column(Float, nullable=True)
