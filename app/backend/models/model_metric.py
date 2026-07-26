from datetime import datetime

from sqlalchemy import DateTime, Float, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from app.backend.database import Base


class ModelMetric(Base):
    __tablename__ = "model_metric"
    __table_args__ = (Index("ix_model_metric_vegetable_model", "vegetable", "model_family", unique=True),)

    id: Mapped[int] = mapped_column(primary_key=True)
    vegetable: Mapped[str] = mapped_column(String(32), index=True)
    model_family: Mapped[str] = mapped_column(String(64), index=True)
    mae: Mapped[float] = mapped_column(Float)
    rmse: Mapped[float] = mapped_column(Float)
    mape: Mapped[float] = mapped_column(Float)
    r2: Mapped[float] = mapped_column(Float)
    interval_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    interval_coverage: Mapped[float | None] = mapped_column(Float, nullable=True)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
