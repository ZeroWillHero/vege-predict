"""Import every ORM model here so Base.metadata sees them all — Alembic's
autogenerate (and any Base.metadata.create_all() call) only picks up models that
have actually been imported somewhere in the process."""

from app.backend.models.forecast import Forecast
from app.backend.models.historical_price import HistoricalPrice
from app.backend.models.model_metric import ModelMetric
from app.backend.models.user import User

__all__ = ["HistoricalPrice", "Forecast", "ModelMetric", "User"]
