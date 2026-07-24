"""FastAPI app for vegepredict: serves pre-computed forecasts, model comparisons,
and historical prices from Postgres (Redis-cached). See CLAUDE.md's "Backend API"
section for architecture notes. Interactive docs at /docs (Swagger UI) and /redoc."""

from fastapi import FastAPI

from app.backend.routers import health, models, predictions, prices, vegetables

TAGS_METADATA = [
    {
        "name": "health",
        "description": "Service liveness — checks Postgres and Redis connectivity.",
    },
    {
        "name": "vegetables",
        "description": "The fixed list of vegetables this system forecasts prices for.",
    },
    {
        "name": "models",
        "description": "Per-vegetable model comparison metrics (MAE, RMSE, MAPE, R²), used to "
        "determine each vegetable's best-performing model family.",
    },
    {
        "name": "predictions",
        "description": "Pre-computed price forecasts — single latest forecast or filtered/"
        "paginated history.",
    },
    {
        "name": "prices",
        "description": "Historical weekly retail/wholesale prices plus the weather and fuel "
        "features used to train the forecasting models.",
    },
]

app = FastAPI(
    title="VegePredict API",
    version="0.1.0",
    description=(
        "Serves **pre-computed** vegetable price forecasts, model comparison metrics, and "
        "historical prices for Sri Lanka's carrot, brinjal, pumpkin, cabbage, snake gourd, and "
        "leeks markets.\n\n"
        "Forecasts come from an offline training pipeline (SARIMAX, LSTM, Random Forest, "
        "XGBoost, CatBoost, and two SARIMAX-residual hybrids) that runs periodically — this API "
        "does not train or run models per-request, it only reads what the last training run "
        "produced. \"Best model\" is always resolved dynamically from the lowest holdout RMSE, "
        "never hardcoded, so it tracks whichever family wins after each retrain.\n\n"
        "All read endpoints are cached in Redis for 7 days (matching the weekly retrain cadence)."
    ),
    openapi_tags=TAGS_METADATA,
)

app.include_router(health.router, tags=["health"])
app.include_router(vegetables.router, tags=["vegetables"])
app.include_router(models.router, tags=["models"])
app.include_router(prices.router, tags=["prices"])
app.include_router(predictions.router, tags=["predictions"])
