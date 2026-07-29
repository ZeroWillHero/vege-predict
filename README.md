# VegePredict

Adaptive, trustworthy vegetable price forecasting for Sri Lanka — UWU ICT 481-6 capstone project.

Fuses historical retail prices, weather, and fuel price data to forecast short-term prices for **carrot, brinjal, pumpkin, cabbage, snake gourd, and leeks**, using nine model families (SARIMAX, LSTM, Random Forest, XGBoost, CatBoost, and four SARIMAX-residual hybrids — XGBoost, CatBoost, Random Forest, and LSTM). Every forecast ships with an 80% prediction interval, not just a point estimate. A FastAPI backend (Postgres + Redis) serves the results.

For full architecture notes, gotchas, and research context, see [CLAUDE.md](CLAUDE.md) — this README only covers how to run things.

## 1. Model training pipeline

### Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Or with [uv](https://docs.astral.sh/uv/) (faster):

```bash
uv venv .venv --python 3.13
uv pip install -r requirements.txt --python .venv/bin/python
```

### Run

```bash
# 1. Build the merged, feature-ready dataset for each vegetable (only needed once, or
#    after raw data changes — see scripts/ below for how to refresh raw data)
python src/data_processing/build_dataset.py --vegetable carrot   # repeat per vegetable, or omit --vegetable for all

# 2. Train every model family for every vegetable in one go (~15 minutes+, SARIMAX/LSTM
#    and their hybrids are the slow parts). Writes results/metrics/all_results.csv,
#    results/metrics/holdout_predictions.csv, and results/tables/model_comparison.csv
python scripts/train_all.py

# Or train a single model family for a single vegetable during development:
python src/models/random_forest/train.py --vegetable carrot

# 3. Genuine future forecast (next N weeks past the last known price date, all 9 model
#    families) — requires trained_models/ artifacts from step 2 above. Writes
#    results/metrics/future_predictions.csv
python scripts/predict_future.py
```

### Refreshing raw data (optional)

```bash
# Incrementally pull new HARTI weekly price bulletins (only fetches weeks not already ingested)
python src/pipeline/scrapers/harti_prices.py update

# Incrementally pull new CEYPETCO diesel price revisions
python src/pipeline/scrapers/cpc_fuel.py update

# Rebuild weather.csv from Open-Meteo (district-wise, 2014-present)
python scripts/fetch_weather_openmeteo.py
```

### Tests

```bash
pytest tests/
```

### Explore results

- `notebooks/01_data_exploration.ipynb` — raw data trends/patterns
- `notebooks/02_feature_engineering.ipynb` — engineered feature inspection
- `notebooks/03_model_results.ipynb` — model comparison tables, forecast-vs-actual plots, prediction interval visualization (needs `results/metrics/all_results.csv` from step 2 above)

## 2. Backend API

Standalone FastAPI service in `app/backend/` — its own virtual environment, separate from the training pipeline's.

### Setup

```bash
uv venv app/backend/.venv --python 3.13
uv pip install -r app/backend/requirements.txt --python app/backend/.venv/bin/python
```

(`python -m venv app/backend/.venv && app/backend/.venv/bin/pip install -r app/backend/requirements.txt` works too if you don't have `uv`.)

### Prerequisites

A local Postgres instance (database `vegepredict`) and Redis instance, both reachable at the defaults in `app/backend/config.py` (override via an `app/backend/.env` if yours differ — see `CLAUDE.md`'s Backend API section for connection details).

### Run

```bash
# 1. Apply schema migrations (only needed once, or after a schema change)
app/backend/.venv/bin/python3 -m alembic -c app/backend/alembic.ini upgrade head

# 2. Seed/refresh Postgres from the CSVs scripts/train_all.py (and, if present,
#    scripts/predict_future.py) produce — safe to re-run after every retrain (idempotent upsert)
app/backend/.venv/bin/python3 app/backend/seed.py

# 3. Start the API (run from the project root, so app.backend.* imports resolve)
app/backend/.venv/bin/python3 -m uvicorn app.backend.main:app --reload --port 8000
```

Then open **http://127.0.0.1:8000/docs** for the interactive Swagger UI (or `/redoc` for ReDoc) — every endpoint, parameter, and response field is documented there, with example values you can run directly against the live server.

### Tests

```bash
app/backend/.venv/bin/python3 -m pytest app/backend/tests/ -c app/backend/pytest.ini
```

## Project layout

See [CLAUDE.md](CLAUDE.md)'s "Folder layout" section for the full breakdown of `src/`, `app/backend/`, `data/`, `results/`, and `research-papers/`.
