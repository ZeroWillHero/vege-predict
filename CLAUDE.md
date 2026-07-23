# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research on vegetable price forecasting using weather data, fuel prices, and historical vegetable prices as inputs. Target vegetables: **carrot, brinjal, pumpkin, cabbage, snake gourd, leeks**. Models under comparison: **SARIMAX, XGBoost, CatBoost**, and two hybrids — **XGBoost+SARIMAX** and **CatBoost+SARIMAX** (SARIMAX models the linear/seasonal component; the ML model is trained on SARIMAX's residuals).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Common commands

There is no build step (pure Python research code). Typical workflow:

```bash
# 1. Merge raw price/weather/fuel data into a per-vegetable processed dataset
python src/data_processing/build_dataset.py --vegetable carrot

# 2. Train a specific model for a specific vegetable
python src/models/sarimax/train.py --vegetable carrot
python src/models/xgboost/train.py --vegetable carrot
python src/models/catboost/train.py --vegetable carrot
python src/models/hybrid_xgboost_sarimax/train.py --vegetable carrot
python src/models/hybrid_catboost_sarimax/train.py --vegetable carrot

# 3. Run tests
pytest tests/
pytest tests/test_features.py::test_lag_features   # single test
```

Model hyperparameters (SARIMAX order/seasonal_order, XGBoost/CatBoost params, exogenous feature lists) live in `configs/config.yaml` — do not hardcode them in training scripts.

## Architecture

**Data flow:** `data/raw/{vegetable_prices,weather,fuel}` (untouched source files, one subfolder per source) → `src/data_processing/build_dataset.py` joins them on date into `data/processed/<vegetable>.csv` → `src/features/feature_engineering.py` adds lags/rolling stats/calendar features (shared by every model, so a feature added here is available to all five model families) → each `src/models/<model>/train.py` consumes the processed+featurized frame and writes a fitted model to `trained_models/<model>/<vegetable>.pkl`.

**Hybrid models** (`src/models/hybrid_*`) depend on the plain SARIMAX fit: they load/re-fit SARIMAX first, compute residuals (actual − SARIMAX prediction), then train the ML model on those residuals plus the exogenous features. Final forecast = SARIMAX prediction + ML residual prediction. Keep this two-stage structure intact — don't collapse it into a single end-to-end model.

**Per-vegetable, per-model artifacts:** every vegetable is modeled independently (no cross-vegetable pooling), and every model family is trained separately per vegetable. Expect `6 vegetables × 5 model families = 30` trained artifacts, plus corresponding metrics rows in `results/metrics/`.

**Evaluation:** `src/evaluation/metrics.py` is the single source of truth for MAE/RMSE/MAPE — all training scripts and notebooks should import from there rather than reimplementing metrics, so comparisons across the 5 model families stay apples-to-apples.

## Folder layout

- `data/raw/` — untouched source data (vegetable_prices, weather, fuel), one subfolder each
- `data/processed/` — merged, per-vegetable model-ready datasets
- `src/data_processing/` — raw → processed merging
- `src/features/` — shared feature engineering used by all models
- `src/models/{sarimax,xgboost,catboost,hybrid_xgboost_sarimax,hybrid_catboost_sarimax}/` — one training module per model family
- `src/evaluation/` — shared metrics and cross-model comparison
- `trained_models/<model_family>/` — serialized fitted models, one file per vegetable
- `results/{figures,tables,metrics}/` — plots, comparison tables, metric outputs for the paper
- `configs/config.yaml` — vegetables list, data paths, and per-model hyperparameters
- `notebooks/` — exploratory analysis; production logic belongs in `src/`, not notebooks
- `research-papers/references/` — literature (PDFs, papers) informing the methodology
- `research-papers/drafts/` — the paper/thesis draft itself
