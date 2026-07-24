# Chapter 3: Methodology

## 3.1 Overview

This chapter describes how the forecasting pipeline was built: the data sources and how they were integrated, the feature engineering shared across every model, the seven forecasting model families under comparison, and the evaluation protocol used to compare them. The implementation follows the two-stage structure set out in Chapter 1 — build a clean multi-source dataset first, then train and evaluate every model family against it identically — so that differences in reported accuracy reflect genuine differences between models rather than differences in the data each model happened to see.

## 3.2 Data Sources and Integration

Three data sources were combined, each weekly and each covering a different aspect of the vegetable market:

**Historical prices.** Weekly wholesale and retail prices for six vegetables — carrot, brinjal, pumpkin, cabbage, snake gourd, and leeks — sourced from Central Bank/HARTI-style price bulletins, spanning 2000-01-03 to 2025-12-22. Retail price was selected as the forecast target (`forecasting.target_column` in `configs/config.yaml`), since it is the figure most directly tied to the farmer- and consumer-facing advisory objective in Chapter 1.

**Fuel prices.** Weekly diesel prices, 2015-01-05 to 2025-12-29. Diesel was used rather than petrol, since diesel powers the lorries and trucks that move produce from farm to market — the transport-cost mechanism identified in the literature (Ranaweera et al., 2023) — while petrol has no direct link to the vegetable supply chain. Petrol prices were excluded from the raw dataset entirely rather than merely unused in code, so the data source matches what the model actually consumes.

**Weather.** Weekly average temperature and rainfall, fetched from the Open-Meteo historical archive API for all 25 administrative districts of Sri Lanka (2013-12-30 to 2026-07-20; `scripts/fetch_weather_openmeteo.py`), aggregated from daily records: average temperature as the mean of each day's (max+min)/2, average rainfall as the mean of each day's precipitation total across the week.

**Vegetable-to-district mapping.** Since price data is reported nationally but weather varies by growing region, each vegetable was mapped to a single representative district reflecting Sri Lanka's up-country/low-country growing-zone split (Weerasekara et al., 2026): carrot, leeks, and cabbage (up-country) to Nuwara Eliya; brinjal to Matale, pumpkin to Anuradhapura, and snake gourd to Kurunegala (low-country, dry-zone-adjacent districts, consistent with the market centres used by Ranaweera et al., 2023). This mapping is stored as data (`weather_district_map` in `configs/config.yaml`), not hardcoded in code, so it can be revisited without a code change.

**Integration and alignment.** `src/data_processing/build_dataset.py` joins the three sources on week-start date, per vegetable, using each vegetable's mapped district for the weather join. The three sources do not share an identical date range — fuel data begins 2015-01-05, and price data ends 2025-12-22 — so the merged dataset is clipped to their intersection, 2015-01-05 to 2025-12-22 (`aligned_start_date`/`aligned_end_date` in config), yielding 571 weekly observations per vegetable with no missing values.

## 3.3 Feature Engineering

`src/features/feature_engineering.py` applies one shared transformation to every vegetable's merged series, so every model family is trained on an identical feature set (only the model architecture differs):

- **Lagged price**: retail price at *t*-1, *t*-2, *t*-4, and *t*-8 weeks.
- **Rolling statistics**: mean and standard deviation of retail price over trailing 4-, 8-, and 12-week windows, computed from `shift(1)` before rolling so no window includes the value being predicted.
- **Calendar features**: month, ISO week-of-year, and quarter.
- **Season flag**: a binary Maha (October-April) / Yala (May-September) indicator, following the two-cultivation-season structure documented by Weerasekara et al. (2026).
- **Exogenous features**: contemporaneous rainfall, temperature, and diesel price, plus each lagged by one week, to capture both same-week and delayed effects.

Rows in the warm-up period (before the longest lag/rolling window has enough history) are dropped, leaving 559 usable rows per vegetable. `get_feature_columns()` derives the final feature list programmatically, excluding the date and target columns and the raw wholesale price (kept in the processed file for reference, but excluded from the feature set to avoid leaking a near-duplicate of the target).

## 3.4 Forecasting Models

Seven model families were implemented, matching the proposal's mandated set (SARIMA/SARIMAX, LSTM, Random Forest, XGBoost, plus hybrids) with CatBoost and its hybrid retained as an extended baseline:

**SARIMAX** (`src/models/sarimax/`). A seasonal ARIMA with exogenous regressors, fit via `statsmodels`' state-space implementation. The seasonal order is fixed at (1,0,1,52) for every vegetable — a 52-week seasonal period matching weekly Sri Lankan price data, following the SARIMA(3,1,2)(0,0,2)[52] precedent for carrot prices reported by Champika and Mugera (cited in Weerasekara et al., 2026). Seasonal differencing (D=1) was tested but produced unstable, divergent out-of-sample forecasts with roughly ten years of training data (~ten seasonal cycles) and was replaced with D=0. `simple_differencing=True` was likewise tested for speed but found to break out-of-sample forecasting once exogenous regressors are present, and was reverted.

The non-seasonal order (p,d,q) is selected per vegetable via a standard Box-Jenkins AIC search (`scripts/select_sarimax_order.py`): every candidate (p,1,q) for p,q ∈ {0,1,2} is fit on the full training series with the seasonal order held fixed, and the AIC-minimizing candidate is adopted (`models.sarimax.order_by_vegetable` in `configs/config.yaml`; full results in `results/tables/sarimax_order_selection.csv`). Five of six vegetables selected (2,1,2); carrot selected (0,1,2) and leeks (1,1,2) — all richer than the (1,1,1) starting point. This AIC-selected order is the statistically correct choice for *in-sample* model fit, but Chapter 4 shows it does not uniformly improve *holdout* forecast accuracy (substantially better for pumpkin and leeks, slightly worse for carrot, brinjal, and snake_gourd) — the expected and well-documented divergence between an information criterion (in-sample fit, penalized for complexity) and out-of-sample generalization. This is why AIC is used only to select SARIMAX's internal order, while cross-model-family comparison (Section 3.5, Chapter 4) uses holdout RMSE throughout: AIC has no standard definition for the tree-based and deep-learning model families in this study, so it cannot serve as a common selection criterion across all seven.

**LSTM** (`src/models/lstm/`). A stacked LSTM (2 layers, 64 units, 0.2 dropout) implemented in TensorFlow/Keras, trained on 30-week sequences of price plus exogenous history, with inputs standardized via a per-fold `StandardScaler` and early stopping on training loss (patience 10).

**Random Forest, XGBoost, CatBoost** (`src/models/{random_forest,xgboost,catboost}/`). Standard tabular regressors trained directly on the engineered feature set from Section 3.3, using `scikit-learn`, `xgboost`, and `catboost` respectively.

**Hybrid XGBoost+SARIMAX, Hybrid CatBoost+SARIMAX** (`src/models/hybrid_{xgboost,catboost}_sarimax/`). A two-stage design: SARIMAX is fit first on the training window to capture the linear/seasonal component; its in-sample residuals (actual − fitted) become the regression target for an XGBoost or CatBoost model trained on the same engineered feature set. At prediction time, the final forecast is the sum of the SARIMAX forecast and the residual model's prediction. This residual-correction pattern follows the precedent of hybrid statistical-ML approaches in the literature (Patil et al., 2023; Zhao et al., 2025) — the intended benefit is combining SARIMAX's explicit handling of trend and seasonality with the residual model's ability to capture the nonlinear structure SARIMAX misses.

A shared runner (`src/models/common.py`) handles the CV/holdout/save plumbing identically across all seven families, so that the only thing that varies between one model's training script and another's is the model-specific fit/predict logic.

## 3.5 Evaluation Protocol

Following the proposal's specification of time-series cross-validation, each vegetable's 559-row feature set is split as follows (`src.evaluation.metrics.time_series_splits`):

1. **Final holdout**: the most recent 52 weeks are set aside and never used for model fitting or selection — only for the final reported metric.
2. **Walk-forward cross-validation**: the remaining ~507 rows are split into 5 expanding-window folds, each fold training on all data up to a point and validating on the following contiguous block, so no fold ever validates on data that precedes its training window.
3. **Production refit**: after CV and holdout evaluation, each model is refit one final time on the complete 559-row series to produce the artifact saved under `trained_models/`.

All models are scored on MAE, RMSE, MAPE, and R² (`src/evaluation/metrics.py`), computed identically regardless of model family, so cross-model comparisons in Chapter 4 are on equal footing.

## 3.6 Implementation Environment

The pipeline is implemented in Python 3.13, in an isolated project `.venv` (not system/Anaconda Python), with `numpy`, `pandas`, `scikit-learn`, `statsmodels`, `xgboost`, `catboost`, and `tensorflow` as the core dependencies (`requirements.txt`). `scripts/train_all.py` orchestrates all 6 vegetables × 7 model families in a single run, writing per-run metrics to `results/metrics/all_results.csv` and a pivoted comparison table to `results/tables/model_comparison.csv`. All results in this thesis were produced on Apple Silicon (M4).
