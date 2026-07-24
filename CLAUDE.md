# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**Adaptive, Trustworthy Vegetable Price Forecasting Using Multi-Source Data and AI in Sri Lanka** — UWU ICT 481-6 capstone (supervisors: Dr. Niranjan W. Gunasekara, Ms. KRR Premathilaka). Fuses historical prices, weather, fuel prices, satellite NDVI (Sentinel-2 via Google Earth Engine), IoT sensor readings, and farmer behavioral survey data to forecast short-term vegetable prices, explain the forecasts, and turn them into farmer-facing advice.

Target vegetables: **carrot, brinjal, pumpkin, cabbage, snake gourd, leeks**.

**Forecasting models** (proposal-mandated: SARIMA, LSTM, Random Forest, XGBoost, plus hybrids; CatBoost and its hybrid are an extended baseline beyond the proposal, kept from the original scaffold):
- **SARIMAX** — linear/seasonal component, weather+fuel as exogenous regressors
- **LSTM** — sequence model over price + exogenous history
- **Random Forest** — lag/rolling/calendar features
- **XGBoost**, **CatBoost** — same feature set as Random Forest
- **Hybrid XGBoost+SARIMAX**, **Hybrid CatBoost+SARIMAX** — SARIMAX models the linear/seasonal component; the ML model is trained on SARIMAX's residuals; final forecast = SARIMAX prediction + residual prediction

**Beyond forecasting**, the proposal's remaining objectives:
- **Explainability** (`src/explainability/`) — SHAP feature attribution across models, so forecasts are transparent to farmers/traders, not a black box
- **Advisory** (`src/advisory/`) — an LLM turns a forecast + its SHAP explanation into a farmer-friendly natural-language recommendation
- **Prototype** (`app/`) — a web/mobile interface surfacing forecasts, SHAP visualizations, and LLM advisory messages

Evaluation is via time-series cross-validation (walk-forward, `src.evaluation.metrics.time_series_splits`) plus a final untouched holdout, comparing models on MAE, RMSE, MAPE, and R².

## Project status

**Phase: forecasting models — implemented and runnable end-to-end.** SHAP explainability, LLM advisory, satellite/IoT/behavioral fusion, and the prototype UI are still deferred (scoped out of this phase deliberately, not forgotten).

Resolved decisions (were open questions earlier; now settled and reflected in code/config):
- **Forecast target**: `retail_price` (not wholesale) — `forecasting.target_column` in config.
- **Weather↔vegetable mapping**: each vegetable mapped to one representative growing district (`weather_district_map` in config) rather than a national average — see the table in the plan/commit history if the rationale needs revisiting.
- **LSTM framework**: TensorFlow/Keras (installed in `.venv`).
- **Fuel feature**: `diesel_price` only — `petrol92_price` was removed from `data/raw/fuel/fuel_data_weekly.csv` entirely (not just unused in code), since diesel is the transport-cost driver for produce trucks and petrol has no link to the vegetable supply chain (matches the literature).
- **Vegetable name mismatch**: raw CSV uses `BRINJALS`/`SNAKE GOURD`/etc.; `vegetable_name_map` in config bridges this to the lowercase config names.
- **Data alignment**: usable range across all 3 raw sources is 2015-01-05 to 2025-12-22 (fuel data is the binding start constraint, price data the binding end constraint) — `aligned_start_date`/`aligned_end_date` in config.

Known technical gotchas (if touching `src/models/sarimax/`):
- `simple_differencing=True` combined with `exog` breaks statsmodels' out-of-sample forecasting — it was tried for speed and produced forecasts that diverged to nonsensical values (confirmed empirically). Do not re-enable it without re-verifying holdout predictions stay in a sane range.
- Never pickle a fitted `SARIMAXResultsWrapper` directly — with `seasonal_order` period 52, it balloons to ~320MB per artifact (the Kalman filter's full state/covariance history). Use `SarimaxArtifact` (`src/models/sarimax/train.py`) instead: it stores just `params` + a small history frame and reconstructs via `.filter(params)`, at ~20KB with byte-identical forecasts. `HybridSarimaxResidualModel` (`src/models/hybrid_xgboost_sarimax/train.py`) already builds on this — don't regress it back to storing the raw results object.

## Model selection criteria

Two different criteria are used deliberately, for two different questions:

- **Best model family per vegetable** (the headline result): **holdout RMSE**. AIC is only well-defined for likelihood-based models like SARIMAX — there's no standard AIC for Random Forest, XGBoost, CatBoost, or an LSTM, so it can't be used to compare across these heterogeneous families. Out-of-sample error is the standard approach for this (as in the M-competitions).
- **SARIMAX's own (p,d,q) order**: **AIC**, via `scripts/select_sarimax_order.py` — standard Box-Jenkins order identification. Searches (p,1,q) for p,q ∈ {0,1,2} per vegetable (d=1 and seasonal_order=(1,0,1,52) held fixed, both already established as necessary/stable — see gotchas above), fits each candidate on the full series, and picks the AIC-minimizer. Results in `results/tables/sarimax_order_selection.csv`; the selected orders are wired into `configs/config.yaml` as `models.sarimax.order_by_vegetable`, which every SARIMAX-based model (`sarimax`, both hybrids) now reads per vegetable via `get_order_for_vegetable()`.
- **Important**: the AIC-selected order does not uniformly beat the arbitrary (1,1,1) baseline it replaced on holdout RMSE — it helped substantially for pumpkin (RMSE 110.8→57.5) and leeks, but slightly hurt carrot, brinjal, and snake_gourd. This is expected, not a bug: AIC measures in-sample fit quality, not out-of-sample generalization. It's still the methodologically correct way to *choose the order* — it just was never going to guarantee the best holdout score, which is exactly why the headline "best model" decision uses holdout RMSE instead.

Environment: project uses an isolated `.venv` (not system/Anaconda Python) — see Setup below. Verified working on Apple Silicon (M4).

**Latest full-grid results** (`results/tables/model_comparison.csv`, holdout RMSE, lower is better — regenerate with `python scripts/train_all.py`; SARIMAX-based columns use the AIC-selected per-vegetable order):

| vegetable | catboost | random_forest | xgboost | lstm | sarimax | hybrid_xgb+sarimax | hybrid_cb+sarimax |
|---|---|---|---|---|---|---|---|
| carrot | 114.4 | **133.1** | 184.2 | 182.9 | 315.8 | 321.2 | 289.0 |
| brinjal | 96.9 | **82.3** | 106.6 | 128.5 | 144.1 | 163.5 | 152.8 |
| pumpkin | 26.8 | **22.8** | 23.3 | 64.0 | 57.5 | 59.3 | 58.7 |
| cabbage | 62.2 | **42.2** | 50.6 | 104.0 | 98.8 | 115.7 | 113.9 |
| snake_gourd | **71.5** | 62.5 | 72.7 | 94.9 | 108.3 | 123.1 | 120.1 |
| leeks | **46.0** | 47.7 | 49.7 | 58.7 | 71.5 | 72.7 | 72.5 |

Random Forest and CatBoost dominate every vegetable; SARIMAX and both SARIMAX-hybrids remain the worst performers even with AIC-selected orders. This is a genuine finding, not a bug: SARIMAX's own fit is poor on the recent (volatile) holdout year (all its R² values are still negative except leeks, ~0.01), and because the hybrids are additive on top of that baseline, they inherit its error rather than correcting it — the residual-correction framing only helps when the underlying statistical model is reasonably good, which is why pumpkin (where AIC-tuning helped SARIMAX most) is also where the hybrids improved most. Worth a discussion-section paragraph in the thesis; don't read it as "the pipeline is broken."

**Forecast prediction rows** (`results/metrics/holdout_predictions.csv`, 2,184 = 6 vegetables × 7 models × 52 weeks) and **AIC order-search rows** (`results/tables/sarimax_order_selection.csv`, 54 = 6 vegetables × 9 candidate orders) should be regenerated together whenever the SARIMAX order changes — run `scripts/select_sarimax_order.py` first, update `order_by_vegetable` if orders shift, then `scripts/train_all.py`.

Still open / deferred to a later phase: LLM provider for the advisory module; how satellite NDVI / IoT / behavioral survey data will actually be sourced (GEE auth, IoT data format, survey instrument); SHAP explainability implementation; prototype UI framework choice.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Or with [uv](https://docs.astral.sh/uv/) (faster):

```bash
uv venv .venv --python 3.13
uv pip install -r requirements.txt --python .venv/bin/python
```

No `pyproject.toml`/`uv.lock` needed or wanted here — `requirements.txt` is the single source of truth for dependencies; don't let `uv init`/`uv add` create project files that duplicate or drift from it.

**Do not add `shap` back to `requirements.txt` casually.** It was in there for the (still-deferred, see Project status) explainability phase, but pulled in `numba`/`llvmlite`, and combined with the rest of this list, `uv`'s resolver picks `llvmlite==0.36.0` — which only supports Python <3.10 and fails to build on 3.13 (confirmed: `shap` alone resolves fine at `llvmlite==0.48.0`; it's a multi-package constraint intersection, not `shap` itself misbehaving). When the SHAP phase actually starts, add it back and re-verify `uv pip install -r requirements.txt` still succeeds before committing.

## Common commands

There is no build step (pure Python research code). Typical workflow:

```bash
# 1. Merge raw price/weather/fuel/satellite/iot/behavioral data into a per-vegetable processed dataset
python src/data_processing/build_dataset.py --vegetable carrot

# 2. Train a specific model for a specific vegetable
python src/models/sarimax/train.py --vegetable carrot
python src/models/lstm/train.py --vegetable carrot
python src/models/random_forest/train.py --vegetable carrot
python src/models/xgboost/train.py --vegetable carrot
python src/models/catboost/train.py --vegetable carrot
python src/models/hybrid_xgboost_sarimax/train.py --vegetable carrot
python src/models/hybrid_catboost_sarimax/train.py --vegetable carrot

# 3. Or train every model family for every vegetable in one go (writes results/metrics/all_results.csv
#    and results/tables/model_comparison.csv)
python scripts/train_all.py

# 4. Rebuild weather.csv from Open-Meteo (district-wise, 2014-present) if it's missing
python scripts/fetch_weather_openmeteo.py

# 5. Run tests
pytest tests/
pytest tests/test_features.py::test_lag_features   # single test
```

Model hyperparameters (SARIMAX order/seasonal_order, LSTM architecture, XGBoost/CatBoost/Random Forest params, exogenous feature lists) live in `configs/config.yaml` — do not hardcode them in training scripts.

## Architecture

**Data flow:** `data/raw/{vegetable_prices,weather,fuel,satellite,iot,behavioral}` (untouched source files, one subfolder per source) → `src/data_processing/build_dataset.py` joins them on date into `data/processed/<vegetable>.csv` → `src/features/feature_engineering.py` adds lags/rolling stats/calendar features (shared by every model, so a feature added here is available to all model families) → each `src/models/<model>/train.py` consumes the processed+featurized frame and writes a fitted model to `trained_models/<model>/<vegetable>.pkl` → `src/explainability/shap_explain.py` computes SHAP attributions on the fitted model → `src/advisory/llm_advisory.py` turns a forecast + its SHAP explanation into a farmer-facing message → `app/` surfaces all of it in a UI.

**Hybrid models** (`src/models/hybrid_*`) depend on the plain SARIMAX fit: they load/re-fit SARIMAX first, compute residuals (actual − SARIMAX prediction), then train the ML model on those residuals plus the exogenous features. Final forecast = SARIMAX prediction + ML residual prediction. Keep this two-stage structure intact — don't collapse it into a single end-to-end model.

**Per-vegetable, per-model artifacts:** every vegetable is modeled independently (no cross-vegetable pooling), and every model family is trained separately per vegetable. Expect `6 vegetables × 7 model families = 42` trained artifacts, plus corresponding metrics rows in `results/metrics/`.

**Evaluation:** `src/evaluation/metrics.py` is the single source of truth for MAE/RMSE/MAPE — all training scripts and notebooks should import from there rather than reimplementing metrics, so comparisons across model families stay apples-to-apples.

**Forecast verification:** `common.run_training` returns both the metrics dict and the holdout-period `(date, actual, predicted)` series; `scripts/train_all.py` concatenates the latter into `results/metrics/holdout_predictions.csv` (2,184 rows = 6 vegetables × 7 models × 52 holdout weeks). `notebooks/03_model_results.ipynb` plots these as forecast-vs-actual charts (and residuals), saved to `results/figures/forecast_vs_actual_*.png` — don't rely on RMSE/R² tables alone to sanity-check a model; look at the actual curve.

## Folder layout

- `data/raw/` — untouched source data (vegetable_prices, weather, fuel, satellite, iot, behavioral), one subfolder each
- `data/processed/` — merged, per-vegetable model-ready datasets
- `src/data_processing/` — raw → processed merging
- `src/features/` — shared feature engineering used by all models
- `src/models/{sarimax,lstm,random_forest,xgboost,catboost,hybrid_xgboost_sarimax,hybrid_catboost_sarimax}/` — one training module per model family
- `src/explainability/` — SHAP-based feature attribution, shared across model families
- `src/advisory/` — LLM-based advisory message generation from forecast + explanation
- `src/evaluation/` — shared metrics and cross-model comparison
- `trained_models/<model_family>/` — serialized fitted models, one file per vegetable
- `results/{figures,tables,metrics,explainability}/` — plots, comparison tables, metric outputs, and SHAP visuals for the paper
- `app/` — prototype forecast + explanation + advisory interface (web/mobile)
- `configs/config.yaml` — vegetables list, data paths, per-model hyperparameters, vegetable/district name mappings
- `scripts/train_all.py` — orchestrates all 7 model families x 6 vegetables, writes the combined results tables
- `scripts/fetch_weather_openmeteo.py` — repopulates `data/raw/weather/weather.csv` from Open-Meteo
- `notebooks/` — exploratory analysis; production logic belongs in `src/`, not notebooks. `01_data_exploration.ipynb`, `02_feature_engineering.ipynb`, `03_model_results.ipynb` (the last needs `results/metrics/all_results.csv` from `scripts/train_all.py` to exist first)
- `research-papers/references/` — literature (PDFs, papers) informing the methodology; gitignored (not checked in), fetch again from `## Literature status` below if missing
- `research-papers/drafts/thesis/` — the thesis chapters (`01_introduction.md`, `02_literature_review.md`, `03_methodology.md`, ...)

## Documentation stays in sync with research

This is a research project, not just a codebase — the docs are part of the deliverable. Whenever research content changes (new papers found, scope/objectives change, a literature claim is added or superseded, methodology decisions are made), update the relevant doc in the same session, don't leave it for later:
- New objectives, model families, or data sources → update `## Project` above and the folder layout
- New papers read → update `research-papers/drafts/thesis/02_literature_review.md` (and its References list) and the `## Literature status` list below
- New introduction/problem-statement framing → update `research-papers/drafts/thesis/01_introduction.md`

## Literature status

Sourced and read (in `research-papers/references/`, cited in `02_literature_review.md`): Madubhashini (2021/2023) — Sri Lanka, first SL-specific ML study; Ranaweera, Rathnayake & Ananda (2023) — Sri Lanka, beans/brinjal/carrot/pumpkin, RF best, same 4 exogenous features (rainfall, temperature, fuel price, production) as this project's config; Ruhunuge et al. (2024) — Sri Lanka, carrot-specific VAR climate causality; Weerasekara et al. (2026) — Sri Lanka, most advanced prior work (multi-market/season/regime XGBoost+LightGBM); Paul et al. (2022) — brinjal, India; Zhao et al. (2025) — TCN-XGBoost hybrid; Patil et al. (2023) — HySALS hybrid SARIMA-LSTM; Mayank, Shelke & Roy (2025); Shree Sanjay & Janarthanan (2025).

**Not yet sourced** — needed before objectives 3–4 can be written up: explainable AI (SHAP) applied to price/agricultural forecasting; LLM-based farmer advisory systems. Do not write those literature subsections from memory — search and read first.
