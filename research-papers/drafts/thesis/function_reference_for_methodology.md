# Function Reference for the Methodology Chapter

This is a working reference, not a thesis chapter itself. It lists every function/class actually
defined in the research pipeline (`src/`, `scripts/`) and says exactly where in `Final Thesis.docx`
each one should be cited, so the methodology chapter accurately describes the real implementation.
Use it as a checklist while editing the thesis by hand.

Two corrections were already applied directly to `Final Thesis.docx` (§3.6 intro, the model table,
and §3.6.2.1's hyperparameter-tuning table/paragraph): the model count was fixed from "seven" to
"nine" (the two newer hybrids — Random Forest-SARIMAX and LSTM-SARIMAX — were missing), and the
hyperparameter-tuning section was rewritten because it previously implied a real validation-based
search happened for Random Forest/XGBoost/CatBoost/LSTM, when in fact **no automated search exists
anywhere in the codebase** (confirmed by grepping for `GridSearchCV`/`optuna`/`hyperopt`/etc. — zero
matches) — every one of those models' hyperparameters is a fixed value read from `configs/config.yaml`.
Only SARIMAX's order underwent a real AIC search. Everything below is new — not yet added to the thesis.

---

## 1. Full function inventory, by pipeline stage

### 1.1 Data collection (`src/pipeline/scrapers/`)

| Function | File | Purpose |
|---|---|---|
| `fetch_pdf(url, retries, timeout)` | `common.py` | Shared HTTP-with-retry PDF download helper used by both scrapers. |
| `extract_numbers(text)` | `common.py` | Pulls numeric tokens out of scraped PDF text. |
| `_fetch_index_html()` | `harti_prices.py` | Loads the HARTI bulletin index page. |
| `list_available_weeks(year, html)` | `harti_prices.py` | Lists bulletin weeks available for a year. |
| `find_bulletin_url(week, year, html)` | `harti_prices.py` | Resolves the actual PDF link for a given week (file names aren't predictable from the date alone). |
| `_parse_row(line)`, `_parse_table_block(...)`, `_month_number(name)` | `harti_prices.py` | Low-level PDF text parsing helpers for the price table. |
| `extract_week_end(pdf_bytes)` | `harti_prices.py` | Reads the bulletin's stated week-ending date. |
| `parse_bulletin(pdf_bytes, week_start)` | `harti_prices.py` | Turns one bulletin PDF into a price DataFrame (midpoint-of-range averaging happens here). |
| `fetch_and_parse(week, year, week_start)` | `harti_prices.py` | Combines fetch + parse for one bulletin. |
| `_state_path/_load_state(...)` | `harti_prices.py` | Tracks which bulletins were already processed, so re-runs only fetch new ones. |
| `update_harti_prices(config, years, dry_run)` | `harti_prices.py` | Top-level driver: fetches all new bulletins and appends to the raw price CSV. |
| `fetch_revision_table()` | `cpc_fuel.py` | Scrapes CEYPETCO's diesel price revision history. |
| `to_weekly_series(revisions, start, end)` | `cpc_fuel.py` | Forward-fills revision-dated prices into a weekly series. |
| `update_fuel_prices(config, dry_run)` | `cpc_fuel.py` | Top-level driver for the fuel dataset. |
| `fetch_report/parse_report/fetch_and_parse` | `cbsl_prices.py` | Equivalent scraper for a CBSL price source (secondary/backup source — confirm with supervisor whether this is actually used in the reported results before citing it). |

### 1.2 Weather collection (`scripts/fetch_weather_openmeteo.py`)

| Function | Purpose |
|---|---|
| `fetch_district_daily(district, lat, lon, end_date, retries)` | Pulls daily max/min temperature + precipitation from Open-Meteo for one district. |
| `to_weekly(df)` | Aggregates daily weather into the weekly average-temperature/rainfall series used by the model. |
| `fetch_district_forecast(district, lat, lon, days)` | Real short-range forecast (used only by future-forecasting, not by training). |
| `climatology_estimate(district, iso_week, weather_csv_path)` | Historical per-ISO-week average, used as a fallback when no real forecast/actual value is available yet. |

### 1.3 Data quality checks (`src/pipeline/quality_checks.py`)

| Function | Purpose |
|---|---|
| `find_missing_weeks(dates, freq)` | Detects gaps in a weekly date index. |
| `detect_outliers(...)` | Flags implausible values in a raw source. |
| `align_price_schema(new_df, config)` | Normalises a newly scraped price batch to the canonical schema before merging. |
| `validate_and_merge(existing_path, new_df, ...)` | Merges new rows into an existing raw CSV with validation. |
| `audit_source(name, path, ...)` | Runs the above checks over one raw source and reports issues. |

### 1.4 Data integration (`src/data_processing/build_dataset.py`)

| Function | Purpose |
|---|---|
| `load_prices(vegetable, config)` | Reads and filters the raw price CSV for one vegetable. |
| `load_fuel(config)` | Reads the weekly diesel price series. |
| `load_weather(vegetable, config)` | Reads and filters the weekly weather series for that vegetable's representative district. |
| `build_dataset(vegetable, config)` | Inner-joins the three sources on week date and clips to the aligned date range — this is **the** data-integration function. |

### 1.5 Feature engineering (`src/features/feature_engineering.py`)

| Function | Purpose |
|---|---|
| `build_features(df, config)` | Builds all 20 model inputs: 4 price lags, 6 rolling mean/std stats, 3 calendar features, 1 season flag, 3 lagged exogenous features (contemporaneous exogenous values are also retained) — drops leakage by shifting before computing rolling stats. |
| `get_feature_columns(df)` | Returns the feature-column name list every model consumes. |

### 1.6 Evaluation metrics (`src/evaluation/metrics.py`)

| Function | Purpose |
|---|---|
| `mae(y_true, y_pred)`, `rmse(...)`, `mape(...)`, `r2(...)` | The four error measures reported throughout Ch.3–4. |
| `evaluate(y_true, y_pred)` | Bundles all four into one dict — this is what every training script actually calls. |
| `time_series_splits(n, n_splits, holdout)` | Builds the walk-forward CV folds **and** the final untouched holdout split — this is the function implementing the "time-based division" described in §3.6.2.2. |
| `prediction_interval(residuals, confidence)` | Empirical quantile interval from pooled CV residuals (the 80% band mentioned in §3.7.2). |
| `interval_coverage(y_true, lower, upper)` | The empirical coverage check reported alongside RMSE/MAE. |

### 1.7 Shared training pipeline (`src/models/common.py`)

| Function | Purpose |
|---|---|
| `load_features(vegetable, config)` | Loads the processed+featurised frame for one vegetable. |
| `run_training(vegetable, model_family, config, fit_predict, fit_final, keras)` | **The core shared procedure every one of the nine model families runs through** — builds CV folds, fits/scores each fold, pools residuals for the prediction interval, refits on the holdout split, refits again on all data for the production artefact, and saves it. This is the single most important function to cite in §3.6.2.2, since it's what actually guarantees every model family was evaluated identically. |
| `run_for_vegetables(vegetables, model_family, config, fit_predict, fit_final, keras)` | Loops `run_training` across all six vegetables for one model family. |

### 1.8 Per-model training modules (`src/models/<family>/train.py`)

Each of the 9 model directories follows the same three-function shape — cite this pattern once
rather than nine times:

| Function pattern | Purpose |
|---|---|
| `_make_model(config)` (RF/XGBoost/CatBoost) or `_build_model(n_features, config)` (LSTM) | Constructs the estimator with the fixed hyperparameters from `configs/config.yaml`. |
| `make_fit_predict(config)` / `make_fit_final(config)` | Adapter closures that give `run_training()` a uniform `fit_predict`/`fit_final` interface regardless of the underlying library (scikit-learn, XGBoost, CatBoost, Keras, statsmodels). |
| `train_all_vegetables(config, vegetables)` | Entry point called by `scripts/train_all.py` for that one model family. |

Model-specific internals worth citing individually:

| Function/class | File | Purpose |
|---|---|---|
| `SarimaxArtifact` (class), `_fit()`, `get_order_for_vegetable()` | `sarimax/train.py` | Fits `statsmodels` SARIMAX; stores a lightweight picklable artefact instead of the full ~320MB results object; looks up the AIC-selected order per vegetable. |
| `_scale()`, `_build_sequences()`, `_inverse_target()`, `_build_model()`, `_fit_lstm()`, `_predict()` | `lstm/train.py` | Scaling, sequence windowing, network construction, training-with-early-stopping, and inverse-scaling for LSTM. |
| `HybridSarimaxResidualModel` (class), `_make_residual_model()`, `_fit_hybrid()` | `hybrid_xgboost_sarimax/train.py` (reused by the CatBoost and Random Forest hybrids) | Fits SARIMAX, computes residuals, fits the residual corrector, and sums both forecasts at prediction time. |
| `HybridSarimaxLstmModel` (class), `_fit_hybrid()` | `hybrid_lstm_sarimax/train.py` | Same idea but for the LSTM residual corrector; includes `resid_clip` to guard against a rare SARIMAX divergence overflowing the LSTM's float32 scaler. |

### 1.9 SARIMAX order search (`scripts/select_sarimax_order.py`)

| Function | Purpose |
|---|---|
| `search_vegetable(vegetable, config)` | Fits all 9 candidate `(p,1,q)` orders (`p,q ∈ {0,1,2}`) with the seasonal order fixed, records AIC/BIC/HQIC — **this is the only real automated hyperparameter search anywhere in the project.** |
| `main()` | Runs the search for every vegetable, writes `results/tables/sarimax_order_selection.csv`, prints the AIC-minimising order per vegetable (which is then hand-copied into `config.yaml`). |

### 1.10 Orchestration and inference

| Function | File | Purpose |
|---|---|---|
| `main()` / `MODEL_TRAINERS` | `scripts/train_all.py` | Runs all 9 families × 6 vegetables, writes the combined metrics/comparison tables. |
| `recursive_forecast()`, `build_future_exog()`, `_build_future_row()`, `_holdout_interval()` | `src/inference/future_forecast.py` | Genuine future forecasting beyond the holdout period (recursive one-step chaining for ML models; single multi-step call for SARIMAX-family models) — relevant to §3.8 (Forecasting Instrument), not §3.6. |

---

### 1.11 Data Collection Functions per Variable (Summary Table)

The three data-collection subsections (§3.3.1–3.3.3) each cover one variable and one source. The
table below consolidates §1.1–§1.2 above into a single per-variable view, showing the exact
sequence of functions each variable's raw data passes through, from retrieval to the final weekly
value used by the model.

| Variable | Source | Function(s) used | File | How it is used |
|---|---|---|---|---|
| Retail / wholesale price | HARTI weekly bulletins | `update_harti_prices()` → `_fetch_index_html()` → `list_available_weeks()` → `find_bulletin_url()` → `fetch_and_parse()` → `parse_bulletin()` | `src/pipeline/scrapers/harti_prices.py` | `update_harti_prices()` drives the whole process: it lists which weekly bulletins are available via `_fetch_index_html()`/`list_available_weeks()`, resolves each bulletin's PDF link with `find_bulletin_url()` (since file names are not predictable from the date alone), and downloads and converts each one to a price row with `fetch_and_parse()`, which internally calls `parse_bulletin()` to read the price table, `extract_week_end()` to confirm the bulletin's week-ending date, and `_parse_table_block()`/`_parse_row()` to extract and average the reported price ranges. `_load_state()`/`_state_path()` record which bulletins were already processed, so a re-run only fetches new ones. |
| Average temperature | Open-Meteo API (daily max/min temperature, per representative district) | `fetch_district_daily()` → `to_weekly()` | `scripts/fetch_weather_openmeteo.py` | `fetch_district_daily()` requests daily maximum and minimum temperature for one district's coordinates; `to_weekly()` averages the daily midpoint temperature (mean of daily max and min) into one weekly average-temperature value per district. |
| Average rainfall | Open-Meteo API (daily precipitation, per representative district) | `fetch_district_daily()` → `to_weekly()` | `scripts/fetch_weather_openmeteo.py` | The same `fetch_district_daily()` call also retrieves daily precipitation alongside temperature; `to_weekly()` sums the daily precipitation values within each week to produce the weekly rainfall total used as a feature. |
| Diesel price | CEYPETCO revision history | `update_fuel_prices()` → `fetch_revision_table()` → `to_weekly_series()` | `src/pipeline/scrapers/cpc_fuel.py` | `update_fuel_prices()` drives the process: `fetch_revision_table()` retrieves the revision-dated diesel price history, and `to_weekly_series()` forward-fills the most recently active price onto every week between one revision date and the next, producing one weekly diesel-price value. |
| All three variables (validation, before use) | — | `align_price_schema()`, `detect_outliers()`, `find_missing_weeks()`, `validate_and_merge()`, `audit_source()` | `src/pipeline/quality_checks.py` | Each newly collected batch is normalised to the project's schema by `align_price_schema()`, checked for outliers with `detect_outliers()` and gaps with `find_missing_weeks()`, then merged into the existing raw CSV by `validate_and_merge()`; `audit_source()` runs this whole check for one source and reports any issue found. |

*Table: Data collection functions used per variable, from retrieval to the final weekly value*

---

## 2. Where to add these in `Final Thesis.docx`

| Thesis section | What to add | Which functions to name |
|---|---|---|
| **§3.3 Data Collection** (3.3.1–3.3.3) | One sentence per subsection naming the driver function, e.g. "retrieved via `update_harti_prices()`, which calls `fetch_and_parse()` per bulletin." | `update_harti_prices()`, `parse_bulletin()`, `update_fuel_prices()`, `to_weekly_series()`, `fetch_district_daily()`, `to_weekly()` |
| **§3.4 Data Preprocessing** (3.4.2 Data Integration, 3.4.3 Feature Engineering) | Name the exact join/feature functions instead of describing the logic only in prose. | `build_dataset()` (3.4.2); `build_features()`, `get_feature_columns()` (3.4.3) |
| **§3.6.1 SARIMAX** (already describes the math) | Add one sentence after the order-selection table: "implemented in `_fit()` and `get_order_for_vegetable()` (`src/models/sarimax/train.py`), with the fitted model stored via the lightweight `SarimaxArtifact` wrapper rather than the full results object." | `_fit()`, `get_order_for_vegetable()`, `SarimaxArtifact` |
| **§3.6.2 Machine-Learning/Deep-Learning/Hybrid Models** (intro, before 3.6.2.1) | Add a short paragraph: "All nine model families were trained through one shared procedure, `run_training()` in `src/models/common.py`, so that every family received an identical time-based split, feature set, and evaluation process — only the model-fitting step (`_make_model()`/`_build_model()`, described per family below) differs." This is the single highest-value addition — it's the function that makes the whole cross-model comparison methodologically sound, and it isn't mentioned anywhere currently. | `run_training()`, `run_for_vegetables()` |
| **§3.6.2.1 Hyperparameter Tuning** (already corrected in the docx) | Optionally add the exact fixed values as a new small table (values below) right after the corrected paragraph, since the existing table only names *which* hyperparameters exist, not their actual values. | n/a — this is data, not a function, but pairs with `_make_model()`/`_build_model()` above |
| **§3.6.2.2 Model Training and Time-Based Validation** | Name the split function directly: "implemented by `time_series_splits()` (`src/evaluation/metrics.py`), which builds the walk-forward folds and the final holdout split from one call." | `time_series_splits()` |
| **§3.7.1 Performance Metrics** | Add one line: "computed by `mae()`, `rmse()`, `mape()`, `r2()`, and bundled by `evaluate()` in `src/evaluation/metrics.py`, so every model family's metrics are computed identically." | `mae()`, `rmse()`, `mape()`, `r2()`, `evaluate()` |
| **§3.7.2 Comparison and Selection** (prediction interval paragraph) | Name the interval functions: "computed by `prediction_interval()` from pooled cross-validation residuals, with `interval_coverage()` reporting the holdout coverage check." | `prediction_interval()`, `interval_coverage()` |
| **§3.8 Development of the Forecasting Instrument** | If this section describes how the prototype produces a forecast beyond the holdout period, name `recursive_forecast()`/`build_future_exog()` here — these aren't part of model training at all, so don't cite them in §3.6. | `recursive_forecast()`, `build_future_exog()` |

### Suggested exact hyperparameter values for the §3.6.2.1 table addition

| Model | Hyperparameter | Value |
|---|---|---|
| Random Forest | `n_estimators` / `max_depth` / `min_samples_leaf` | 500 / 10 / 2 |
| XGBoost | `n_estimators` / `max_depth` / `learning_rate` | 500 / 6 / 0.05 |
| CatBoost | `iterations` / `depth` / `learning_rate` | 1000 / 6 / 0.05 |
| LSTM | `lookback` / `layers` / `units` / `dropout` / `batch_size` / `epochs` / `learning_rate` | 30 / 2 / 64 / 0.2 / 32 / 100 (early-stopped) / 0.001 |
| SARIMAX | `seasonal_order` (fixed) / non-seasonal order (AIC-selected) | (1,0,1,52) / see §3.6.1.2 table |

---

## 3. Drafted sentences, ready to insert

These are complete sentences, not a description of what to write — they can be copied directly
into the corresponding subsection of `Final Thesis.docx`. All are written in past passive tense
without a stated subject, matching the voice already used throughout the methodology chapter.

### §3.3.1 Vegetable Price Data from HARTI

New weekly bulletins were retrieved and parsed by `update_harti_prices()`, which located each
unprocessed bulletin, downloaded it, and passed it to `fetch_and_parse()`. Each bulletin was
converted into a price table by `parse_bulletin()`, using `extract_week_end()` to read the
bulletin's stated week-ending date and `_parse_table_block()` to extract the relevant rows. Prices
already processed in an earlier run were tracked by `_load_state()`, so that only new bulletins
were fetched on each subsequent run.

### §3.3.2 Weather Data from Open-Meteo

Daily temperature and precipitation data were retrieved for each representative district by
`fetch_district_daily()`, and were aggregated into the weekly average-temperature and rainfall
series used by the model by `to_weekly()`.

### §3.3.3 Diesel Price Data from CEYPETCO

The diesel price revision history was retrieved by `fetch_revision_table()`, and was converted
into a continuous weekly series by `to_weekly_series()`, which forward-filled each revision-dated
price until the next revision.

### §3.4.2 Data Integration

The three weekly sources were merged by `build_dataset()`, which called `load_prices()`,
`load_weather()`, and `load_fuel()` in turn, joined the results on the common week date, and
restricted the merged frame to the aligned date range shared by all three sources.

### §3.4.3 Data Transformation and Feature Engineering

The twenty model inputs listed in Table 3.2 were generated by `build_features()`, and the
resulting feature-column names were returned by `get_feature_columns()` for use by every model
family. Rolling and lagged statistics were computed only from values shifted one week backward
within `build_features()`, so that no feature was derived from information unavailable at the
time of the forecast.

### §3.6.1.2 / 3.6.1.3 SARIMAX Parameter Selection and Fitting

The non-seasonal order for each vegetable was selected by `search_vegetable()`
(`scripts/select_sarimax_order.py`), which fitted every candidate `(p, 1, q)` order for
`p, q ∈ {0, 1, 2}` and recorded its AIC, BIC, and HQIC. The order that minimised AIC was then set
for each vegetable and read at training time by `get_order_for_vegetable()`. The SARIMAX model
itself was fitted by `_fit()` (`src/models/sarimax/train.py`), and the fitted result was stored as
a lightweight `SarimaxArtifact` rather than the full results object, to avoid the large file size
associated with the fitted seasonal state.

### §3.6.2 Machine-Learning, Deep-Learning, and Hybrid Models (introductory paragraph)

All nine model families were trained through one shared procedure, `run_training()`
(`src/models/common.py`), so that every family was evaluated with an identical time-based split,
feature set, and evaluation process, and only the model-fitting step was allowed to differ between
families. This procedure was repeated across all six vegetables by `run_for_vegetables()`.

### §3.6.2.1 Hyperparameter Tuning

Random Forest, XGBoost, and CatBoost regressors were constructed by `_make_model()`, and the LSTM
network was constructed by `_build_model()`, each using the fixed hyperparameter values declared
in the project configuration file rather than values selected by an automated search. The
`fit_predict` and `fit_final` interfaces required by `run_training()` were produced for each model
family by `make_fit_predict()` and `make_fit_final()`.

### §3.6.2.2 Model Training and Time-Based Validation

The walk-forward cross-validation folds and the final holdout split were constructed by
`time_series_splits()` (`src/evaluation/metrics.py`), so that chronological order was preserved
throughout training and evaluation.

### §3.7.1 Performance Metrics

Mean Absolute Error, Root Mean Square Error, Mean Absolute Percentage Error, and the coefficient of
determination were computed by `mae()`, `rmse()`, `mape()`, and `r2()` respectively, and were
bundled into a single result by `evaluate()`, so that every model family's metrics were computed in
an identical manner.

### §3.7.2 Comparison and Selection of Forecasting Models

The 80% prediction interval was derived from pooled cross-validation residuals by
`prediction_interval()`, and the proportion of holdout actuals falling within that interval was
measured by `interval_coverage()`.

### §3.8 Development of the Forecasting Instrument

Forecasts beyond the holdout period were produced by `recursive_forecast()`
(`src/inference/future_forecast.py`), with future weather and diesel values resolved by
`build_future_exog()` and each new weekly row constructed by `_build_future_row()`.

---

## 4. Style note (how to cite a function in running academic prose)

Standard practice in applied CS/ML theses is an inline citation the first time a function is
introduced, not a footnote: name the function in `code font`, followed by its file path in
parentheses the first time only, e.g.

> "...each model family's hyperparameters were fixed values read by `_make_model()`
> (`src/models/random_forest/train.py`) directly from the project configuration file, rather than
> selected by an automated search."

Don't repeat the file path on every later mention of the same function — name it once per section.
Keep function names in a monospace/code font throughout (consistent with how `Final Thesis.docx`
already renders code-like terms), and reserve full signatures (with parameter lists) for a table,
not for prose.

---

Compiled from the project's source code and mapped to their corresponding methodology subsections, so that each programmatic function is incorporated into the proper place within the thesis.

The programmatic functions identified above were added to their corresponding sections of the methodology chapter, in accordance with the mapping presented in Section 2.
