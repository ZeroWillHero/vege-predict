# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**Adaptive, Trustworthy Vegetable Price Forecasting Using Multi-Source Data and AI in Sri Lanka** — UWU ICT 481-6 capstone (supervisors: Dr. Niranjan W. Gunasekara, Ms. KRR Premathilaka). Fuses historical prices, weather, fuel prices, satellite NDVI (Sentinel-2 via Google Earth Engine), IoT sensor readings, and farmer behavioral survey data to forecast short-term vegetable prices, explain the forecasts, and turn them into farmer-facing advice.

Target vegetables: **carrot, brinjal, pumpkin, cabbage, snake gourd, leeks**.

**Forecasting models** (proposal-mandated: SARIMA, LSTM, Random Forest, XGBoost, plus hybrids; CatBoost and its hybrid are an extended baseline beyond the proposal, kept from the original scaffold; the SARIMAX+Random Forest and SARIMAX+LSTM hybrids were added afterward at supervisor request):
- **SARIMAX** — linear/seasonal component, weather+fuel as exogenous regressors
- **LSTM** — sequence model over price + exogenous history
- **Random Forest** — lag/rolling/calendar features
- **XGBoost**, **CatBoost** — same feature set as Random Forest
- **Hybrid XGBoost+SARIMAX**, **Hybrid CatBoost+SARIMAX**, **Hybrid Random Forest+SARIMAX**, **Hybrid LSTM+SARIMAX** — one shared design across all four: SARIMAX models the linear/seasonal component; the second model (XGBoost, CatBoost, Random Forest, or LSTM) is trained on SARIMAX's residuals; final forecast = SARIMAX prediction + residual prediction. The LSTM variant (`hybrid_lstm_sarimax`) reuses the plain LSTM's scaling/windowing code unchanged, just handing it a frame where `"target"` is the SARIMAX residual instead of price, and clips the residual to a bounded range before scaling — SARIMAX's out-of-sample forecast occasionally diverges on a small/early CV fold (a known SARIMAX fragility, see gotchas below), and the tree-based residual models tolerate that silently while the LSTM's float32 scaling path doesn't without the clip. See `research-papers/drafts/thesis/03_methodology.md` §3.10.6 for the full write-up.

**Beyond forecasting**, the proposal's remaining objectives:
- **Explainability** (`src/explainability/`) — SHAP feature attribution across models, so forecasts are transparent to farmers/traders, not a black box
- **Advisory** (`src/advisory/`) — an LLM turns a forecast + its SHAP explanation into a farmer-friendly natural-language recommendation
- **Prototype** (`app/`) — a web/mobile interface surfacing forecasts, SHAP visualizations, and LLM advisory messages. Its backend (`app/backend/`) is built: a FastAPI + Postgres + Redis API serving pre-computed forecasts, model comparisons, and historical prices — see `## Backend API` below. The frontend itself is still deferred.

Evaluation is via time-series cross-validation (walk-forward, `src.evaluation.metrics.time_series_splits`) plus a final untouched holdout, comparing models on MAE, RMSE, MAPE, and R².

## Project status

**Phase: forecasting models — implemented and runnable end-to-end. Backend API (`app/backend/`) — implemented and runnable end-to-end. Genuine future forecasting (`scripts/predict_future.py`) — implemented and runnable end-to-end.** SHAP explainability, LLM advisory, satellite/IoT/behavioral fusion, the auto-retrain pipeline, and the prototype frontend UI are still deferred (scoped out of this phase deliberately, not forgotten).

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
- SARIMAX's out-of-sample `.forecast()` can occasionally diverge to an extreme value on a small/early walk-forward CV fold (confirmed empirically: cabbage's second CV fold produced a forecast around 3.9e38 with the AIC-selected order) — the tree-based residual hybrids tolerate this silently (one bad-but-finite fold score), but `hybrid_lstm_sarimax` (`src/models/hybrid_lstm_sarimax/train.py`) casts the residual to float32 for scaling, and a value that large overflows to `inf`, which crashes `StandardScaler`. `HybridSarimaxLstmModel.resid_clip` (10× the in-sample residual std, computed at fit time) bounds this before scaling — don't remove it without re-verifying `scripts/train_all.py` runs clean across all 6 vegetables.
- **Genuine future forecasting must call SARIMAX-based `.forecast()` exactly once per vegetable/family, covering the whole horizon in one call — never once per week.** `SarimaxArtifact.forecast(future_df)` reconstructs the filter from the original training data and calls `get_forecast(steps=len(future_df), ...)`; calling it repeatedly with a 1-row `future_df` would make every future week look like the very next week after training ends (a 1-step forecast), not N steps ahead — silently wrong, no error raised. `src/inference/future_forecast.py::recursive_forecast()` gets this right (confirmed by testing: its SARIMAX-family output matches a direct single multi-step `.forecast()` call exactly) — preserve this when touching that module.

## Future forecasting (`scripts/predict_future.py`)

Genuinely predicts `forecasting.future_horizon_weeks` weeks (default 8, ~2 months) past **today's real date**, for all 9 model families, reusing each family's already-persisted production artifact (`trained_models/<family>/<vegetable>.*` from `scripts/train_all.py` — no retraining). Implemented in `src/inference/future_forecast.py`:
- **Horizon is anchored to today, not to the last known price date** — `build_future_exog()` computes `weeks_to_bridge = (today_week_start - last_known_date) // 7` and generates `weeks_to_bridge + future_horizon_weeks` total weeks. This matters because the price data can lag behind real time by a month or more (confirmed: at one point price data ended 2026-06-22 while today was 2026-07-29, a 5-week gap) — anchoring purely to the last known date would silently produce "future" weeks that are already in the past by the time anyone actually looks at them. The gap weeks are bridged automatically (recursive lag features can't skip them anyway) rather than skipped, so the response includes both the catch-up weeks and the genuinely-forward ones — don't "simplify" this back to a fixed offset from the last known date.
- **Recursive one-step chaining**: each week's own prediction is fed back in as the next week's `price_lag_1`/etc. (via `_build_future_row()`, which replicates `feature_engineering.py`'s lag/rolling/calendar formulas exactly for a single new row — reusing the constants `LAG_WEEKS`/`ROLLING_WINDOWS`/`EXOG_COLUMNS` directly rather than duplicating them). SARIMAX-based families are the one exception — see the gotcha above.
- **Future weather/diesel resolved in a three-tier cascade**, cheapest/most-accurate first, in `build_future_exog()`: (1) `data/raw/weather/weather.csv`/`fuel_data_weekly.csv` are refreshed independently of price data and are often already ahead of it — if a future week's real value is already sitting there, use it directly, no estimation at all; (2) Open-Meteo's real forecast endpoint (`scripts/fetch_weather_openmeteo.py::fetch_district_forecast()`, ~16 days out, fetched lazily/at most once — only called if tier 1 doesn't cover everything); (3) `climatology_estimate()` — that ISO week's historical average per district — for anything further out. Diesel price forward-fills the last known CPC revision if the horizon exceeds available fuel data (diesel only changes on periodic revisions, so this is usually accurate, not a guess). Deliberately not replaced with dedicated forecasting models for either: Open-Meteo's forecast is already better than anything an in-house model could produce from this project's own small weather archive, and diesel is a policy-driven step function with no learnable signal in its own price history alone (no forex/subsidy/global-oil-price inputs in this project's data) — carry-forward is the correct expectation between revisions, not a shortcut.
- **Prediction interval uses holdout residuals, not CV-fold residuals** — `_holdout_interval()` derives the ± offset from `results/metrics/holdout_predictions.csv`'s already-persisted `actual - predicted` for that vegetable/family, via the same `prediction_interval()` function the backtest path uses. CV-fold residuals aren't persisted anywhere and recomputing them would mean re-running the full walk-forward CV — a deliberate, documented substitution, not an oversight.
- Output: `results/metrics/future_predictions.csv`, same shape as `holdout_predictions.csv` minus `actual` (genuinely unknown). `app/backend/seed.py::seed_forecasts()` reads this file too (if present) and upserts into the same `forecast` table — `actual_price` ends up `NULL` for these rows (confirmed via the live API), not `NaN`, because of the `pd.isna()` guard added alongside this feature (a real pre-existing bug: `actual_price` used to be assigned straight from the CSV with no NaN check, unlike `predicted_lower`/`predicted_upper` a few lines above it — a future row's missing actual would have silently become Postgres `NaN` instead of `NULL`, breaking the "null for unresolved weeks" API contract before anything ever exercised that path). No Alembic migration was needed — `actual_price` was already nullable.

## Model selection criteria

Two different criteria are used deliberately, for two different questions:

- **Best model family per vegetable** (the headline result): **holdout RMSE**. AIC is only well-defined for likelihood-based models like SARIMAX — there's no standard AIC for Random Forest, XGBoost, CatBoost, or an LSTM, so it can't be used to compare across these heterogeneous families. Out-of-sample error is the standard approach for this (as in the M-competitions).
- **SARIMAX's own (p,d,q) order**: **AIC**, via `scripts/select_sarimax_order.py` — standard Box-Jenkins order identification. Searches (p,1,q) for p,q ∈ {0,1,2} per vegetable (d=1 and seasonal_order=(1,0,1,52) held fixed, both already established as necessary/stable — see gotchas above), fits each candidate on the full series, and picks the AIC-minimizer. Results in `results/tables/sarimax_order_selection.csv`; the selected orders are wired into `configs/config.yaml` as `models.sarimax.order_by_vegetable`, which every SARIMAX-based model (`sarimax`, both hybrids) now reads per vegetable via `get_order_for_vegetable()`.
- **Important**: the AIC-selected order does not uniformly beat the arbitrary (1,1,1) baseline it replaced on holdout RMSE — it helped substantially for pumpkin (RMSE 110.8→57.5) and leeks, but slightly hurt carrot, brinjal, and snake_gourd. This is expected, not a bug: AIC measures in-sample fit quality, not out-of-sample generalization. It's still the methodologically correct way to *choose the order* — it just was never going to guarantee the best holdout score, which is exactly why the headline "best model" decision uses holdout RMSE instead.

Environment: project uses an isolated `.venv` (not system/Anaconda Python) — see Setup below. Verified working on Apple Silicon (M4).

**Latest full-grid results** (`results/tables/model_comparison.csv`, holdout RMSE, lower is better — regenerate with `python scripts/train_all.py`; SARIMAX-based columns use the AIC-selected per-vegetable order; holdout is always the most recent 52 weeks of whatever data exists, so it shifts forward as new data is ingested — not directly comparable to earlier snapshots of this table):

| vegetable | catboost | random_forest | xgboost | lstm | sarimax | hybrid_xgb+sarimax | hybrid_cb+sarimax | hybrid_rf+sarimax | hybrid_lstm+sarimax |
|---|---|---|---|---|---|---|---|---|---|
| carrot | 112.7 | **100.9** | 108.2 | 216.2 | 245.7 | 254.6 | 243.9 | 249.9 | 469.5 |
| brinjal | 98.0 | **90.7** | 100.5 | 149.8 | 124.2 | 128.3 | 131.4 | 128.7 | 213.9 |
| pumpkin | 26.8 | **22.8** | 23.3 | 49.5 | 57.5 | 59.3 | 58.7 | 58.9 | 109.4 |
| cabbage | 62.0 | **48.2** | 49.9 | 112.7 | 110.8 | 116.8 | 115.1 | 114.3 | 203.6 |
| snake_gourd | 71.5 | **62.5** | 72.7 | 91.6 | 108.3 | 123.1 | 120.1 | 122.0 | 181.6 |
| leeks | **48.5** | 50.3 | 51.8 | 68.6 | 95.7 | 98.7 | 95.4 | 95.4 | 159.6 |

Random Forest wins 5/6 vegetables; CatBoost wins leeks narrowly. SARIMAX and all four SARIMAX-hybrids remain the worst performers on every vegetable even with AIC-selected orders — still a genuine finding, not a bug: SARIMAX's own fit is poor on the (still volatile) holdout year, all its R² values are negative, and the hybrids inherit that error rather than correcting it. The two new hybrids (`hybrid_random_forest_sarimax`, `hybrid_lstm_sarimax`, added at supervisor request) both confirm this pattern rather than escaping it. `hybrid_random_forest_sarimax` lands right alongside `hybrid_xgboost_sarimax`/`hybrid_catboost_sarimax` — all three tree-based residual correctors produce nearly identical, only-slightly-worse-than-plain-SARIMAX numbers, exactly as expected from the shared architecture. `hybrid_lstm_sarimax` is a more striking finding: it is the single worst model family in the entire grid on every vegetable, clearly worse than plain SARIMAX itself (e.g. carrot 469.5 vs. SARIMAX's own 245.7) — this was root-caused during development, not just observed: SARIMAX's out-of-sample forecast for the holdout year is heavily biased (nearly flat, failing to track the actual price swings), so the "residual" the LSTM is asked to correct has a training/test distribution shift far larger than what the tree-based residual models face; a scaled, autoregressive LSTM amplifies that shift instead of damping it the way a tree regressor's bounded predictions do. A residual-clipping safeguard (`HybridSarimaxLstmModel.resid_clip`) was added purely to keep training numerically stable (SARIMAX occasionally diverges to an extreme value on a small early CV fold, e.g. cabbage fold 2 — this crashed training before the clip was added), not to fix the underlying accuracy problem, which is architectural. See `research-papers/drafts/thesis/03_methodology.md` §3.10.6 for the full write-up. Don't read any of these model-ranking numbers as fixed — they move with the holdout window, which moves with the data; re-run `scripts/train_all.py` after any data update and expect the table to shift. (LSTM and `hybrid_lstm_sarimax` are the only stochastic model families in the grid, seeded by LSTM weight initialization; the other seven are deterministic given the same input.)

**Forecast prediction rows** (`results/metrics/holdout_predictions.csv`, 2,808 = 6 vegetables × 9 models × 52 weeks) and **AIC order-search rows** (`results/tables/sarimax_order_selection.csv`, 54 = 6 vegetables × 9 candidate orders) should be regenerated together whenever the SARIMAX order changes — run `scripts/select_sarimax_order.py` first, update `order_by_vegetable` if orders shift, then `scripts/train_all.py`.

Still open / deferred to a later phase: LLM provider for the advisory module; how satellite NDVI / IoT / behavioral survey data will actually be sourced (GEE auth, IoT data format, survey instrument); SHAP explainability implementation; prototype UI framework choice.

## Backend API

`app/backend/` is a standalone FastAPI service — its own `requirements.txt` and `.venv`, deliberately separate from the research pipeline's (`app/backend/.venv` vs. the root `.venv`; don't run backend code with the root venv or vice versa). It serves **pre-computed** forecasts (not live per-request predictions), model comparisons, and historical prices out of Postgres, cache-aside through Redis. Notifications are explicitly deferred (planned, not built).

**Infrastructure** (both pre-existing, not started by this project's tooling — a local Docker Postgres instance, container name `postgres`, image `postgres:15.17-trixie`; and a local Docker Redis instance, container name `Redis-main`):
- Postgres: `localhost:5432`, database `vegepredict`, user `postgres` / password `pg123` (local dev only — do not reuse these credentials anywhere non-local). Connect via `docker exec postgres psql -U postgres -d vegepredict`.
- Redis: `localhost:6379/0`. Connect via `docker exec Redis-main redis-cli`.
- Connection strings default to the local values above in `app/backend/config.py` (pydantic-settings); override via an `app/backend/.env` (gitignored) with `VEGEPREDICT_DATABASE_URL`/`VEGEPREDICT_REDIS_URL` if pointing elsewhere. Deliberately separate from `configs/config.yaml`, which holds the research pipeline's domain config, not deployment connection strings.

**Schema** (SQLAlchemy 2.0 async ORM + Alembic migrations, `app/backend/models/`, `app/backend/alembic/versions/`):
- `historical_price` — one row per (vegetable, week_start): wholesale/retail price, temperature, rainfall, diesel price. Mirrors `data/processed/<vegetable>.csv`.
- `forecast` — one row per (vegetable, model_family, forecast_date): predicted vs. actual price, `predicted_lower`/`predicted_upper` (the prediction interval — see Architecture below), `generated_at` (timezone-aware timestamp — see gotcha below). Seeded from `results/metrics/holdout_predictions.csv`.
- `model_metric` — one row per (vegetable, model_family): MAE/RMSE/MAPE/R², `interval_confidence`/`interval_coverage` (nominal vs. empirically observed prediction-interval calibration), `evaluated_at`. Seeded from `results/metrics/all_results.csv`. **"Best model per vegetable" is always derived by querying for the lowest RMSE, never a denormalized flag** — avoids staleness when metrics update after a retrain.

**Gotcha:** `generated_at`/`evaluated_at` must be `DateTime(timezone=True)` (Postgres `timestamptz`), not plain `DateTime` — the seed script populates them with `datetime.now(timezone.utc)` (timezone-aware), and asyncpg rejects a tz-aware Python value against a `TIMESTAMP WITHOUT TIME ZONE` column. Confirmed by a real failure the first time this was built; fixed via `alembic revision --autogenerate`, not by stripping tzinfo in Python.

**ETL** (`app/backend/seed.py`): idempotent upsert (`INSERT ... ON CONFLICT DO UPDATE` on each table's natural-key unique index) from the same three CSVs `scripts/train_all.py` produces. Safe to re-run after every retrain — not yet wired into an automated pipeline (the natural hook is `auto_retrain.py`'s promotion step, once that exists, see the still-pending auto-retrain tasks). Run manually: `app/backend/.venv/bin/python3 app/backend/seed.py`.

**API endpoints** (`app/backend/routers/`, `app/backend/services/` for the query logic):
- `GET /health` — DB + Redis connectivity check
- `GET /vegetables` — the 6 target vegetables
- `GET /models?vegetable=` — per-vegetable model comparison (MAE/RMSE/MAPE/R²), sorted by RMSE ascending
- `GET /predictions/{vegetable}?model=best|<family>` — single latest forecast; `model=best` (default) resolves to the lowest-RMSE model family. Selects purely by `forecast_date`, so this can return a genuinely future, unresolved row (`actual_price: null`) if one exists
- `GET /predictions/{vegetable}/future?model=best|<family>` — every not-yet-resolved week (`actual_price IS NULL`) for a vegetable, oldest first — the dedicated way to read "the upcoming forecast" as a trajectory rather than filtering the general list endpoint. Empty list (not 404) if `scripts/predict_future.py` hasn't been run yet
- `GET /predictions?vegetable=&model=&year=&start_date=&end_date=&limit=&offset=` — filtered/paginated forecast history, including by date range
- `GET /prices/{vegetable}/history?start_date=&end_date=&limit=&offset=` — historical prices
- `POST /auth/login`, `GET /auth/me` — JWT login and current-user identity (see User Management & Auth below)
- `POST /users`, `GET /users`, `GET /users/{id}`, `PATCH /users/{id}` — user account management, role-gated (see User Management & Auth below)

All read endpoints are cache-aside through Redis (`app/backend/cache.py` for the generic get/set/invalidate-by-prefix, `app/backend/services/cache_service.py` for key conventions), TTL 7 days (matches the weekly retrain cadence). **Gotcha, confirmed by a real bug:** a bare Postgres reseed does *not* invalidate Redis — the API kept serving pre-retrain cached JSON (missing the newly added prediction-interval fields) until `seed.py` started calling `cache_service.invalidate_vegetable()` for every vegetable after its commit. Any future write path that changes `forecast`/`model_metric`/`historical_price` data (the still-pending `auto_retrain.py` included) must invalidate the same way — don't reintroduce a bare commit with no cache invalidation. Guarded by `app/backend/tests/test_cache_invalidation.py`.

**Interactive docs**: Swagger UI at `/docs`, ReDoc at `/redoc` (both come free from FastAPI's OpenAPI generation). Every endpoint has a `summary`/`description`/`response_description`, every query/path param has a `description` + example value, and every Pydantic schema field (`app/backend/schemas/`) has a `Field(description=...)` — this is deliberately kept up to date so `/docs` is enough for another developer to understand and try the API without reading the source. When adding or changing an endpoint, add the same level of description, don't leave it bare.

**Tests** (`app/backend/tests/`, `pytest` + `httpx.AsyncClient` against the live app, hitting the real seeded Postgres/Redis rather than mocks): run with `app/backend/.venv/bin/python3 -m pytest app/backend/tests/ -c app/backend/pytest.ini`. Coverage includes edge cases per endpoint, not just the happy path: unknown/case-mismatched vegetables, unknown model families (404 on the single-forecast endpoint, empty list rather than an error on the filtered list endpoint — deliberately different, since only `vegetable` is validated against a closed set), out-of-range/invalid pagination (`limit`/`offset` boundaries → 422), invalid date ranges and formats, SQL-injection-shaped input (safe due to parameterized queries — verified the table is still queryable afterward), pagination-page disjointness, and a cross-endpoint consistency check (`/predictions/{veg}?model=best` matches the lowest-RMSE row from `/models`). **Gotcha:** `app/backend/pytest.ini` sets both `asyncio_default_fixture_loop_scope = session` and `asyncio_default_test_loop_scope = session` — required because `cache.py`'s Redis client and `database.py`'s SQLAlchemy engine are module-level singletons created once at import time; pytest-asyncio's default per-test event loop tears down and reopens a loop for every test, which breaks a singleton connection pool opened on an earlier (now-closed) loop with `RuntimeError: Event loop is closed`. Both ini keys are needed — fixture scope alone does not also cover the per-test loop.

Run the API locally: `app/backend/.venv/bin/python3 -m uvicorn app.backend.main:app --reload --port 8000` (from the project root, so the `app.backend.*` absolute imports resolve).

### User Management & Auth

Three roles — `superadmin`, `admin`, `farmer` — stored on the `users` table (`app/backend/models/user.py`: first_name, last_name, email (unique), hashed_password, role, is_active, created_at/updated_at). Auth is stateless JWT (HS256, `app/backend/security.py`), issued by `POST /auth/login` and passed as `Authorization: Bearer <token>` — deliberately not cookie/session-based, since the API's CORS is `allow_origins=["*"]` with `allow_credentials=False` (a wildcard-origin + credentialed-cookie combination browsers reject anyway). `VEGEPREDICT_JWT_SECRET`/`VEGEPREDICT_JWT_EXPIRE_MINUTES` in `.env` control signing; `config.py`'s default secret is dev-only and must be overridden in production.

**Existing read endpoints (`/vegetables`, `/models`, `/predictions*`, `/prices*`) stay fully public — auth applies only to `/auth/*` and `/users*`.** There is no public self-registration; the very first account must be created via the standalone `app/backend/create_superadmin.py` script (mirrors `seed.py`'s style — `VEGEPREDICT_SUPERADMIN_PASSWORD` env var or an interactive prompt, never a CLI arg, to avoid shell-history leakage). Every subsequent account is created through `POST /users` by an existing admin/superadmin.

Permission matrix (enforced partly at the router via `security.require_role()`, partly in `services/user_service.py` since several rules depend on the relationship between the caller and the *target* user, not just the caller's role alone):

| Endpoint | Roles | Notes |
|---|---|---|
| `POST /auth/login` | public | generic 401 on any failure — doesn't distinguish unknown email / wrong password / deactivated account |
| `GET /auth/me` | any authenticated | self only |
| `POST /users` | superadmin, admin | superadmin: any role. admin: `role` must be `farmer`, else 403 |
| `GET /users` | superadmin, admin | superadmin: all users. admin: farmers + self only |
| `GET /users/{id}` | superadmin, admin | admin requesting another admin/superadmin gets 404, not 403 — avoids confirming the account exists |
| `PATCH /users/{id}` | superadmin, admin, self | only superadmin may change `role`; admin may edit/deactivate farmers only; farmer may edit only their own name/password |

No hard delete — deactivation is `PATCH .../is_active=false`. Full coverage of this matrix lives in `app/backend/tests/test_auth.py`/`test_users.py`, using `conftest.py`'s `superadmin_user`/`admin_user`/`farmer_user` + matching `*_token` fixtures (insert a `User` row directly, then log in through the real `/auth/login` endpoint to get a token — same real-Postgres/no-mocks convention as the rest of the suite).

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

**Backend API** (`app/backend/`, own venv — see `## Backend API` above for details):

```bash
# Seed/refresh Postgres from the CSVs scripts/train_all.py produces
app/backend/.venv/bin/python3 app/backend/seed.py

# Run the API locally (from the project root, so app.backend.* imports resolve)
app/backend/.venv/bin/python3 -m uvicorn app.backend.main:app --reload --port 8000

# Run backend tests (hits the real seeded Postgres/Redis)
app/backend/.venv/bin/python3 -m pytest app/backend/tests/ -c app/backend/pytest.ini
```

Model hyperparameters (SARIMAX order/seasonal_order, LSTM architecture, XGBoost/CatBoost/Random Forest params, exogenous feature lists) live in `configs/config.yaml` — do not hardcode them in training scripts.

## Architecture

**Data flow:** `data/raw/{vegetable_prices,weather,fuel,satellite,iot,behavioral}` (untouched source files, one subfolder per source) → `src/data_processing/build_dataset.py` joins them on date into `data/processed/<vegetable>.csv` → `src/features/feature_engineering.py` adds lags/rolling stats/calendar features (shared by every model, so a feature added here is available to all model families) → each `src/models/<model>/train.py` consumes the processed+featurized frame and writes a fitted model to `trained_models/<model>/<vegetable>.pkl` → `src/explainability/shap_explain.py` computes SHAP attributions on the fitted model → `src/advisory/llm_advisory.py` turns a forecast + its SHAP explanation into a farmer-facing message → `app/` surfaces all of it in a UI.

**Hybrid models** (`src/models/hybrid_*`) depend on the plain SARIMAX fit: they load/re-fit SARIMAX first, compute residuals (actual − SARIMAX prediction), then train the ML model on those residuals plus the exogenous features. Final forecast = SARIMAX prediction + ML residual prediction. Keep this two-stage structure intact — don't collapse it into a single end-to-end model.

**Per-vegetable, per-model artifacts:** every vegetable is modeled independently (no cross-vegetable pooling), and every model family is trained separately per vegetable. Expect `6 vegetables × 9 model families = 54` trained artifacts, plus corresponding metrics rows in `results/metrics/`.

**Evaluation:** `src/evaluation/metrics.py` is the single source of truth for MAE/RMSE/MAPE — all training scripts and notebooks should import from there rather than reimplementing metrics, so comparisons across model families stay apples-to-apples.

**Prediction intervals:** every model's holdout predictions also carry a `predicted_lower`/`predicted_upper` band — model-agnostic, not model-specific (no quantile regression, no MC dropout, no per-family special-casing). `common.run_training` pools the walk-forward CV folds' out-of-sample residuals (actual − predicted, concatenated across all 5 folds — *not* the holdout's own residuals, to avoid a circular interval), takes the empirical percentiles at `forecasting.interval_confidence` (default 0.8 → 10th/90th percentile) via `metrics.prediction_interval()`, and adds that offset to the holdout point forecasts. `metrics.interval_coverage()` then reports what fraction of holdout actuals actually landed inside the band — this is the real calibration check, logged per vegetable/model as `interval_coverage` in `results/metrics/all_results.csv` alongside MAE/RMSE/MAPE/R². See `research-papers/drafts/thesis/03_methodology.md` Section 3.6 for the full write-up, including the known limitation (pooling residuals equally across differently-sized expanding CV folds).

**Calibration varies a lot by vegetable, and that's a real finding, not noise**: for each vegetable's *best* (lowest-RMSE) model, empirical coverage against the nominal 0.8 ranges from ~0.54 (carrot, brinjal — the interval is too narrow, understating uncertainty) to ~0.81 (pumpkin — well-calibrated). Random Forest, the winner on 5/6 vegetables, apparently has more volatile out-of-holdout errors than its CV-fold residuals suggest for the harder-to-forecast vegetables. Don't quote "80% interval" as if it's uniformly true — check `interval_coverage` per vegetable before making a calibration claim in the thesis.

**Forecast verification:** `common.run_training` returns both the metrics dict and the holdout-period `(date, actual, predicted, predicted_lower, predicted_upper)` series; `scripts/train_all.py` concatenates the latter into `results/metrics/holdout_predictions.csv` (2,808 rows = 6 vegetables × 9 models × 52 holdout weeks). `notebooks/03_model_results.ipynb` plots these as forecast-vs-actual charts (and residuals), saved to `results/figures/forecast_vs_actual_*.png` — don't rely on RMSE/R² tables alone to sanity-check a model; look at the actual curve.

## Folder layout

- `data/raw/` — untouched source data (vegetable_prices, weather, fuel, satellite, iot, behavioral), one subfolder each
- `data/processed/` — merged, per-vegetable model-ready datasets
- `src/data_processing/` — raw → processed merging
- `src/features/` — shared feature engineering used by all models
- `src/models/{sarimax,lstm,random_forest,xgboost,catboost,hybrid_xgboost_sarimax,hybrid_catboost_sarimax,hybrid_random_forest_sarimax,hybrid_lstm_sarimax}/` — one training module per model family
- `src/explainability/` — SHAP-based feature attribution, shared across model families
- `src/advisory/` — LLM-based advisory message generation from forecast + explanation
- `src/evaluation/` — shared metrics and cross-model comparison
- `trained_models/<model_family>/` — serialized fitted models, one file per vegetable
- `results/{figures,tables,metrics,explainability}/` — plots, comparison tables, metric outputs, and SHAP visuals for the paper
- `app/` — prototype forecast + explanation + advisory interface (web/mobile)
  - `app/backend/` — FastAPI + Postgres + Redis API (own `requirements.txt`/`.venv`, separate from the root ones). `main.py` (app + router registration), `config.py`/`database.py`/`cache.py`/`security.py` (infra setup — `security.py` owns password hashing, JWT issuing/verification, and the `get_current_user`/`require_role` auth dependencies), `constants.py` (vegetable list), `models/` (SQLAlchemy ORM: `historical_price`, `forecast`, `model_metric`, `user`), `schemas/` (Pydantic response models), `routers/` + `services/` (one pair per resource: vegetables, models, prices, predictions, auth, users, plus `health`), `seed.py` (idempotent CSV → Postgres ETL), `create_superadmin.py` (standalone script to bootstrap the first account, mirrors `seed.py`'s style), `alembic/` (schema migrations), `tests/` (pytest + httpx against the live app). See `## Backend API` above for endpoints and gotchas, and `### User Management & Auth` for the role/permission model.
- `configs/config.yaml` — vegetables list, data paths, per-model hyperparameters, vegetable/district name mappings
- `scripts/train_all.py` — orchestrates all 9 model families x 6 vegetables, writes the combined results tables
- `scripts/fetch_weather_openmeteo.py` — repopulates `data/raw/weather/weather.csv` from Open-Meteo
- `notebooks/` — exploratory analysis; production logic belongs in `src/`, not notebooks. `01_data_exploration.ipynb`, `02_feature_engineering.ipynb`, `03_model_results.ipynb` (the last needs `results/metrics/all_results.csv` from `scripts/train_all.py` to exist first)
- `research-papers/references/` — literature (PDFs, papers) informing the methodology; gitignored (not checked in), fetch again from `## Literature status` below if missing
- `research-papers/drafts/thesis/` — the thesis chapters (`01_introduction.md`, `02_literature_review.md`, `03_methodology.md`, ...)

## Documentation stays in sync with research

This is a research project, not just a codebase — the docs are part of the deliverable. Whenever research content changes (new papers found, scope/objectives change, a literature claim is added or superseded, methodology decisions are made), update the relevant doc in the same session, don't leave it for later. This isn't optional or best-effort: after finishing any implementation task, explicitly check each bullet below against what just changed, and write the update if one applies — silence is only correct when none of them apply, not the default.
- New objectives, model families, or data sources → update `## Project` above and the folder layout
- New papers read → update `research-papers/drafts/thesis/02_literature_review.md` (and its References list) and the `## Literature status` list below
- New introduction/problem-statement framing → update `research-papers/drafts/thesis/01_introduction.md`
- New methodology decisions — evaluation protocol changes (e.g. prediction intervals, new CV scheme), hyperparameter tuning approach, model-selection criteria, or any other "how we did it" decision that isn't just a code detail → update `research-papers/drafts/thesis/03_methodology.md`. If the decision changed a result worth reporting (a metric, a comparison table), update the relevant numbers there too, not just the narrative.

## Literature status

Sourced and read (in `research-papers/references/`, cited in `02_literature_review.md`): Madubhashini (2021/2023) — Sri Lanka, first SL-specific ML study; Ranaweera, Rathnayake & Ananda (2023) — Sri Lanka, beans/brinjal/carrot/pumpkin, RF best, same 4 exogenous features (rainfall, temperature, fuel price, production) as this project's config; Ruhunuge et al. (2024) — Sri Lanka, carrot-specific VAR climate causality; Weerasekara et al. (2026) — Sri Lanka, most advanced prior work (multi-market/season/regime XGBoost+LightGBM); Paul et al. (2022) — brinjal, India; Zhao et al. (2025) — TCN-XGBoost hybrid; Patil et al. (2023) — HySALS hybrid SARIMA-LSTM; Mayank, Shelke & Roy (2025); Shree Sanjay & Janarthanan (2025).

**MLOps / automated retraining** (Section 2.10, sourced for the `feature/auto-train-MLIOPs` application-side work): Garg et al. (2022) — CI/CD/CT/CM MLOps lifecycle; Meisenbacher et al. (2022) — automated forecasting pipeline review, most cover only 2-3 of 5 stages; Pham et al. (2024) — data vs. concept drift, semi-supervised detection; Katalay, Dimandja & Masakuna (2025) — multi-criteria statistical drift-triggered retraining (PSI, KL divergence); Jain et al. (2020) — crop price context-based retraining + data-quality monitoring (IBM, India); Zelingher (2025) — AGRICAF, explainable ML + econometrics for farmer-accessible commodity forecasts. Key takeaway used in the auto-train plan: tree-based models (RF/XGBoost/CatBoost) have no incremental-learning mode and need full refits; SARIMAX refitting is the computational bottleneck (confirmed empirically in this project); given this project's scale (single machine, weekly data, ~10-15min full retrain), a scheduled retrain + validation gate is the right-sized design, not full drift-detection infrastructure.

**Explainable AI (SHAP) in agricultural price forecasting** (Section 2.9, sourced for objective 3): Patro et al. (2025) — EXACT-FARM, SARIMAX-XGBoost-LSTM hybrid + SHAP + counterfactual analysis, 15-20% accuracy gain over non-hybrid baselines; Jain, Lwin, Dal & Oo (2026) — SHAP-explained ensemble across 22 Indian commodities/7 states, Random Forest strongest (RMSE 1.49, R² 0.9995) — independently converges with this project's own finding that Random Forest wins most vegetables.

**LLM-based farmer advisory** (Section 2.9, sourced for objective 4): Tzachor et al. (2023, *Nature Food*) — LLMs can scale personalized ag-extension advice but risk generic/unsupported output if ungrounded; Sawant, Nair & Hariharan (2026) — RAG advisory over best-practice docs, evaluated across Llama3.1/Mistral/Phi3/Qwen2.5; Samuel et al. (2026) — AgroLLM, RAG + symbolic domain-knowledge layer, 95.2% accuracy on a 504-question benchmark with reduced hallucination. Consistent lesson: ground LLM output in structured, verifiable data (here: the forecast + its SHAP explanation), not an open-ended prompt.

**Satellite/IoT signal for vegetable crops** (Section 2.8, sourced for objective 1 — motivates the NDVI/IoT fusion but neither paper forecasts price): Darra et al. (2023) — Sentinel-2 NDVI ensemble ML for processing tomato yield, ML beat statistical baseline; Ayall et al. (2025) — AI-based "backcasting" for incomplete IoT sensor coverage across growing seasons (strawberry), relevant if this project's own IoT data has coverage gaps.

**SARIMAX+RandomForest / SARIMAX+LSTM hybrid precedent** (Section 2.5, sourced for the `hybrid_random_forest_sarimax` and `hybrid_lstm_sarimax` model families added at supervisor request): Patil et al. (2023) — HySALS, SARIMA-LSTM — is a direct structural precedent for `hybrid_lstm_sarimax` (fit SARIMA(X) first, LSTM corrects its residuals); Ranaweera et al. (2023) — RF strongest among several ML models for Sri Lankan vegetables — motivates Random Forest as a residual-correction choice for `hybrid_random_forest_sarimax`, alongside the existing XGBoost/CatBoost options. Phung & Trinh (2025) — SARIMA-LSTM-Random Forest three-stage hybrid for gold price forecasting — remains cited as a related but structurally different precedent (a three-stage chain, vs. this project's two independent two-stage hybrids).

Note: Darra et al. (2023), Tzachor et al. (2023), and Phung & Trinh (2025) are cited from published metadata only — no local PDF (Sensors blocks automated retrieval despite CC BY; Nature Food is paywalled; Cogent Economics & Finance is gold open-access but Cloudflare-blocks automated retrieval).
