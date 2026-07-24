# Chapter 1: Introduction

## 1.1 Background

Agriculture underpins a substantial share of Sri Lanka's economy and rural livelihoods, with roughly a third of households depending on farming for income. Vegetables occupy a central place in this system, yet their prices are exceptionally volatile from one week to the next. This volatility is not incidental: Sri Lanka's vegetable market is largely import-isolated, so any disruption on the supply side — a flood, a drought, a fuel shortage, a transport bottleneck — feeds directly into retail prices with little buffering from imports. Production is also split structurally between two monsoon-driven cultivation seasons, *Maha* (October–April) and *Yala* (May–September), and between up-country crops such as carrot and leeks and low-country crops such as brinjal and pumpkin, each with different growing conditions and supply dynamics.

For farmers, this volatility translates directly into poor outcomes. Without reliable forward-looking price information, planting and harvesting decisions are made on incomplete information, often guided by informal networks rather than data. The result is a familiar cycle: oversupply crashes prices in some seasons, undersupply spikes them in others, and post-harvest losses erode already thin margins. Institutions such as the Hector Kobbekaduwa Agrarian Research and Training Institute (HARTI) and the Central Bank of Sri Lanka publish historical price bulletins, but these are retrospective — they describe what prices *were*, not what they are likely to *become*.

## 1.2 Problem Statement

Two distinct gaps motivate this research.

The first is a **forecasting accuracy gap**. Early work on Sri Lankan vegetable prices relied on classical statistical models such as ARIMA and SARIMA fitted to a single price series, without exogenous drivers. More recent work has shown that gradient-boosted machine learning models, trained on richer feature sets that include origin-zone weather, diesel costs, and exchange rates, and that explicitly separate the Maha and Yala seasons, substantially outperform these classical baselines and remain accurate even under an unseen, structurally different economic regime. This confirms that price movements in Sri Lanka's import-isolated market are not random, but it also shows that a single historical-price series alone is not enough — supply-chain-aware, multi-source features are what make the difference.

The second is a **trust and usability gap**. Even where accurate forecasting models exist, they remain research artifacts: outputs are numbers in a spreadsheet or a research paper, not something a farmer or trader can act on. None of the forecasting systems reviewed in Chapter 2 pair their predictions with an explanation of *why* the model expects a price to move, and none translate a forecast into a plain-language recommendation. A farmer given a number with no explanation has little more basis for trust than a farmer given no forecast at all.

Sri Lanka currently lacks a system that closes both gaps at once: one that fuses multiple relevant data sources (market prices, weather, satellite crop-condition indices, IoT sensor readings, and farmer behavioral data), forecasts prices with models suited to this market's structure, explains those forecasts in terms a non-technical user can verify, and converts them into actionable advisory messages.

## 1.3 Research Aim and Objectives

**Aim:** To build an adaptive, explainable AI-based system for forecasting Sri Lankan vegetable prices from multi-source data, and to make those forecasts usable by farmers through explanation and advisory generation.

**Specific objectives:**

1. Construct an integrated dataset combining historical vegetable prices, weather observations, fuel prices, satellite NDVI, IoT sensor readings, and farmer behavioral data.
2. Develop and compare forecasting models — SARIMA/SARIMAX, LSTM, Random Forest, XGBoost, CatBoost, and hybrid statistical–ML models — for six target vegetables: carrot, brinjal, pumpkin, cabbage, snake gourd, and leeks.
3. Apply SHAP-based explainable AI to make each model's forecasts transparent and auditable.
4. Build an LLM-based advisory module that converts a forecast and its explanation into a farmer-friendly recommendation.
5. Deliver a working prototype interface that surfaces forecasts, explanations, and advisory messages.

## 1.4 Scope

The study covers six vegetables chosen to represent both up-country (carrot, leeks, cabbage) and low-country (brinjal, pumpkin, snake gourd) growing conditions. Forecasting is evaluated via time-series cross-validation on RMSE, MAPE, and R², consistent with standard practice in the agricultural price forecasting literature reviewed in Chapter 2. The explainability and advisory components are scoped to the six target vegetables and the models trained in objective 2; broader questions of real-time deployment, IoT hardware provisioning, and large-scale farmer adoption are outside the scope of this study and are noted as future work.

## 1.5 Significance

An accurate, explainable, and actionable forecasting system has direct value for multiple stakeholders: farmers gain a basis for planting, harvesting, and selling decisions; traders and market intermediaries gain earlier visibility into price movements; and policymakers gain an early-warning tool for market interventions. More broadly, reducing price-driven post-harvest losses and improving farmer income stability contributes to national food security.

## 1.6 Thesis Structure

Chapter 2 reviews the literature on statistical, machine learning, and hybrid approaches to agricultural and vegetable price forecasting, with particular attention to Sri Lankan and South Asian studies, and identifies the specific gap this research addresses. Chapter 3 describes the methodology: data sources and integration, feature engineering, model architectures, and the explainability and advisory pipeline. Chapter 4 presents experimental results and model comparisons. Chapter 5 discusses findings, limitations, and implications. Chapter 6 concludes and outlines future work.
