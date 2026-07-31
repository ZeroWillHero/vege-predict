# Introduction

## 1.1 Background of the Study

Agriculture is a central pillar of the Sri Lankan economy, and the vegetable subsector contributes importantly to food and nutritional security, rural business, and farm income. Vegetables are cultivated across the country and form a major livelihood source for many farmers, even though they are not a major food. Recent evidence shows that vegetable prices in Sri Lanka have fluctuated radically in the last decade due to unpredictable climate change, natural disasters, economic crises, and other structural challenges, exposing farmers and consumers to considerable risk. Because vegetable crops are short-lived and their prices are noisy and nonlinear, forecasting is a difficult yet crucial task, especially in a context where systematic price repayment mechanisms are limited. Accurate price forecasting is therefore essential to guide farmers’ production decisions, to support traders and policymakers in managing market risk, and to protect household well-being (Ranaweera et al., 2023).

Vegetable production is commonly classified into up‑country and low‑country zones, reflecting differences in elevation, climate and production systems. Up-country vegetables such as carrots (Daucus carota subsp. sativus) and cabbage (Brassica oleracea var. capitata) are grown at higher altitudes with cooler temperatures and distinct yala and maha seasons, which shape cropping calendars and yield variability. Low‑country vegetables, including pumpkin(Cucurbita pepo) and brinjal(Solanum melongena), are produced in warmer areas where rainfall patterns, temperature and pest/disease dynamics differ. These area differences translate into distinct production cycles, yield stability and market access and, hence, into different seasonal price patterns and volatility. For example, the progress of up-country vegetable cultivation has been much less in area than low-country cultivation, which can amplify local supply shortages and complicate price behaviour across markets (Champika & Mugera, 2023).

## 1.2 Vegetable Price Fluctuation and Its Challenges in Sri Lanka

Vegetable prices in Sri Lanka vary widely throughout the year, with clear seasonal peaks. Analysis of weekly price data shows that prices of key vegetables such as carrot, cabbage are lowest during peak harvesting periods and highest during lean seasons, reflecting strong seasonal supply effects. These fluctuations are driven by natural climate variability, production changes, market supply, consumer demand, government regulations and input and fuel costs. Because vegetables are highly perishable and storage and cold‑chain architecture remain limited, even short‑term disorder in production or logistics can cause sharp price swings (Ranaweera et al., 2023).

Such volatility has serious suggestions for different stakeholders. Small‑scale farmers face unstable incomes and are often forced to sell immediately after harvest when prices are lowest, reinforcing debt cycles and discouraging investment in improved technologies. Traders and other intermediaries must manage purchasing and inventory decisions under high price risk and may respond by storing or withdrawing from volatile markets. Consumers, especially low‑income households that spend a large share of their budgets on food, are sensitive to sudden price increases that reduce vegetable intake below recommended levels. Policymakers struggle to design effective price‑stabilisation and support policies in the absence of reliable forecasting and early‑warning systems. At present, official market information systems focus largely on reporting historical and current prices, and there is limited dissemination of short-term or medium-term forecasts for short-lived vegetables. This information gap contributes to substandard decisions by farmers, traders and consumers and strengthens the case for strong vegetable price forecasting tools (Champika & Mugera, 2023).

## 1.3 Factors Influencing Vegetable Prices

Evidence studies highlight a range of factors that affect vegetable price behaviour in Sri Lanka. Weather and climate variables, particularly rainfall and temperature, are key factors of yields and harvest timing; rainfall or sudden temperature changes can affect crop growth at any stage and lead to supply shocks that translate into price volatility. Seasonal production cycles associated with the Maha and Yala seasons generate recurring patterns in prices, with two price spikes corresponding to lean periods and two notable price drops during peak harvesting seasons. Market supply and demand respond to changes in production, consumer preferences and policy measures, while transport and fuel costs directly affect marketing margins because vegetables are moved long distances by lorries and trucks (Ranaweera et al., 2023).

Physical and structural constraints, including inadequate storage, weak market integration and variable road and transport conditions, also contribute to price variability and spatial price differences. Studies have shown that vegetable price variability is mainly associated with climatic factors, seasonal supply gaps and rising input costs, all of which complicate farmers’ production and marketing decisions. In addition, broader macroeconomic developments such as exchange‑rate movements and energy price shocks can trigger structural breaks in price series, changing the underlying volatility behaviour over time.

### 1.3.1 Importance of External Factors in Price Forecasting

Given this complexity, including external factors into price forecasting models is essential. Research on the Sri Lankan vegetable market and on other agricultural products demonstrates that models using only historical prices often fail to capture large movements driven by weather anomalies, fuel price changes, or policy negotiations. Including external variables such as rainfall, temperature, and fuel prices improves the descriptive power of forecasting models and allows them to identify causal mechanisms behind price changes rather than merely temporal correlations.

For instance, multivariate SARIMAX models and machine‑learning approaches that incorporate climatic variables have achieved lower error measures—such as root mean square error and mean absolute percentage error—than univariate ARIMA models in price forecasting studies for tomatoes and other crops. In Sri Lanka, machine‑learning models trained on vegetable prices, rainfall, temperature, diesel prices and production data have shown that external variables significantly enhance predictive accuracy, especially for crops with strong climate sensitivity. Such models can also be used to generate scenario‑based forecasts under alternative weather or cost trajectories, providing actionable information for farmers, traders and policymakers to plan cropping, harvesting, storage and market interventions (Sunny Kumar et al., 2025).

## 1.4 Existing Vegetable Price Forecasting Approaches

Existing work on vegetable price forecasting in Sri Lanka and abroad has used both traditional time‑series models and newer machine‑learning techniques. Time‑series models such as ARIMA and SARIMA are widely applied to capture trends, seasonality and autocorrelation in agricultural price data, including carrot and cabbage prices in Sri Lanka. In these studies, ARIMA models were able to forecast weekly retail prices of carrot and cabbage with around 71% and 55% accuracy, respectively, although residual volatility and structural breaks remained. To address the influence of external drivers, SARIMAX models have been developed that integrate climatic variables such as rainfall and temperature into the forecasting process; these models have produced lower forecast errors for tomato prices than their univariate counterparts.

Aside from time‑series models, a growing literature applies machine‑learning and data‑mining techniques to crop price prediction. In the Sri Lankan vegetable market, tree‑based algorithms such as Random Forest, as well as linear regression, SMO regression and multilayer artificial neural networks, have been evaluated using rainfall, temperature, diesel prices and production as predictors. Results indicate that Random Forest and related tree‑based models consistently outperform other classifiers in terms of mean absolute error and root mean square error, particularly for pumpkins and other selected vegetables. These findings are broadly consistent with international studies where decision‑tree‑based approaches and, at the same time, other methods have shown strong performance in forecasting crop prices and yields (Sunny Kumar et al., 2025).

Artificial neural networks and other deep‑learning models have also been applied to vegetable price forecasting, capturing nonlinear relationships in noisy time series. While some studies report high predictive accuracy for neural networks, results from Sri Lankan data suggest that multilayer perceptron models may exhibit higher error rates when multiple external variables are included and that tree‑based algorithms can be strong under limited and noisy datasets. Across all methods, data quality and availability remain key limitations: missing records, irregular reporting and restricted spatial coverage reduce model reliability, and rare extreme events are difficult to model without representative training data. These constraints highlight the need to adapt and compare forecasting approaches under Sri Lankan conditions and to develop integrated systems that can support early‑warning and decision‑making in the vegetable sector.  (Ranaweera et al., 2023).

# Methodology

## 3.0 Materials and Methods

This chapter explains how the vegetable price forecasting research was carried out from the beginning to the final prototype. It describes the research design, data sources, data collection process, data cleaning, exploratory analysis, feature engineering, model training, model validation, final testing, technology selection, system development, and evaluation. The purpose of this chapter is to provide a clear and repeatable record of the work. Another researcher should be able to understand the complete process and repeat the study using the same data sources and methods.

The project forecasts the weekly retail prices of six vegetables in Sri Lanka: carrot, brinjal, pumpkin, cabbage, snake gourd, and leeks. A separate forecasting process is used for each vegetable because each vegetable has a different price pattern, growing area, seasonal behaviour, and level of market volatility. The main output is a one-step-ahead weekly retail price forecast. Historical retail prices are the main input, while temperature, rainfall, and diesel price are used as external variables that may influence production and transport costs.

The methodology follows a complete data science pipeline. First, data are collected from official or trusted sources. Second, the data are cleaned and converted to a common weekly format. Third, the datasets are joined by week and vegetable. Fourth, the price behaviour is explored using charts and statistical methods. Fifth, past-price, rolling, calendar, weather, and fuel features are created. Sixth, seven forecasting model families are trained and compared. Finally, the best model for each vegetable is selected and prepared for use in the forecasting prototype.

Overall Research and System Development Process

*Figure 3.1: Overall research and system development process*

## 3.1 Research Design

This study uses a quantitative, experimental, and comparative research design. It is quantitative because the analysis is based on numerical data such as vegetable prices in Sri Lankan rupees, rainfall in millimetres, temperature in degrees Celsius, and diesel price in Sri Lankan rupees. The study does not depend on personal opinions or interview responses. The conclusions are made by analysing numerical patterns and model performance values.

The study is experimental because different forecasting models are built, trained, adjusted, and tested. Each model receives historical data and produces a future price prediction. The predicted values are compared with the real prices that occurred later. This process allows the researcher to measure how accurately each model works.

The study is comparative because it does not assume that one forecasting method is always the best. Classical time-series, machine-learning, deep-learning, and hybrid models are compared under the same time-based evaluation process. The comparison is completed separately for each vegetable. The research is also applied research because the final objective is not only to study forecasting methods. The selected models are calculated to support a practical system that can provide future price information to farmers, traders, consumers, and decision makers. The methodology therefore includes both model development and prototype system development.

## 3.2 Research Scope and Unit of Analysis

The unit of analysis is one vegetable during one week. Each record in the final dataset represents a weekly observation for a selected vegetable. The main target variable is the weekly retail price. Wholesale price is retained where available for analysis, but the retail price is used as the prediction target because it directly represents the price paid by consumers and observed by small traders.

The six vegetables were selected because they represent both up-country and low-country production patterns and show different levels of price variation. Carrot, cabbage, and leeks commonly represent up-country vegetables, while brinjal, pumpkin, and snake gourd represent other production environments. Treating the vegetables separately prevents the model from incorrectly assuming that all vegetables follow the same market behaviour.

The exploration files contain price records from an earlier period than the fuel and weather records. However, a forecasting model that uses all three data sources can only be trained during the period where the required variables overlap. Therefore, the processed model datasets begin from the common period in which price, weather, and diesel information are available together. This avoids creating false values for years where an external variable was not available.

## 3.3 Data Sources and Data Collection

All data used in this study are secondary data. The data already existed and were published by official organisations or a trusted weather service. No questionnaire, interview, or human experiment was used. Three main sources were used: HARTI (Hector Kobbekaduwa Agrarian Research and Training Institute) for vegetable prices, CEYPETCO (Ceylon Petroleum Corporation) for diesel prices, and Open-Meteo for historical weather.

| Data source | Data collected | Reason for use | Original frequency |
|---|---|---|---|
| Hector Kobbekaduwa Agrarian Research and Training Institute (HARTI) | Wholesale and retail vegetable prices | Provides Sri Lankan market price information for the selected vegetables | Weekly bulletins |
| Ceylon Petroleum Corporation (CEYPETCO) | Lanka Auto Diesel price | Represents an important transport-cost factor in the vegetable supply chain | Irregular revision dates, converted to weekly |
| Open-Meteo historical weather API | Average Temperature from Maximum and Minimum Temperatures | Represents growing conditions in the selected production districts | Daily, converted to weekly |

*Table 3.1: Main data sources used in the research*

### 3.3.1 Vegetable Price Data from HARTI

HARTI publishes weekly food-item bulletins containing market price tables. The bulletins are mainly provided as PDF files. The relevant vegetable tables include wholesale and retail price information collected from reporting markets.

A Python data collection script was used to visit the HARTI bulletin page, identify the correct weekly bulletin link, download the PDF, and read the required price tables. Searching the bulletin page is necessary because the file names and links are not always predictable. The script therefore reads the web page and identifies the available bulletin links instead of constructing a file name based only on a date.

Prices in the bulletins may be shown as ranges. For example, a market may report a carrot price range of LKR 120 to LKR 150. A single value is required for weekly modelling. Therefore, the midpoint of the range is calculated. In this example, the midpoint is LKR 135. The midpoints from the available reporting markets are then averaged to create one weekly price value for the vegetable.

Vegetable names are standardised because the same vegetable may appear using different spellings, capitalisation, or plural forms. A mapping configuration changes the source names into one internal name, such as converting different forms of brinjal into the internal label brinjal. This step prevents the same vegetable from being separated into multiple categories.

The collection script works step by step. It records the bulletins that were already processed and collects only new bulletins during the next run. This reduces unnecessary downloads and makes it possible to update the dataset regularly. The raw price file contains the vegetable name, week start date, wholesale price, and retail price.

### 3.3.2 Diesel Price Data from CEYPETCO

CEYPETCO publishes historical fuel prices. Unlike the vegetable data, fuel prices do not change every week. A price is introduced on a revision date and normally remains valid until the next revision. The research uses Lanka Auto Diesel because diesel is widely connected with the transport of vegetables from farms to markets.

A Python script reads the revision dates and prices and converts them into a weekly series. For every weekly date, the script selects the latest diesel price that was active on or before that week. This is a forward-fill process based on official price validity. For example, when a diesel price becomes active in June, and the next revision occurs in August, the June value is used for the weeks between those dates.

Petrol was not included as a model feature because diesel is more directly connected with the larger vehicles used in agricultural transport. The weekly fuel dataset contains the week's date and diesel price. Using a weekly series makes the fuel data compatible with the weekly vegetable price records.

### 3.3.3 Weather Data from Open-Meteo

Historical weather data were collected from the Open-Meteo API. An API allows a program to request data automatically. Representative coordinates were defined for Sri Lankan districts. The script requested daily maximum temperature, daily minimum temperature, and we get the average temperature from them.

Daily weather data were converted into weekly data because the target vegetable prices are weekly. The daily midpoint temperature was calculated as the average of the daily maximum and minimum temperatures. The weekly average temperature was then calculated from the daily midpoint values in that week. Daily precipitation values were also summarised into the weekly rainfall variable used by the project.

Each vegetable was linked to a representative production district through a configuration file. This is better than using one national weather value because growing conditions differ between districts. In the supplied project, carrot, cabbage, and leeks share the Nuwara Eliya weather series. Other vegetables are mapped to their relevant production districts. Storing the mapping in a configuration file makes it easier to update the district assignment later without rewriting the full program.

## 3.4 Data Storage and File Organisation

The project separates data into raw and processed locations. Raw files contain the collected information before major transformations. Processed files contain cleaned and merged weekly datasets prepared for analysis and modelling. This separation is important because it preserves the original information and allows the preprocessing process to be repeated when needed.

During the research stage, CSV files were used because they are simple, readable, and supported by all selected Python libraries. A separate processed CSV file was created for each vegetable. The example carrot dataset in the feature-engineering notebook contained the date, retail price, wholesale price, average temperature, average rainfall, and diesel price before model features were created.

Configuration values, including vegetable names, weather-district mappings, feature settings, and model settings, were stored outside the main scripts in a YAML configuration file. This improves reproducibility because the researcher can change an option in one location instead of editing many scripts.

## 3.5 Data Preparation and Preprocessing

Data cleaning was completed before model training. The main objective was to create a reliable weekly time series with consistent names, valid dates, numerical values, and one record per vegetable per week. The following checks were required during preprocessing.

*Figure 3.2: Data Cleaning and Preprocessing*

The three sources were merged using the week date. Weather was first selected according to the representative district of the vegetable. The price record, district weather record, and diesel record for the same week were then joined. The result was one integrated weekly dataset for each vegetable.

Missing values must be handled carefully in a time-series study. Future values must never be used to fill earlier records because this would create data leakage. Forward filling is methodologically valid for diesel because the official price remains active until the next revision. Other missing price or weather values should be handled according to the actual source-code rule and documented clearly.

### 3.5.2 Data Integration

After cleaning, the price, weather, and diesel datasets were combined using the common weekly date. First, the correct district weather series was selected for each vegetable. Next, the vegetable price record, district weather record, and diesel price record for the same week were joined. A separate integrated weekly dataset was then produced for each vegetable.

Using only the common period ensured that a model using external variables did not receive invented historical weather or diesel values. The final integrated records included the date, retail price, wholesale price where available, average temperature, rainfall, and diesel price before feature engineering.

### 3.5.3 Data Transformation and Feature Engineering

Feature engineering changed the integrated weekly data into model inputs. The same feature-building function was used by the training scripts so that every applicable model received a consistent set of variables. Twenty input features were created, as shown in Table 3.2.

| Feature group | Features | Purpose |
|---|---|---|
| Current external variables | `average_temperature`, `average_rainfall`, `diesel_price` | Represent current weather and transport-cost conditions. |
| Price lags | `price_lag_1`, `price_lag_2`, `price_lag_4`, `price_lag_8` | Provide prices from 1, 2, 4, and 8 earlier weeks. |
| Rolling averages | `price_rolling_mean_4`, `price_rolling_mean_8`, `price_rolling_mean_12` | Describe recent average price levels. |
| Rolling volatility | `price_rolling_std_4`, `price_rolling_std_8`, `price_rolling_std_12` | Describe recent price stability or volatility. |
| Calendar variables | `month`, `week_of_year`, `quarter` | Help the model learn repeating calendar patterns. |
| Agricultural season | `is_maha_season` | Marks the Maha season according to the project configuration. |
| Lagged external variables | `average_temperature_lag_1`, `average_rainfall_lag_1`, `diesel_price_lag_1` | Represent external conditions from the previous week. |

*Table 3.2: Features created for the forecasting models*

The first rows of each dataset could not contain complete lag and rolling-window values. These warm-up rows were removed before modelling. Data leakage was also checked. At week t, lag and rolling features were calculated only from information available before week t. The current target price was never placed inside its own input features.

## 3.6 Exploratory Data Analysis (EDA)

Exploratory data analysis was completed before model training to understand the structure and quality of the time series. Python, pandas, NumPy, Matplotlib, Seaborn, and statsmodels were used to create summary statistics and visualisations. The analysis examined price behaviour, external variables, data quality, and possible relationships that could support feature and model selection.

### 3.6.1 Trend Analysis

Weekly retail price line charts were prepared separately for each vegetable. These charts were used to identify long-term increases or decreases, changes in the average price level, and periods with sudden market movement. Rolling averages were also examined because they reduce short-term noise and make the general direction of the series easier to see.

### 3.6.2 Seasonality Analysis

Seasonality analysis examined whether similar price patterns repeated during particular months or weeks of the year. Prices were compared by month and week of year, and curves from different calendar years were placed on a common annual scale. Autocorrelation plots were also used to check whether prices were related to values from earlier seasonal lags. Because the data are weekly, an annual seasonal cycle of 52 weeks was considered.

### 3.6.3 Price Volatility Analysis

Price volatility describes how strongly prices move over time. Weekly changes and rolling standard deviations were calculated to identify stable and unstable periods. A 12-week rolling standard deviation was used during exploration. A high value indicated that the vegetable price had moved widely around its recent average. This analysis was important because a model that performs well during stable periods may perform differently during a price shock.

### 3.6.4 Correlation Analysis Between Price and External Factors

Correlation analysis was used to examine the relationship between retail price, temperature, rainfall, and diesel price. The analysis considered both the current week and delayed effects at earlier lags. This was necessary because weather conditions may affect production and market supply after several weeks, while a diesel revision may affect transport costs immediately or after a short delay. Correlation was treated as an exploratory measure and not as proof that one variable directly caused a price change.

### 3.6.5 Time Series Decomposition Analysis

STL (Seasonal and Trend) decomposition was applied using a 52-week seasonal period. STL separates the observed price series into three parts: trend, seasonal, and residual. The trend component shows the longer-term direction, the seasonal component shows repeating annual behaviour, and the residual component contains irregular movements that are not explained by trend or seasonality. This analysis helped to show whether a classical seasonal model was suitable and whether strong unexplained movements remained.

## 3.7 Forecasting Model Development

Seven model families were trained and compared for each vegetable. They included a classical statistical model, three tree-based machine-learning models, one deep-learning model, and two hybrid models.

| Model | Model type | Role in the research |
|---|---|---|
| SARIMAX | Classical statistical time-series model | Learns autoregressive, moving-average, seasonal, differenced, and external-variable relationships. |
| Random Forest | Bagged decision-tree regression | Averages many trees to learn non-linear relationships with reduced variance. |
| XGBoost | Gradient-boosted decision trees | Builds trees in sequence so that later trees correct earlier errors. |
| CatBoost | Gradient-boosted decision trees | Uses boosting and regularisation to learn complex tabular relationships. |
| LSTM | Deep-learning sequence model | Reads sequences of earlier weeks and learns short- and longer-term dependencies. |
| Hybrid XGBoost-SARIMAX | Hybrid statistical and machine-learning model | Combines a SARIMAX forecast with an XGBoost-based correction. |
| Hybrid CatBoost-SARIMAX | Hybrid statistical and machine-learning model | Combines a SARIMAX forecast with a CatBoost-based correction. |

*Table 3.3 summarises their roles in the research*

### 3.7.1 Time Series Forecasting Models

ARIMA means Autoregressive Integrated Moving Average. It represents relationships with earlier values, differencing, and earlier errors. SARIMA extends ARIMA by adding seasonal terms. SARIMAX further extends SARIMA by including external variables. In this project, SARIMAX was the classical model included in the final seven-model comparison because it could use weekly temperature, rainfall, and diesel price together with the historical price series.

Stationarity was checked before fitting the classical model. A stationary series has a broadly stable mean and variance over time. The Augmented Dickey-Fuller test was used as the formal stationarity test. A p-value below 0.05 was used as evidence against the null hypothesis of a unit root. First-order differencing was used to reduce non-stationarity where required.

Autocorrelation Function and Partial Autocorrelation Function plots were examined to understand price relationships at earlier lags. Candidate SARIMAX orders were fitted and compared using the Akaike Information Criterion (AIC). A lower AIC was preferred within the SARIMAX family because it balances statistical fit and model complexity. The annual seasonal length was 52 weeks.

The selected non-seasonal orders were (0, 1, 2) for carrot, (1, 1, 2) for leeks, and (2, 1, 2) for brinjal, pumpkin, cabbage, and snake gourd. The seasonal order used in the selection results was (1, 0, 1, 52). Residual plots, residual autocorrelation, and the Ljung-Box test were used to check whether important patterns remained after fitting (Ruhunuge et al., 2023).

### 3.7.2 Machine Learning Forecasting Models

Random Forest, XGBoost, and CatBoost were trained using the engineered weekly feature table. Random Forest creates many decision trees from different samples and averages their predictions. This reduces the dependence on a single tree and allows the model to learn non-linear relationships.

XGBoost and CatBoost are gradient-boosting models. They build decision trees in sequence. Each new tree focuses on part of the error that remains after the earlier trees. The models can learn interactions among lagged prices, rolling statistics, calendar variables, weather, and diesel prices. Regularisation and validation-based tuning were used to reduce overfitting.

The project also included hybrid XGBoost-SARIMAX and hybrid CatBoost-SARIMAX models. In these models, SARIMAX represents the main linear and seasonal structure, while the boosting model learns a correction for remaining non-linear error. The final hybrid forecast combines the statistical forecast and the machine-learning correction.

### 3.7.3 Deep Learning Models

The deep-learning model used in the final comparison was Long Short-Term Memory (LSTM). LSTM is a type of artificial neural network developed for sequential data. Unlike a tree-based model that receives one feature row, an LSTM receives a sequence containing several earlier weeks and predicts the following weekly price.

The input features were scaled before training so that variables with different numerical ranges could be processed more reliably. The sequences were passed through LSTM memory units and then an output layer that returned one predicted price. The network weights were updated by minimising a training loss. Validation loss and early stopping were used to reduce overfitting. The final LSTM structure and training values were controlled by the project model configuration.

## 3.8 Model Training and Validation

Every model was trained separately for each vegetable. A common automated experiment was used so that the data split, features, metrics, and result-saving process were consistent across the model families. The chronological order of the observations was maintained throughout the experiment.

### 3.8.1 Training and Testing Data Division

A time-based division was used instead of a random train-test split. The older part of the data was used for model fitting and validation. The most recent 52 weeks were reserved as the final unseen test period. These holdout observations were not used to choose model settings.

Walk-forward validation was used within the earlier data. An initial historical period was used for training, and the model predicted the next validation block. The training period was then moved forward or expanded, the model was trained again, and the next future block was predicted. This produced performance measurements from several periods and provided a better test of model stability than a single validation split.

### 3.8.2 Hyperparameter Tuning

Model settings were selected using validation performance, not the final test data. For Random Forest, the tuning process considered the number of trees, tree depth, minimum samples, and the number of features used at a split. For XGBoost and CatBoost, it considered the number of trees or repetitions, learning rate, depth, sampling, and regularisation. For LSTM, the adjustable settings included sequence length, number of layers and units, dropout, batch size, learning rate, number of periods, and early-stopping settings.

SARIMAX tuning followed a different process because its parameters describe the statistical structure of the series. Candidate non-seasonal and seasonal orders were compared using AIC on the training data. After a configuration was selected, the model was evaluated with the same future validation and test periods used for the other model families.

### 3.8.3 Forecasting Procedure

The research used one-step-ahead weekly forecasting. The procedure was repeated in sequential order as follows:

1. Use only the price and external information available up to the current forecasting point.
2. Create the required lag, rolling, calendar, weather, and diesel input features or the LSTM sequence.
3. Load or fit the required model for the selected vegetable.
4. Predict the retail price for the following week.
5. Move the forecasting point forward and repeat the same procedure for the next evaluation week.
6. Save the actual value, predicted value, error measures, and prediction limits for analysis.

## 3.9 Model Performance Evaluation

The models were evaluated using mean absolute error (MAE), root mean square error (RMSE), mean absolute percentage error (MAPE), and the coefficient of determination. The same measures were calculated for every model and vegetable on the unseen test period.

Mean Absolute Error (MAE) is the average absolute difference between the actual price and the predicted price. It is measured in Sri Lankan rupees and gives a direct explanation of the average forecasting error. A lower MAE is better.

**Formula:** `MAE = (1 / n) x sum of |Actual Price - Predicted Price|`

Root Mean Square Error (RMSE) is the square root of the average squared error. Because errors are squared, large mistakes receive a greater penalty. RMSE was used as the main measure for final model selection. A lower RMSE is better.

**Formula:** `RMSE = square root of [(1 / n) x sum of (Actual Price - Predicted Price)^2]`

Mean Absolute Percentage Error (MAPE) expresses the average error as a percentage of the actual price. It supports comparison across vegetables with different normal price levels. A lower MAPE is better.

**Formula:** `MAPE = (100 / n) x sum of |(Actual Price - Predicted Price) / Actual Price|`

R-squared (R²) measures how much of the variation in the test prices is explained by the model compared with a mean-based reference. A value closer to 1 indicates better explanatory performance. A negative value means that the model performed worse than the reference during that period.

The evaluation also included actual-versus-predicted plots, residual analysis, and an 80 percent prediction interval. The interval provides a lower and upper forecast limit and shows the uncertainty around the point forecast. Interval coverage was calculated as the percentage of actual holdout prices that fell inside the interval.

## 3.10 Model Comparison and Selection

All seven model families were compared on the same final holdout period. The comparison was carried out separately for carrot, brinjal, pumpkin, cabbage, snake gourd, and leeks. This avoided selecting one model only because it performed well for a different vegetable.

The model with the lowest holdout RMSE was selected as the main forecasting model for each vegetable. MAE, MAPE, R², interval coverage, and visual forecast behaviour were examined as supporting evidence. This combined approach reduced the risk of selecting a model from one performance number without checking its practical behaviour.

AIC and RMSE had different purposes. AIC was used only to choose the internal SARIMAX order. It was not used to compare SARIMAX with Random Forest, XGBoost, CatBoost, LSTM, or the hybrid models because those model families do not share one comparable AIC definition. Out-of-sample holdout RMSE was therefore used for the final cross-model decision.

## 3.11 Forecasting System / Prototype Development

After model comparison, the selected model for each vegetable was prepared for use in a forecasting prototype. The prototype connects data preparation, the saved model, forecast generation, and a user-facing response. It demonstrates that the research method can be used as a practical system and not only as an offline experiment.

**Technologies used.** Python was the main language for collection scripts, preprocessing, analysis, modelling, evaluation, and backend logic. Jupyter Notebook was used for interactive exploration and result checking. pandas and NumPy were used for data handling and numerical work. Matplotlib and Seaborn were used for graphs. statsmodels was used for STL, ADF, ACF/PACF, SARIMAX, AIC, and residual diagnostics. scikit-learn was used for Random Forest, preprocessing, and metrics. The XGBoost and CatBoost libraries were used for the boosting models, and a Python deep-learning framework was used for LSTM. YAML configuration files stored vegetable mappings and model settings. CSV files were used during research development. Git was used for version control. The prototype backend used FastAPI, PostgreSQL was used for structured data storage, and Redis was used to cache recent forecasts.

| Technology | Use in the project |
|---|---|
| Python | Data collection, preprocessing, modelling, evaluation, and backend logic. |
| Jupyter Notebook | Exploratory analysis, feature inspection, result analysis, and figures. |
| pandas and NumPy | Data cleaning, joining, weekly aggregation, numerical calculation, and feature tables. |
| Matplotlib and Seaborn | Time-series plots, correlation views, model comparisons, and diagnostics. |
| statsmodels | STL, stationarity analysis, ACF/PACF, SARIMAX, AIC, and residual tests. |
| scikit-learn | Random Forest, preprocessing utilities, validation support, and evaluation metrics. |
| XGBoost and CatBoost | Gradient-boosted tree forecasting models and hybrid corrections. |
| Deep-learning framework | LSTM model construction and training. |
| FastAPI | Forecasting backend and request-response endpoints. |
| PostgreSQL | Structured storage for weekly data and model-related records. |
| Redis | Caching recent forecasts for faster repeated requests. |
| YAML, CSV, and Git | Configuration, research data files, saved outputs, and version control. |

*Table 3.4: Main technologies and their use*

**System architecture.** The architecture follows this path: HARTI, CEYPETCO, and Open-Meteo data sources feed the data-collection scripts; the raw data move through cleaning, weekly transformation, and integration; the processed data are used for feature engineering and model training; the selected models are saved; the FastAPI backend receives a forecast request, retrieves the latest required data from PostgreSQL, prepares the correct input, produces a forecast, stores or reads recent results through Redis, and returns the result to the interface.

**Data input and forecasting process.** A request identifies the vegetable for which a next-week forecast is required. The backend loads the selected model and the latest required price, weather, and diesel information. It creates the correct feature row or sequence, produces the point forecast and forecast limits, and returns the vegetable, forecast date, predicted price, lower and upper limits, and update information.

**User interface development.** The user interface presents the forecast in a simple form. A user selects a vegetable and views the predicted weekly retail price, prediction interval, recent price trend, forecast date, and model update time. The interface must also state that the forecast is an estimate and can be affected by sudden market or weather events.

## 3.12 Model Testing and Validation

Testing was completed at data, feature, model, and prototype levels. Data tests checked date formats, weekly ordering, required columns, duplicate records, missing target values, and impossible numerical values. Integration tests checked that each vegetable week received the correct district weather record and active diesel price.

Feature tests checked that lag and rolling variables were based only on earlier weeks. For example, price_lag_1 at week t was compared with the target price at week t-1. Similar checks were applied to rolling statistics and lagged weather and fuel variables. These tests were important for preventing data leakage.

Scientific model testing used walk-forward validation and the final unseen 52-week holdout period. SARIMAX reliability was checked through residual plots, residual autocorrelation, and the Ljung-Box test. LSTM training and validation loss were monitored for overfitting. For all models, actual-versus-predicted plots, error distributions, and 80 percent interval coverage were reviewed.

Prototype testing sent sample requests for each vegetable and checked that the response contained the correct forecast fields and finite numerical values. Offline model predictions were compared with backend results to make sure that the same input produced the same forecast. Invalid vegetable names, missing recent data, response stability, and normal response time were also checked so that the system could return a clear message instead of failing unexpectedly.

## 3.13 Limitations

The study depends on secondary data. Therefore, its quality is limited by the completeness and consistency of the information published by HARTI, CEYPETCO, and Open-Meteo. PDF table formats can change, and some weeks or variables may be missing. Wholesale prices are not available consistently for every selected vegetable.

Weather is represented by a selected district coordinate and cannot describe the exact conditions experienced by every farm that supplies the market. Diesel price represents an important transport-cost factor, but it does not capture every transport expense. The model also does not include all possible causes of vegetable price change, such as production quantity, market arrivals, imports, exports, road conditions, storage, holidays, policy changes, and sudden supply shocks.

The common modelling period is shorter than the complete historical vegetable price period because the required external data do not cover every early week. Deep-learning models can also be limited by the number of weekly observations available. In addition, future weather and fuel conditions may change unexpectedly.

Forecasting always contains uncertainty. The selected model is based on patterns found in historical data, but future market conditions can be different. The system therefore provides an estimate and prediction interval rather than a guaranteed future price. Models should be retrained and monitored when new weekly data become available.
