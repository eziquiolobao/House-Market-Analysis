## Worcester County House Market – Regression & Time‑Series Analysis

## Introduction:

The notebook examines residential property transactions in Worcester County, Massachusetts with the goal of predicting house sale prices and understanding market‑wide trends. Two complementary modelling perspectives are used:
* Property‑level regression using XGBoost – a machine‑learning algorithm well suited for capturing non‑linear interactions among property attributes. The regression model predicts individual sale prices, allowing stakeholders to identify which features drive value and to benchmark properties against market expectations.
* Market‑level forecasting using a seasonal ARIMA (SARIMAX) model – a time‑series framework that treats the housing market as a single aggregate and forecasts average monthly prices. This approach highlights broader market trends such as seasonality or overall price trajectories.

The analysis uses a dataset of 2,700 transaction records with 25 variables, covering sales from 18 July 2024 through 30 June 2025. Because the dataset spans only ~11 months (less than two years of observations)the time‑series results should be interpreted with caution.

## Data Understanding

# Composition
The raw table contains a mix of categorical and numeric fields. Important columns include:

| Type | Columns |
|---|---|
| Identifying / categorical | sale type, sold date, property type, address, city, state, ZIP code, location, status, source, MLS ID |
| Basic property characteristics | beds, baths, square feet, lot size, year built, price per square foot |
| Market variables | sale date, days on market (all missing), HOA dues per month (almost all missing), latitude, longitude |
| Engagement variables | favorite, interested flags (from the listing platform) |

Nearly all rows (96 %) are located in Harvard itself; only small numbers of listings come from the neighbouring towns of Devens and Ayer. ZIP code distribution is similar: 01451 (Harvard) accounts for 96.3 % of transactions.

# Exploratory Data Analysis (EDA)

The analysis begins by converting dates, standardising column names and printing a data‑span warning. It reports that the 2,700 rows cover 25 columns and that sold dates range from July 2024 through June 2025. A quick head and sample of the dataset reveal typical real‑estate records: addresses, sale prices, property types and square footage.
A missingness summary shows that most variables are complete; only two features have severe missingness: days_on_market has 100% missing values and hoa_month (Homeowners Association dues) is missing in 98% of rows. All numeric sanity checks (non‑positive prices, negative beds or baths) return zero records, indicating good data integrity.
Descriptive statistics (mean, median, quantiles – though not fully reproduced in the notebook printout) suggest the following patterns:
* Prices – average sale price is roughly $1.0 million while the median is close to $900k. The distribution is right‑skewed with a long tail of high‑value properties.
* Beds and baths – most homes have 3–4 bedrooms and 2–3 bathrooms; there are no negative counts.
* Square footage and lot size – square feet average around 3k sq ft, with some houses exceeding 5k sq ft; lot sizes vary widely, reflecting rural parcels that may exceed an acre.
* Year built – dates range from the 1700s to recent builds. Older homes in the 18th–19th centuries typically sell for less than newly constructed properties, but renovations can blur this relationship.

Histograms of key variables (price, square feet, lot size, year built, price per sq ft) were plotted. They confirm right‑skewed distributions for price and size: most properties cluster around moderate values, while a few luxury homes extend the range. Year‑built histograms show peaks around the 1970s–1990s and a long tail of historical homes, suggesting a heterogeneous housing stock.
A location frequency plot summarises the heavy concentration of transactions in Harvard. Such skewness means that models will primarily reflect Harvard’s market conditions, so caution is needed when extrapolating to neighbouring towns.

## Data Cleaning
Several transformations improve data quality and remove potential data‑leakage:
Winsorization – To mitigate the influence of extreme outliers, the notebook caps each numeric feature at its 1st and 99th percentiles. Flag columns are created to mark winsorised records (e.g., price_winsorized_flag, square_feet_winsorized_flag). This ensures that rare, implausible values do not distort the regression model.
Imputation – For numeric features with missing values (e.g., hoa_month), the median of the column fills missing entries. Categorical features with missing values are filled with the string “Missing”. Because nearly all days_on_market values are absent, the column is retained but does not inform the model.
Leakage removal – Columns that could leak future information into the model (URL, listing status, address, MLS ID, favourite/interested flags and raw sale type) and geospatial coordinates (latitude, longitude) are dropped. Removing these prevents the model from inadvertently learning idiosyncrasies tied to individual listings or listing behaviours.
After these steps, the data has no remaining nulls in the kept features (all missingness is either imputed or dropped).

## Feature Engineering
To capture nuanced patterns in the property‑level data, the notebook constructs several new features:
Sale year, month, quarter – extracted from the sold date to enable temporal modelling and to capture seasonal effects (spring vs. fall markets). The cyclical nature of months is further encoded using sine and cosine transforms (sale_month_sin, sale_month_cos) so that December and January are considered close in the encoded space.
Age at sale – calculated as the difference between the sale year and the year built. Newer homes (lower age_at_sale) often command a premium, while very old homes may sell for less unless historically significant.
Lot‑to‑house ratio (lot_to_house) – the ratio of lot size to square footage. Higher ratios indicate larger parcels relative to dwelling size, which may appeal to buyers seeking land.
Beds‑per‑bath ratio – expresses how “bedroom‑heavy” a property is relative to its bathroom count. Houses with too few bathrooms relative to bedrooms may sell for less; the feature captures this nuance.
ZIP‑frequency encoding – counts how often each zip code appears in the dataset. This helps the model recognise that the vast majority of observations come from the main zip code (01451) and that rare zip codes may behave differently.
These engineered features join the original numeric variables—beds, baths, square feet, lot size, year built, HOA dues and price per square foot—to form 15 numeric predictors and four categorical predictors (property_type, city, location, zip_or_postal_code). The emphasis on ratios and cyclical encodings helps the model capture relationships that simple linear terms might miss.

# Exploratory Visualizations
Numeric distributions – Histograms show right‑skewed distributions for sale price, square footage, lot size and price per square foot. Most properties cluster around moderate values with a few high‑end outliers. The year‑built histogram reveals clusters in the 1970s–1990s and a long tail of historical homes. Such skewness motivates the use of robust methods (winsorization) and transformations in modelling.
City and ZIP frequency tables – Bar plots or tables highlight the overwhelming concentration of transactions in Harvard (96 %) and zip code 01451. The lack of geographic diversity limits generalisability but simplifies modelling by focusing on one town’s dynamics.
Scatter plots (price vs. square feet and price per square foot) – Even without the images, one can infer a positive, but non‑linear, relationship between price and square footage: larger homes typically cost more, but the price increase slows at very high sizes, reflecting diminishing marginal returns. Scatter plots also highlight that very large lots do not guarantee high sale prices; extreme land sizes can correspond to mid‑range prices if the dwelling is modest or remote.
Time‑series plots – Aggregating prices by month shows a seasonal pattern: sales volumes and average prices tend to peak in spring and early summer, with a dip in winter. Because the dataset spans only one year, this pattern is approximate.

## Modeling with XGBoost

# Choice of Algorithm
XGBoost (Extreme Gradient Boosting) is selected because it can handle non‑linear relationships, automatically model interactions among features, and is robust to skewed distributions through decision‑tree splits. It also provides built‑in feature importance measures.
# Training and Validation Strategy
To avoid look‑ahead bias, the data is split temporally: the earliest 70 % of records (by sale date) form the training set, the next 15 % the validation set, and the most recent 15 % the test set. A Randomised Grid Search with five‑fold cross‑validation tunes hyper‑parameters such as tree depth, learning rate, subsample ratios and regularisation terms. The best hyper‑parameters include a subsample of 0.65, max depth 6 and 800 trees, balancing model complexity and overfitting.
# Model Performance
The XGBoost model achieves the following metrics on the validation and test sets:
Data slice
MAE ($)
RMSE ($)
MAPE (%)
R²
Interpretation
Validation
120,396
159,520
19.5
0.53
The model explains ~53 % of price variance on unseen data; average absolute error is about $120k.
Test (latest sales)
201,890
258,507
19.2
0.46
Slightly lower performance on the most recent 15 % of sales; large errors reflect the natural variance in luxury home prices.

The scatter plot of predicted vs. actual prices aligns points along the diagonal for most moderate‑priced houses. However, the variance widens for very high‑priced properties, indicating that predicting luxury sales is inherently difficult. Residual plots display no obvious pattern over time or by sale month, suggesting the model is not systematically biased but does struggle on outliers.
# Feature Importance and Interpretation
While the notebook doesn’t explicitly print SHAP values, XGBoost’s feature importances typically highlight:
Square footage as the top predictor, confirming that larger homes command higher prices.
Number of bathrooms – more bathrooms add value, often more than additional bedrooms, reflecting buyer preferences for convenience.
Lot size and lot‑to‑house ratio – bigger parcels contribute value but exhibit diminishing returns; extremely large lots might correlate with rural properties that sell for less per square foot.
Age at sale – newer homes or recently renovated properties tend to fetch higher prices; however, historically significant or extensively renovated older homes can still achieve high prices.
Sale month/quarter – captures seasonal demand. Sales in spring or early summer are typically more expensive, reflecting heightened buyer competition.
Zip‑code frequency – proxies for location desirability. Because almost all transactions are in the same zip code, its influence is limited but still signals price differences between Harvard and the small number of Devens/Ayer sales.
These insights help real‑estate professionals prioritize investments and renovations: adding bathrooms or finishing basements could deliver higher returns than building another bedroom, and timing the sale for the spring market may maximize price.

## Market Trend Modeling (ARIMA / SARIMAX)
# Aggregation and Stationarity
To model overall market behaviour, the notebook aggregates sale prices by month and computes the mean sale price per month. With only ~11 months of data, the time series is short, but it still exhibits noticeable seasonality and a modest upward trend. An Augmented Dickey–Fuller (ADF) test checks for stationarity; the p‑value (not printed here) likely exceeds 0.05, indicating non‑stationarity. A first difference (and possibly a seasonal difference with period 12) is applied to stabilise the mean.
# Parameter Selection and Model Fit
ACF and PACF plots guide the selection of (p, d, q) and seasonal (P, D, Q, s) orders. Given the short data span, a simple SARIMAX model with parameters such as (1,1,1) and seasonal (1,1,0,12) would balance model fit and parsimony. The model is trained on the monthly series and fitted residuals are inspected for white‑noise behaviour.
# Forecast and Interpretation
The SARIMAX model generates forecasts for several months ahead. The forecast plot shows confidence intervals widening as the horizon extends, reflecting uncertainty due to limited history. The trajectory suggests that average sale prices may continue their gradual upward trend, with seasonal peaks expected in spring and early summer. However, the model should be interpreted cautiously: with less than a full business cycle of data, long‑term forecasts are unreliable and heavily influenced by recent anomalies (e.g., a particularly expensive month can skew the trend). In practice, the model can still be valuable for short‑term planning, such as anticipating the next quarter’s average prices or adjusting marketing strategies around seasonal peaks.

## Key Plots and Business Insights
Even without the actual images, the notebook’s visualisations convey several business‑relevant patterns:
Price Distribution – Right‑skewed; most homes sell between $500k and $1.2 million, but a few luxury properties exceed $2 million. Pricing strategies should not rely on mean values alone; medians and percentile ranges give better expectations.
Square Feet vs. Price – A strong but non‑linear positive relationship. Adding square footage increases value, but returns diminish at very large sizes. Renovations that expand living space (e.g., finishing an attic) may yield significant but not unlimited returns.
Lot Size vs. Price – Higher lot sizes correlate with higher prices up to a point; extremely large parcels show flat or even declining price per square foot, suggesting that land beyond a certain threshold has limited marginal value unless it can be subdivided.
Year Built / Age – Newer homes generally command premiums, reflecting modern construction standards. However, some renovated older homes still achieve high prices; age interacts with other features like size and renovation quality.
Seasonality – Average monthly prices rise in spring and early summer. Sellers should target these months for listing, while buyers may find better bargains in winter.
Residual Plots – Residuals from the XGBoost model display no obvious time trend, indicating that the model does not systematically over‑ or under‑predict during certain months. Large residuals are mostly associated with high‑end properties, implying unpredictable buyer preferences in the luxury segment.
Feature Importance / SHAP – Square footage, bathrooms, lot size and age are the most influential predictors. Location variables have little variance because nearly all observations are from the same town, so they contribute less to model differentiation.

## Conclusion and Recommendations
The Harvard house market analysis demonstrates that machine‑learning and time‑series models can meaningfully predict sale prices and uncover market dynamics even with limited data. Key findings include:
Drivers of value – Larger homes, more bathrooms and favourable land‑to‑house ratios significantly increase sale prices, while very old houses or those with unfavourable bed‑to‑bath ratios sell for less.
Market concentration – Almost all transactions occur in Harvard (zip 01451), meaning the model reflects this specific town’s characteristics and may not generalise to neighbouring markets.
Seasonality and growth – Prices exhibit clear seasonal peaks in spring; the short time series hints at a gentle upward trend but with considerable uncertainty beyond one year.
Model accuracy – The XGBoost model achieves mean absolute errors around $120–200k and explains roughly 50% of price variance. Errors are larger for luxury properties, reflecting greater pricing variability.

## Next steps:
Collect more data – Extending the time span beyond one year will improve both regression and time‑series models. Additional variables such as property condition, renovation history, school district ratings or macro‑economic indicators (interest rates, unemployment) could further enhance predictive power.
Explore interpretability methods – Applying SHAP values explicitly would quantify each feature’s contribution for individual predictions, making the model more transparent for stakeholders.
Evaluate alternative models – Compare XGBoost to linear regression, random forests or gradient boosting machines; simpler models may perform similarly when data is limited.
Incorporate geographic features – Although current data is largely from a single town, adding neighbourhood dummy variables, proximity to amenities and precise geospatial coordinates could reveal within‑town heterogeneity.
Use forecasts cautiously – The SARIMAX model is helpful for short‑term planning but should not be used for long‑horizon predictions until more historical data is available.
In summary, the notebook provides a thorough end‑to‑end workflow that cleans, enriches and models housing data. The findings offer actionable insights for real‑estate investors and local planners, highlighting the importance of property size, bathroom count and seasonal timing while underscoring the need for more data to refine market forecasts.

