# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A small ML project that predicts single-family home sale prices around Harvard, MA (Worcester County). It has three pieces: a data module (`data_pipeline.py`), an analysis/training notebook (`harvard_house_market_analysis.ipynb`), and a Streamlit app (`app.py`) deployed on Streamlit Community Cloud at https://mahousepriceprediction.streamlit.app/.

## Commands

```bash
# Setup (the local .venv already exists and uses Python 3.13)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the app
streamlit run app.py

# Retrain: run the notebook top to bottom from the repo root. It writes model/best_model.pkl.
jupyter nbconvert --to notebook --execute --inplace harvard_house_market_analysis.ipynb

# Load data on its own (uses a cache less than 7 days old, else fetches from Redfin, else falls back to the snapshot)
python -c "import data_pipeline; print(data_pipeline.load_data().shape)"
```

There is no test suite, linter, or build step. `requirements.txt` holds the runtime dependencies for Streamlit Cloud. Jupyter is installed in `.venv` but is not listed in `requirements.txt`. Keep it that way unless the deploy needs it.

## Architecture

**Data flow:** Redfin Stingray CSV endpoint → `data_pipeline.load_data()` → committed snapshot `data/harvard_sales_2026-03.csv` → notebook (via `load_snapshot()`) → `model/best_model.pkl` → `app.py`.

- The notebook trains on the frozen snapshot, not on live data, so its outputs and the numbers written in its markdown stay reproducible. All randomness is seeded (`RANDOM_STATE = 42`).
- **If you change the data, the features or the models, re-execute the notebook and update the hard-coded numbers** in its TL;DR, takeaways and conclusions, and in the README's Key Results.
- `data_pipeline.load_data()` tries three sources in order:
  1. `data/redfin_data.csv`, if it is less than 7 days old. Only this cache file is gitignored. The snapshot next to it is committed.
  2. A live fetch through `fetch_worcester_county_data`. This walks the towns in the `REGIONS` dict, pages through results, and waits between requests on purpose. The result is kept only if at least half the rows are from MA, which catches wrong region IDs.
  3. The committed snapshot, which is already cleaned.
- `clean_and_format()` converts column names to snake_case, coerces numeric types, drops rows without a price (Redfin adds a disclaimer row), deduplicates on address/city/sold_date/price, and drops columns that are more than 90% empty. Downstream code depends on the snake_case names it produces (`square_feet`, `lot_size`, `sold_date`, `price`, and so on).
- `REGIONS` currently holds only Harvard, even though the README and the app's City dropdown mention Devens and Ayer. City is not a model feature. The dropdown is display-only.

**The model artifact is the contract between the notebook and the app.** `best_model.pkl` is a joblib dict with these keys: `pipeline` (any fitted estimator with `predict`), `features`, `model_name`, `cv_mae`, `price_stats` (min/max/median/mean), and `n_samples`. `app.py` reads all of them. If you change the export cell (the last code cell of the notebook), update `app.py` to match.

**Features are defined once.** `data_pipeline.FEATURES` and `data_pipeline.add_features()` are used by both the notebook and `app.py` (`build_feature_vector()`). `add_features()` expects `year_built`, `sale_year` and `sale_month`, and computes `house_age = sale_year - year_built` plus the sin/cos month encoding. The app passes the current year as `sale_year`.

**Modeling details worth knowing:**
- The modeling set is complete single-family rows only (`property_type == "Single Family Residential"`), which leaves 47 rows.
- The candidates are a `DummyRegressor` median baseline, Linear Regression, Ridge (`RidgeCV`, which tunes alpha internally), Random Forest and XGBoost. All are scored with `RepeatedKFold(5, 10)`. There is no time-based holdout, because it would contain only about 10 sales.
- The model with the lowest CV MAE wins. It's refit on all rows before export, and `cv_mae` is the app's ± range.

**Deployment:** Streamlit Community Cloud serves `app.py` from `main` and uses the committed `model/best_model.pkl`. Retraining changes nothing in production until the new `.pkl` is committed and pushed. The app's "Refresh Data" button only refreshes the local cache. The notebook never reads that cache, so a new snapshot has to be written on purpose.
