"""
Data pipeline for Worcester County house price data.

Fetches sold-home data from Redfin's Stingray CSV endpoint (matching the
approach in Redfin_api.py), cleans it, and caches locally. Falls back to
static house_data.csv if the network fetch fails.
"""

import os
import time
import datetime
import re
import warnings
from io import StringIO

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Redfin region IDs — discovered from working Redfin_api.py and Redfin URLs
# region_type: 6 = city/town
# ---------------------------------------------------------------------------
REGIONS = {
    "Harvard": {"id": 29550, "type": 6},
}

_BASE_URL = (
    "https://www.redfin.com/stingray/api/gis-csv?"
    "al=1&market=boston&ord=redfin-recommended-asc"
    "&region_id={region_id}&region_type={region_type}"
    "&sold_within_days={sold_within_days}"
    "&status=9&uipt=1,2,3,4,5,6&v=8"
)

_HEADERS = {
    "User-Agent": "Mozilla/5.0",
}

_PAGE_SIZE = 100
_MAX_PAGES = 50


# ---- Fetching ---------------------------------------------------------------

def fetch_redfin_data(
    region_id: int,
    region_type: int,
    sold_within_days: int = 365,
    retries: int = 2,
) -> pd.DataFrame:
    """Fetch all pages of sold-home CSV from Redfin for one region."""
    base = _BASE_URL.format(
        region_id=region_id,
        region_type=region_type,
        sold_within_days=sold_within_days,
    )
    frames = []
    for page in range(1, _MAX_PAGES + 1):
        url = f"{base}&num_homes={_PAGE_SIZE}&offset={(page - 1) * _PAGE_SIZE}"
        last_err = None
        for attempt in range(1, retries + 1):
            try:
                resp = requests.get(url, headers=_HEADERS, timeout=30)
                if resp.status_code != 200 or not resp.text.strip():
                    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
                df = pd.read_csv(StringIO(resp.text))
                if df.empty:
                    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
                frames.append(df)
                last_err = None
                break
            except Exception as exc:
                last_err = exc
                if attempt < retries:
                    time.sleep(3)
        if last_err:
            warnings.warn(f"Page {page} failed for region_id={region_id}: {last_err}")
            break
        time.sleep(1)  # polite delay between pages

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_worcester_county_data(
    sold_within_days: int = 365,
    regions: dict | None = None,
) -> pd.DataFrame:
    """Fetch and concatenate data for Worcester County towns."""
    regions = regions or REGIONS
    frames = []
    for name, info in regions.items():
        print(f"  Fetching {name} (region_id={info['id']}) …")
        df = fetch_redfin_data(info["id"], info["type"], sold_within_days)
        if not df.empty:
            print(f"    Got {len(df)} rows")
            frames.append(df)
        else:
            print(f"    No data returned")
        time.sleep(2)  # polite delay between regions
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---- Cleaning ----------------------------------------------------------------

def _snake_case(name: str) -> str:
    """Convert a column name to snake_case."""
    name = re.sub(r"\(.*?\)", "", name).strip()
    name = name.replace("$/", "price_per_").replace("/", "_")
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", "_", name.strip()).lower()
    return name


def clean_and_format(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise columns, parse types, deduplicate, drop sparse cols."""
    df = df.copy()

    # --- column names ---
    df.columns = [_snake_case(c) for c in df.columns]

    # Normalise common aliases
    renames = {
        "state_or_province": "state",
        "zip_or_postal_code": "zip_code",
        "price_per_square_feet": "price_per_sqft",
        "hoa_month": "hoa_month",
        "mls": "mls_id",
    }
    df.rename(columns={k: v for k, v in renames.items() if k in df.columns}, inplace=True)

    # --- parse sold_date ---
    if "sold_date" in df.columns:
        df["sold_date"] = pd.to_datetime(df["sold_date"], format="mixed", dayfirst=False)

    # --- numeric coercion ---
    for col in ["price", "square_feet", "lot_size", "beds", "baths",
                "year_built", "days_on_market", "price_per_sqft",
                "latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r"[,$]", "", regex=True),
                errors="coerce",
            )

    # --- deduplication ---
    subset_cols = [c for c in ["address", "city", "sold_date", "price"] if c in df.columns]
    if subset_cols:
        before = len(df)
        df.drop_duplicates(subset=subset_cols, inplace=True)
        dropped = before - len(df)
        if dropped:
            print(f"  Dropped {dropped} duplicate rows.")

    # --- drop columns with >90% missing ---
    thresh = 0.10 * len(df)
    sparse = [c for c in df.columns if df[c].notna().sum() < thresh]
    if sparse:
        df.drop(columns=sparse, inplace=True)
        print(f"  Dropped sparse columns: {sparse}")

    # --- drop metadata / non-useful columns ---
    drop_if_present = ["favorite", "interested", "source", "status", "sale_type"]
    df.drop(columns=[c for c in drop_if_present if c in df.columns], inplace=True)

    df.reset_index(drop=True, inplace=True)
    return df


# ---- Main entry point --------------------------------------------------------

_STATIC_FALLBACK = os.path.join(os.path.dirname(__file__), "house_data.csv")
_DEFAULT_CACHE = os.path.join(os.path.dirname(__file__), "data", "redfin_data.csv")
_CACHE_MAX_AGE_DAYS = 7


def load_data(
    use_cache: bool = True,
    cache_path: str = _DEFAULT_CACHE,
    sold_within_days: int = 365,
) -> pd.DataFrame:
    """
    Load Worcester County housing data.

    Priority:
    1. Fresh cache (< 7 days old)
    2. Live Redfin fetch → save to cache
    3. Fallback to static house_data.csv
    """
    # --- try cache ---
    if use_cache and os.path.exists(cache_path):
        age = datetime.datetime.now() - datetime.datetime.fromtimestamp(
            os.path.getmtime(cache_path)
        )
        if age.days < _CACHE_MAX_AGE_DAYS:
            print(f"Loading cached data ({age.days}d old): {cache_path}")
            df = pd.read_csv(cache_path, parse_dates=["sold_date"])
            return df

    # --- try live fetch ---
    try:
        print("Fetching fresh data from Redfin …")
        raw = fetch_worcester_county_data(sold_within_days=sold_within_days)
        if not raw.empty:
            df = clean_and_format(raw)
            # Validate: check that fetched data contains expected state
            if "state" in df.columns and len(df) > 0:
                ma_frac = (df["state"] == "MA").mean()
                if ma_frac < 0.5:
                    warnings.warn(
                        f"Fetched data has only {ma_frac:.0%} MA records. "
                        "Region IDs may be wrong. Falling back to static CSV."
                    )
                    raise ValueError("Bad region IDs — wrong geography")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            df.to_csv(cache_path, index=False)
            print(f"Cached {len(df)} rows → {cache_path}")
            return df
    except Exception as exc:
        warnings.warn(f"Live fetch failed: {exc}")

    # --- fallback to static CSV ---
    if os.path.exists(_STATIC_FALLBACK):
        print(f"Falling back to static file: {_STATIC_FALLBACK}")
        df = pd.read_csv(_STATIC_FALLBACK)
        df = clean_and_format(df)
        return df

    raise FileNotFoundError("No data source available.")
