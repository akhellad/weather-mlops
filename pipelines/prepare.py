"""Aggregate raw weather data, build features and target, split into reference and current sets."""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RAW_PATH = Path(os.environ.get("PREPARE_INPUT_PATH", "data/raw/weather.parquet"))
REFERENCE_PATH = Path(os.environ.get("PREPARE_REFERENCE_PATH", "data/processed/reference.parquet"))
CURRENT_PATH = Path(os.environ.get("PREPARE_CURRENT_PATH", "data/processed/current.parquet"))
SPLIT_DATE = pd.Timestamp(os.environ.get("PREPARE_SPLIT_DATE", "2023-01-01"))

LAGS = (1, 2, 3, 7)
FEATURE_COLUMNS = ["temp_c", "dewp_c", "wind_ms", "prcp_mm"]


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.drop(columns=["station_id", "slp_hpa"])
        .groupby("date", as_index=False)
        .mean()
        .sort_values("date")
        .reset_index(drop=True)
    )
    full_index = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily = daily.set_index("date").reindex(full_index).rename_axis("date").reset_index()
    daily[FEATURE_COLUMNS] = daily[FEATURE_COLUMNS].interpolate(method="linear", limit=3)
    return daily


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in FEATURE_COLUMNS:
        for lag in LAGS:
            out[f"{col}_lag{lag}"] = out[col].shift(lag)

    out["day_of_year"] = out["date"].dt.dayofyear.astype("float64")
    out["month"] = out["date"].dt.month.astype("float64")
    radians = 2 * math.pi * out["day_of_year"] / 365.25
    out["doy_sin"] = np.sin(radians)
    out["doy_cos"] = np.cos(radians)

    out["target_temp"] = out["temp_c"].shift(-1)

    return out.dropna().reset_index(drop=True)


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw data not found: {RAW_PATH}. Run ingest.py first.")

    raw = pd.read_parquet(RAW_PATH)
    log.info("Loaded %d raw rows", len(raw))

    daily = aggregate_daily(raw)
    log.info("Aggregated to %d daily rows", len(daily))

    features = build_features(daily)
    log.info("Built features, %d usable rows after lag/target trimming", len(features))

    reference = features[features["date"] < SPLIT_DATE]
    current = features[features["date"] >= SPLIT_DATE]

    if reference.empty or current.empty:
        raise RuntimeError(f"Split at {SPLIT_DATE} produced an empty set")

    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    reference.to_parquet(REFERENCE_PATH, index=False)
    current.to_parquet(CURRENT_PATH, index=False)
    log.info("Wrote %s (%d rows)", REFERENCE_PATH, len(reference))
    log.info("Wrote %s (%d rows)", CURRENT_PATH, len(current))


if __name__ == "__main__":
    main()