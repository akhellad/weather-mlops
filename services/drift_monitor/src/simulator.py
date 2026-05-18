"""Simulate seasonal drift by shifting the current dataset towards summer values."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

NUMERIC_FEATURE_PREFIXES = ("temp_c", "dewp_c", "wind_ms", "prcp_mm")


def inject_seasonal_drift(
    current: pd.DataFrame,
    temp_shift_c: float = 6.0,
    dewp_shift_c: float = 4.0,
    prcp_factor: float = 1.6,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = current.copy()

    temp_cols = [c for c in out.columns if c.startswith("temp_c")]
    dewp_cols = [c for c in out.columns if c.startswith("dewp_c")]
    prcp_cols = [c for c in out.columns if c.startswith("prcp_mm")]

    out[temp_cols] = out[temp_cols] + temp_shift_c + rng.normal(0, 0.5, size=(len(out), len(temp_cols)))
    out[dewp_cols] = out[dewp_cols] + dewp_shift_c + rng.normal(0, 0.5, size=(len(out), len(dewp_cols)))
    out[prcp_cols] = (out[prcp_cols] * prcp_factor).clip(lower=0)

    log.info(
        "Injected seasonal drift: temp+%.1f°C, dewp+%.1f°C, prcp×%.1f",
        temp_shift_c, dewp_shift_c, prcp_factor,
    )
    return out