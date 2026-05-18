"""Test that drift simulation is deterministic and shifts the right columns."""

from __future__ import annotations


def test_simulator_is_deterministic_with_seed(synthetic_features):
    from src.simulator import inject_seasonal_drift

    a = inject_seasonal_drift(synthetic_features.drop(columns=["date", "target_temp"]), seed=42)
    b = inject_seasonal_drift(synthetic_features.drop(columns=["date", "target_temp"]), seed=42)

    assert (a == b).all().all()


def test_simulator_shifts_temperature_upward(synthetic_features):
    from src.simulator import inject_seasonal_drift

    base = synthetic_features.drop(columns=["date", "target_temp"])
    shifted = inject_seasonal_drift(base, temp_shift_c=6.0, seed=42)

    assert shifted["temp_c"].mean() > base["temp_c"].mean() + 5
    assert shifted["doy_sin"].equals(base["doy_sin"])
    assert shifted["month"].equals(base["month"])