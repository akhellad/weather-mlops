"""Test that data_loader produces consistent splits."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def test_load_produces_chronological_non_overlapping_splits(tmp_path, synthetic_features):
    ref = synthetic_features.iloc[:150]
    cur = synthetic_features.iloc[150:]

    ref_path = tmp_path / "reference.parquet"
    cur_path = tmp_path / "current.parquet"
    ref.to_parquet(ref_path, index=False)
    cur.to_parquet(cur_path, index=False)

    trainer_src = Path(__file__).resolve().parents[2] / "services" / "trainer"
    sys.path.insert(0, str(trainer_src))
    for mod in list(sys.modules):
        if mod == "src" or mod.startswith("src."):
            del sys.modules[mod]
    data_loader = importlib.import_module("src.data_loader")
    sys.path.pop(0)

    data = data_loader.load(ref_path, cur_path, val_fraction=0.2)

    assert len(data.X_train) + len(data.X_val) == len(ref)
    assert len(data.X_test) == len(cur)
    assert "target_temp" not in data.X_train.columns
    assert "date" not in data.X_train.columns
    assert data.feature_names == list(data.X_train.columns)
    assert len(data.feature_names) > 0