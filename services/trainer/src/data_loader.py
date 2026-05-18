"""Load prepared datasets and produce train/val/test splits."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

TARGET = "target_temp"
NON_FEATURE_COLUMNS = ("date", TARGET)


@dataclass(frozen=True)
class Dataset:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_names: list[str]


def _split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=list(NON_FEATURE_COLUMNS))
    y = df[TARGET]
    return X, y


def load(reference_path: Path, current_path: Path, val_fraction: float = 0.2) -> Dataset:
    reference = pd.read_parquet(reference_path).sort_values("date").reset_index(drop=True)
    current = pd.read_parquet(current_path).sort_values("date").reset_index(drop=True)

    split_idx = int(len(reference) * (1 - val_fraction))
    train_df = reference.iloc[:split_idx]
    val_df = reference.iloc[split_idx:]

    X_train, y_train = _split_features_target(train_df)
    X_val, y_val = _split_features_target(val_df)
    X_test, y_test = _split_features_target(current)

    log.info(
        "Loaded splits: train=%d val=%d test=%d, features=%d",
        len(X_train), len(X_val), len(X_test), X_train.shape[1],
    )
    return Dataset(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=list(X_train.columns),
    )