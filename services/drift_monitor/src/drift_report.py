"""Compute data drift between reference and current datasets using Evidently."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

log = logging.getLogger(__name__)

_VALUE_DRIFT_RE = re.compile(r"ValueDrift\(column=([^,]+),method=([^,]+),threshold=([\d.]+)\)")


@dataclass
class DriftSummary:
    drift_detected: bool
    drift_share: float
    n_features: int
    n_drifted: int
    threshold: float
    per_column: dict[str, dict[str, float | bool]]


def _extract_summary(report_dict: dict, threshold: float) -> DriftSummary:
    metrics = report_dict.get("metrics", [])

    dataset_metric = next(
        (m for m in metrics if m.get("metric_name", "").startswith("DriftedColumnsCount")),
        None,
    )
    if dataset_metric is None:
        raise RuntimeError("Could not locate DriftedColumnsCount metric in Evidently report")

    value = dataset_metric["value"]
    n_drifted = int(value["count"])
    drift_share = float(value["share"])

    per_column: dict[str, dict] = {}
    for m in metrics:
        match = _VALUE_DRIFT_RE.match(m.get("metric_name", ""))
        if not match:
            continue
        col, _method, col_threshold = match.groups()
        score = float(m["value"])
        per_column[col] = {
            "drift_score": score,
            "threshold": float(col_threshold),
            "drifted": score > float(col_threshold),
        }

    n_features = len(per_column)
    drift_detected = drift_share >= threshold

    return DriftSummary(
        drift_detected=drift_detected,
        drift_share=drift_share,
        n_features=n_features,
        n_drifted=n_drifted,
        threshold=threshold,
        per_column=per_column,
    )


def compute_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    threshold: float = 0.3,
) -> DriftSummary:
    report = Report([DataDriftPreset()])
    snapshot = report.run(current_data=current, reference_data=reference)
    report_dict = snapshot.dict()
    summary = _extract_summary(report_dict, threshold)
    log.info(
        "Drift: detected=%s share=%.2f drifted=%d/%d threshold=%.2f",
        summary.drift_detected, summary.drift_share,
        summary.n_drifted, summary.n_features, summary.threshold,
    )
    return summary