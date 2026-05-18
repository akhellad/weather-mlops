"""Test the Evidently report parsing logic on a fixture."""

from __future__ import annotations

SAMPLE_REPORT = {
    "metrics": [
        {
            "metric_name": "DriftedColumnsCount(drift_share=0.5)",
            "value": {"count": 3.0, "share": 0.75},
        },
        {
            "metric_name": "ValueDrift(column=temp_c,method=Wasserstein distance (normed),threshold=0.1)",
            "value": 0.95,
        },
        {
            "metric_name": "ValueDrift(column=wind_ms,method=Wasserstein distance (normed),threshold=0.1)",
            "value": 0.20,
        },
        {
            "metric_name": "ValueDrift(column=month,method=Wasserstein distance (normed),threshold=0.1)",
            "value": 0.005,
        },
        {
            "metric_name": "ValueDrift(column=doy_sin,method=Wasserstein distance (normed),threshold=0.1)",
            "value": 0.005,
        },
    ]
}


def test_extract_summary_flags_dataset_drift_above_threshold():
    from src.drift_report import _extract_summary

    summary = _extract_summary(SAMPLE_REPORT, threshold=0.3)

    assert summary.drift_detected is True
    assert summary.drift_share == 0.75
    assert summary.n_drifted == 3
    assert summary.n_features == 4
    assert summary.per_column["temp_c"]["drifted"] is True
    assert summary.per_column["month"]["drifted"] is False


def test_extract_summary_respects_threshold():
    from src.drift_report import _extract_summary

    report = {
        "metrics": [
            {
                "metric_name": "DriftedColumnsCount(drift_share=0.5)",
                "value": {"count": 1.0, "share": 0.1},
            },
            {
                "metric_name": "ValueDrift(column=temp_c,method=Wasserstein distance (normed),threshold=0.1)",
                "value": 0.5,
            },
        ]
    }
    summary = _extract_summary(report, threshold=0.3)

    assert summary.drift_detected is False
    assert summary.drift_share == 0.1