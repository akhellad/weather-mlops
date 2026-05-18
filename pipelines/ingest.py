"""Fetch daily weather observations from NOAA GSOD via BigQuery and persist to parquet."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUTPUT_PATH = Path(os.environ.get("INGEST_OUTPUT_PATH", "data/raw/weather.parquet"))
START_YEAR = int(os.environ.get("INGEST_START_YEAR", "2018"))
END_YEAR = int(os.environ.get("INGEST_END_YEAR", "2023"))
STATIONS = tuple(
    s.strip() for s in os.environ.get("INGEST_STATIONS", "071490,071560,071570").split(",")
)

QUERY = """
SELECT
    DATE(CAST(year AS INT64), CAST(mo AS INT64), CAST(da AS INT64)) AS date,
    CONCAT(stn, '-', wban) AS station_id,
    IF(temp = 9999.9, NULL, (temp - 32) * 5 / 9) AS temp_c,
    IF(dewp = 9999.9, NULL, (dewp - 32) * 5 / 9) AS dewp_c,
    IF(wdsp = '999.9', NULL, SAFE_CAST(wdsp AS FLOAT64) * 0.514444) AS wind_ms,
    IF(prcp = 99.99, 0, prcp * 25.4) AS prcp_mm,
    IF(slp = 9999.9, NULL, slp) AS slp_hpa
FROM `bigquery-public-data.noaa_gsod.gsod*`
WHERE
    _TABLE_SUFFIX BETWEEN @start_year AND @end_year
    AND stn IN UNNEST(@stations)
ORDER BY date, station_id
"""


def fetch() -> pd.DataFrame:
    client = bigquery.Client(project=os.environ.get("BQ_PROJECT_ID"))
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_year", "STRING", str(START_YEAR)),
            bigquery.ScalarQueryParameter("end_year", "STRING", str(END_YEAR)),
            bigquery.ArrayQueryParameter("stations", "STRING", list(STATIONS)),
        ]
    )
    log.info("Querying GSOD for stations=%s years=%d-%d", STATIONS, START_YEAR, END_YEAR)
    df = client.query(QUERY, job_config=job_config).to_dataframe()
    log.info("Fetched %d rows", len(df))
    return df


def main() -> None:
    df = fetch()
    if df.empty:
        raise RuntimeError("Empty result set; check station codes and year range")

    df["date"] = pd.to_datetime(df["date"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    log.info("Wrote %s (%d rows, %d stations)", OUTPUT_PATH, len(df), df["station_id"].nunique())


if __name__ == "__main__":
    main()