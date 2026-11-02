"""Airflow DAG: nightly ETL -> training -> quality gate -> API reload.

    extract >> transform >> load >> train >> reload_api

* Data moves between tasks as Parquet in data/staging (never through XCom), and
  every step works partition by partition.
* Every run is written to the etl_runs audit table: `load` records a successful
  ETL, and a failure callback on every task records a failed run with the task
  name and error.
* Problems are visible as failed tasks: the run fails if the transform drops
  more than 5% of rows, if the new model is rejected by the quality gate (the
  current champion keeps serving), or if the API cannot be told to reload.
Set REPO_DIR to the repository path on the Airflow worker.
"""
import os
import sys
import time
from datetime import datetime, timedelta, timezone

from airflow.sdk.exceptions import AirflowFailException
from airflow.sdk import dag, task

REPO_DIR = os.getenv("REPO_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_DIR)
STAGING = os.path.join(REPO_DIR, "data", "staging")
API_URL = os.getenv("API_URL", "http://localhost:8000")


def _record_failure(context):
    """on_failure_callback: write the failed run to the etl_runs audit table."""
    from etl.load import record_run
    ti, run = context["ti"], context["dag_run"]
    record_run(run.start_date or datetime.now(timezone.utc), 0, "failed", {},
               {"dag_run": run.run_id, "task": ti.task_id, "error": repr(context.get("exception"))})


default_args = {
    "owner": "sina",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=30),
    "on_failure_callback": _record_failure,
}


@dag(dag_id="superstore_etl_mlops", schedule="0 2 * * *", start_date=datetime(2026, 11, 1),
     catchup=False, max_active_runs=1, default_args=default_args,
     tags=["dds8530", "etl", "mlops"])
def superstore_etl_mlops():

    @task
    def extract() -> dict:
        from etl.extract import extract_all
        t0 = time.perf_counter()
        _, n, hol = extract_all(os.path.join(STAGING, "orders"))
        hol.to_parquet(os.path.join(STAGING, "holidays.parquet"), index=False)
        return {"orders": n, "holidays": len(hol), "seconds": round(time.perf_counter() - t0, 3)}

    @task
    def transform(extracted: dict) -> dict:
        import pandas as pd
        from etl.transform import transform as run_transform
        t0 = time.perf_counter()
        hol = pd.read_parquet(os.path.join(STAGING, "holidays.parquet"))
        _, report = run_transform(os.path.join(STAGING, "orders"), hol,
                                  os.path.join(STAGING, "features"))
        if report["rows_out"] < 0.95 * extracted["orders"]:            # data-quality gate
            raise AirflowFailException(f"too many rows dropped: {report}")
        report["seconds"] = round(time.perf_counter() - t0, 3)
        report["extract"] = extracted
        return report

    @task
    def load(report: dict, ds=None, dag_run=None) -> int:
        import dask.dataframe as dd
        from etl.load import load_s3, load_warehouse, record_run
        t0 = time.perf_counter()
        features = dd.read_parquet(os.path.join(STAGING, "features"))
        n = load_warehouse(features)
        report["s3_keys"] = load_s3(features, ds)
        timings = {"extract": report["extract"]["seconds"], "transform": report["seconds"],
                   "load": round(time.perf_counter() - t0, 3)}
        record_run(dag_run.start_date, n, "success", timings, {**report, "dag_run": dag_run.run_id})
        return n

    @task
    def train(rows: int) -> dict:
        from ml.train import main
        if main() != 0:            # quality gate rejected the new model
            raise AirflowFailException("quality gate rejected the new model; "
                                       "the current @champion stays in service (see train log)")
        return {"promoted": True}

    @task
    def reload_api(_: dict) -> str:
        import requests
        r = requests.post(f"{API_URL}/reload", timeout=60)   # errors fail the task (and retry)
        r.raise_for_status()
        return f"API now serving v{r.json()['version']}"

    reload_api(train(load(transform(extract()))))


superstore_etl_mlops()
