"""Run the full ETL pipeline (full refresh):  python -m etl.run_etl

Airflow calls the same three functions as separate tasks (dags/superstore_pipeline.py).
"""
from datetime import date, datetime, timezone

from config import DATA_DIR
from etl.extract import extract_all
from etl.load import load_s3, load_warehouse, record_run
from etl.logging_utils import get_logger, timed_step
from etl.transform import transform

log = get_logger("etl")
FEATURES_DIR = DATA_DIR / "staging" / "features"


def run() -> dict:
    started, timings, report, rows = datetime.now(timezone.utc), {}, {}, 0
    try:
        with timed_step(log, "extract", timings):
            staging, _, holidays_df = extract_all()
        with timed_step(log, "transform", timings):
            features, report = transform(staging, holidays_df, FEATURES_DIR)
        with timed_step(log, "load", timings):
            rows = load_warehouse(features)
            report["s3_keys"] = load_s3(features, date.today().isoformat())
    except Exception:
        record_run(started, rows, "failed", timings, report)
        raise
    record_run(started, rows, "success", timings, report)
    log.info("ETL finished: %d rows, timings %s", rows, timings)
    return {"rows": rows, "timings": timings, "report": report}


if __name__ == "__main__":
    run()
