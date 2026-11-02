"""Extract phase: three sources, each read in batches.

1. Flat file (CSV)        - archived 2014-2016 orders, read in chunks with pandas
2. Relational database    - current orders, read in chunks through SQLAlchemy
3. Web API (JSON)         - US public holidays from the Nager.Date REST API

Each batch is written straight to its own Parquet file in the staging area, so
only one batch is in memory at a time; the transform phase reads the staging
area with Dask. Product names are free text (the unstructured part of the
data); they are kept as-is here and turned into features in the transform phase.
"""
import json
import shutil
from pathlib import Path
from typing import Iterator

import dask.dataframe as dd
import pandas as pd
import requests
from sqlalchemy import create_engine, text

from config import CHUNK_SIZE, DATA_DIR, HISTORY_CSV, HOLIDAY_API, RAW_DIR, SOURCE_DB_URL
from etl.logging_utils import get_logger

log = get_logger("extract")

STAGING_ORDERS = DATA_DIR / "staging" / "orders"
COLUMNS = ["row_id", "order_id", "order_date", "ship_date", "ship_mode",
           "customer_id", "customer_name", "segment", "country", "city",
           "state", "postal_code", "region", "product_id", "category",
           "sub_category", "product_name", "sales", "quantity", "discount",
           "profit"]
NUMERIC = ["row_id", "sales", "quantity", "discount", "profit"]


def _standardize(batch: pd.DataFrame, source: str) -> pd.DataFrame:
    """Same column names and types for every batch from every source, so the
    Parquet files form one dataset."""
    batch = batch.copy()
    batch.columns = COLUMNS
    for col in COLUMNS:
        if col in NUMERIC:
            batch[col] = pd.to_numeric(batch[col], errors="coerce").astype("float64")
        else:
            batch[col] = batch[col].astype("string")
    batch["source"] = source
    return batch


def _write_batches(batches: Iterator[pd.DataFrame], source: str, out_dir: Path) -> int:
    n = 0
    for i, batch in enumerate(batches, start=1):
        part = _standardize(batch, source)
        part.to_parquet(out_dir / f"{source}-{i:05d}.parquet", index=False)
        n += len(part)
        log.info("%s batch %d: %d rows written (running total %d)", source, i, len(part), n)
    return n


def extract_csv(out_dir: Path, path=HISTORY_CSV, chunksize: int = CHUNK_SIZE) -> int:
    """Stream the CSV in fixed-size chunks; memory use does not grow with file size."""
    reader = pd.read_csv(path, chunksize=chunksize, dtype=str,
                         encoding="utf-8", encoding_errors="replace")
    return _write_batches(reader, "csv", out_dir)


def extract_db(out_dir: Path, url: str = SOURCE_DB_URL, since: str | None = None,
               chunksize: int = CHUNK_SIZE) -> int:
    """Stream orders from the operational database. `since` gives an incremental read."""
    engine = create_engine(url)
    sql = "SELECT * FROM orders"
    params = {}
    if since:
        sql += " WHERE order_date >= :since"
        params["since"] = since
    with engine.connect().execution_options(stream_results=True) as conn:
        reader = pd.read_sql_query(text(sql), conn, params=params, chunksize=chunksize)
        return _write_batches(reader, "database", out_dir)


def extract_holidays(years, timeout: float = 10) -> pd.DataFrame:
    """Call the holiday API once per year and cache each JSON response in data/raw.
    If the API cannot be reached, fall back to the cached file, then to the
    `holidays` package, so one network failure does not stop the pipeline."""
    rows = []
    for year in sorted(set(int(y) for y in years)):
        cache = RAW_DIR / f"holidays_{year}.json"
        try:
            resp = requests.get(HOLIDAY_API.format(year=year), timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            cache.write_text(json.dumps(data, indent=1))
            origin = "api"
        except requests.RequestException as exc:
            log.warning("holiday API failed for %d (%s)", year, exc.__class__.__name__)
            if cache.exists():
                data, origin = json.loads(cache.read_text()), "cache"
            else:
                import holidays
                data = [{"date": d.isoformat(), "localName": n, "name": n}
                        for d, n in holidays.US(years=year).items()]
                origin = "holidays-package"
        log.info("holidays %d: %d dates (%s)", year, len(data), origin)
        rows += [{"date": h["date"], "holiday": h["name"]} for h in data]
    out = pd.DataFrame(rows).drop_duplicates("date")
    out["date"] = pd.to_datetime(out["date"])
    return out


def extract_all(out_dir: Path = STAGING_ORDERS):
    """Write all order batches to `out_dir` and return (out_dir, row count, holidays)."""
    out_dir = Path(out_dir)
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    n = extract_csv(out_dir) + extract_db(out_dir)
    dates = dd.read_parquet(out_dir, columns=["order_date"])["order_date"]
    years = dates.map_partitions(lambda s: pd.to_datetime(s, format="mixed", errors="coerce").dt.year,
                                 meta=("order_date", "float64")).dropna().unique().compute()
    holidays_df = extract_holidays(years)
    log.info("extracted %d order lines into %s and %d holidays", n, out_dir, len(holidays_df))
    return out_dir, n, holidays_df
