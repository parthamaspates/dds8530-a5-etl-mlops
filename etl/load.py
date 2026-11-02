"""Load phase: SQLite/PostgreSQL warehouse through SQLAlchemy, plus Parquet in S3.

Features arrive as a Dask DataFrame and are written one partition at a time, so
loading never needs the whole table in memory. The warehouse table is written
to a staging table first and swapped in inside one transaction, so readers (the
training step) never see a half-written table.
"""
import io
import json
from datetime import datetime, timezone

from sqlalchemy import (Column, DateTime, Float, Integer, MetaData, String, Table, Text,
                        create_engine, inspect, text)

from config import AWS_REGION, S3_BUCKET, S3_PREFIX, WAREHOUSE_URL
from etl.logging_utils import get_logger

log = get_logger("load")
TABLE = "order_features"

metadata = MetaData()
etl_runs = Table(
    "etl_runs", metadata,
    Column("run_id", Integer, primary_key=True, autoincrement=True),
    Column("started_at", DateTime), Column("finished_at", DateTime),
    Column("rows_loaded", Integer), Column("status", String(20)),
    Column("seconds_total", Float), Column("details", Text),
)


def _partitions(ddf):
    for i, part in enumerate(ddf.to_delayed()):
        yield i, part.compute()


def load_warehouse(ddf, url: str = WAREHOUSE_URL) -> int:
    engine = create_engine(url)
    staging = f"{TABLE}_staging"
    for i, part in _partitions(ddf):
        part.to_sql(staging, engine, if_exists="replace" if i == 0 else "append",
                    index=False, chunksize=1000)
    with engine.begin() as conn:                       # one transaction
        conn.execute(text(f"DROP TABLE IF EXISTS {TABLE}"))
        conn.execute(text(f"ALTER TABLE {staging} RENAME TO {TABLE}"))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS ix_{TABLE}_year ON {TABLE}(order_year)"))
        n = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE}")).scalar_one()
    log.info("warehouse: %d rows in %s from %d partitions (%s)",
             n, TABLE, ddf.npartitions, url.split("://")[0])
    return n


def load_s3(ddf, run_date: str, bucket: str = S3_BUCKET) -> list[str]:
    """Write Parquet to s3://bucket/ml/features/run_date=.../order_year=YYYY/part-NNNNN.parquet"""
    if not bucket:
        log.info("S3_BUCKET not set: skipping S3 upload")
        return []
    import boto3
    s3 = boto3.client("s3", region_name=AWS_REGION)
    keys = []
    for i, part in _partitions(ddf):
        for year, chunk in part.groupby("order_year"):
            key = f"{S3_PREFIX}/run_date={run_date}/order_year={year}/part-{i:05d}.parquet"
            buf = io.BytesIO()
            chunk.to_parquet(buf, index=False)
            s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
            keys.append(key)
    log.info("S3: wrote %d Parquet files to s3://%s/%s/run_date=%s/",
             len(keys), bucket, S3_PREFIX, run_date)
    return keys


def record_run(started: datetime, rows: int, status: str, timings: dict, report: dict,
               url: str = WAREHOUSE_URL) -> None:
    engine = create_engine(url)
    metadata.create_all(engine, tables=[etl_runs])
    with engine.begin() as conn:
        conn.execute(etl_runs.insert().values(
            started_at=started, finished_at=datetime.now(timezone.utc), rows_loaded=rows,
            status=status, seconds_total=round(sum(timings.values()), 3),
            details=json.dumps({"timings": timings, "report": report}, default=str)))


def table_exists(url: str = WAREHOUSE_URL) -> bool:
    return inspect(create_engine(url)).has_table(TABLE)
