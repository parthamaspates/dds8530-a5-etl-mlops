"""Load-phase tests: warehouse swap and the S3 Parquet layout (S3 mocked with moto)."""
import io

import boto3
import dask.dataframe as dd
import pandas as pd
import pytest
from sqlalchemy import create_engine, text

moto = pytest.importorskip("moto")

from etl.load import load_s3, load_warehouse

FRAME = pd.DataFrame({"row_id": [1, 2, 3, 4], "order_year": [2016, 2016, 2017, 2017],
                      "sales": [10.0, 20.0, 30.0, 40.0], "profitable": [1, 0, 1, 1]})


def test_warehouse_loads_every_partition(tmp_path):
    url = f"sqlite:///{tmp_path / 'wh.db'}"
    n = load_warehouse(dd.from_pandas(FRAME, npartitions=2), url=url)
    assert n == 4
    with create_engine(url).connect() as conn:
        tables = {r[0] for r in conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))}
    assert "order_features" in tables and "order_features_staging" not in tables


def test_s3_writes_parquet_partitioned_by_year(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    with moto.mock_aws():
        s3 = boto3.client("s3", region_name="us-west-1")
        s3.create_bucket(Bucket="test-bucket",
                         CreateBucketConfiguration={"LocationConstraint": "us-west-1"})
        keys = load_s3(dd.from_pandas(FRAME, npartitions=2), "2026-11-02", bucket="test-bucket")
        assert sorted(keys) == [
            "ml/features/run_date=2026-11-02/order_year=2016/part-00000.parquet",
            "ml/features/run_date=2026-11-02/order_year=2017/part-00001.parquet",
        ]
        body = s3.get_object(Bucket="test-bucket", Key=keys[0])["Body"].read()
        assert len(pd.read_parquet(io.BytesIO(body))) == 2
