"""Send real 2017 orders to the running API to exercise it and its metrics.

    python scripts/load_test.py --url http://localhost:8000 --requests 300 --batch 5
"""
import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import requests
from sqlalchemy import create_engine

from config import WAREHOUSE_URL
from ml.features import RAW_FIELDS

p = argparse.ArgumentParser()
p.add_argument("--url", default="http://localhost:8000")
p.add_argument("--requests", type=int, default=300)
p.add_argument("--batch", type=int, default=5)
p.add_argument("--bad-every", type=int, default=25, help="send an invalid request every N")
args = p.parse_args()

df = pd.read_sql_query("SELECT * FROM order_features WHERE order_year = 2017",
                       create_engine(WAREHOUSE_URL))
df["order_date"] = pd.to_datetime(df["order_date"]).dt.date.astype(str)
df["ship_date"] = pd.to_datetime(df["ship_date"]).dt.date.astype(str)
records = df[RAW_FIELDS + ["profitable"]].to_dict("records")
random.seed(42)

correct = total = 0
latencies, codes = [], {}
for i in range(1, args.requests + 1):
    batch = random.sample(records, args.batch)
    payload = {"orders": [{k: r[k] for k in RAW_FIELDS} for r in batch]}
    if args.bad_every and i % args.bad_every == 0:
        payload["orders"][0]["discount"] = 1.5            # invalid on purpose -> 422
    t0 = time.perf_counter()
    resp = requests.post(f"{args.url}/predict", json=payload, timeout=10)
    latencies.append(time.perf_counter() - t0)
    codes[resp.status_code] = codes.get(resp.status_code, 0) + 1
    if resp.ok:
        for r, pred in zip(batch, resp.json()["predictions"]):
            correct += int((pred["label"] == "profitable") == bool(r["profitable"]))
            total += 1

lat = pd.Series(latencies) * 1000
print(f"requests: {args.requests}, status codes: {codes}")
print(f"latency ms: median {lat.median():.1f}, p95 {lat.quantile(0.95):.1f}, max {lat.max():.1f}")
print(f"live accuracy on {total} scored 2017 orders: {correct / total:.4f}")
