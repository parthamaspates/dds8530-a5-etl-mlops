"""Create the two local sources from the Kaggle Superstore file:

* data/raw/orders_history_2014_2016.csv  - archived orders exported to a flat file
* data/source_orders.db (table `orders`) - the operational database with 2017 orders

This mirrors Assignment 3, where the orders lived in Amazon RDS (MySQL). To read
from RDS instead, set SOURCE_DB_URL=mysql+pymysql://user:pw@host/superstore.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sqlalchemy import create_engine

from config import HISTORY_CSV, SOURCE_DB_URL, SUPERSTORE_CSV

df = pd.read_csv(SUPERSTORE_CSV, encoding="cp1252")
year = pd.to_datetime(df["Order Date"], format="%m/%d/%Y").dt.year

history = df[year <= 2016]
current = df[year == 2017].copy()
history.to_csv(HISTORY_CSV, index=False)

# Operational DB uses snake_case column names, as in the Assignment 3 schema.
current.columns = [c.lower().replace(" ", "_").replace("-", "_") for c in current.columns]
engine = create_engine(SOURCE_DB_URL)
current.to_sql("orders", engine, if_exists="replace", index=False)
print(f"history CSV: {len(history)} rows -> {HISTORY_CSV}")
print(f"source DB:   {len(current)} rows -> {SOURCE_DB_URL}")
