"""Model-side feature engineering, packaged inside the MLflow model.

Training and the API both call `add_features` through the saved pipeline, so a
request is transformed with exactly the code used in training (no
training/serving skew). Inputs are the raw order fields a client would know.
"""
import holidays
import numpy as np
import pandas as pd

RAW_FIELDS = ["order_date", "ship_date", "ship_mode", "segment", "region", "category",
              "sub_category", "product_name", "sales", "quantity", "discount"]
NUMERIC = ["log_sales", "quantity", "discount", "log_unit_list_price", "days_to_ship",
           "order_month", "order_dow", "near_holiday", "name_word_count", "name_has_digits"]
CATEGORICAL = ["ship_mode", "segment", "region", "category", "sub_category", "brand"]
TEXT = "product_name"
HOLIDAY_WINDOW_DAYS = 3


def _near_holiday(dates: pd.Series) -> pd.Series:
    years = dates.dt.year.dropna().unique().astype(int).tolist()
    days = pd.to_datetime(list(holidays.US(years=years).keys()))
    window = {d + pd.Timedelta(days=k) for d in days
              for k in range(-HOLIDAY_WINDOW_DAYS, HOLIDAY_WINDOW_DAYS + 1)}
    return dates.dt.normalize().isin(window).astype(int)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    order = pd.to_datetime(df["order_date"], format="mixed")
    ship = pd.to_datetime(df["ship_date"], format="mixed")
    sales = df["sales"].astype(float)
    qty = df["quantity"].astype(float)
    disc = df["discount"].astype(float)
    df["log_sales"] = np.log1p(sales)
    df["log_unit_list_price"] = np.log1p(sales / (qty * (1 - disc)))
    df["days_to_ship"] = (ship - order).dt.days
    df["order_month"] = order.dt.month
    df["order_dow"] = order.dt.dayofweek
    df["near_holiday"] = _near_holiday(order)
    name = df[TEXT].fillna("").astype(str)
    df[TEXT] = name
    df["brand"] = name.str.split(" ").str[0].str.lower()
    df["name_word_count"] = name.str.split().str.len()
    df["name_has_digits"] = name.str.contains(r"\d", regex=True).astype(int)
    return df[NUMERIC + CATEGORICAL + [TEXT]]
