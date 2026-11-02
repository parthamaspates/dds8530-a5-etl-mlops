import pandas as pd

from etl.extract import COLUMNS
from etl.transform import transform


def _row(row_id, **kw):
    base = dict(row_id=row_id, order_id="CA-2017-1", order_date="11/20/2017",
                ship_date="11/24/2017", ship_mode="Standard Class", customer_id="C1",
                customer_name="A B", segment="Consumer", country="United States",
                city="Los Angeles", state="California", postal_code="90001", region="West",
                product_id="P1", category="Office Supplies", sub_category="Paper",
                product_name="Xerox 1967", sales=25.92, quantity=4, discount=0.0, profit=12.4)
    base.update(kw)
    return base


def _run(rows):
    orders = pd.DataFrame(rows)[COLUMNS]
    hol = pd.DataFrame({"date": pd.to_datetime(["2017-11-23"]), "holiday": ["Thanksgiving"]})
    features, report = transform(orders, hol)
    return features.compute().sort_values("row_id").reset_index(drop=True), report


def test_duplicates_and_invalid_rows_removed():
    rows = [_row(1), _row(1),                              # duplicate row_id (CSV + DB overlap)
            _row(2, quantity=0),                           # invalid quantity
            _row(3, ship_date="11/01/2017"),               # ships before it was ordered
            _row(4, order_id=None)]                        # missing key
    out, report = _run(rows)
    assert out["row_id"].tolist() == [1]
    assert report["rows_dropped"] == 4


def test_missing_values_filled():
    rows = [_row(1, sales=None, profit=-5.0), _row(2, sales=10.0), _row(3, sales=30.0),
            _row(4, segment=None)]
    out, report = _run(rows)
    assert report["missing_before"] == 2
    assert out.loc[out.row_id == 1, "sales"].item() == 25.92      # median of the column
    assert out.loc[out.row_id == 4, "segment"].item() == "Unknown"


def test_features_and_normalization():
    rows = [_row(1), _row(2, order_date="06/01/2017", ship_date="06/03/2017",
                          product_name="Hon 4070 Chair", discount=0.2, profit=-3.0)]
    out, _ = _run(rows)
    first = out.set_index("row_id").loc[1]
    assert first["days_to_ship"] == 4
    assert first["near_holiday"] == 1                     # within 3 days of Thanksgiving
    assert first["brand"] == "xerox"
    assert first["profitable"] == 1
    assert out.set_index("row_id").loc[2, "profitable"] == 0
    assert abs(out["sales_z"].mean()) < 1e-9              # z-scores are centered
