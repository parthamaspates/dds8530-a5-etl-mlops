"""API tests. They need a registered @champion model (run ml/train.py first);
in CI the workflow trains the model before running these."""
import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from api.app import app

ORDER = {"order_date": "2017-11-20", "ship_date": "2017-11-24", "ship_mode": "Standard Class",
         "segment": "Consumer", "region": "West", "category": "Office Supplies",
         "sub_category": "Paper", "product_name": "Xerox 1967", "sales": 25.92,
         "quantity": 4, "discount": 0.0}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        if not c.get("/health").json()["model_loaded"]:
            pytest.skip("no champion model registered yet")
        yield c


def test_predict_returns_probability(client):
    r = client.post("/predict", json={"orders": [ORDER]})
    assert r.status_code == 200
    pred = r.json()["predictions"][0]
    assert 0 <= pred["probability_profitable"] <= 1
    assert pred["label"] in ("profitable", "loss")


def test_heavy_discount_table_predicted_loss(client):
    order = {**ORDER, "category": "Furniture", "sub_category": "Tables", "sales": 714.31,
             "quantity": 3, "discount": 0.5,
             "product_name": "Bretford CR4500 Series Slim Rectangular Table"}
    r = client.post("/predict", json={"orders": [order]})
    assert r.json()["predictions"][0]["label"] == "loss"


def test_invalid_input_rejected(client):
    assert client.post("/predict", json={"orders": [{**ORDER, "discount": 1.5}]}).status_code == 422
    bad_dates = {**ORDER, "ship_date": "2017-11-01"}
    assert client.post("/predict", json={"orders": [bad_dates]}).status_code == 422


def test_metrics_exposed(client):
    text = client.get("/metrics").text
    assert "model_predictions_total" in text and "api_request_latency_seconds" in text
