"""REST API serving the champion model from the MLflow Model Registry.

    uvicorn api.app:app --port 8000

Endpoints: GET /health, GET /model-info, POST /predict, POST /reload, GET /metrics
(Prometheus format; scraped by monitoring/prometheus.yml).
"""
import json
import os
import time
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Literal

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from mlflow.tracking import MlflowClient
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field, model_validator

from config import MLFLOW_TRACKING_URI, MODEL_ALIAS, MODEL_NAME
from etl.logging_utils import get_logger
from ml.features import RAW_FIELDS

log = get_logger("api")
MODEL_DIR = os.getenv("MODEL_DIR", "")

# ---- Prometheus metrics -------------------------------------------------------
REQUESTS = Counter("api_requests_total", "HTTP requests", ["endpoint", "method", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"],
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
PREDICTIONS = Counter("model_predictions_total", "Predictions served", ["label"])
PROBABILITY = Histogram("model_prediction_probability", "Predicted P(profitable)",
                        buckets=[i / 10 for i in range(11)])
ERRORS = Counter("model_prediction_errors_total", "Failed prediction requests")
MODEL_VERSION = Gauge("model_version_info", "Registered model version being served")


# ---- request / response schemas ------------------------------------------------
class Order(BaseModel):
    order_date: date
    ship_date: date
    ship_mode: Literal["Same Day", "First Class", "Second Class", "Standard Class"]
    segment: Literal["Consumer", "Corporate", "Home Office"]
    region: Literal["Central", "East", "South", "West"]
    category: Literal["Furniture", "Office Supplies", "Technology"]
    sub_category: str
    product_name: str
    sales: float = Field(gt=0)
    quantity: int = Field(gt=0)
    discount: float = Field(ge=0, lt=1)

    @model_validator(mode="after")
    def ship_after_order(self):
        if self.ship_date < self.order_date:
            raise ValueError("ship_date must be on or after order_date")
        return self


class PredictRequest(BaseModel):
    orders: list[Order] = Field(min_length=1, max_length=1000)


# ---- model loading -----------------------------------------------------------
class ModelHolder:
    def __init__(self):
        self.model, self.info = None, {}

    def load(self):
        if MODEL_DIR:        # container: model exported by scripts/export_model.py
            self.model = mlflow.sklearn.load_model(MODEL_DIR)
            self.info = json.loads((Path(MODEL_DIR) / "model_info.json").read_text())
        else:                # local / Airflow: read the alias from the registry
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
            self.model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
            self.info = {"name": MODEL_NAME, "alias": MODEL_ALIAS, "version": int(mv.version),
                         "run_id": mv.run_id, "tags": mv.tags}
        MODEL_VERSION.set(int(self.info["version"]))
        log.info("loaded %s v%s (@%s)", MODEL_NAME, self.info["version"], MODEL_ALIAS)


holder = ModelHolder()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        holder.load()
    except Exception:
        log.exception("could not load model at startup; /predict returns 503 until /reload")
    yield


app = FastAPI(title="Superstore Profitability API", version="1.0", lifespan=lifespan)


@app.middleware("http")
async def observe(request: Request, call_next):
    t0 = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        path = request.url.path
        if path != "/metrics":
            LATENCY.labels(path).observe(time.perf_counter() - t0)
            REQUESTS.labels(path, request.method, str(status)).inc()


@app.get("/health")
def health():
    return {"status": "ok" if holder.model is not None else "degraded",
            "model_loaded": holder.model is not None}


@app.get("/model-info")
def model_info():
    if holder.model is None:
        raise HTTPException(503, "model not loaded")
    return holder.info


@app.post("/predict")
def predict(req: PredictRequest):
    if holder.model is None:
        raise HTTPException(503, "model not loaded")
    try:
        X = pd.DataFrame([o.model_dump() for o in req.orders])[RAW_FIELDS]
        X["order_date"] = X["order_date"].astype(str)
        X["ship_date"] = X["ship_date"].astype(str)
        proba = holder.model.predict_proba(X)[:, 1]
    except Exception as exc:
        ERRORS.inc()
        log.exception("prediction failed")
        raise HTTPException(500, f"prediction failed: {exc.__class__.__name__}")
    out = []
    for p in proba:
        label = "profitable" if p >= 0.5 else "loss"
        PREDICTIONS.labels(label).inc()
        PROBABILITY.observe(float(p))
        out.append({"label": label, "probability_profitable": round(float(p), 4)})
    log.info("predicted %d orders with v%s", len(out), holder.info["version"])
    return {"model_version": holder.info["version"], "predictions": out}


@app.post("/reload")
def reload():
    """Called by the CD workflow after a new version is promoted to @champion."""
    holder.load()
    return holder.info


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
