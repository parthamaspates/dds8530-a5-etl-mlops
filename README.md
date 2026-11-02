# Superstore ETL and MLOps Pipeline

DDS-8530 Big Data Integration, Assignment 5 (Sina Zareian, National University)

This project builds a scalable ETL pipeline for the Sample Superstore orders and connects it to an MLOps pipeline. The ETL output feeds a model that predicts whether an order line will be profitable. That model is trained and versioned with MLflow, served by a FastAPI REST API, orchestrated by Apache Airflow, deployed by GitHub Actions, and monitored with Prometheus and Grafana.

```
CSV (2014-2016) ─┐
SQL DB (2017)  ──┼─► extract ─► transform (Dask) ─► load ─► SQLite/PostgreSQL warehouse
Holiday API    ──┘   (batches)  clean, normalize,          + Parquet in S3
                                 features
                                                     │
                              train (scikit-learn + MLflow) ─► Model Registry (@champion)
                                                     │
                     FastAPI /predict  ◄─────────────┘        Prometheus ─► Grafana
                     /metrics ───────────────────────────────────┘
Airflow DAG (daily 02:00): extract >> transform >> load >> train >> reload_api
GitHub Actions: test ─► ETL ─► train + quality gate ─► API tests ─► Docker image to GHCR
```

## Repository layout

| Path | Purpose |
|---|---|
| `config.py` | All settings; each one can be overridden with an environment variable |
| `scripts/seed_sources.py` | Builds the two local sources from `data/raw/superstore.csv`: a 2014–2016 CSV archive and a 2017 operational database |
| `etl/extract.py` | Streams the CSV and the database in 2,000-row batches (pandas chunks, SQLAlchemy), writing each batch straight to Parquet in `data/staging/orders/`, and calls the holiday REST API |
| `etl/transform.py` | Dask, reading the staging Parquet: cleaning, deduplication, missing values, validity rules, feature engineering, z-score normalization; writes partitioned Parquet |
| `etl/load.py` | Partition-by-partition load into the warehouse (staging table, then atomic swap), Parquet to S3, and an `etl_runs` audit table |
| `etl/run_etl.py` | Runs the full ETL with step timings and logging |
| `ml/features.py` | Model-side feature code that is packaged inside the MLflow model |
| `ml/train.py` | Trains 7 candidate models, logs them to MLflow, registers the best one, and applies the quality gate (floor and champion comparison) |
| `api/app.py` | FastAPI service: `/predict`, `/health`, `/model-info`, `/reload`, `/metrics` |
| `dags/superstore_pipeline.py` | Airflow DAG |
| `.github/workflows/ci-cd.yml` | CI/CD pipeline |
| `Dockerfile`, `docker-compose.yml`, `docker-compose.deploy.yml` | API image; the local stack (API, Prometheus, Grafana); the deployed stack, which runs the published image |
| `monitoring/` | Prometheus scrape config, alert rules, and the provisioned Grafana dashboard |
| `tests/` | Unit tests (transform) and API tests |
| `scripts/load_test.py`, `scripts/traffic.sh` | Send real 2017 orders to the API (Python load test; curl-only traffic script used by the deploy job) |
| `scripts/export_model.py` | Exports the `@champion` model for the Docker image |

## Setup

Python 3.12 or 3.13.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run the pipeline by hand

```bash
python scripts/seed_sources.py        # create the CSV and database sources
python -m etl.run_etl                 # extract -> transform -> load (logs/pipeline.log)
python -m ml.train                    # MLflow experiment + registry; exit code 1 if the gate fails
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000     # http://localhost:5000
uvicorn api.app:app --port 8000       # http://localhost:8000/docs
python scripts/load_test.py           # exercise the API
pytest                                # 12 tests
```

Example request:

```bash
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"orders": [{
  "order_date": "2017-11-20", "ship_date": "2017-11-24", "ship_mode": "Second Class",
  "segment": "Corporate", "region": "Central", "category": "Furniture",
  "sub_category": "Tables", "product_name": "Bretford CR4500 Series Slim Rectangular Table",
  "sales": 714.31, "quantity": 3, "discount": 0.5}]}'
```

## Optional: AWS S3 and Amazon RDS

* **S3:** set `S3_BUCKET` and the load step also writes Parquet to `s3://$S3_BUCKET/ml/features/run_date=YYYY-MM-DD/order_year=YYYY/`. The `superstore-etl` IAM user from Assignment 3 already has `s3:PutObject` on `superstore-datalake-sina-2026`.

  ```bash
  S3_BUCKET=superstore-datalake-sina-2026 AWS_REGION=us-west-1 python -m etl.run_etl
  ```

* **RDS:** point the database source at the Assignment 3 RDS MySQL instance. This needs `pip install pymysql`.

  ```bash
  SOURCE_DB_URL='mysql+pymysql://USER:PASSWORD@HOST/superstore'
  ```

* **PostgreSQL warehouse:**

  ```bash
  WAREHOUSE_URL='postgresql+psycopg://user:pw@host/db'
  ```

## Airflow

Install Airflow in a separate virtual environment (see `requirements-airflow.txt`), then:

```bash
export AIRFLOW_HOME=$PWD/airflow_home AIRFLOW__CORE__DAGS_FOLDER=$PWD/dags AIRFLOW__CORE__LOAD_EXAMPLES=False
airflow db migrate
airflow dags test superstore_etl_mlops 2026-11-02       # one full run
airflow standalone                                      # scheduler + UI on :8080
```

The DAG runs daily at 02:00 UTC. Each task retries twice. Problems show up as failed tasks:

* the transform drops more than 5% of rows;
* the quality gate rejects the new model, in which case the current `@champion` keeps serving;
* the API cannot be told to reload.

Every run is written to the `etl_runs` audit table. `load` records a successful ETL, and a failure callback on every task records a failed run with the task name and the error.

### Quality gate

A new model becomes `@champion` only if both of these hold:

1. its 2017 test ROC AUC is at least `MIN_ROC_AUC` (default 0.85);
2. it is at least as good as the current champion, re-scored on the same test set.

The decision is stored as a `quality_gate` tag on the model version.

## CI/CD (GitHub Actions)

The workflow has three jobs.

1. **`pipeline`** runs on every push and pull request:
   1. unit tests (transform, load with mocked S3, quality gate)
   2. ETL
   3. training with the quality gate
   4. API tests against the new model
   5. model export
2. **`publish`** runs on `main`. It builds the API image with that exact model and smoke-tests it (health, model info, predictions). It then pushes `linux/amd64` and `linux/arm64` images to `ghcr.io/<owner>/<repo>/api`, tagged with the commit SHA and `latest`.
3. **`deploy`** (continuous deployment) runs on a self-hosted runner on the deployment machine. It pulls the new image and restarts the API, Prometheus and Grafana with `docker-compose.deploy.yml`, so the stack keeps running after the job ends. It then verifies the release: the API is healthy, 300 requests are served, Prometheus reports the API target as up, and Grafana is healthy.

A weekly scheduled run retrains the model.

### Enabling deployment (one-time setup, Mac or Linux with Docker)

1. Install and start Docker Desktop. `docker --version` should print a version.
2. In the GitHub repository go to **Settings → Actions → Runners → New self-hosted runner**. Pick your OS and architecture (Apple chip = ARM64), run the commands GitHub shows in Terminal, then run `./run.sh` and leave that window open.
3. In **Settings → Secrets and variables → Actions → Variables**, add the repository variable `DEPLOY_ENABLED` = `true`.
4. In **Actions → ci-cd → Run workflow**, run the workflow on `main`. All three jobs should turn green.
5. Open:
   * http://localhost:8000/docs
   * http://localhost:9090/targets (`superstore-api` should be UP)
   * http://localhost:3000 → Dashboards → MLOps → *Superstore Model API*

   Run `bash scripts/traffic.sh` for more traffic.
6. When finished:
   * stop the stack with `docker compose -f docker-compose.deploy.yml down`
   * stop the runner with Ctrl+C

Self-hosted runners execute workflow code on your machine, so keep the runner stopped when you are not deploying.

### Rollback

* **Registry:** point `@champion` at the previous version and call `POST /reload`.
* **Container:** re-run the `deploy` job of an earlier commit, which redeploys its image.

## Monitoring

```bash
python -m ml.train && python scripts/export_model.py
docker compose up --build
```

* **Prometheus** (http://localhost:9090) scrapes `/metrics` every 15 s. Alert rules cover:
  * the API being down
  * p95 latency above 500 ms
  * an error rate above 2%
  * a shift in the share of predicted losses
* **Grafana** (http://localhost:3000, login admin/admin) opens the provisioned *Superstore Model API* dashboard.

Application logs go to `logs/pipeline.log`. Each ETL run, successful or failed, is recorded in the `etl_runs` table.

## Data

The Sample Superstore dataset (9,994 order lines, 2014–2017), as used in Assignments 1–4.
