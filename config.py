"""Central settings. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
LOG_DIR = Path(os.getenv("LOG_DIR", ROOT / "logs"))

# Sources
SUPERSTORE_CSV = RAW_DIR / "superstore.csv"                 # original Kaggle file
HISTORY_CSV = Path(os.getenv("HISTORY_CSV", RAW_DIR / "orders_history_2014_2016.csv"))
SOURCE_DB_URL = os.getenv("SOURCE_DB_URL", f"sqlite:///{DATA_DIR / 'source_orders.db'}")
HOLIDAY_API = os.getenv("HOLIDAY_API", "https://date.nager.at/api/v3/PublicHolidays/{year}/US")

# Targets
WAREHOUSE_URL = os.getenv("WAREHOUSE_URL", f"sqlite:///{DATA_DIR / 'warehouse.db'}")
S3_BUCKET = os.getenv("S3_BUCKET", "")                      # e.g. superstore-datalake-sina-2026
S3_PREFIX = os.getenv("S3_PREFIX", "ml/features")
AWS_REGION = os.getenv("AWS_REGION", "us-west-1")

# Processing
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2000"))           # rows per extraction batch
NPARTITIONS = int(os.getenv("NPARTITIONS", "4"))            # Dask partitions

# MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "superstore-profitability")
MODEL_NAME = os.getenv("MODEL_NAME", "superstore-profit-classifier")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
MIN_ROC_AUC = float(os.getenv("MIN_ROC_AUC", "0.85"))       # quality gate for promotion
