"""Copy the @champion model out of the MLflow registry into build/model, so the
Docker image carries the exact registered version (used by CI/CD).

    python scripts/export_model.py
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mlflow
from mlflow.tracking import MlflowClient

from config import MLFLOW_TRACKING_URI, MODEL_ALIAS, MODEL_NAME

out = ROOT / "build" / "model"
shutil.rmtree(out, ignore_errors=True)
out.parent.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}",
                                    dst_path=str(out))
info = {"name": MODEL_NAME, "alias": MODEL_ALIAS, "version": int(mv.version),
        "run_id": mv.run_id, "tags": mv.tags}
(out / "model_info.json").write_text(json.dumps(info, indent=2))
print(f"exported {MODEL_NAME} v{mv.version} -> {out}")
