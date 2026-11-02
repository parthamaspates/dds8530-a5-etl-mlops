"""Train, compare and register a profitability classifier with MLflow.

    python -m ml.train

* Data: order_features table in the warehouse (output of the ETL).
* Target: profitable = 1 if the order line made a profit.
* Split by time: train on 2014-2016 orders, test on 2017 orders.
* Every candidate is a nested MLflow run (parameters, 5-fold CV ROC AUC, test metrics).
* The best candidate by CV ROC AUC is registered in the MLflow Model Registry.
* Quality gate: it gets the alias "champion" only if (1) its test ROC AUC is at least
  MIN_ROC_AUC and (2) it is at least as good as the current champion, re-scored on the
  same test set. Otherwise the current champion stays in service and the exit code is 1.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sqlalchemy import create_engine

from config import (EXPERIMENT_NAME, MIN_ROC_AUC, MLFLOW_TRACKING_URI, MODEL_ALIAS,
                    MODEL_NAME, WAREHOUSE_URL)
from etl.logging_utils import get_logger
from ml.features import CATEGORICAL, NUMERIC, RAW_FIELDS, TEXT, add_features

log = get_logger("train")

# Types skops must be told to trust: our own feature function and the tree
# structure used by HistGradientBoosting (we created both, so they are safe).
TRUSTED_TYPES = ["ml.features.add_features",
                 "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]

CANDIDATES = {
    "logreg_C0.1": LogisticRegression(C=0.1, max_iter=2000),
    "logreg_C1": LogisticRegression(C=1.0, max_iter=2000),
    "logreg_C10": LogisticRegression(C=10.0, max_iter=2000),
    "rf_depth12": RandomForestClassifier(n_estimators=300, max_depth=12, n_jobs=-1, random_state=42),
    "rf_full": RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, random_state=42),
    "hgb_lr0.05": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, random_state=42),
    "hgb_lr0.1": HistGradientBoostingClassifier(learning_rate=0.1, max_iter=300, random_state=42),
}


def build_pipeline(clf) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20), CATEGORICAL),
        ("txt", TfidfVectorizer(max_features=300, ngram_range=(1, 2), min_df=5), TEXT),
    ], sparse_threshold=0)
    return Pipeline([("features", FunctionTransformer(add_features)),
                     ("pre", pre), ("clf", clf)])


def load_data():
    # fixed row order, so cross-validation folds (and results) are reproducible
    df = pd.read_sql_query("SELECT * FROM order_features ORDER BY row_id", create_engine(WAREHOUSE_URL))
    train, test = df[df.order_year <= 2016], df[df.order_year == 2017]
    return (train[RAW_FIELDS], train["profitable"], test[RAW_FIELDS], test["profitable"])


def test_metrics(y, pred, proba) -> dict:
    return {"test_accuracy": accuracy_score(y, pred), "test_precision": precision_score(y, pred),
            "test_recall": recall_score(y, pred), "test_f1": f1_score(y, pred),
            "test_roc_auc": roc_auc_score(y, proba)}


def gate_decision(new_auc: float, champion_auc: float | None, floor: float) -> str:
    """Promote only if the new model clears the floor AND is no worse than the champion."""
    if new_auc < floor:
        return "rejected: below MIN_ROC_AUC"
    if champion_auc is not None and new_auc < champion_auc:
        return "rejected: worse than champion"
    return "promoted"


def current_champion_auc(X_te, y_te):
    """Re-score the model now serving (@champion) on today's test set; None if there is none."""
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    except MlflowException:
        return None, None
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    return int(mv.version), roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])


def main() -> int:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    X_tr, y_tr, X_te, y_te = load_data()
    log.info("train %d rows (2014-2016), test %d rows (2017)", len(X_tr), len(X_te))
    champ_version, champ_auc = current_champion_auc(X_te, y_te)
    if champ_auc is not None:
        log.info("current champion v%s scores test ROC AUC %.4f", champ_version, champ_auc)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    with mlflow.start_run(run_name="model-selection") as parent:
        mlflow.log_params({"train_rows": len(X_tr), "test_rows": len(X_te),
                           "split": "time: train<=2016, test=2017", "cv_folds": 5})
        for name, clf in CANDIDATES.items():
            with mlflow.start_run(run_name=name, nested=True) as run:
                pipe = build_pipeline(clf)
                cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=1)
                pipe.fit(X_tr, y_tr)
                proba = pipe.predict_proba(X_te)[:, 1]
                m = test_metrics(y_te, (proba >= 0.5).astype(int), proba)
                mlflow.log_param("model_type", type(clf).__name__)
                mlflow.log_params({k: v for k, v in clf.get_params().items()
                                   if k in ("C", "max_depth", "n_estimators", "learning_rate", "max_iter")})
                mlflow.log_metrics({"cv_roc_auc_mean": cv_auc.mean(), "cv_roc_auc_std": cv_auc.std(), **m})
                results.append({"name": name, "run_id": run.info.run_id, "pipe": pipe,
                                "cv_auc": cv_auc.mean(), **m})
                log.info("%-12s cv_auc=%.4f test_auc=%.4f test_f1=%.4f",
                         name, cv_auc.mean(), m["test_roc_auc"], m["test_f1"])

        best = max(results, key=lambda r: r["cv_auc"])
        log.info("best by CV ROC AUC: %s", best["name"])
        mlflow.set_tag("best_candidate", best["name"])
        mlflow.log_metrics({"best_cv_roc_auc": best["cv_auc"], "best_test_roc_auc": best["test_roc_auc"]})
        if champ_auc is not None:
            mlflow.log_metric("champion_test_roc_auc", champ_auc)

        # final model: log with signature, input example and the feature code, then register
        fig, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay.from_estimator(best["pipe"], X_te, y_te, ax=ax, colorbar=False)
        ax.set_title(f"{best['name']} - 2017 test set")
        mlflow.log_figure(fig, "confusion_matrix.png")
        table = pd.DataFrame([{k: v for k, v in r.items() if k != "pipe"} for r in results])
        mlflow.log_text(table.round(4).to_csv(index=False), "candidate_comparison.csv")
        sig = infer_signature(X_tr.head(50), best["pipe"].predict_proba(X_tr.head(50))[:, 1])
        info = mlflow.sklearn.log_model(best["pipe"], name="model", signature=sig,
                                        input_example=X_tr.head(3), code_paths=[str(Path(__file__).parent)],
                                        skops_trusted_types=TRUSTED_TYPES,
                                        registered_model_name=MODEL_NAME)
        version = info.registered_model_version

    client = MlflowClient()
    new_auc = best["test_roc_auc"]
    gate = {"min_roc_auc": MIN_ROC_AUC, "champion_version": champ_version,
            "champion_test_roc_auc": None if champ_auc is None else round(champ_auc, 4)}
    gate["result"] = gate_decision(new_auc, champ_auc, MIN_ROC_AUC)
    client.set_model_version_tag(MODEL_NAME, version, "candidate", best["name"])
    client.set_model_version_tag(MODEL_NAME, version, "test_roc_auc", f"{new_auc:.4f}")
    client.set_model_version_tag(MODEL_NAME, version, "quality_gate", gate["result"])
    if gate["result"] == "promoted":
        client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, version)
        log.info("v%s promoted to @%s: test AUC %.4f >= %.2f and >= champion %s",
                 version, MODEL_ALIAS, new_auc, MIN_ROC_AUC, gate["champion_test_roc_auc"])
        status = 0
    else:
        log.warning("v%s %s (test AUC %.4f, champion v%s %.4f, floor %.2f): @%s unchanged",
                    version, gate["result"], new_auc, champ_version, champ_auc or float("nan"),
                    MIN_ROC_AUC, MODEL_ALIAS)
        status = 1
    print(json.dumps({"parent_run": parent.info.run_id, "best": best["name"], "version": version,
                      "test_roc_auc": round(new_auc, 4), "gate": gate}))
    return status


if __name__ == "__main__":
    sys.exit(main())
