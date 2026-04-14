# mlflow_tracking.py
# ─────────────────────────────────────────────────────────────────────────────
# Wraps the training pipeline with MLflow tracking so every run is recorded.
# This makes it easy to compare experiments and roll back to a previous version
# if a new training run performs worse.
#
# To view results after running this:
#   mlflow ui --port 5000
# Then open http://localhost:5000
# ─────────────────────────────────────────────────────────────────────────────

import mlflow
import mlflow.sklearn
import mlflow.pytorch
import pandas as pd
import numpy as np
import logging
import os
import json
from datetime import datetime

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")

EXPERIMENT = "clinical-lab-predictor"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")


def setup():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    log.info(f"MLflow experiment: '{EXPERIMENT}'  |  URI: {TRACKING_URI}")


def load_data():
    return (
        pd.read_csv("data/X_train.csv"),
        pd.read_csv("data/X_test.csv"),
        pd.read_csv("data/y_train.csv").squeeze(),
        pd.read_csv("data/y_test.csv").squeeze(),
    )


def track_random_forest(X_train, X_test, y_train, y_test):
    """Log the RF training run — params, metrics, feature importance, artifacts."""
    import joblib
    from sklearn.metrics import roc_auc_score, classification_report

    rf_meta = json.load(open("models/rf_metadata.json")) if os.path.exists("models/rf_metadata.json") else {}

    run_name = f"RF_{datetime.now().strftime('%m%d_%H%M')}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({
            "model_type": "RandomForestClassifier",
            "dataset":    "pima_indians_diabetes",
            "author":     "Mayur Patil",
        })

        # log hyperparameters
        params = rf_meta.get("best_params", {})
        mlflow.log_params(params)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train",    len(X_train))

        # log metrics
        metrics = rf_meta.get("test_metrics", {})
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.log_metric("cv_auc", rf_meta.get("cv_auc", 0))

        # log feature importance as individual metrics — makes it easy to compare across runs
        for feat, imp in rf_meta.get("feature_importance", {}).items():
            mlflow.log_metric(f"fi_{feat}", round(imp, 5))

        # log model and artifacts
        model = joblib.load("models/rf_model.pkl")
        mlflow.sklearn.log_model(model, "rf_model", registered_model_name="ClinicalPredictor_RF")

        if os.path.exists("docs/eval_report.json"):
            mlflow.log_artifact("docs/eval_report.json")
        if os.path.exists("docs/model_card.md"):
            mlflow.log_artifact("docs/model_card.md")

        run_id = mlflow.active_run().info.run_id
        log.info(f"RF run logged: {run_id}")
        return run_id


def track_deep_learning():
    """Log the DL training run — architecture, epoch metrics, final performance."""
    import torch
    import sys
    sys.path.insert(0, "src")

    dl_meta = json.load(open("models/dl_metadata.json")) if os.path.exists("models/dl_metadata.json") else {}

    run_name = f"DL_{datetime.now().strftime('%m%d_%H%M')}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({
            "model_type": "ClinicalNet_PyTorch",
            "dataset":    "pima_indians_diabetes",
            "author":     "Mayur Patil",
        })

        mlflow.log_params({
            "architecture": dl_meta.get("architecture", ""),
            "n_features":   dl_meta.get("n_features", 0),
            "optimizer":    "Adam",
            "lr":           1e-3,
            "weight_decay": 1e-4,
            "batch_size":   32,
            "max_epochs":   120,
            "patience":     10,
        })

        metrics = dl_meta.get("test_metrics", {})
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.log_metric("best_val_auc", dl_meta.get("best_val_auc", 0))
        mlflow.log_metric("epochs_run",   dl_meta.get("epochs_run", 0))

        run_id = mlflow.active_run().info.run_id
        log.info(f"DL run logged: {run_id}")
        return run_id


def print_run_comparison():
    """Quick summary table of all runs in the experiment."""
    client = mlflow.tracking.MlflowClient()
    exp    = client.get_experiment_by_name(EXPERIMENT)
    if not exp:
        return

    runs = client.search_runs(exp.experiment_id, order_by=["metrics.auc_roc DESC"])
    log.info("\n" + "─" * 60)
    log.info(f"{'Run':<30} {'AUC':>8} {'Accuracy':>10}")
    log.info("─" * 60)
    for r in runs[:10]:
        name = r.data.tags.get("mlflow.runName", r.info.run_id[:8])
        auc  = r.data.metrics.get("auc_roc", 0)
        acc  = r.data.metrics.get("accuracy", 0)
        log.info(f"{name:<30} {auc:>8.4f} {acc:>10.4f}")


def run():
    log.info("=" * 55)
    log.info("MLFLOW EXPERIMENT TRACKING")
    log.info("=" * 55)

    setup()
    X_train, X_test, y_train, y_test = load_data()
    track_random_forest(X_train, X_test, y_train, y_test)
    track_deep_learning()
    print_run_comparison()

    log.info("\nView results: mlflow ui --port 5000")
    log.info("Done.\n")


if __name__ == "__main__":
    run()
