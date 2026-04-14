# train_rf.py
# ─────────────────────────────────────────────────────────────────────────────
# Random Forest is my go-to first model for tabular clinical data.
# It handles mixed feature types well, gives you feature importance for free,
# and doesn't need much preprocessing beyond what we already did.
#
# Using GridSearch here because the dataset is small enough that it won't take
# forever, and getting the max_depth right matters more than people think —
# unconstrained trees overfit badly on medical data.
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import joblib
import json
import logging
import os
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, average_precision_score
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")


def load_data():
    X_train = pd.read_csv("data/X_train.csv")
    X_test  = pd.read_csv("data/X_test.csv")
    y_train = pd.read_csv("data/y_train.csv").squeeze()
    y_test  = pd.read_csv("data/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def tune_and_train(X_train, y_train):
    """
    5-fold stratified CV with a focused param grid.
    I'm not doing an exhaustive search here — just the hyperparameters
    that actually matter for Random Forest on small tabular datasets.
    """
    param_grid = {
        "n_estimators":     [100, 200, 300],
        "max_depth":        [None, 8, 15, 25],
        "min_samples_leaf": [1, 2, 4],
        "max_features":     ["sqrt", "log2"],
        "class_weight":     ["balanced"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)

    log.info("Starting GridSearchCV (this takes a minute)...")
    search = GridSearchCV(
        base_model, param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    search.fit(X_train, y_train)

    log.info(f"Best params:  {search.best_params_}")
    log.info(f"Best CV AUC:  {search.best_score_:.4f}")

    return search.best_estimator_, search.best_params_, search.best_score_


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc  = roc_auc_score(y_test, y_prob)
    apr  = average_precision_score(y_test, y_prob)
    rep  = classification_report(y_test, y_pred, output_dict=True)
    cm   = confusion_matrix(y_test, y_pred).tolist()

    # false negative rate matters a lot clinically — missing a diabetic patient is bad
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fnr = fn / (fn + tp)

    metrics = {
        "auc_roc":            round(auc, 4),
        "avg_precision":      round(apr, 4),
        "accuracy":           round(rep["accuracy"], 4),
        "f1_weighted":        round(rep["weighted avg"]["f1-score"], 4),
        "precision_positive": round(rep["1"]["precision"], 4),
        "recall_positive":    round(rep["1"]["recall"], 4),
        "false_negative_rate": round(fnr, 4),
        "confusion_matrix":   cm,
    }

    log.info(f"Test AUC-ROC:          {metrics['auc_roc']}")
    log.info(f"Test Accuracy:         {metrics['accuracy']}")
    log.info(f"False Negative Rate:   {metrics['false_negative_rate']}")

    return metrics


def get_feature_importance(model, feature_names):
    importance = dict(zip(feature_names, model.feature_importances_))
    # sort descending so it's easy to read
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


def save_model(model, metrics, best_params, feature_importance, cv_auc):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/rf_model.pkl")

    meta = {
        "model":             "RandomForestClassifier",
        "version":           "1.0.0",
        "trained_at":        datetime.utcnow().isoformat(),
        "cv_auc":            round(cv_auc, 4),
        "best_params":       best_params,
        "test_metrics":      metrics,
        "feature_importance": feature_importance,
    }
    with open("models/rf_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info("Model saved → models/rf_model.pkl")
    log.info("Metadata saved → models/rf_metadata.json")
    return meta


def run():
    log.info("=" * 55)
    log.info("RANDOM FOREST TRAINING")
    log.info("=" * 55)

    X_train, X_test, y_train, y_test = load_data()
    model, best_params, cv_auc = tune_and_train(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    importance = get_feature_importance(model, X_train.columns.tolist())
    meta = save_model(model, metrics, best_params, importance, cv_auc)

    log.info("Done.\n")
    return model, meta


if __name__ == "__main__":
    run()
# tuning 
# importance 
