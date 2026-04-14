# evaluate.py
# ─────────────────────────────────────────────────────────────────────────────
# Model evaluation + bias/fairness analysis.
#
# The fairness piece is important: the Pima dataset is entirely Pima Indian women
# from Arizona. A model trained on it might not generalize to other populations,
# and even within this dataset it could perform differently across age groups.
# Flagging that explicitly so whoever deploys this knows what they're working with.
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import joblib
import json
import logging
import os
from datetime import datetime
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, average_precision_score,
    roc_curve, precision_recall_curve
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")


def load_artifacts():
    model  = joblib.load("models/rf_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv").squeeze()
    return model, scaler, X_test, y_test


def overall_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        "auc_roc":           round(roc_auc_score(y_test, y_prob), 4),
        "avg_precision":     round(average_precision_score(y_test, y_prob), 4),
        "accuracy":          round((tp + tn) / (tp + tn + fp + fn), 4),
        "sensitivity":       round(tp / (tp + fn), 4),   # recall for positive class
        "specificity":       round(tn / (tn + fp), 4),
        "ppv":               round(tp / (tp + fp), 4),   # precision
        "npv":               round(tn / (tn + fn), 4),
        "false_negative_rate": round(fn / (fn + tp), 4), # missing a diabetic patient
        "false_positive_rate": round(fp / (fp + tn), 4),
        "confusion_matrix":  [[int(tn), int(fp)], [int(fn), int(tp)]],
    }

    log.info(f"AUC-ROC:      {metrics['auc_roc']}")
    log.info(f"Sensitivity:  {metrics['sensitivity']}  (how often we catch diabetic patients)")
    log.info(f"Specificity:  {metrics['specificity']}  (how often we correctly clear non-diabetic)")
    log.info(f"FNR:          {metrics['false_negative_rate']}  (rate of missed diagnoses)")

    return metrics


def fairness_audit(model, X_test, y_test):
    """
    Check if the model performs consistently across age groups.
    A large AUC gap between groups would be a red flag — it means the model
    is systematically better or worse for some patients than others.

    WHO definition of fairness in clinical ML: no group should have a
    false negative rate more than 10% higher than the best-performing group.
    """
    df = X_test.copy()
    df["_label"] = y_test.values
    df["_prob"]  = model.predict_proba(X_test)[:, 1]
    df["_pred"]  = model.predict(X_test)

    # use AgeGroup if it was created by feature engineering, otherwise bucket Age
    if "AgeGroup" in df.columns:
        df["_group"] = df["AgeGroup"].map({0: "<30", 1: "30–45", 2: "45–60", 3: "60+"})
    else:
        df["_group"] = pd.cut(df["Age"], bins=[0, 30, 45, 60, 120],
                              labels=["<30", "30–45", "45–60", "60+"])

    group_results = {}
    for group_name, gdf in df.groupby("_group"):
        if len(gdf) < 10 or gdf["_label"].nunique() < 2:
            continue  # skip groups too small to be meaningful

        auc = roc_auc_score(gdf["_label"], gdf["_prob"])
        rep = classification_report(gdf["_label"], gdf["_pred"], output_dict=True, zero_division=0)
        cm  = confusion_matrix(gdf["_label"], gdf["_pred"])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        group_results[str(group_name)] = {
            "n":           len(gdf),
            "positive_rate": round(float(gdf["_label"].mean()), 3),
            "auc_roc":     round(auc, 4),
            "accuracy":    round(rep["accuracy"], 4),
            "sensitivity": round(tp / (tp + fn) if (tp + fn) > 0 else 0, 4),
            "fnr":         round(fn / (fn + tp) if (fn + tp) > 0 else 0, 4),
        }
        log.info(
            f"  {group_name:7s}  n={len(gdf):3d}  "
            f"AUC={auc:.3f}  sens={group_results[str(group_name)]['sensitivity']:.3f}  "
            f"FNR={group_results[str(group_name)]['fnr']:.3f}"
        )

    # summary: compute max disparity across groups
    aucs = [v["auc_roc"] for v in group_results.values()]
    fnrs = [v["fnr"]     for v in group_results.values()]
    auc_gap = round(max(aucs) - min(aucs), 4) if aucs else 0
    fnr_gap = round(max(fnrs) - min(fnrs), 4) if fnrs else 0

    group_results["_summary"] = {
        "auc_gap":  auc_gap,
        "fnr_gap":  fnr_gap,
        "auc_flag": "REVIEW" if auc_gap > 0.1 else "OK",
        "fnr_flag": "REVIEW" if fnr_gap > 0.1 else "OK",
    }

    if auc_gap > 0.1:
        log.warning(f"Fairness flag: AUC gap = {auc_gap:.3f} across age groups")
    else:
        log.info(f"Fairness OK: AUC gap = {auc_gap:.3f}")

    return group_results


def save_report(overall, fairness):
    os.makedirs("docs", exist_ok=True)
    report = {
        "evaluated_at":      datetime.utcnow().isoformat(),
        "model":             "RandomForestClassifier v1.0.0",
        "overall_metrics":   overall,
        "fairness_audit":    fairness,
        "governance_status": "APPROVED" if overall["auc_roc"] >= 0.75 else "NEEDS_REVIEW",
    }
    with open("docs/eval_report.json", "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Evaluation report saved → docs/eval_report.json")
    log.info(f"Governance status: {report['governance_status']}")
    return report


def run():
    log.info("=" * 55)
    log.info("EVALUATION + FAIRNESS AUDIT")
    log.info("=" * 55)

    model, scaler, X_test, y_test = load_artifacts()
    overall  = overall_metrics(model, X_test, y_test)
    fairness = fairness_audit(model, X_test, y_test)
    report   = save_report(overall, fairness)

    log.info("Done.\n")
    return report


if __name__ == "__main__":
    run()
# fairness 
