# drift_monitor.py
# ─────────────────────────────────────────────────────────────────────────────
# Checks whether the distribution of incoming predictions has drifted
# from what the model saw during training.
#
# Two tests:
#   1. KS test — detects any change in distribution shape
#   2. PSI (Population Stability Index) — the industry standard for
#      credit/clinical model monitoring. < 0.1 = stable, > 0.2 = retrain.
#
# Run this on a schedule (e.g., daily via cron or the Docker service).
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import json
import logging
import os
from datetime import datetime
from scipy import stats

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")

# PSI thresholds from the literature on clinical model monitoring
PSI_STABLE   = 0.10
PSI_MODERATE = 0.20

ALERT_PATH  = "monitoring/drift_alerts.json"
REPORT_PATH = "monitoring/drift_report.json"


# ── Statistical tests ─────────────────────────────────────────────────────────

def ks_test(ref: np.ndarray, cur: np.ndarray):
    """Two-sample KS test. Low p-value = distributions are different."""
    stat, p = stats.ks_2samp(ref, cur)
    return round(stat, 4), round(p, 4), "DRIFT" if p < 0.05 else "STABLE"


def psi(ref: np.ndarray, cur: np.ndarray, n_bins=10):
    """
    Population Stability Index.
    Adding a small epsilon to avoid log(0) — standard practice.
    """
    lo = min(ref.min(), cur.min())
    hi = max(ref.max(), cur.max())
    edges = np.linspace(lo, hi, n_bins + 1)

    ref_counts = np.histogram(ref, bins=edges)[0]
    cur_counts = np.histogram(cur, bins=edges)[0]

    eps       = 1e-8
    ref_pct   = (ref_counts + eps) / len(ref)
    cur_pct   = (cur_counts + eps) / len(cur)
    psi_val   = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

    if psi_val < PSI_STABLE:
        label = "STABLE"
    elif psi_val < PSI_MODERATE:
        label = "MODERATE — investigate"
    else:
        label = "HIGH — consider retraining"

    return round(psi_val, 4), label


# ── Load data ─────────────────────────────────────────────────────────────────

def load_reference():
    if not os.path.exists("data/X_train.csv"):
        raise FileNotFoundError("Training data not found. Run feature_engineering.py first.")
    return pd.read_csv("data/X_train.csv")


def load_predictions():
    path = "monitoring/predictions.jsonl"
    if not os.path.exists(path):
        return pd.DataFrame()

    rows = [json.loads(l) for l in open(path)]
    if not rows:
        return pd.DataFrame()

    # flatten the input dict into columns
    records = [{**r.get("input", {}), "probability": r["probability"], "prediction": r["prediction"]}
               for r in rows]
    return pd.DataFrame(records)


# ── Feature drift analysis ────────────────────────────────────────────────────

# Maps API field names → training column names
FEATURE_MAP = {
    "glucose":       "Glucose",
    "bmi":           "BMI",
    "age":           "Age",
    "blood_pressure": "BloodPressure",
    "insulin":       "Insulin",
}


def analyze_features(ref_df, cur_df):
    results = {}
    for cur_col, ref_col in FEATURE_MAP.items():
        if cur_col not in cur_df.columns or ref_col not in ref_df.columns:
            continue
        ref_vals = ref_df[ref_col].dropna().values
        cur_vals = cur_df[cur_col].dropna().values
        if len(cur_vals) < 15:
            continue  # not enough data to test meaningfully

        ks_stat, ks_p, ks_label = ks_test(ref_vals, cur_vals)
        psi_val, psi_label       = psi(ref_vals, cur_vals)

        results[cur_col] = {
            "ks_stat":   ks_stat,
            "ks_p":      ks_p,
            "ks_verdict": ks_label,
            "psi":        psi_val,
            "psi_verdict": psi_label,
            "ref_mean":   round(float(ref_vals.mean()), 2),
            "cur_mean":   round(float(cur_vals.mean()), 2),
            "mean_delta": round(float(cur_vals.mean() - ref_vals.mean()), 2),
        }
        log.info(f"  {cur_col:18s}  KS={ks_label:7s}  PSI={psi_val:.3f} ({psi_label.split(' ')[0]})")

    return results


def analyze_prediction_rate(ref_df, cur_df):
    """
    Compare the proportion of positive predictions to the training label rate.
    A big shift here usually means something meaningful has changed.
    """
    ref_rate = ref_df.get("Outcome", pd.Series(dtype=float)).mean() if "Outcome" in ref_df else 0.35
    cur_rate = cur_df["prediction"].mean() if "prediction" in cur_df.columns else 0

    delta   = abs(cur_rate - ref_rate)
    verdict = "DRIFT" if delta > 0.15 else "STABLE"

    return {
        "ref_positive_rate": round(float(ref_rate), 4),
        "cur_positive_rate": round(float(cur_rate), 4),
        "delta":             round(float(delta), 4),
        "verdict":           verdict,
    }


# ── Alerting ──────────────────────────────────────────────────────────────────

def build_alerts(feature_results, pred_results):
    alerts = []

    for feat, r in feature_results.items():
        if r["ks_verdict"] == "DRIFT":
            alerts.append({
                "severity": "HIGH",
                "type":     "FEATURE_DRIFT",
                "feature":  feat,
                "message":  f"Significant distribution shift in {feat} (KS p={r['ks_p']})",
                "action":   "Investigate data pipeline and consider retraining",
            })
        elif "HIGH" in r["psi_verdict"]:
            alerts.append({
                "severity": "HIGH",
                "type":     "PSI_DRIFT",
                "feature":  feat,
                "message":  f"PSI = {r['psi']} for {feat}",
                "action":   "Review incoming data distribution",
            })

    if pred_results["verdict"] == "DRIFT":
        alerts.append({
            "severity": "CRITICAL",
            "type":     "PREDICTION_DRIFT",
            "feature":  "positive_rate",
            "message":  f"Positive rate shifted by {pred_results['delta']:.1%}",
            "action":   "URGENT: Review model performance and retrain if needed",
        })

    return alerts


def save_report(feature_results, pred_results, alerts):
    os.makedirs("monitoring", exist_ok=True)
    report = {
        "checked_at":       datetime.utcnow().isoformat(),
        "feature_drift":    feature_results,
        "prediction_drift": pred_results,
        "alerts":           alerts,
        "status":           "ALERT" if alerts else "OK",
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Drift report saved → {REPORT_PATH}")

    # append any new alerts to the alert log
    if alerts:
        existing = json.load(open(ALERT_PATH)) if os.path.exists(ALERT_PATH) else []
        for a in alerts:
            a["ts"] = datetime.utcnow().isoformat()
        existing.extend(alerts)
        with open(ALERT_PATH, "w") as f:
            json.dump(existing, f, indent=2)

    return report


def run():
    log.info("=" * 55)
    log.info("DRIFT MONITORING")
    log.info("=" * 55)

    try:
        ref_df = load_reference()
    except FileNotFoundError as e:
        log.error(str(e))
        return

    cur_df = load_predictions()
    if cur_df.empty:
        log.warning("No predictions logged yet — nothing to compare against")
        return

    log.info(f"Reference: {len(ref_df)} rows  |  Current: {len(cur_df)} predictions")

    feature_results = analyze_features(ref_df, cur_df)
    pred_results    = analyze_prediction_rate(ref_df, cur_df)
    alerts          = build_alerts(feature_results, pred_results)
    report          = save_report(feature_results, pred_results, alerts)

    log.info(f"Status: {report['status']}  |  Alerts: {len(alerts)}")
    for a in alerts:
        log.warning(f"  [{a['severity']}] {a['message']}")

    log.info("Done.\n")
    return report


if __name__ == "__main__":
    run()
# drift alert logging 
