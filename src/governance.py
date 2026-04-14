# governance.py
# ─────────────────────────────────────────────────────────────────────────────
# Model governance isn't just paperwork — it's how you prove the model is
# trustworthy enough to use in a clinical setting. This module handles:
#   - Version registry (so you can roll back if something goes wrong)
#   - Audit log (immutable record of every training run)
#   - Model card (human-readable summary for non-technical stakeholders)
#
# The format is loosely inspired by Google's Model Cards and Hugging Face's
# model card standard, adapted for a healthcare context.
# ─────────────────────────────────────────────────────────────────────────────

import json
import hashlib
import logging
import os
from datetime import datetime

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")

REGISTRY_PATH  = "docs/model_registry.json"
AUDIT_LOG_PATH = "docs/audit_log.json"


def file_hash(path):
    """SHA-256 of a file. Used to verify the data hasn't changed between runs."""
    if not os.path.exists(path):
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


def load_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def build_version_entry():
    """Pull together everything we know about this training run."""
    rf_meta = load_json("models/rf_metadata.json", {})
    dl_meta = load_json("models/dl_metadata.json", {})
    report  = load_json("docs/eval_report.json", {})

    return {
        "version":    "1.0.0",
        "created_at": datetime.utcnow().isoformat(),
        "data_hash":  file_hash("data/clean.csv"),
        "models": {
            "random_forest": {
                "file":    "models/rf_model.pkl",
                "cv_auc":  rf_meta.get("cv_auc"),
                "metrics": rf_meta.get("test_metrics", {}),
                "params":  rf_meta.get("best_params", {}),
            },
            "deep_learning": {
                "file":         "models/dl_best.pt",
                "best_val_auc": dl_meta.get("best_val_auc"),
                "metrics":      dl_meta.get("test_metrics", {}),
                "architecture": dl_meta.get("architecture", ""),
            },
        },
        "evaluation": {
            "auc_roc":           report.get("overall_metrics", {}).get("auc_roc"),
            "governance_status": report.get("governance_status", "PENDING"),
            "auc_fairness_gap":  report.get("fairness_audit", {}).get("_summary", {}).get("auc_gap"),
        },
        "intended_use":  "Research prototype. Predicts diabetes risk from clinical lab values.",
        "not_for":       "Direct clinical decision-making without physician review.",
        "limitations": [
            "Trained exclusively on Pima Indian women from Arizona — limited generalizability",
            "Insulin missingness (~49% of original data) may affect insulin-related predictions",
            "Does not incorporate longitudinal patient history or medication data",
        ],
    }


def update_registry(entry):
    registry = load_json(REGISTRY_PATH, [])
    registry.append(entry)
    os.makedirs("docs", exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)
    log.info(f"Registry updated → {REGISTRY_PATH}  ({len(registry)} version(s))")


def append_audit_log(entry):
    """Audit log is append-only — we never overwrite entries."""
    audit = load_json(AUDIT_LOG_PATH, [])
    audit.append({
        "timestamp":         datetime.utcnow().isoformat(),
        "event":             "MODEL_TRAINED",
        "version":           entry["version"],
        "data_hash":         entry["data_hash"],
        "auc_roc":           entry["evaluation"]["auc_roc"],
        "governance_status": entry["evaluation"]["governance_status"],
        "run_by":            "Mayur Patil",
    })
    with open(AUDIT_LOG_PATH, "w") as f:
        json.dump(audit, f, indent=2)
    log.info(f"Audit log updated → {AUDIT_LOG_PATH}")


def write_model_card(entry):
    """
    Model card in plain Markdown. Written to be readable by a clinician,
    not just a data scientist.
    """
    auc = entry["evaluation"].get("auc_roc", "N/A")
    status = entry["evaluation"].get("governance_status", "PENDING")

    card = f"""# Model Card — Clinical Lab Abnormality Predictor

## Overview
- **Task:** Binary classification — predict diabetes risk from clinical lab values
- **Version:** {entry['version']}
- **Date:** {entry['created_at'][:10]}
- **Author:** Mayur Patil, M.S. Data Science, George Washington University
- **Status:** {status}

## Dataset
- **Source:** Pima Indians Diabetes Dataset (UCI Machine Learning Repository)
- **Population:** Pima Indian women aged ≥21, Maricopa County, Arizona
- **Size:** ~750 records after cleaning
- **Features:** Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age, Pregnancies

## Performance
| Model | AUC-ROC |
|-------|---------|
| Random Forest | {entry['models']['random_forest'].get('metrics', {}).get('auc_roc', 'N/A')} |
| PyTorch DL    | {entry['models']['deep_learning'].get('metrics', {}).get('auc_roc', 'N/A')} |

## Intended Use
This model is a **research prototype** designed to demonstrate end-to-end clinical ML practices.
It should not be used as a standalone diagnostic tool. Any clinical application would require:
- Prospective validation on a diverse patient population
- Review and sign-off by qualified clinicians
- Integration with existing clinical workflows

## Limitations
{chr(10).join(f'- {l}' for l in entry['limitations'])}

## Fairness
Bias checks were performed across patient age groups. See `docs/eval_report.json` for details.
AUC fairness gap: {entry['evaluation'].get('auc_fairness_gap', 'N/A')}

## Governance
- Model versioned in `docs/model_registry.json`
- All training runs logged in `docs/audit_log.json`
- Data fingerprint: `{entry['data_hash']}`
"""

    with open("docs/model_card.md", "w", encoding="utf-8") as f:
        f.write(card)
    log.info("Model card written → docs/model_card.md")


def run():
    log.info("=" * 55)
    log.info("MODEL GOVERNANCE")
    log.info("=" * 55)

    entry = build_version_entry()
    update_registry(entry)
    append_audit_log(entry)
    write_model_card(entry)

    log.info(f"Governance status: {entry['evaluation']['governance_status']}")
    log.info("Done.\n")
    return entry


if __name__ == "__main__":
    run()
