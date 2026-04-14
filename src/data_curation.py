# data_curation.py
# ─────────────────────────────────────────────────────────────────────────────
# First thing I do with any clinical dataset is figure out what's broken.
# The Pima dataset has a fun quirk: zeros in columns like Glucose or BMI
# are biologically impossible, so they're really just missing values in disguise.
# This module handles all of that — loading, cleaning, and basic QC.
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import logging
import os
import json
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# These columns can't be zero — a glucose of 0 means the patient is dead,
# not that the value is missing. Replacing with NaN so we can impute later.
BIOLOGICALLY_INVALID_ZEROS = [
    "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
]

COLUMN_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

DATA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
)


def load_raw_data(source=DATA_URL):
    """Pull the dataset from the URL or a local path."""
    log.info(f"Loading data from: {source}")
    df = pd.read_csv(source, header=None, names=COLUMN_NAMES)
    log.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


def fix_impossible_zeros(df):
    """
    Replace zeros that can't be real with NaN.
    Insulin has the most — nearly half the rows — which makes sense
    since fasting insulin wasn't always measured.
    """
    df = df.copy()
    for col in BIOLOGICALLY_INVALID_ZEROS:
        n_zeros = (df[col] == 0).sum()
        if n_zeros > 0:
            df[col] = df[col].replace(0, np.nan)
            log.info(f"  {col}: replaced {n_zeros} zeros with NaN")
    return df


def impute_with_group_median(df):
    """
    Fill NaNs using the median within each outcome group (diabetic vs not).
    This is better than a global median because the two groups have
    meaningfully different distributions — e.g., diabetics have higher glucose.
    """
    df = df.copy()
    for col in BIOLOGICALLY_INVALID_ZEROS:
        if df[col].isnull().sum() == 0:
            continue
        group_medians = df.groupby("Outcome")[col].median()
        df[col] = df.apply(
            lambda row: group_medians[row["Outcome"]] if pd.isnull(row[col]) else row[col],
            axis=1
        )
        log.info(f"  {col}: imputed — non-diabetic={group_medians[0]:.1f}, diabetic={group_medians[1]:.1f}")
    return df


def drop_extreme_outliers(df, z_cutoff=3.5):
    """
    Remove rows where any feature is more than `z_cutoff` standard deviations
    from the mean. Using 3.5 instead of 3 to be a bit more conservative —
    clinical data is noisy and we don't want to throw away real edge cases.
    """
    features = df.columns.drop("Outcome")
    z = (df[features] - df[features].mean()) / df[features].std()
    clean_mask = (z.abs() < z_cutoff).all(axis=1)
    n_dropped = (~clean_mask).sum()
    df = df[clean_mask].reset_index(drop=True)
    log.info(f"Outlier removal: dropped {n_dropped} rows, {len(df)} remaining")
    return df


def run_qc(df):
    """Quick sanity check before we hand this off to feature engineering."""
    report = {
        "run_at": datetime.utcnow().isoformat(),
        "n_rows": len(df),
        "n_missing": int(df.isnull().sum().sum()),
        "n_duplicates": int(df.duplicated().sum()),
        "positive_rate": round(float(df["Outcome"].mean()), 3),
        "column_stats": df.describe().round(2).to_dict(),
    }

    if report["n_missing"] > 0:
        log.warning(f"QC: {report['n_missing']} missing values remain after imputation")
    if report["n_duplicates"] > 0:
        log.warning(f"QC: {report['n_duplicates']} duplicate rows detected")

    log.info(
        f"QC passed — {report['n_rows']} rows, "
        f"{report['positive_rate']:.1%} positive rate"
    )
    return report


def save(df, path="data/clean.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    log.info(f"Saved clean data → {path}")


def run():
    log.info("=" * 55)
    log.info("DATA CURATION")
    log.info("=" * 55)

    df = load_raw_data()
    df = fix_impossible_zeros(df)
    df = impute_with_group_median(df)
    df = drop_extreme_outliers(df)
    qc = run_qc(df)
    save(df)

    log.info("Done.\n")
    return df, qc


if __name__ == "__main__":
    run()
# zero fix 
# imputation added 
# deploy script 
