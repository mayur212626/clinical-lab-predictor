# feature_engineering.py
# ─────────────────────────────────────────────────────────────────────────────
# Building features that actually make clinical sense.
# The raw Pima columns are fine, but a few derived features help the model
# pick up on patterns that aren't obvious from individual values alone.
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import joblib
import logging
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")


def add_clinical_features(df):
    """
    A few features that endocrinologists actually look at:

    - Glucose/Insulin ratio: rough proxy for insulin resistance. When glucose
      is high but insulin is low, it suggests the pancreas isn't keeping up.
    - BMI category: the WHO cutoffs (underweight/normal/overweight/obese)
      matter clinically and give the tree-based models cleaner splits.
    - Age group: diabetes risk jumps significantly after 45, so bucketing
      helps non-linear models without having to learn it from scratch.
    """
    df = df.copy()

    # Adding 1 to avoid division by zero — Insulin of 0 still exists after imputation
    # in some edge cases, so this is a safety measure
    df["GlucoseInsulinRatio"] = df["Glucose"] / (df["Insulin"] + 1)

    df["BMICategory"] = pd.cut(
        df["BMI"],
        bins=[0, 18.5, 25.0, 30.0, 100.0],
        labels=[0, 1, 2, 3]   # underweight, normal, overweight, obese
    ).astype(int)

    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 30, 45, 60, 120],
        labels=[0, 1, 2, 3]   # young, middle, pre-senior, senior
    ).astype(int)

    log.info(f"Added 3 clinical features → {list(df.columns)}")
    return df


def split(df, test_size=0.2, seed=42):
    """Standard 80/20 split, stratified so both splits have the same positive rate."""
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )
    log.info(f"Split: {len(X_train)} train / {len(X_test)} test")
    log.info(f"  Train positive rate: {y_train.mean():.1%}")
    log.info(f"  Test positive rate:  {y_test.mean():.1%}")
    return X_train, X_test, y_train, y_test


def scale(X_train, X_test):
    """
    Fit a StandardScaler on the training set only — never peek at test data.
    Save the scaler so we can apply the exact same transformation at inference.
    """
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    log.info("Scaler fitted and saved → models/scaler.pkl")

    return X_train_s, X_test_s, scaler


def balance_with_smote(X_train, y_train, seed=42):
    """
    The dataset is about 65/35 (non-diabetic/diabetic), which isn't terrible,
    but SMOTE helps the model learn the minority class better without just
    duplicating rows. It generates synthetic samples in feature space.
    """
    before = y_train.value_counts().to_dict()
    sm = SMOTE(random_state=seed)
    X_bal, y_bal = sm.fit_resample(X_train, y_train)
    after = pd.Series(y_bal).value_counts().to_dict()

    log.info(f"SMOTE: {before} → {after}")
    return X_bal, y_bal


def save_splits(X_train, X_test, y_train, y_test):
    os.makedirs("data", exist_ok=True)
    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    pd.Series(y_train).to_csv("data/y_train.csv", index=False)
    pd.Series(y_test).to_csv("data/y_test.csv", index=False)
    log.info("Train/test splits saved → data/")


def run():
    log.info("=" * 55)
    log.info("FEATURE ENGINEERING")
    log.info("=" * 55)

    df = pd.read_csv("data/clean.csv")
    df = add_clinical_features(df)

    X_train, X_test, y_train, y_test = split(df)
    X_train_s, X_test_s, _ = scale(X_train, X_test)
    X_train_bal, y_train_bal = balance_with_smote(X_train_s, y_train)

    save_splits(X_train_bal, X_test_s, y_train_bal, y_test)

    log.info("Done.\n")
    return X_train_bal, X_test_s, y_train_bal, y_test


if __name__ == "__main__":
    run()
# split added 
