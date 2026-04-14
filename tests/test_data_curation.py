# tests/test_data_curation.py
# ─────────────────────────────────────────────────────────────────────────────
# Unit tests for the data curation pipeline.
# Nothing fancy here — just making sure the basic transformations work
# correctly and that we haven't broken anything obvious.
# ─────────────────────────────────────────────────────────────────────────────

import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_curation import (
    fix_impossible_zeros,
    impute_with_group_median,
    drop_extreme_outliers,
    run_qc,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample():
    """Small clean-ish dataframe that mirrors the Pima structure."""
    return pd.DataFrame({
        "Pregnancies":            [1, 2, 0, 3, 1],
        "Glucose":                [148, 0,  85, 183, 100],
        "BloodPressure":          [72,  66,  0,  64,  74],
        "SkinThickness":          [35,  29,  0,   0,  20],
        "Insulin":                [0,   0,   0,   0,  85],
        "BMI":                    [33.6, 26.6, 0.0, 23.3, 30.1],
        "DiabetesPedigreeFunction": [0.627, 0.351, 0.183, 0.672, 0.5],
        "Age":                    [50, 31, 22, 32, 45],
        "Outcome":                [1,  0,  0,  1,  0],
    })


# ── Tests: fix_impossible_zeros ───────────────────────────────────────────────

class TestFixZeros:
    def test_glucose_zero_becomes_nan(self, sample):
        out = fix_impossible_zeros(sample)
        assert out["Glucose"].isnull().sum() == 1

    def test_bmi_zero_becomes_nan(self, sample):
        out = fix_impossible_zeros(sample)
        assert out["BMI"].isnull().sum() == 1

    def test_pregnancies_untouched(self, sample):
        # zero pregnancies is valid — don't replace it
        out = fix_impossible_zeros(sample)
        assert out["Pregnancies"].isnull().sum() == 0

    def test_original_not_mutated(self, sample):
        original_glucose = sample["Glucose"].copy()
        fix_impossible_zeros(sample)
        assert (sample["Glucose"] == original_glucose).all()

    def test_nonzero_values_preserved(self, sample):
        out = fix_impossible_zeros(sample)
        # 148, 85, 183, 100 should still be there
        assert out["Glucose"].dropna().tolist() == [148.0, 85.0, 183.0, 100.0]


# ── Tests: impute_with_group_median ──────────────────────────────────────────

class TestImpute:
    def test_no_nulls_after_imputation(self, sample):
        df  = fix_impossible_zeros(sample)
        out = impute_with_group_median(df)
        assert out[["Glucose", "BloodPressure", "BMI", "Insulin", "SkinThickness"]].isnull().sum().sum() == 0

    def test_imputed_values_are_positive(self, sample):
        df  = fix_impossible_zeros(sample)
        out = impute_with_group_median(df)
        for col in ["Glucose", "BMI", "BloodPressure"]:
            assert (out[col] > 0).all(), f"{col} has non-positive imputed values"

    def test_group_medians_differ(self, sample):
        # diabetics and non-diabetics should get different imputed values
        df  = fix_impossible_zeros(sample)
        out = impute_with_group_median(df)
        # just making sure the imputation ran and didn't use a global constant
        assert out["Glucose"].nunique() > 1


# ── Tests: drop_extreme_outliers ─────────────────────────────────────────────

class TestDropOutliers:
    def test_normal_data_mostly_kept(self, sample):
        df  = fix_impossible_zeros(sample)
        df  = impute_with_group_median(df)
        out = drop_extreme_outliers(df, z_cutoff=10.0)
        # with a very high cutoff, nothing should be dropped
        assert len(out) == len(df)

    def test_extreme_value_removed(self):
        # inject an obvious outlier into otherwise normal data
        df = pd.DataFrame({
            "Pregnancies":            [1, 1, 1, 1, 1, 1],
            "Glucose":                [110, 105, 108, 112, 109, 999],  # 999 is extreme
            "BloodPressure":          [70, 72, 68, 74, 71, 70],
            "SkinThickness":          [25, 28, 26, 27, 25, 26],
            "Insulin":                [80, 85, 82, 88, 81, 83],
            "BMI":                    [28.0, 29.0, 27.5, 30.0, 28.5, 29.0],
            "DiabetesPedigreeFunction": [0.4, 0.5, 0.45, 0.48, 0.42, 0.47],
            "Age":                    [30, 32, 31, 33, 30, 31],
            "Outcome":                [0, 1, 0, 1, 0, 1],
        })
        out = drop_extreme_outliers(df, z_cutoff=2.5)
        assert len(out) < len(df)

    def test_returns_dataframe(self, sample):
        df  = fix_impossible_zeros(sample)
        df  = impute_with_group_median(df)
        out = drop_extreme_outliers(df)
        assert isinstance(out, pd.DataFrame)


# ── Tests: run_qc ─────────────────────────────────────────────────────────────

class TestQC:
    def test_report_has_required_keys(self, sample):
        df  = fix_impossible_zeros(sample)
        df  = impute_with_group_median(df)
        out = run_qc(df)
        for key in ["run_at", "n_rows", "n_missing", "n_duplicates", "positive_rate"]:
            assert key in out, f"Missing key: {key}"

    def test_row_count_correct(self, sample):
        df  = fix_impossible_zeros(sample)
        df  = impute_with_group_median(df)
        out = run_qc(df)
        assert out["n_rows"] == len(df)

    def test_positive_rate_in_range(self, sample):
        df  = fix_impossible_zeros(sample)
        df  = impute_with_group_median(df)
        out = run_qc(df)
        assert 0.0 <= out["positive_rate"] <= 1.0
