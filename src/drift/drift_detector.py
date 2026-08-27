import numpy as np
import pandas as pd

from scipy.stats import ks_2samp


# ============================================================
# CONFIGURATION
# ============================================================

NUMERIC_DRIFT_THRESHOLD = 0.10
CATEGORICAL_DRIFT_THRESHOLD = 0.10
OVERALL_DRIFT_THRESHOLD = 0.20


# ============================================================
# NUMERIC DRIFT
# ============================================================

def calculate_numeric_drift(
    reference,
    current
):
    """
    Calculates distribution drift for a numerical feature
    using the Kolmogorov-Smirnov (KS) statistic.

    Args:
        reference: Reference/training feature values.
        current: Current/test feature values.

    Returns:
        Dictionary containing drift statistics.
    """

    reference = pd.Series(reference).dropna()
    current = pd.Series(current).dropna()

    if len(reference) == 0 or len(current) == 0:
        return {
            "metric": "KS Statistic",
            "score": np.nan,
            "drifted": False
        }

    statistic, p_value = ks_2samp(
        reference,
        current
    )

    return {
        "metric": "KS Statistic",
        "score": float(statistic),
        "p_value": float(p_value),
        "drifted": bool(
            statistic >= NUMERIC_DRIFT_THRESHOLD
        )
    }


# ============================================================
# CATEGORICAL DRIFT
# ============================================================

def calculate_categorical_drift(
    reference,
    current
):
    """
    Calculates distribution drift for a categorical feature
    using Total Variation Distance.

    Total Variation Distance measures how different the
    category distributions are between two datasets.

    Range:

        0.0 -> identical distributions
        1.0 -> completely different distributions
    """

    reference = pd.Series(reference).fillna(
        "__MISSING__"
    )

    current = pd.Series(current).fillna(
        "__MISSING__"
    )

    reference_distribution = (
        reference.value_counts(normalize=True)
    )

    current_distribution = (
        current.value_counts(normalize=True)
    )

    categories = set(
        reference_distribution.index
    ).union(
        current_distribution.index
    )

    reference_distribution = (
        reference_distribution.reindex(
            categories,
            fill_value=0
        )
    )

    current_distribution = (
        current_distribution.reindex(
            categories,
            fill_value=0
        )
    )

    score = (
        0.5
        * np.abs(
            reference_distribution
            - current_distribution
        ).sum()
    )

    return {
        "metric": "Total Variation Distance",
        "score": float(score),
        "drifted": bool(
            score >= CATEGORICAL_DRIFT_THRESHOLD
        )
    }


# ============================================================
# COMPLETE DRIFT ANALYSIS
# ============================================================

def detect_data_drift(
    reference_data,
    current_data
):
    """
    Compares reference and current datasets feature by feature.

    Numerical columns use the KS statistic.

    Categorical columns use Total Variation Distance.

    Returns:
        A dictionary containing per-feature results and
        overall drift statistics.
    """

    common_columns = [
        column
        for column in reference_data.columns
        if column in current_data.columns
    ]

    results = []

    for column in common_columns:

        reference_column = reference_data[column]
        current_column = current_data[column]

        # ----------------------------------------------------
        # Numerical feature
        # ----------------------------------------------------

        if pd.api.types.is_numeric_dtype(
            reference_column
        ):

            result = calculate_numeric_drift(
                reference_column,
                current_column
            )

            results.append({

                "feature": column,

                "data_type": "Numerical",

                "metric": result["metric"],

                "drift_score": result["score"],

                "p_value": result.get(
                    "p_value",
                    np.nan
                ),

                "drifted": result["drifted"]

            })

        # ----------------------------------------------------
        # Categorical feature
        # ----------------------------------------------------

        else:

            result = calculate_categorical_drift(
                reference_column,
                current_column
            )

            results.append({

                "feature": column,

                "data_type": "Categorical",

                "metric": result["metric"],

                "drift_score": result["score"],

                "p_value": np.nan,

                "drifted": result["drifted"]

            })

    results_df = pd.DataFrame(
        results
    )

    if len(results_df) == 0:

        drifted_features = 0
        total_features = 0
        drift_percentage = 0.0

    else:

        drifted_features = int(
            results_df["drifted"].sum()
        )

        total_features = len(
            results_df
        )

        drift_percentage = (
            drifted_features
            / total_features
        )

    overall_drift = (
        drift_percentage
        >= OVERALL_DRIFT_THRESHOLD
    )

    return {

        "feature_results": results_df,

        "drifted_features": drifted_features,

        "total_features": total_features,

        "drift_percentage": drift_percentage,

        "overall_drift": overall_drift

    }