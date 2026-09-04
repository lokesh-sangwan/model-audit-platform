from src.config.settings import settings
import numpy as np
import pandas as pd

from scipy.stats import ks_2samp


# ============================================================
# CONFIGURATION
# ============================================================

HIGH_CARDINALITY_THRESHOLD = 0.5
HIGH_MISSINGNESS_THRESHOLD = 0.5

NUMERIC_DRIFT_THRESHOLD = settings.drift_monitor_threshold
CATEGORICAL_DRIFT_THRESHOLD = settings.drift_monitor_threshold
OVERALL_DRIFT_THRESHOLD = settings.drift_monitor_threshold


# ============================================================
# FEATURE QUALITY CHECKS
# ============================================================

def _is_identifier_column(column_name):
    """
    Determines whether a column is likely to be an identifier.
    """

    name = (
        str(column_name)
        .lower()
        .replace("_", "")
        .replace(" ", "")
    )

    identifier_keywords = [
        "id",
        "identifier"
    ]

    return any(
        keyword in name
        for keyword in identifier_keywords
    )


def _is_high_cardinality(
    data,
    column
):
    """
    Determines whether a categorical column has a high
    proportion of unique values.
    """

    unique_ratio = (
        data[column].nunique(dropna=True)
        / max(len(data), 1)
    )

    return unique_ratio >= HIGH_CARDINALITY_THRESHOLD


def _is_high_missingness(
    data,
    column
):
    """
    Determines whether a column has a high percentage
    of missing values.
    """

    missing_ratio = data[column].isna().mean()

    return missing_ratio >= HIGH_MISSINGNESS_THRESHOLD


# ============================================================
# NUMERICAL DRIFT
# ============================================================

def calculate_numeric_drift(
    reference,
    current
):
    """
    Calculates numerical distribution drift using the
    Kolmogorov-Smirnov statistic.
    """

    reference = pd.Series(reference).dropna()
    current = pd.Series(current).dropna()

    if len(reference) == 0 or len(current) == 0:

        return {
            "metric": "KS Statistic",
            "score": np.nan,
            "p_value": np.nan,
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
    Calculates categorical distribution drift using
    Total Variation Distance.
    """

    reference = pd.Series(reference).fillna(
        "__MISSING__"
    )

    current = pd.Series(current).fillna(
        "__MISSING__"
    )

    reference_distribution = (
        reference.value_counts(
            normalize=True
        )
    )

    current_distribution = (
        current.value_counts(
            normalize=True
        )
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
        "p_value": np.nan,
        "drifted": bool(
            score >= CATEGORICAL_DRIFT_THRESHOLD
        )
    }


# ============================================================
# FEATURE PROFILING
# ============================================================

def profile_features(
    reference_data
):
    """
    Identifies features that should participate in drift
    analysis and features that should be excluded because
    they are identifiers, highly missing, or high-cardinality.

    Returns:
        usable_features
        excluded_features
    """

    usable_features = []
    excluded_features = []

    for column in reference_data.columns:

        # ----------------------------------------------------
        # Identifier
        # ----------------------------------------------------

        if _is_identifier_column(column):

            excluded_features.append({
                "feature": column,
                "reason": "Identifier column"
            })

            continue

        # ----------------------------------------------------
        # High-cardinality categorical
        # ----------------------------------------------------

        if (
            not pd.api.types.is_numeric_dtype(
                reference_data[column]
            )
            and _is_high_cardinality(
                reference_data,
                column
            )
        ):

            excluded_features.append({
                "feature": column,
                "reason": "High-cardinality categorical feature"
            })

            continue

        # ----------------------------------------------------
        # Highly missing categorical
        # ----------------------------------------------------

        if (
            not pd.api.types.is_numeric_dtype(
                reference_data[column]
            )
            and _is_high_missingness(
                reference_data,
                column
            )
        ):

            excluded_features.append({
                "feature": column,
                "reason": "High missingness"
            })

            continue

        usable_features.append(
            column
        )

    return (
        usable_features,
        excluded_features
    )


# ============================================================
# COMPLETE DRIFT ANALYSIS
# ============================================================

def detect_data_drift(
    reference_data,
    current_data
):
    """
    Performs feature-level and overall drift analysis.

    Only model-relevant features are included in the drift
    calculation.

    Identifier-like, highly-missing, and high-cardinality
    categorical features are reported separately.
    """

    # --------------------------------------------------------
    # Find common columns
    # --------------------------------------------------------

    common_columns = [
        column
        for column in reference_data.columns
        if column in current_data.columns
    ]

    reference_common = reference_data[
        common_columns
    ]

    # --------------------------------------------------------
    # Profile features
    # --------------------------------------------------------

    (
        usable_features,
        excluded_features
    ) = profile_features(
        reference_common
    )

    results = []

    # --------------------------------------------------------
    # Analyze usable features
    # --------------------------------------------------------

    for column in usable_features:

        reference_column = (
            reference_data[column]
        )

        current_column = (
            current_data[column]
        )

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

                "p_value": result["p_value"],

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

                "p_value": result["p_value"],

                "drifted": result["drifted"]

            })

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Overall drift calculation
    # --------------------------------------------------------

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

        "excluded_features": pd.DataFrame(
            excluded_features
        ),

        "drifted_features": drifted_features,

        "total_features": total_features,

        "drift_percentage": drift_percentage,

        "overall_drift": overall_drift

    }