import pandas as pd


def validate_target(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str
):
    """
    Validates whether the selected target is technically
    compatible with the selected machine learning problem.

    Returns:
        {
            "valid": bool,
            "message": str,
            "warnings": list[str]
        }
    """

    warnings = []


    # ========================================================
    # TARGET COLUMN CHECK
    # ========================================================

    if target_column not in df.columns:

        return {
            "valid": False,
            "message": (
                f"Target column '{target_column}' "
                "does not exist in the dataset."
            ),
            "warnings": warnings
        }


    target = df[target_column]


    # ========================================================
    # MISSING TARGET VALUES
    # ========================================================

    missing_count = target.isna().sum()


    if missing_count > 0:

        return {
            "valid": False,
            "message": (
                f"Target column contains {missing_count} "
                "missing value(s). Target values cannot be "
                "missing during model training."
            ),
            "warnings": warnings
        }


    # ========================================================
    # DATASET SIZE
    # ========================================================

    sample_count = len(target)


    if sample_count < 10:

        return {
            "valid": False,
            "message": (
                "The dataset must contain at least "
                "10 rows."
            ),
            "warnings": warnings
        }


    # ========================================================
    # CLASSIFICATION
    # ========================================================

    if problem_type == "Classification":

        unique_count = target.nunique(
            dropna=True
        )


        # ----------------------------------------------------
        # Need at least two classes
        # ----------------------------------------------------

        if unique_count < 2:

            return {
                "valid": False,
                "message": (
                    "Classification requires at least "
                    "two distinct target classes."
                ),
                "warnings": warnings
            }


        # ----------------------------------------------------
        # Check class distribution
        # ----------------------------------------------------

        class_counts = target.value_counts()

        smallest_class_count = class_counts.min()


        if smallest_class_count < 2:

            return {
                "valid": False,
                "message": (
                    "Each target class must contain at "
                    "least two samples so the data can be "
                    "split reliably for training and testing."
                ),
                "warnings": warnings
            }


        # ----------------------------------------------------
        # High-cardinality warning
        # ----------------------------------------------------

        if unique_count > 20:

            warnings.append(
                "The selected classification target has "
                f"{unique_count} unique classes. This is a "
                "high-cardinality target and may require "
                "specialized modeling."
            )


        # ----------------------------------------------------
        # Imbalanced target warning
        # ----------------------------------------------------

        class_proportions = (
            class_counts / sample_count
        )

        smallest_class_proportion = (
            class_proportions.min()
        )


        if smallest_class_proportion < 0.05:

            warnings.append(
                "The target classes are highly imbalanced. "
                "Consider evaluating precision, recall, "
                "and F1 score carefully."
            )


    # ========================================================
    # REGRESSION
    # ========================================================

    elif problem_type == "Regression":

        # ----------------------------------------------------
        # Regression requires numerical target
        # ----------------------------------------------------

        if not pd.api.types.is_numeric_dtype(
            target
        ):

            return {
                "valid": False,
                "message": (
                    "Regression requires a numerical "
                    "target column."
                ),
                "warnings": warnings
            }


        # ----------------------------------------------------
        # Need more than one unique value
        # ----------------------------------------------------

        unique_count = target.nunique(
            dropna=True
        )


        if unique_count < 2:

            return {
                "valid": False,
                "message": (
                    "Regression requires the target to "
                    "contain more than one distinct value."
                ),
                "warnings": warnings
            }


        # ----------------------------------------------------
        # Warn about discrete numeric targets
        # ----------------------------------------------------

        if unique_count <= 10:

            warnings.append(
                "The selected numerical target contains "
                f"only {unique_count} unique values. "
                "Regression is technically possible, but "
                "Classification may be more appropriate "
                "if these values represent discrete classes."
            )


    # ========================================================
    # UNKNOWN PROBLEM TYPE
    # ========================================================

    else:

        return {
            "valid": False,
            "message": (
                f"Unsupported problem type: "
                f"'{problem_type}'."
            ),
            "warnings": warnings
        }


    # ========================================================
    # VALID
    # ========================================================

    return {
        "valid": True,
        "message": "Target is valid.",
        "warnings": warnings
    }