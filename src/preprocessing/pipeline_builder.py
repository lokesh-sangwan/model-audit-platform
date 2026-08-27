from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)


# ============================================================
# CONFIGURATION
# ============================================================

HIGH_CARDINALITY_THRESHOLD = 0.5
HIGH_MISSINGNESS_THRESHOLD = 0.5


# ============================================================
# COLUMN DETECTION
# ============================================================

def _is_identifier_column(column_name):
    """
    Determines whether a column is likely to be an identifier.

    Identifier columns generally should not be used as predictive
    features because they identify a record rather than describe
    a meaningful characteristic of that record.
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


def _find_high_cardinality_columns(
    X,
    categorical_columns
):
    """
    Finds categorical columns with a high proportion
    of unique values.
    """

    high_cardinality_columns = []

    for column in categorical_columns:

        unique_ratio = (
            X[column].nunique(dropna=True)
            / max(len(X), 1)
        )

        if unique_ratio >= HIGH_CARDINALITY_THRESHOLD:

            high_cardinality_columns.append(
                column
            )

    return high_cardinality_columns


def _find_high_missingness_columns(
    X,
    categorical_columns
):
    """
    Finds categorical columns with a high percentage
    of missing values.
    """

    high_missingness_columns = []

    for column in categorical_columns:

        missing_ratio = X[column].isna().mean()

        if missing_ratio >= HIGH_MISSINGNESS_THRESHOLD:

            high_missingness_columns.append(
                column
            )

    return high_missingness_columns


# ============================================================
# MAIN PIPELINE BUILDER
# ============================================================

def build_preprocessing_pipeline(X):
    """
    Builds a preprocessing pipeline based on the structure
    of the input feature dataframe.

    Processing steps:

    1. Exclude obvious identifier columns.
    2. Detect numeric and categorical columns.
    3. Exclude high-cardinality categorical columns.
    4. Exclude highly-missing categorical columns.
    5. Impute missing numeric values.
    6. Scale numeric features.
    7. Impute missing categorical values.
    8. One-hot encode usable categorical features.

    Args:
        X: Feature dataframe.

    Returns:
        sklearn ColumnTransformer
    """

    # --------------------------------------------------------
    # Identify identifier columns
    # --------------------------------------------------------

    identifier_columns = [
        column
        for column in X.columns
        if _is_identifier_column(column)
    ]

    X_model = X.drop(
        columns=identifier_columns,
        errors="ignore"
    )

    # --------------------------------------------------------
    # Detect column types
    # --------------------------------------------------------

    numeric_columns = list(
        X_model.select_dtypes(
            include=["int64", "float64"]
        ).columns
    )

    categorical_columns = list(
        X_model.select_dtypes(
            include=["object"]
        ).columns
    )

    # --------------------------------------------------------
    # Detect problematic categorical columns
    # --------------------------------------------------------

    high_cardinality_columns = (
        _find_high_cardinality_columns(
            X_model,
            categorical_columns
        )
    )

    high_missingness_columns = (
        _find_high_missingness_columns(
            X_model,
            categorical_columns
        )
    )

    # --------------------------------------------------------
    # Keep only suitable categorical features
    # --------------------------------------------------------

    excluded_categorical_columns = set(
        high_cardinality_columns
        + high_missingness_columns
    )

    usable_categorical_columns = [
        column
        for column in categorical_columns
        if column not in excluded_categorical_columns
    ]

    # --------------------------------------------------------
    # Numeric preprocessing
    # --------------------------------------------------------

    numeric_pipeline = Pipeline(

        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="mean"
                )
            ),

            (
                "scaler",
                StandardScaler()
            )

        ]

    )

    # --------------------------------------------------------
    # Categorical preprocessing
    # --------------------------------------------------------

    categorical_pipeline = Pipeline(

        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="if_binary"
                )
            )

        ]

    )

    # --------------------------------------------------------
    # Build transformers
    # --------------------------------------------------------

    transformers = []

    if numeric_columns:

        transformers.append(

            (
                "numeric",
                numeric_pipeline,
                numeric_columns
            )

        )

    if usable_categorical_columns:

        transformers.append(

            (
                "categorical",
                categorical_pipeline,
                usable_categorical_columns
            )

        )

    # --------------------------------------------------------
    # Final ColumnTransformer
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(

        transformers=transformers,

        remainder="drop"

    )

    return preprocessor