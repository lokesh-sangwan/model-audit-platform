from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)


def build_preprocessing_pipeline(X):
    """
    Builds preprocessing pipeline based on column types.

    Args:
        X: Feature dataframe

    Returns:
        sklearn ColumnTransformer
    """


    numeric_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns


    categorical_columns = X.select_dtypes(
        include=["object"]
    ).columns



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
                    handle_unknown="ignore"
                )
            )

        ]

    )



    preprocessor = ColumnTransformer(

        transformers=[

            (
                "numeric",
                numeric_pipeline,
                numeric_columns
            ),


            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )

        ]

    )


    return preprocessor