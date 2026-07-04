from src.preprocessing.pipeline_builder import (
    build_preprocessing_pipeline
)


def preprocess_data(
    X_train,
    X_test
):
    """
    Applies preprocessing pipeline.

    Returns:
        processed data
        fitted preprocessor
    """


    preprocessor = build_preprocessing_pipeline(
        X_train
    )


    X_train_processed = (
        preprocessor.fit_transform(
            X_train
        )
    )


    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )


    return (
        X_train_processed,
        X_test_processed,
        preprocessor
    )