from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np


def evaluate_regressor(
    model,
    X_test,
    y_test
):
    """
    Evaluates a regression model.

    Args:
        model: trained regression model
        X_test: processed test features
        y_test: test target values

    Returns:
        metrics: dictionary containing regression metrics
        predictions: model predictions
    """

    predictions = model.predict(
        X_test
    )


    mae = mean_absolute_error(
        y_test,
        predictions
    )


    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )


    r2 = r2_score(
        y_test,
        predictions
    )


    metrics = {

        "mae": mae,

        "rmse": rmse,

        "r2_score": r2

    }


    return metrics, predictions