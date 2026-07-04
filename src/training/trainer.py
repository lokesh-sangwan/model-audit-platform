def train_model(
    model,
    X_train,
    y_train
):
    """
    Trains a machine learning model.

    Args:
        model: sklearn estimator
        X_train: training features
        y_train: labels

    Returns:
        trained model
    """


    model.fit(
        X_train,
        y_train
    )


    return model