from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)



def evaluate_classifier(
    model,
    X_test,
    y_test
):
    """
    Evaluates classification model performance.


    Args:
        model: trained ML model
        X_test: testing features
        y_test: true labels


    Returns:
        Dictionary of metrics
    """


    predictions = model.predict(
        X_test
    )


    metrics = {


        "accuracy": accuracy_score(
            y_test,
            predictions
        ),


        "precision": precision_score(
            y_test,
            predictions,
            average="weighted"
        ),


        "recall": recall_score(
            y_test,
            predictions,
            average="weighted"
        ),


        "f1_score": f1_score(
            y_test,
            predictions,
            average="weighted"
        )

    }


    return (
        metrics,
        predictions
    )