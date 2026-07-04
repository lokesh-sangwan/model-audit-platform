import plotly.express as px

from sklearn.metrics import confusion_matrix



def create_confusion_matrix_plot(
    y_true,
    y_pred
):
    """
    Creates confusion matrix visualization.


    Args:
        y_true: Actual labels
        y_pred: Model predictions


    Returns:
        Plotly figure
    """


    matrix = confusion_matrix(
        y_true,
        y_pred
    )


    figure = px.imshow(

        matrix,

        text_auto=True,

        labels={

            "x": "Predicted",

            "y": "Actual"

        }

    )


    figure.update_layout(

        title="Confusion Matrix"

    )


    return figure