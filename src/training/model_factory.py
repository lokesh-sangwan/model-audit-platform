from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def get_model(
    model_name: str
):
    """
    Creates ML model based on user choice.

    Args:
        model_name: selected algorithm

    Returns:
        sklearn model object
    """


    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=1000
        ),


        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),


        "Random Forest": RandomForestClassifier(
            random_state=42
        )

    }


    if model_name not in models:

        raise ValueError(
            "Unsupported model selected"
        )


    return models[model_name]