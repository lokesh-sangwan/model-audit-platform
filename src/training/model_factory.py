from src.config.settings import settings

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression
)

from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)


def get_model(
    model_name: str
):
    """
    Creates an ML model based on user choice.

    Args:
        model_name: selected algorithm

    Returns:
        sklearn model object
    """

    models = {

        # ====================================================
        # CLASSIFICATION MODELS
        # ====================================================

        "Logistic Regression": LogisticRegression(
            max_iter=1000
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=settings.random_state
        ),

        "Random Forest": RandomForestClassifier(
            random_state=settings.random_state
        ),


        # ====================================================
        # REGRESSION MODELS
        # ====================================================

        "Linear Regression": LinearRegression(),

        "Decision Tree Regressor": DecisionTreeRegressor(
            random_state=settings.random_state
        ),

        "Random Forest Regressor": RandomForestRegressor(
            random_state=settings.random_state
        )

    }


    if model_name not in models:

        raise ValueError(
            "Unsupported model selected"
        )


    return models[model_name]