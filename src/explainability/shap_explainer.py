import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import shap

from scipy import sparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


def _to_dense(data):
    """Convert sparse data to a dense array."""

    if sparse.issparse(data):
        return data.toarray()

    return data


def _get_feature_names(preprocessor, number_of_features):
    """Get transformed feature names from the fitted preprocessor."""

    try:
        feature_names = preprocessor.get_feature_names_out()

        if len(feature_names) == number_of_features:
            return list(feature_names)

    except Exception:
        pass

    return [
        f"Feature {index + 1}"
        for index in range(number_of_features)
    ]


def _get_explainer(model, background_data):
    """Select the appropriate SHAP explainer."""

    if isinstance(
        model,
        (
            RandomForestClassifier,
            DecisionTreeClassifier
        )
    ):
        return shap.TreeExplainer(model)

    if isinstance(model, LogisticRegression):
        return shap.LinearExplainer(
            model,
            background_data
        )

    return shap.Explainer(
        model,
        background_data
    )


def _select_class_shap_values(
    shap_values,
    class_index=1
):
    """
    Select SHAP values for one classification output.

    For binary/multiclass classifiers, recent SHAP versions
    may return values with shape:

        samples × features × classes

    We select one class so that the summary plot represents
    the contribution toward that class.
    """

    if isinstance(shap_values, list):

        if len(shap_values) > class_index:
            return shap_values[class_index]

        return shap_values[0]

    if isinstance(shap_values, shap.Explanation):

        values = shap_values.values

        if values.ndim == 3:

            selected_values = values[:, :, class_index]

            if shap_values.base_values is not None:

                base_values = shap_values.base_values

                if base_values.ndim == 2:

                    base_values = base_values[:, class_index]

            else:
                base_values = None

            return shap.Explanation(
                values=selected_values,
                base_values=base_values,
                data=shap_values.data,
                feature_names=shap_values.feature_names
            )

        return shap_values

    if hasattr(shap_values, "ndim"):

        if shap_values.ndim == 3:
            return shap_values[:, :, class_index]

    return shap_values


def generate_shap_summary(
    model,
    X_train,
    X_test,
    preprocessor
):
    """
    Generate a standard SHAP summary plot for a
    classification model.

    The plot shows how each feature contributes
    toward the selected class.
    """

    X_train_dense = _to_dense(X_train)
    X_test_dense = _to_dense(X_test)

    X_train_dense = X_train_dense.astype(float)
    X_test_dense = X_test_dense.astype(float)

    feature_names = _get_feature_names(
        preprocessor,
        X_test_dense.shape[1]
    )

    X_train_display = pd.DataFrame(
        X_train_dense,
        columns=feature_names
    )

    X_test_display = pd.DataFrame(
        X_test_dense,
        columns=feature_names
    )

    explainer = _get_explainer(
        model,
        X_train_display
    )

    # Generate SHAP explanation.
    if isinstance(
        model,
        (
            RandomForestClassifier,
            DecisionTreeClassifier
        )
    ):

        raw_shap_values = explainer(
            X_test_display,
            check_additivity=False
        )

    else:

        raw_shap_values = explainer(
            X_test_display
        )

    # For classification, select the positive class.
    shap_values = _select_class_shap_values(
        raw_shap_values,
        class_index=1
    )

    # Create the actual SHAP summary plot.
    plt.close("all")

    shap.summary_plot(
        shap_values,
        X_test_display,
        plot_type="dot",
        show=False,
        max_display=15
    )

    figure = plt.gcf()

    figure.set_size_inches(
        10,
        7
    )

    figure.tight_layout()

    # Convert the actual figure to PNG bytes.
    image_buffer = io.BytesIO()

    figure.savefig(
        image_buffer,
        format="png",
        dpi=150,
        bbox_inches="tight"
    )

    image_buffer.seek(0)

    image_bytes = image_buffer.getvalue()

    plt.close(figure)

    return shap_values, image_bytes