import streamlit as st

from src.evaluation.classification import (
    evaluate_classifier
)

from src.evaluation.regression import (
    evaluate_regressor
)

from src.visualization.confusion_matrix import (
    create_confusion_matrix_plot
)


st.title("Model Evaluation")


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results():

    evaluation = st.session_state["evaluation"]

    metrics = evaluation["metrics"]

    predictions = evaluation["predictions"]

    problem_type = st.session_state.get(
        "problem_type",
        "Classification"
    )


    st.success(
        "Evaluation completed."
    )


    # ========================================================
    # CLASSIFICATION RESULTS
    # ========================================================

    if problem_type == "Classification":

        col1, col2 = st.columns(2)

        col1.metric(
            "Accuracy",
            f"{metrics['accuracy']:.3f}"
        )

        col2.metric(
            "Precision",
            f"{metrics['precision']:.3f}"
        )


        col3, col4 = st.columns(2)

        col3.metric(
            "Recall",
            f"{metrics['recall']:.3f}"
        )

        col4.metric(
            "F1 Score",
            f"{metrics['f1_score']:.3f}"
        )


        figure = create_confusion_matrix_plot(

            st.session_state["y_test"],

            predictions

        )


        st.plotly_chart(
            figure,
            use_container_width=True,
            key="evaluation_confusion_matrix"
        )


    # ========================================================
    # REGRESSION RESULTS
    # ========================================================

    else:

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "MAE",
            f"{metrics['mae']:.3f}"
        )

        col2.metric(
            "RMSE",
            f"{metrics['rmse']:.3f}"
        )

        col3.metric(
            "R² Score",
            f"{metrics['r2_score']:.3f}"
        )


# ============================================================
# PREREQUISITE CHECK
# ============================================================

if "trained_model" not in st.session_state:

    st.warning(
        "Please train a model first."
    )

    st.stop()


# ============================================================
# PROBLEM TYPE
# ============================================================

problem_type = st.session_state.get(
    "problem_type",
    "Classification"
)


st.info(
    f"Evaluation Type: **{problem_type}**"
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def run_evaluation():

    if problem_type == "Classification":

        metrics, predictions = evaluate_classifier(

            st.session_state["trained_model"],

            st.session_state["X_test_processed"],

            st.session_state["y_test"]

        )

    else:

        metrics, predictions = evaluate_regressor(

            st.session_state["trained_model"],

            st.session_state["X_test_processed"],

            st.session_state["y_test"]

        )


    st.session_state["evaluation"] = {

        "metrics": metrics,

        "predictions": predictions

    }


    st.session_state["predictions"] = predictions


# ============================================================
# ALREADY EVALUATED
# ============================================================

if "evaluation" in st.session_state:

    display_results()

    st.divider()

    if st.button(
        "🔄 Re-run Evaluation"
    ):

        run_evaluation()

        st.rerun()


# ============================================================
# FIRST EVALUATION
# ============================================================

else:

    st.info(
        "Model has not been evaluated yet."
    )


    if st.button(
        "Run Evaluation"
    ):

        run_evaluation()

        st.rerun()