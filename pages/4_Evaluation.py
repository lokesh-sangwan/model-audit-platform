import streamlit as st

from src.evaluation.classification import (
    evaluate_classifier
)

from src.visualization.confusion_matrix import (
    create_confusion_matrix_plot
)


st.title("Model Evaluation")


def display_results():

    evaluation = st.session_state["evaluation"]

    metrics = evaluation["metrics"]

    predictions = evaluation["predictions"]


    st.success("Evaluation completed.")


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


if "trained_model" not in st.session_state:

    st.warning(
        "Please train a model first."
    )

    st.stop()


# ============================
# Already evaluated
# ============================

if "evaluation" in st.session_state:

    display_results()

    st.divider()

    if st.button("🔄 Re-run Evaluation"):

        metrics, predictions = evaluate_classifier(

            st.session_state["trained_model"],

            st.session_state["X_test_processed"],

            st.session_state["y_test"]

        )

        st.session_state["evaluation"] = {

            "metrics": metrics,

            "predictions": predictions

        }

        st.rerun()

# ============================
# First evaluation
# ============================

else:

    st.info("Model has not been evaluated yet.")

    if st.button("Run Evaluation"):

        metrics, predictions = evaluate_classifier(

            st.session_state["trained_model"],

            st.session_state["X_test_processed"],

            st.session_state["y_test"]

        )

        st.session_state["evaluation"] = {

            "metrics": metrics,

            "predictions": predictions

        }

        st.rerun()