import streamlit as st

from src.evaluation.classification import (
    evaluate_classifier
)

from src.visualization.confusion_matrix import (
    create_confusion_matrix_plot
)


st.title(
    "Model Evaluation"
)


if "trained_model" not in st.session_state:

    st.warning(
        "Please train a model first."
    )


else:

    st.subheader(
        "Evaluate Model Performance"
    )


    if st.button(
        "Run Evaluation"
    ):

        (
            metrics,
            predictions
        ) = evaluate_classifier(

            st.session_state["trained_model"],

            st.session_state["X_test_processed"],

            st.session_state["y_test"]

        )


        st.session_state["evaluation_metrics"] = (
            metrics
        )


        st.session_state["predictions"] = (
            predictions
        )


        st.success(
            "Evaluation completed"
        )


        col1, col2 = st.columns(2)


        col1.metric(

            "Accuracy",

            round(
                metrics["accuracy"],
                3
            )

        )


        col2.metric(

            "Precision",

            round(
                metrics["precision"],
                3
            )

        )


        col3, col4 = st.columns(2)


        col3.metric(

            "Recall",

            round(
                metrics["recall"],
                3
            )

        )


        col4.metric(

            "F1 Score",

            round(
                metrics["f1_score"],
                3
            )

        )


        confusion_matrix_plot = (
            create_confusion_matrix_plot(

                st.session_state["y_test"],

                predictions

            )
        )


        st.plotly_chart(

            confusion_matrix_plot,

            use_container_width=True

        )