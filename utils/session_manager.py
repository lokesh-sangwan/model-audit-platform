import streamlit as st


PIPELINE_KEYS = [
    "X_train_processed",
    "X_test_processed",
    "y_train",
    "y_test",
    "preprocessor",
    "trained_model",
    "model_name",
    "evaluation",
    "predictions",
    "shap_values",
    "drift_report",
    "deployment_decision",
    "audit_report"
]


def _next_uploader_key():
    st.session_state["uploader_key"] = (
        st.session_state.get("uploader_key", 0) + 1
    )


def reset_pipeline():

    for key in PIPELINE_KEYS:
        st.session_state.pop(key, None)


def clear_dataset():

    reset_pipeline()

    st.session_state.pop("dataset", None)
    st.session_state.pop("dataset_name", None)

    _next_uploader_key()