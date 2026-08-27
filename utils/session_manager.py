import streamlit as st


# ============================================================
# PIPELINE STATE
# ============================================================

PIPELINE_KEYS = [

    # Raw split data
    "X_train_raw",
    "X_test_raw",

    # Processed data
    "X_train_processed",
    "X_test_processed",

    # Target data
    "y_train",
    "y_test",

    # Preprocessing
    "preprocessor",

    # Model
    "trained_model",
    "model_name",

    # Evaluation
    "evaluation_metrics",
    "predictions",
    "evaluation",

    # Explainability
    "shap_values",
    "shap_image_bytes",
    "shap_sample_size",

    # Drift
    "drift_report",

    # Future modules
    "deployment_decision",
    "audit_report"
]


# ============================================================
# RESET PIPELINE
# ============================================================

def reset_pipeline():
    """
    Clears every artifact derived from the current dataset
    while keeping the uploaded dataset itself.
    """

    for key in PIPELINE_KEYS:

        st.session_state.pop(
            key,
            None
        )


# ============================================================
# CLEAR DATASET
# ============================================================

def clear_dataset():
    """
    Clears the entire current audit session.
    """

    reset_pipeline()

    st.session_state.pop(
        "dataset",
        None
    )

    st.session_state.pop(
        "dataset_name",
        None
    )