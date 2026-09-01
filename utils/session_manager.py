import streamlit as st


# ============================================================
# STATE GROUPS
# ============================================================

DATASET_KEYS = [
    "dataset",
    "dataset_name"
]


PREPROCESSING_KEYS = [
    "target_column",
    "problem_type",
    "target_validation_warnings",
    "X_train_raw",
    "X_test_raw",
    "X_train_processed",
    "X_test_processed",
    "y_train",
    "y_test",
    "preprocessor"
]


MODEL_KEYS = [
    "trained_model",
    "model_name"
]


EVALUATION_KEYS = [
    "evaluation_metrics",
    "predictions",
    "evaluation"
]


EXPLAINABILITY_KEYS = [
    "shap_values",
    "shap_image_bytes",
    "shap_sample_size"
]


DRIFT_KEYS = [
    "drift_report"
]


DECISION_KEYS = [
    "deployment_decision"
]


AUDIT_KEYS = [
    "audit_report",
    "current_audit_id"
]


# ============================================================
# INTERNAL HELPER
# ============================================================

def _clear_keys(keys):
    """
    Removes the supplied keys from Streamlit session state.
    """

    for key in keys:
        st.session_state.pop(
            key,
            None
        )


# ============================================================
# RESET FROM PREPROCESSING
# ============================================================

def reset_from_preprocessing():
    """
    Clears everything that depends on preprocessing.

    Keeps:
        - Dataset
        - Dataset name

    Clears:
        - Preprocessing
        - Model
        - Evaluation
        - Explainability
        - Drift
        - Decision
        - Audit
    """

    _clear_keys(
        PREPROCESSING_KEYS
        + MODEL_KEYS
        + EVALUATION_KEYS
        + EXPLAINABILITY_KEYS
        + DRIFT_KEYS
        + DECISION_KEYS
        + AUDIT_KEYS
    )


# ============================================================
# RESET FROM MODEL
# ============================================================

def reset_from_model():
    """
    Clears everything that depends on the trained model.

    Keeps:
        - Dataset
        - Preprocessing

    Clears:
        - Model
        - Evaluation
        - Explainability
        - Drift
        - Decision
        - Audit
    """

    _clear_keys(
        MODEL_KEYS
        + EVALUATION_KEYS
        + EXPLAINABILITY_KEYS
        + DRIFT_KEYS
        + DECISION_KEYS
        + AUDIT_KEYS
    )


# ============================================================
# RESET FROM EVALUATION
# ============================================================

def reset_from_evaluation():
    """
    Clears everything that depends on model evaluation.

    Keeps:
        - Dataset
        - Preprocessing
        - Trained model

    Clears:
        - Evaluation
        - Explainability
        - Drift
        - Decision
        - Audit
    """

    _clear_keys(
        EVALUATION_KEYS
        + EXPLAINABILITY_KEYS
        + DRIFT_KEYS
        + DECISION_KEYS
        + AUDIT_KEYS
    )


# ============================================================
# RESET FROM EXPLAINABILITY
# ============================================================

def reset_from_explainability():
    """
    Clears explainability and downstream audit artifacts.
    """

    _clear_keys(
        EXPLAINABILITY_KEYS
        + DECISION_KEYS
        + AUDIT_KEYS
    )


# ============================================================
# RESET FROM DRIFT
# ============================================================

def reset_from_drift():
    """
    Clears drift and downstream decision/report artifacts.
    """

    _clear_keys(
        DRIFT_KEYS
        + DECISION_KEYS
        + AUDIT_KEYS
    )


# ============================================================
# RESET FROM DECISION
# ============================================================

def reset_from_decision():
    """
    Clears deployment recommendation and audit report.
    """

    _clear_keys(
        DECISION_KEYS
        + AUDIT_KEYS
    )


# ============================================================
# RESET ENTIRE PIPELINE
# ============================================================

def reset_pipeline():
    """
    Clears every artifact derived from the current dataset
    while keeping the uploaded dataset itself.
    """

    _clear_keys(
        PREPROCESSING_KEYS
        + MODEL_KEYS
        + EVALUATION_KEYS
        + EXPLAINABILITY_KEYS
        + DRIFT_KEYS
        + DECISION_KEYS
        + AUDIT_KEYS
    )


# ============================================================
# CLEAR DATASET
# ============================================================

def clear_dataset():
    """
    Clears the entire current audit session, including
    the uploaded dataset.
    """

    reset_pipeline()

    _clear_keys(
        DATASET_KEYS
    )