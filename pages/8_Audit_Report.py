import streamlit as st

from src.audit.audit_report import (
    generate_audit_report
)

from src.history.audit_history import (
    save_audit_record,
    audit_already_saved
)


st.title(
    "ML Audit Report"
)


# ============================================================
# PREREQUISITE CHECK
# ============================================================

required_keys = [
    "evaluation",
    "drift_report",
    "deployment_decision"
]


missing_keys = [
    key
    for key in required_keys
    if key not in st.session_state
]


if missing_keys:

    st.warning(
        "Please complete Model Evaluation, Data Drift, "
        "and Deployment Recommendation first."
    )

    st.stop()


# ============================================================
# PROBLEM TYPE
# ============================================================

problem_type = st.session_state.get(
    "problem_type",
    "Classification"
)


# ============================================================
# GENERATE REPORT
# ============================================================

if "audit_report" not in st.session_state:

    explainability_available = (
        "shap_image_bytes"
        in st.session_state
    )


    dataset_name = st.session_state.get(
        "dataset_name",
        "Unknown Dataset"
    )


    model_name = st.session_state.get(
        "model_name",
        "Unknown Model"
    )


    report = generate_audit_report(

        dataset_name,

        model_name,

        st.session_state[
            "evaluation"
        ],

        st.session_state[
            "drift_report"
        ],

        st.session_state[
            "deployment_decision"
        ],

        explainability_available

    )


    # --------------------------------------------------------
    # Store problem type in report
    # --------------------------------------------------------

    report[
        "problem_type"
    ] = problem_type


    st.session_state[
        "audit_report"
    ] = report


# ============================================================
# LOAD REPORT
# ============================================================

report = st.session_state[
    "audit_report"
]


metadata = report[
    "audit_metadata"
]


performance = report[
    "model_performance"
]


drift = report[
    "data_drift"
]


explainability = report[
    "explainability"
]


recommendation = report[
    "deployment_recommendation"
]


# ============================================================
# AUDIT INFORMATION
# ============================================================

st.subheader(
    "Audit Information"
)


col1, col2 = st.columns(2)


col1.metric(
    "Dataset",
    metadata["dataset_name"]
)


col2.metric(
    "Model",
    metadata["model_name"]
)


st.caption(
    f"Problem Type: {problem_type}"
)


st.caption(
    f"Generated at: {metadata['generated_at']}"
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()


st.subheader(
    "Model Performance"
)


if problem_type == "Classification":

    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "Accuracy",
        f"{performance['accuracy']:.3f}"
    )


    col2.metric(
        "Precision",
        f"{performance['precision']:.3f}"
    )


    col3.metric(
        "Recall",
        f"{performance['recall']:.3f}"
    )


    col4.metric(
        "F1 Score",
        f"{performance['f1_score']:.3f}"
    )


else:

    col1, col2, col3 = st.columns(3)


    col1.metric(
        "MAE",
        f"{performance['mae']:.3f}"
    )


    col2.metric(
        "RMSE",
        f"{performance['rmse']:.3f}"
    )


    col3.metric(
        "R² Score",
        f"{performance['r2_score']:.3f}"
    )


# ============================================================
# DATA DRIFT
# ============================================================

st.divider()


st.subheader(
    "Data Drift"
)


col1, col2, col3 = st.columns(3)


col1.metric(
    "Usable Features",
    drift["total_features"]
)


col2.metric(
    "Drifted Features",
    drift["drifted_features"]
)


col3.metric(
    "Drift Percentage",
    f"{drift['drift_percentage']:.1%}"
)


if drift["overall_drift"]:

    st.error(
        "Overall data drift detected."
    )

else:

    st.success(
        "No significant overall data drift detected."
    )


# ============================================================
# EXPLAINABILITY
# ============================================================

st.divider()


st.subheader(
    "Explainability"
)


if explainability["available"]:

    st.success(
        "SHAP explanation available."
    )

else:

    st.warning(
        "SHAP explanation was not generated."
    )


# ============================================================
# FINAL RECOMMENDATION
# ============================================================

st.divider()


st.subheader(
    "Deployment Recommendation"
)


decision = recommendation[
    "decision"
]


if decision == "DEPLOY":

    st.success(
        "🟢 DEPLOY"
    )

elif decision == "MONITOR":

    st.warning(
        "🟡 MONITOR"
    )

else:

    st.error(
        "🔴 BLOCK"
    )


st.write(
    f"**Assessment Level:** "
    f"{recommendation['severity']}"
)


# ============================================================
# REASONING
# ============================================================

st.subheader(
    "Decision Reasoning"
)


for reason in recommendation[
    "reasons"
]:

    st.write(
        f"• {reason}"
    )


# ============================================================
# SAVE AUDIT
# ============================================================

st.divider()


st.subheader(
    "Audit History"
)


if audit_already_saved(
    report
):

    st.success(
        "This audit has already been saved to history."
    )

else:

    if st.button(
        "Save Audit to History"
    ):

        audit_id = save_audit_record(
            report
        )


        st.session_state[
            "current_audit_id"
        ] = audit_id


        st.success(
            f"Audit saved successfully. "
            f"Audit ID: {audit_id}"
        )


# ============================================================
# COMPLETE AUDIT DATA
# ============================================================

with st.expander(
    "View Complete Audit Data"
):

    st.json(
        report
    )