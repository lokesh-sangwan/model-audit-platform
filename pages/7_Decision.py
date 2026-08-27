import streamlit as st

from src.decision.decision_engine import (
    make_deployment_decision,
    DEPLOY_THRESHOLD,
    MONITOR_THRESHOLD,
    DRIFT_MONITOR_THRESHOLD,
    DRIFT_BLOCK_THRESHOLD
)


st.title(
    "Deployment Recommendation"
)


# ============================================================
# PREREQUISITE CHECK
# ============================================================

required_keys = [
    "evaluation",
    "drift_report"
]


missing_keys = [
    key
    for key in required_keys
    if key not in st.session_state
]


if missing_keys:

    st.warning(
        "Please complete Model Evaluation and "
        "Data Drift Detection first."
    )

    st.stop()


# ============================================================
# CURRENT MODEL
# ============================================================

if "model_name" in st.session_state:

    st.info(
        f"Model under review: "
        f"**{st.session_state['model_name']}**"
    )


# ============================================================
# DECISION RULES
# ============================================================

with st.expander(
    "Decision criteria"
):

    st.write(
        "### Deploy"
    )

    st.write(
        f"All evaluation metrics ≥ "
        f"{DEPLOY_THRESHOLD:.0%}, "
        f"drift < {DRIFT_MONITOR_THRESHOLD:.0%}, "
        "and explainability available."
    )

    st.write(
        "### Monitor"
    )

    st.write(
        f"Metrics between "
        f"{MONITOR_THRESHOLD:.0%} and "
        f"{DEPLOY_THRESHOLD:.0%}, "
        f"or drift ≥ {DRIFT_MONITOR_THRESHOLD:.0%}, "
        "or explainability is unavailable."
    )

    st.write(
        "### Block"
    )

    st.write(
        f"Any metric < {MONITOR_THRESHOLD:.0%}, "
        f"or drift ≥ {DRIFT_BLOCK_THRESHOLD:.0%}, "
        "or required audit information is missing."
    )


# ============================================================
# EXPLAINABILITY STATUS
# ============================================================

explainability_available = (
    "shap_image_bytes"
    in st.session_state
)


# ============================================================
# GENERATE DECISION
# ============================================================

if "deployment_decision" not in st.session_state:

    st.info(
        "No deployment recommendation has been generated yet."
    )

    if st.button(
        "Generate Deployment Recommendation"
    ):

        with st.spinner(
            "Evaluating deployment readiness..."
        ):

            decision = make_deployment_decision(

                st.session_state[
                    "evaluation"
                ],

                st.session_state[
                    "drift_report"
                ],

                explainability_available

            )

            st.session_state[
                "deployment_decision"
            ] = decision

        st.rerun()


# ============================================================
# DISPLAY DECISION
# ============================================================

else:

    decision = st.session_state[
        "deployment_decision"
    ]

    decision_name = decision[
        "decision"
    ]

    # --------------------------------------------------------
    # Main decision
    # --------------------------------------------------------

    st.subheader(
        "Final Recommendation"
    )

    if decision_name == "DEPLOY":

        st.success(
            "🟢 DEPLOY"
        )

    elif decision_name == "MONITOR":

        st.warning(
            "🟡 MONITOR"
        )

    else:

        st.error(
            "🔴 BLOCK"
        )

    # --------------------------------------------------------
    # Severity
    # --------------------------------------------------------

    st.write(
        f"**Assessment Level:** "
        f"{decision['severity']}"
    )

    st.divider()

    # --------------------------------------------------------
    # Model metrics
    # --------------------------------------------------------

    st.subheader(
        "Model Performance"
    )

    metrics = st.session_state[
        "evaluation"
    ]["metrics"]

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

    # --------------------------------------------------------
    # Drift
    # --------------------------------------------------------

    st.subheader(
        "Data Drift"
    )

    drift_report = st.session_state[
        "drift_report"
    ]

    col1, col2 = st.columns(2)

    col1.metric(
        "Drifted Features",
        drift_report[
            "drifted_features"
        ]
    )

    col2.metric(
        "Drift Percentage",
        f"{drift_report['drift_percentage']:.1%}"
    )

    # --------------------------------------------------------
    # Explainability
    # --------------------------------------------------------

    st.subheader(
        "Explainability"
    )

    if explainability_available:

        st.success(
            "SHAP explanation available."
        )

    else:

        st.warning(
            "SHAP explanation has not been generated."
        )

    # --------------------------------------------------------
    # Decision reasons
    # --------------------------------------------------------

    st.subheader(
        "Decision Reasoning"
    )

    for reason in decision[
        "reasons"
    ]:

        st.write(
            f"• {reason}"
        )

    # --------------------------------------------------------
    # Re-run
    # --------------------------------------------------------

    st.divider()

    if st.button(
        "🔄 Recalculate Recommendation"
    ):

        with st.spinner(
            "Recalculating deployment readiness..."
        ):

            decision = make_deployment_decision(

                st.session_state[
                    "evaluation"
                ],

                st.session_state[
                    "drift_report"
                ],

                explainability_available

            )

            st.session_state[
                "deployment_decision"
            ] = decision

        st.rerun()