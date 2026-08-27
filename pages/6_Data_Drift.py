import streamlit as st

from src.drift.drift_detector import (
    detect_data_drift,
    NUMERIC_DRIFT_THRESHOLD,
    CATEGORICAL_DRIFT_THRESHOLD,
    OVERALL_DRIFT_THRESHOLD
)


st.title(
    "Data Drift Detection"
)


# ============================================================
# PREREQUISITE CHECK
# ============================================================

required_keys = [
    "X_train_raw",
    "X_test_raw"
]


missing_keys = [
    key
    for key in required_keys
    if key not in st.session_state
]


if missing_keys:

    st.warning(
        "Please complete preprocessing first."
    )

    st.stop()


# ============================================================
# EXPLANATION
# ============================================================

st.info(
    """
    This module compares the training dataset with the
    held-out test dataset to simulate a data-drift analysis.

    In a production environment, the current dataset would
    normally come from incoming production data rather than
    the test split.
    """
)


# ============================================================
# THRESHOLD INFORMATION
# ============================================================

with st.expander(
    "Drift thresholds"
):

    st.write(
        f"Numerical feature threshold: "
        f"{NUMERIC_DRIFT_THRESHOLD:.2f}"
    )

    st.write(
        f"Categorical feature threshold: "
        f"{CATEGORICAL_DRIFT_THRESHOLD:.2f}"
    )

    st.write(
        f"Overall drift threshold: "
        f"{OVERALL_DRIFT_THRESHOLD:.0%} "
        "of features"
    )


# ============================================================
# GENERATE REPORT
# ============================================================

if "drift_report" not in st.session_state:

    st.info(
        "No drift analysis has been generated yet."
    )

    if st.button(
        "Generate Drift Analysis"
    ):

        with st.spinner(
            "Analyzing feature distributions..."
        ):

            report = detect_data_drift(

                st.session_state[
                    "X_train_raw"
                ],

                st.session_state[
                    "X_test_raw"
                ]

            )

            st.session_state[
                "drift_report"
            ] = report

        st.rerun()


# ============================================================
# DISPLAY REPORT
# ============================================================

else:

    report = st.session_state[
        "drift_report"
    ]

    # --------------------------------------------------------
    # Overall status
    # --------------------------------------------------------

    if report["overall_drift"]:

        st.error(
            "⚠️ Overall data drift detected."
        )

    else:

        st.success(
            "✅ No significant overall data drift detected."
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Features",
        report["total_features"]
    )

    col2.metric(
        "Drifted Features",
        report["drifted_features"]
    )

    col3.metric(
        "Drift Percentage",
        f"{report['drift_percentage']:.1%}"
    )

    st.divider()

    # --------------------------------------------------------
    # Feature-level results
    # --------------------------------------------------------

    st.subheader(
        "Feature-level Drift"
    )

    results = report[
        "feature_results"
    ].copy()

    display_results = results.copy()

    display_results[
        "drift_score"
    ] = display_results[
        "drift_score"
    ].round(4)

    display_results[
        "p_value"
    ] = display_results[
        "p_value"
    ].round(4)

    display_results[
        "drifted"
    ] = display_results[
        "drifted"
    ].map(
        {
            True: "⚠️ Drifted",
            False: "✅ Stable"
        }
    )

    st.dataframe(
        display_results,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Drifted feature summary
    # --------------------------------------------------------

    drifted_features = results[
        results["drifted"]
    ]

    st.subheader(
        "Drift Summary"
    )

    if drifted_features.empty:

        st.success(
            "No individual features crossed the configured "
            "drift thresholds."
        )

    else:

        st.warning(
            f"{len(drifted_features)} feature(s) "
            "showed potential distribution drift."
        )

        st.dataframe(
            drifted_features[
                [
                    "feature",
                    "data_type",
                    "metric",
                    "drift_score"
                ]
            ],
            use_container_width=True
        )

    st.divider()

    # --------------------------------------------------------
    # Re-run
    # --------------------------------------------------------

    if st.button(
        "🔄 Re-run Drift Analysis"
    ):

        with st.spinner(
            "Recalculating drift..."
        ):

            report = detect_data_drift(

                st.session_state[
                    "X_train_raw"
                ],

                st.session_state[
                    "X_test_raw"
                ]

            )

            st.session_state[
                "drift_report"
            ] = report

        st.rerun()