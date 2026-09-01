import streamlit as st

from src.drift.drift_detector import (
    detect_data_drift,
    NUMERIC_DRIFT_THRESHOLD,
    CATEGORICAL_DRIFT_THRESHOLD,
    OVERALL_DRIFT_THRESHOLD
)

from utils.session_manager import (
    reset_from_drift
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

    Identifier-like, highly-missing, and high-cardinality
    categorical features are excluded from the main drift
    calculation and reported separately.
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
        "of usable features"
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
    # Summary metrics
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)


    col1.metric(
        "Usable Features",
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


    if results.empty:

        st.warning(
            "No usable features were available for drift analysis."
        )

    else:

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

    st.subheader(
        "Drift Summary"
    )


    drifted_features = results[
        results["drifted"]
    ]


    if drifted_features.empty:

        st.success(
            "No usable features crossed the configured "
            "drift thresholds."
        )

    else:

        st.warning(
            f"{len(drifted_features)} usable feature(s) "
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


    # --------------------------------------------------------
    # Excluded feature analysis
    # --------------------------------------------------------

    excluded_features = report[
        "excluded_features"
    ]


    if not excluded_features.empty:

        st.divider()


        st.subheader(
            "Features Excluded from Drift Analysis"
        )


        st.caption(
            "These features are not included in the overall "
            "drift score because they are identifier-like, "
            "high-cardinality, or highly missing."
        )


        st.dataframe(
            excluded_features,
            use_container_width=True
        )


    # --------------------------------------------------------
    # Re-run
    # --------------------------------------------------------

    st.divider()


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


        # ----------------------------------------------------
        # Clear previous drift-dependent artifacts
        # ----------------------------------------------------

        reset_from_drift()


        # ----------------------------------------------------
        # Store new drift report
        # ----------------------------------------------------

        st.session_state[
            "drift_report"
        ] = report


        st.rerun()