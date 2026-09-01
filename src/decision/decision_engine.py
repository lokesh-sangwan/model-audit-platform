# ============================================================
# DECISION ENGINE
# ============================================================

DEPLOY_THRESHOLD = 0.80
MONITOR_THRESHOLD = 0.60

DRIFT_MONITOR_THRESHOLD = 0.20
DRIFT_BLOCK_THRESHOLD = 0.50


# ============================================================
# CLASSIFICATION PERFORMANCE EVALUATION
# ============================================================

def evaluate_classification_performance(metrics):
    """
    Evaluates classification model performance.

    Uses:
        Accuracy
        Precision
        Recall
        F1 Score

    Returns:
        Dictionary containing performance status,
        failed metrics, and critical metrics.
    """

    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1_score"
    ]

    failed_metrics = []
    critical_metrics = []

    for metric_name in metric_names:

        value = metrics.get(
            metric_name
        )

        if value is None:

            critical_metrics.append(
                metric_name
            )

            continue

        if value < MONITOR_THRESHOLD:

            critical_metrics.append(
                metric_name
            )

        elif value < DEPLOY_THRESHOLD:

            failed_metrics.append(
                metric_name
            )

    return {
        "failed_metrics": failed_metrics,
        "critical_metrics": critical_metrics
    }


# ============================================================
# REGRESSION PERFORMANCE EVALUATION
# ============================================================

def evaluate_regression_performance(metrics):
    """
    Evaluates regression model performance.

    R² is used as the primary deployment-readiness
    performance metric because MAE and RMSE are
    scale-dependent and cannot have universal thresholds
    across arbitrary datasets.

    Returns:
        Dictionary containing performance status,
        failed metrics, and critical metrics.
    """

    r2_score = metrics.get(
        "r2_score"
    )

    failed_metrics = []
    critical_metrics = []

    if r2_score is None:

        critical_metrics.append(
            "r2_score"
        )

    elif r2_score < MONITOR_THRESHOLD:

        critical_metrics.append(
            "r2_score"
        )

    elif r2_score < DEPLOY_THRESHOLD:

        failed_metrics.append(
            "r2_score"
        )

    return {
        "failed_metrics": failed_metrics,
        "critical_metrics": critical_metrics
    }


# ============================================================
# PERFORMANCE EVALUATION
# ============================================================

def evaluate_model_performance(
    metrics,
    problem_type="Classification"
):
    """
    Evaluates model performance according to
    the selected machine learning problem type.
    """

    if problem_type == "Regression":

        return evaluate_regression_performance(
            metrics
        )

    return evaluate_classification_performance(
        metrics
    )


# ============================================================
# MAIN DECISION FUNCTION
# ============================================================

def make_deployment_decision(
    evaluation,
    drift_report,
    explainability_available,
    problem_type="Classification"
):
    """
    Generates a deployment recommendation using:

        - Model performance
        - Data drift
        - Explainability availability

    Possible decisions:

        DEPLOY
        MONITOR
        BLOCK
    """

    # --------------------------------------------------------
    # Validate required inputs
    # --------------------------------------------------------

    if not evaluation:

        return {
            "decision": "BLOCK",
            "reasons": [
                "Model evaluation has not been completed."
            ],
            "severity": "Critical"
        }


    if not drift_report:

        return {
            "decision": "BLOCK",
            "reasons": [
                "Data drift analysis has not been completed."
            ],
            "severity": "Critical"
        }


    metrics = evaluation.get(
        "metrics",
        {}
    )


    performance = evaluate_model_performance(
        metrics,
        problem_type
    )


    failed_metrics = performance[
        "failed_metrics"
    ]

    critical_metrics = performance[
        "critical_metrics"
    ]


    # --------------------------------------------------------
    # Read drift information
    # --------------------------------------------------------

    drift_percentage = drift_report.get(
        "drift_percentage",
        0.0
    )


    # --------------------------------------------------------
    # BLOCK conditions
    # --------------------------------------------------------

    if critical_metrics:

        if problem_type == "Regression":

            threshold_description = (
                f"R² is below the critical threshold "
                f"of {MONITOR_THRESHOLD:.2f}."
            )

        else:

            threshold_description = (
                "One or more evaluation metrics are below "
                f"the critical threshold of "
                f"{MONITOR_THRESHOLD:.0%}."
            )


        reasons = [
            threshold_description
        ]


        reasons.append(
            "Critical metrics: "
            + ", ".join(
                critical_metrics
            )
        )


        return {

            "decision": "BLOCK",

            "reasons": reasons,

            "severity": "Critical"

        }


    if drift_percentage >= DRIFT_BLOCK_THRESHOLD:

        return {

            "decision": "BLOCK",

            "reasons": [

                f"Data drift affects "
                f"{drift_percentage:.1%} of usable features, "
                f"which exceeds the "
                f"{DRIFT_BLOCK_THRESHOLD:.0%} block threshold."

            ],

            "severity": "Critical"

        }


    # --------------------------------------------------------
    # MONITOR conditions
    # --------------------------------------------------------

    monitor_reasons = []


    if failed_metrics:

        if problem_type == "Regression":

            monitor_reasons.append(

                "R² is below the "
                f"{DEPLOY_THRESHOLD:.2f} deployment threshold."

            )

        else:

            monitor_reasons.append(

                "The following metrics are below the "
                f"{DEPLOY_THRESHOLD:.0%} deployment threshold: "
                + ", ".join(
                    failed_metrics
                )

            )


    if drift_percentage >= DRIFT_MONITOR_THRESHOLD:

        monitor_reasons.append(

            f"Data drift affects "
            f"{drift_percentage:.1%} of usable features."

        )


    if not explainability_available:

        monitor_reasons.append(

            "Model explainability has not been generated."

        )


    if monitor_reasons:

        return {

            "decision": "MONITOR",

            "reasons": monitor_reasons,

            "severity": "Warning"

        }


    # --------------------------------------------------------
    # DEPLOY
    # --------------------------------------------------------

    return {

        "decision": "DEPLOY",

        "reasons": [

            (
                "All required model performance criteria "
                "meet the deployment threshold."
            ),

            "Data drift is below the monitoring threshold.",

            "Model explainability is available."

        ],

        "severity": "Ready"

    }