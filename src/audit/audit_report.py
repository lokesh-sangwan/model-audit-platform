from datetime import datetime


def generate_audit_report(
    dataset_name,
    model_name,
    evaluation,
    drift_report,
    deployment_decision,
    explainability_available
):
    """
    Builds a structured audit report from all completed
    model-audit stages.

    Args:
        dataset_name: Name of the evaluated dataset.
        model_name: Name of the trained model.
        evaluation: Evaluation results.
        drift_report: Data drift results.
        deployment_decision: Final deployment recommendation.
        explainability_available: Whether SHAP explanation
            was successfully generated.

    Returns:
        Dictionary containing the complete audit record.
    """

    metrics = evaluation.get(
        "metrics",
        {}
    )

    report = {

        "audit_metadata": {

            "dataset_name": dataset_name,

            "model_name": model_name,

            "generated_at": datetime.now().isoformat(
                timespec="seconds"
            )

        },

        "model_performance": {

            "accuracy": metrics.get(
                "accuracy"
            ),

            "precision": metrics.get(
                "precision"
            ),

            "recall": metrics.get(
                "recall"
            ),

            "f1_score": metrics.get(
                "f1_score"
            )

        },

        "data_drift": {

            "total_features": drift_report.get(
                "total_features",
                0
            ),

            "drifted_features": drift_report.get(
                "drifted_features",
                0
            ),

            "drift_percentage": drift_report.get(
                "drift_percentage",
                0.0
            ),

            "overall_drift": drift_report.get(
                "overall_drift",
                False
            )

        },

        "explainability": {

            "available": explainability_available

        },

        "deployment_recommendation": {

            "decision": deployment_decision.get(
                "decision"
            ),

            "severity": deployment_decision.get(
                "severity"
            ),

            "reasons": deployment_decision.get(
                "reasons",
                []
            )

        }

    }

    return report