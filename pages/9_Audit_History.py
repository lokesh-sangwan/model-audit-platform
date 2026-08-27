import streamlit as st

from src.history.audit_history import (
    load_audit_history
)


st.title(
    "Audit History"
)


# ============================================================
# LOAD HISTORY
# ============================================================

history = load_audit_history()


# ============================================================
# EMPTY STATE
# ============================================================

if not history:

    st.info(
        "No audit records have been saved yet."
    )

    st.stop()


# ============================================================
# SUMMARY
# ============================================================

st.metric(
    "Total Audits",
    len(history)
)


st.divider()


# ============================================================
# HISTORY TABLE
# ============================================================

history_rows = []


for record in history:

    report = record[
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

    recommendation = report[
        "deployment_recommendation"
    ]

    history_rows.append({

        "Audit ID": record[
            "audit_id"
        ],

        "Dataset": metadata[
            "dataset_name"
        ],

        "Model": metadata[
            "model_name"
        ],

        "Accuracy": round(
            performance["accuracy"],
            3
        ),

        "F1 Score": round(
            performance["f1_score"],
            3
        ),

        "Drift": (
            f"{drift['drift_percentage']:.1%}"
        ),

        "Decision": recommendation[
            "decision"
        ],

        "Generated At": metadata[
            "generated_at"
        ]

    })


st.dataframe(
    history_rows,
    use_container_width=True
)


# ============================================================
# INDIVIDUAL AUDIT DETAILS
# ============================================================

st.divider()

st.subheader(
    "Audit Details"
)


audit_options = {

    record["audit_id"]: record

    for record in history
}


selected_audit_id = st.selectbox(
    "Select an audit",
    list(
        audit_options.keys()
    )
)


selected_record = audit_options[
    selected_audit_id
]


selected_report = selected_record[
    "audit_report"
]


# ============================================================
# DISPLAY SELECTED AUDIT
# ============================================================

metadata = selected_report[
    "audit_metadata"
]

performance = selected_report[
    "model_performance"
]

drift = selected_report[
    "data_drift"
]

recommendation = selected_report[
    "deployment_recommendation"
]


st.write(
    f"### {metadata['model_name']} "
    f"on {metadata['dataset_name']}"
)

st.caption(
    f"Audit ID: {selected_audit_id}"
)

st.caption(
    f"Generated at: {metadata['generated_at']}"
)


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


st.write(
    f"**Drift:** "
    f"{drift['drift_percentage']:.1%}"
)

st.write(
    f"**Decision:** "
    f"{recommendation['decision']}"
)


st.subheader(
    "Decision Reasoning"
)

for reason in recommendation[
    "reasons"
]:

    st.write(
        f"• {reason}"
    )


with st.expander(
    "View Complete Audit"
):

    st.json(
        selected_report
    )