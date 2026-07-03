import streamlit as st

st.set_page_config(
    page_title="ModelAudit",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ ModelAudit")
st.subheader("ML Deployment Readiness & Risk Assessment Platform")

st.markdown("---")

st.markdown("""
### Welcome

This platform simulates an internal Machine Learning audit system used before deploying a model into production.

The audit workflow includes:

- Dataset Validation
- Data Preprocessing
- Model Training
- Model Evaluation
- Explainability
- Data Drift Detection
- Deployment Decision
- Audit History

Use the sidebar to navigate through the modules.
""")