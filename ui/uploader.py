import streamlit as st


def upload_csv():

    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0

    return st.file_uploader(
        "Upload your dataset (CSV)",
        type=["csv"],
        help="Upload a CSV dataset for model auditing",
        key=f"dataset_uploader_{st.session_state['uploader_key']}"
    )