import streamlit as st


def upload_csv():
    """
    Creates a CSV upload component.

    Returns:
        Uploaded file object
    """

    uploaded_file = st.file_uploader(
        label="Upload your dataset (CSV)",
        type=["csv"],
        help="Upload a CSV dataset for model auditing"
    )

    return uploaded_file