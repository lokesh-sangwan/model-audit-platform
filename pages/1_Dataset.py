import streamlit as st
import pandas as pd

from ui.uploader import upload_csv
from utils.validators import validate_dataset

from src.dataset.profiler import (
    get_dataset_overview,
    get_column_details
)


st.title("Dataset Upload & Validation")


uploaded_file = upload_csv()


if uploaded_file:


    try:

        df = pd.read_csv(uploaded_file)


        is_valid, message = validate_dataset(df)


        if not is_valid:

            st.error(message)


        else:

            st.success(message)


            st.session_state["dataset"] = df


            st.subheader("Dataset Preview")


            st.dataframe(
                df.head()
            )


            overview = get_dataset_overview(df)


            st.subheader("Dataset Overview")


            col1, col2, col3, col4 = st.columns(4)


            col1.metric(
                "Rows",
                overview["rows"]
            )


            col2.metric(
                "Columns",
                overview["columns"]
            )


            col3.metric(
                "Missing Values",
                overview["missing_values"]
            )


            col4.metric(
                "Duplicate Rows",
                overview["duplicate_rows"]
            )


            st.subheader(
                "Column Details"
            )


            st.dataframe(
                get_column_details(df)
            )


    except Exception:

        st.error(
            "Error processing dataset."
        )