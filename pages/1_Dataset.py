import streamlit as st
import pandas as pd

from ui.uploader import upload_csv
from utils.validators import validate_dataset

from src.dataset.profiler import (
    get_dataset_overview,
    get_column_details
)

from src.storage.file_manager import (
    save_uploaded_dataset
)


st.title(
    "Dataset Upload & Validation"
)


# Function to display dataset information
def display_dataset_info(df):

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        df.head()
    )


    overview = get_dataset_overview(
        df
    )


    st.subheader(
        "Dataset Overview"
    )


    col1, col2, col3, col4 = st.columns(
        4
    )


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



# If dataset already exists in session
if "dataset" in st.session_state:


    st.success(
        "Dataset already loaded"
    )


    if "dataset_name" in st.session_state:

        st.info(
            f"Current dataset: {st.session_state['dataset_name']}"
        )


    df = st.session_state["dataset"]


    display_dataset_info(
        df
    )



# First time upload
else:


    uploaded_file = upload_csv()


    if uploaded_file:


        try:


            df = pd.read_csv(
                uploaded_file
            )


            is_valid, message = validate_dataset(
                df
            )


            if not is_valid:


                st.error(
                    message
                )


            else:


                st.success(
                    message
                )


                # Store dataframe
                st.session_state["dataset"] = df


                # Store filename
                st.session_state["dataset_name"] = (
                    uploaded_file.name
                )


                # Save physically
                saved_path = save_uploaded_dataset(
                    df,
                    uploaded_file.name
                )


                st.info(
                    f"Dataset saved: {saved_path}"
                )


                display_dataset_info(
                    df
                )


        except Exception as error:


            st.error(
                f"Error processing dataset: {error}"
            )