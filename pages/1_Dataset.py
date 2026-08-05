import streamlit as st
import pandas as pd

from ui.uploader import upload_csv

from utils.validators import validate_dataset
from utils.session_manager import (
    reset_pipeline,
    clear_dataset
)

from src.dataset.profiler import (
    get_dataset_overview,
    get_column_details
)

from src.storage.file_manager import (
    save_uploaded_dataset
)


st.title("Dataset Upload & Validation")


uploaded_file = upload_csv()


if uploaded_file is not None:

    current_name = st.session_state.get("dataset_name")

    if current_name != uploaded_file.name:

        try:

            df = pd.read_csv(uploaded_file)

            valid, message = validate_dataset(df)

            if valid:

                reset_pipeline()

                st.session_state["dataset"] = df
                st.session_state["dataset_name"] = uploaded_file.name

                save_uploaded_dataset(
                    df,
                    uploaded_file.name
                )

                st.rerun()

            else:

                st.error(message)

        except Exception as e:

            st.error(str(e))


if "dataset" not in st.session_state:

    st.info("Upload a dataset to begin.")

    st.stop()


df = st.session_state["dataset"]

overview = get_dataset_overview(df)

st.success(
    f"Current Dataset: {st.session_state['dataset_name']}"
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Rows", overview["rows"])
c2.metric("Columns", overview["columns"])
c3.metric("Missing", overview["missing_values"])
c4.metric("Duplicates", overview["duplicate_rows"])

st.subheader("Preview")

st.dataframe(
    df.head(),
    use_container_width=True
)

st.subheader("Column Details")

st.dataframe(
    get_column_details(df),
    use_container_width=True
)

st.divider()

if st.button("🗑 Remove Dataset"):

    clear_dataset()

    st.rerun()