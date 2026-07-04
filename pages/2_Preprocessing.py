import streamlit as st

from src.preprocessing.splitter import split_dataset
from src.preprocessing.processor import preprocess_data


st.title(
    "Data Preprocessing"
)


if "dataset" not in st.session_state:

    st.warning(
        "Please upload a dataset first."
    )


else:

    df = st.session_state["dataset"]


    st.subheader(
        "Dataset Preview"
    )


    st.dataframe(
        df.head()
    )


    target_column = st.selectbox(
        "Select Target Column",
        df.columns
    )


    if st.button(
        "Run Preprocessing"
    ):


        try:


            (
                X_train,
                X_test,
                y_train,
                y_test
            ) = split_dataset(
                df,
                target_column
            )


            (
                X_train_processed,
                X_test_processed,
                preprocessor
            ) = preprocess_data(
                X_train,
                X_test
            )


            st.session_state["X_train_processed"] = (
                X_train_processed
            )


            st.session_state["X_test_processed"] = (
                X_test_processed
            )


            st.session_state["y_train"] = y_train


            st.session_state["y_test"] = y_test


            st.session_state["preprocessor"] = (
                preprocessor
            )


            st.success(
                "Preprocessing completed successfully"
            )


            col1, col2 = st.columns(2)


            col1.metric(
                "Training Samples",
                X_train_processed.shape[0]
            )


            col2.metric(
                "Testing Samples",
                X_test_processed.shape[0]
            )


        except Exception as error:


            st.error(
                f"Preprocessing failed: {error}"
            )