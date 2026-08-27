import streamlit as st

from src.preprocessing.splitter import split_dataset
from src.preprocessing.processor import preprocess_data


st.title("Data Preprocessing")


# ============================================================
# DISPLAY SUMMARY
# ============================================================

def display_preprocessing_summary():

    st.success(
        "Preprocessing completed successfully."
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Training Samples",
        st.session_state[
            "X_train_processed"
        ].shape[0]
    )

    col2.metric(
        "Testing Samples",
        st.session_state[
            "X_test_processed"
        ].shape[0]
    )


# ============================================================
# PREREQUISITE CHECK
# ============================================================

if "dataset" not in st.session_state:

    st.warning(
        "Please upload a dataset first."
    )

    st.stop()


df = st.session_state["dataset"]


# ============================================================
# DATASET PREVIEW
# ============================================================

st.subheader(
    "Dataset Preview"
)

st.dataframe(
    df.head(),
    use_container_width=True
)


# ============================================================
# TARGET COLUMN
# ============================================================

target_column = st.selectbox(
    "Select Target Column",
    df.columns
)


# ============================================================
# PREPROCESSING FUNCTION
# ============================================================

def run_preprocessing():

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

    # --------------------------------------------------------
    # Store raw data for drift detection
    # --------------------------------------------------------

    st.session_state[
        "X_train_raw"
    ] = X_train.copy()

    st.session_state[
        "X_test_raw"
    ] = X_test.copy()

    # --------------------------------------------------------
    # Store processed data
    # --------------------------------------------------------

    st.session_state[
        "X_train_processed"
    ] = X_train_processed

    st.session_state[
        "X_test_processed"
    ] = X_test_processed

    # --------------------------------------------------------
    # Store targets
    # --------------------------------------------------------

    st.session_state[
        "y_train"
    ] = y_train

    st.session_state[
        "y_test"
    ] = y_test

    # --------------------------------------------------------
    # Store preprocessor
    # --------------------------------------------------------

    st.session_state[
        "preprocessor"
    ] = preprocessor

    # --------------------------------------------------------
    # Store target column
    # --------------------------------------------------------

    st.session_state[
        "target_column"
    ] = target_column

    # --------------------------------------------------------
    # Clear downstream artifacts
    # --------------------------------------------------------

    st.session_state.pop(
        "trained_model",
        None
    )

    st.session_state.pop(
        "model_name",
        None
    )

    st.session_state.pop(
        "evaluation",
        None
    )

    st.session_state.pop(
        "evaluation_metrics",
        None
    )

    st.session_state.pop(
        "predictions",
        None
    )

    st.session_state.pop(
        "shap_values",
        None
    )

    st.session_state.pop(
        "shap_image_bytes",
        None
    )

    st.session_state.pop(
        "shap_sample_size",
        None
    )

    st.session_state.pop(
        "drift_report",
        None
    )


# ============================================================
# ALREADY PREPROCESSED
# ============================================================

if "X_train_processed" in st.session_state:

    display_preprocessing_summary()

    st.divider()

    if st.button(
        "🔄 Re-run Preprocessing"
    ):

        run_preprocessing()

        st.success(
            "Preprocessing completed again."
        )

        st.rerun()


# ============================================================
# FIRST RUN
# ============================================================

else:

    st.info(
        "Dataset has not been preprocessed yet."
    )

    if st.button(
        "Run Preprocessing"
    ):

        run_preprocessing()

        st.success(
            "Preprocessing completed successfully."
        )

        st.rerun()