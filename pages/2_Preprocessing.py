import streamlit as st

from src.preprocessing.splitter import split_dataset
from src.preprocessing.processor import preprocess_data
from src.validation.target_validator import validate_target
from utils.session_manager import reset_from_preprocessing


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
# PROBLEM TYPE
# ============================================================

st.subheader(
    "Machine Learning Problem Type"
)

problem_type = st.radio(
    "Select the type of prediction problem",
    [
        "Classification",
        "Regression"
    ],
    horizontal=True
)


if problem_type == "Classification":

    st.caption(
        "Use Classification when the target represents "
        "discrete classes or categories."
    )

else:

    st.caption(
        "Use Regression when the target is a continuous "
        "numerical value, such as price, salary, or demand."
    )


# ============================================================
# TARGET INFORMATION
# ============================================================

target_data = df[target_column]

target_dtype = target_data.dtype

unique_values = target_data.nunique(
    dropna=True
)


st.write(
    f"**Selected target:** `{target_column}`"
)


col1, col2 = st.columns(2)


col1.write(
    f"**Data type:** `{target_dtype}`"
)


col2.write(
    f"**Unique values:** `{unique_values}`"
)


# ============================================================
# TARGET VALIDATION
# ============================================================

validation_result = validate_target(

    df,

    target_column,

    problem_type

)


target_valid = validation_result[
    "valid"
]


# ============================================================
# DISPLAY VALIDATION RESULT
# ============================================================

if target_valid:

    st.success(
        validation_result["message"]
    )

    for warning in validation_result[
        "warnings"
    ]:

        st.warning(
            warning
        )

else:

    st.error(
        validation_result["message"]
    )


# ============================================================
# PREPROCESSING FUNCTION
# ============================================================

def run_preprocessing():

    # --------------------------------------------------------
    # Clear everything derived from the previous
    # preprocessing configuration
    # --------------------------------------------------------

    reset_from_preprocessing()


    # --------------------------------------------------------
    # Split dataset
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_dataset(
        df,
        target_column
    )


    # --------------------------------------------------------
    # Apply preprocessing
    # --------------------------------------------------------

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
    # Store problem type
    # --------------------------------------------------------

    st.session_state[
        "problem_type"
    ] = problem_type


    # --------------------------------------------------------
    # Store target validation warnings
    # --------------------------------------------------------

    st.session_state[
        "target_validation_warnings"
    ] = validation_result[
        "warnings"
    ]


# ============================================================
# ALREADY PREPROCESSED
# ============================================================

if "X_train_processed" in st.session_state:

    display_preprocessing_summary()

    st.divider()


    if st.button(
        "🔄 Re-run Preprocessing"
    ):

        if not target_valid:

            st.error(
                "Please correct the target selection "
                "before running preprocessing."
            )

            st.stop()


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

        if not target_valid:

            st.error(
                "Please correct the target selection "
                "before running preprocessing."
            )

            st.stop()


        run_preprocessing()


        st.success(
            "Preprocessing completed successfully."
        )


        st.rerun()