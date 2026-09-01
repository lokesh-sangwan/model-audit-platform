import streamlit as st

from src.training.model_factory import get_model
from src.training.trainer import train_model
from utils.session_manager import reset_from_model


st.title(
    "Model Training"
)


# ============================================================
# PREREQUISITE CHECK
# ============================================================

if "X_train_processed" not in st.session_state:

    st.warning(
        "Please complete preprocessing first."
    )

    st.stop()


# ============================================================
# PROBLEM TYPE
# ============================================================

problem_type = st.session_state.get(
    "problem_type",
    "Classification"
)


st.info(
    f"Machine Learning Problem: **{problem_type}**"
)


# ============================================================
# CURRENT MODEL
# ============================================================

if "trained_model" in st.session_state:

    st.success(
        "Model already trained"
    )

    st.info(
        f"Current Model: "
        f"{st.session_state['model_name']}"
    )


# ============================================================
# MODEL OPTIONS
# ============================================================

st.subheader(
    "Choose Machine Learning Model"
)


if problem_type == "Classification":

    model_options = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ]

else:

    model_options = [
        "Linear Regression",
        "Decision Tree Regressor",
        "Random Forest Regressor"
    ]


model_name = st.selectbox(
    "Model",
    model_options
)


# ============================================================
# TRAIN MODEL
# ============================================================

if st.button(
    "Train Model"
):

    try:

        with st.spinner(
            f"Training {model_name}..."
        ):

            model = get_model(
                model_name
            )

            trained_model = train_model(

                model,

                st.session_state[
                    "X_train_processed"
                ],

                st.session_state[
                    "y_train"
                ]

            )


        # ----------------------------------------------------
        # Clear previous model-dependent artifacts
        # ----------------------------------------------------

        reset_from_model()


        # ----------------------------------------------------
        # Store new trained model
        # ----------------------------------------------------

        st.session_state[
            "trained_model"
        ] = trained_model


        st.session_state[
            "model_name"
        ] = model_name


        st.session_state[
            "problem_type"
        ] = problem_type


        st.success(
            f"{model_name} trained successfully."
        )


    except Exception as e:

        st.error(
            "Model training failed."
        )

        st.exception(
            e
        )