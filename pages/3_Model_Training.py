import streamlit as st


from src.training.model_factory import get_model
from src.training.trainer import train_model



st.title(
    "Model Training"
)



if "X_train_processed" not in st.session_state:


    st.warning(
        "Please complete preprocessing first."
    )



else:


    st.subheader(
        "Choose Machine Learning Model"
    )


    model_name = st.selectbox(

        "Model",

        [

            "Logistic Regression",

            "Decision Tree",

            "Random Forest"

        ]

    )



    if st.button(
        "Train Model"
    ):


        model = get_model(
            model_name
        )



        trained_model = train_model(

            model,

            st.session_state["X_train_processed"],

            st.session_state["y_train"]

        )



        st.session_state["trained_model"] = (
            trained_model
        )


        st.session_state["model_name"] = (
            model_name
        )



        st.success(

            f"{model_name} trained successfully"

        )