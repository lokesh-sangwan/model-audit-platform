import streamlit as st

from src.explainability.shap_explainer import (
    generate_shap_summary
)


st.title("Model Explainability")


# =====================================================
# CLEAN UP OLD SHAP STATE
# =====================================================

# Previous versions of the application stored a
# Matplotlib Figure in "shap_image". That object is
# no longer used.

st.session_state.pop("shap_image", None)


# =====================================================
# PREREQUISITE CHECK
# =====================================================

required_keys = [
    "trained_model",
    "X_train_processed",
    "X_test_processed",
    "preprocessor"
]

missing_keys = [
    key
    for key in required_keys
    if key not in st.session_state
]


if missing_keys:

    st.warning(
        "Please complete preprocessing and model training first."
    )

    st.stop()


# =====================================================
# EXISTING EXPLANATION
# =====================================================

if (
    "shap_values" in st.session_state
    and "shap_image_bytes" in st.session_state
):

    st.success(
        "SHAP explanation generated successfully."
    )

    st.image(
        st.session_state["shap_image_bytes"],
        use_container_width=True
    )

    st.divider()

    if st.button("🔄 Re-generate SHAP"):

        with st.spinner(
            "Generating SHAP explanation..."
        ):

            shap_values, image_bytes = (
                generate_shap_summary(

                    st.session_state["trained_model"],

                    st.session_state["X_train_processed"],

                    st.session_state["X_test_processed"],

                    st.session_state["preprocessor"]

                )
            )

            st.session_state["shap_values"] = (
                shap_values
            )

            st.session_state["shap_image_bytes"] = (
                image_bytes
            )

        st.rerun()


# =====================================================
# NO EXISTING EXPLANATION
# =====================================================

else:

    # Remove any incomplete/stale SHAP state.
    st.session_state.pop(
        "shap_values",
        None
    )

    st.session_state.pop(
        "shap_image_bytes",
        None
    )

    st.info(
        "No explainability report has been generated yet."
    )

    if st.button(
        "Generate SHAP Explanation"
    ):

        with st.spinner(
            "Generating SHAP explanation..."
        ):

            shap_values, image_bytes = (
                generate_shap_summary(

                    st.session_state["trained_model"],

                    st.session_state["X_train_processed"],

                    st.session_state["X_test_processed"],

                    st.session_state["preprocessor"]

                )
            )

            st.session_state["shap_values"] = (
                shap_values
            )

            st.session_state["shap_image_bytes"] = (
                image_bytes
            )

        st.rerun()