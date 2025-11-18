import streamlit as st

st.set_page_config(page_title="Clinical Trial Matching PoC", page_icon="🧪")

st.title("🧪 Clinical Trial Patient Matching – PoC")

st.write(
    """
    This is a **minimal Streamlit app** for the clinical trial patient matching
    proof of concept.

    Next steps will be:
    1. Generate or load synthetic patient data.
    2. Enter or upload mock protocol criteria.
    3. Score and rank patients based on how well they match.
    """
)

st.success("If you can see this in Streamlit Cloud, your GitHub → Streamlit connection works!")
