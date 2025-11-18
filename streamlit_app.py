import streamlit as st
import pandas as pd
import numpy as np

# -----------------------------
# Synthetic patient generator
# -----------------------------
def generate_synthetic_patients(n=300, seed=42):
    rng = np.random.default_rng(seed)

    genders = ["Female", "Male", "Other"]
    diagnoses = [
        "Type 2 Diabetes",
        "Hypertension",
        "Breast Cancer",
        "COPD",
        "Obesity",
        "Hyperlipidemia",
    ]
    meds = ["Metformin", "Lisinopril", "Atorvastatin", "Albuterol", "Insulin", "None"]
    regions = ["US-East", "US-West", "EU-North", "EU-South", "APAC"]

    df = pd.DataFrame(
        {
            "PatientID": [f"P{100000+i}" for i in range(n)],
            "Age": rng.integers(18, 85, size=n),
            "Gender": rng.choice(genders, size=n, p=[0.49, 0.49, 0.02]),
            "Diagnosis": rng.choice(diagnoses, size=n),
            # Two generic biomarkers for PoC (kept generic on purpose)
            "Biomarker_A": np.round(rng.normal(loc=2.0, scale=0.7, size=n), 2),
            "Biomarker_B": np.round(rng.normal(loc=50, scale=15, size=n), 1),
            "Smoker": rng.choice([0, 1], size=n, p=[0.8, 0.2]),
            "Medications": rng.choice(meds, size=n),
            "Location": rng.choice(regions, size=n),
        }
    )

    # Add a little missingness so it feels realistic
    mask_a = rng.random(n) < 0.05
    mask_b = rng.random(n) < 0.05
    df.loc[mask_a, "Biomarker_A"] = np.nan
    df.loc[mask_b, "Biomarker_B"] = np.nan

    return df


# -----------------------------
# Streamlit app
# -----------------------------
st.set_page_config(page_title="Clinical Trial Matching PoC", page_icon="🧪", layout="wide")

st.title("🧪 Clinical Trial Patient Matching – PoC")

st.markdown(
    """
This app will eventually demonstrate **AI-assisted clinical trial patient matching**
using synthetic EHR-style data.

### Current step (1/4)
Right now we're focusing on:
1. Generating a synthetic patient dataset
2. Exploring it in the UI

Next steps will be:
- Adding protocol criteria input
- Parsing criteria (NLP-lite)
- Scoring and ranking patients by match
"""
)

# Sidebar controls
st.sidebar.header("⚙️ Patient Data Settings")
n_patients = st.sidebar.slider("Number of synthetic patients", 50, 1000, 300, step=50)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)

# Generate data
patients_df = generate_synthetic_patients(n=n_patients, seed=seed)

st.subheader("👥 Synthetic Patient Dataset")
st.caption("These records are fully synthetic and generated on the fly for PoC purposes.")

st.dataframe(patients_df, use_container_width=True, height=400)

st.markdown("**Basic summary statistics**")
st.write(patients_df.describe(include="all"))

st.success(
    "✅ Synthetic patient data is now flowing through the app.\n\n"
    "Next, we'll add protocol criteria input and matching logic on top of this dataset."
)
