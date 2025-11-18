import streamlit as st
import pandas as pd
import numpy as np
import re

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
# NLP-lite: protocol parsing
# -----------------------------

# Age patterns: "Age between 40 and 70", "Age 40-70"
AGE_RANGE_RE = re.compile(
    r"age\s*(?:between|from)?\s*(\d{1,3})\s*(?:-|to|and)\s*(\d{1,3})",
    re.IGNORECASE,
)

# "Age >= 40", "Age at least 40"
AGE_MIN_RE = re.compile(
    r"age\s*(?:>=|≥|at least|minimum|min)\s*(\d{1,3})",
    re.IGNORECASE,
)

# "Age <= 70", "Age at most 70"
AGE_MAX_RE = re.compile(
    r"age\s*(?:<=|≤|at most|maximum|max)\s*(\d{1,3})",
    re.IGNORECASE,
)

# Diagnosis / condition line: "Diagnosis: Type 2 Diabetes"
DIAG_RE = re.compile(
    r"(diagnosis|condition)\s*:\s*([a-z0-9 \-\&\/]+)",
    re.IGNORECASE,
)

# Biomarker lines: "Biomarker A >= 1.8", "Biomarker B <= 60"
BIOMARKER_RE = re.compile(
    r"biomarker\s*(A|B)\s*(>=|≤|<=|≥|>|<|=)\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)

# "Exclude smokers" or "No smokers"
EXCLUDE_SMOKER_RE = re.compile(
    r"(exclude|no)\s*smokers",
    re.IGNORECASE,
)


def parse_protocol_text(txt: str):
    """
    Very simple, rule-based parser that extracts structured criteria
    from free-text protocol descriptions.
    """
    txt = (txt or "").strip()

    criteria = {
        "age_min": None,
        "age_max": None,
        "diagnosis_required": None,
        "biomarkers": [],  # list of {"name": "A"/"B", "op": ">=", "value": float}
        "exclude_smokers": False,
    }

    # Age range
    m_range = AGE_RANGE_RE.search(txt)
    if m_range:
        criteria["age_min"] = int(m_range.group(1))
        criteria["age_max"] = int(m_range.group(2))
    else:
        # Age min only
        mmin = AGE_MIN_RE.search(txt)
        if mmin:
            criteria["age_min"] = int(mmin.group(1))

        # Age max only
        mmax = AGE_MAX_RE.search(txt)
        if mmax:
            criteria["age_max"] = int(mmax.group(1))

    # Diagnosis / condition
    m_diag = DIAG_RE.search(txt)
    if m_diag:
        criteria["diagnosis_required"] = m_diag.group(2).strip()

    # Biomarkers
    for bm in BIOMARKER_RE.finditer(txt):
        name, op, val = bm.group(1).upper(), bm.group(2), float(bm.group(3))
        # Normalize ops like ≥, ≤ to >=, <=
        norm_map = {"≥": ">=", "<=": "<=", "≤": "<=", ">=": ">=", ">": ">", "<": "<", "=": "="}
        op = norm_map.get(op, op)
        criteria["biomarkers"].append(
            {
                "name": name,  # "A" or "B"
                "op": op,      # ">=", "<=", etc.
                "value": val,
            }
        )

    # Exclude smokers
    if EXCLUDE_SMOKER_RE.search(txt):
        criteria["exclude_smokers"] = True

    return criteria


# -----------------------------
# Streamlit app
# -----------------------------
st.set_page_config(
    page_title="Clinical Trial Matching PoC",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Clinical Trial Patient Matching – PoC")

st.markdown(
    """
This app will eventually demonstrate **AI-assisted clinical trial patient matching**
using synthetic EHR-style data.

We are building it step-by-step.

**Current focus**
1. Generate a synthetic patient dataset
2. Accept mock protocol criteria as text (or uploaded .txt file)
3. Parse the protocol text into structured criteria (NLP-lite)

Next steps:
- Use the parsed criteria to score and rank patients by match
"""
)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("⚙️ Patient Data Settings")
n_patients = st.sidebar.slider(
    "Number of synthetic patients",
    50,
    1000,
    300,
    step=50
)
seed = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    max_value=999999,
    value=42,
    step=1
)

st.sidebar.markdown("---")
st.sidebar.header("📄 Protocol Upload (optional)")
uploaded_protocol = st.sidebar.file_uploader(
    "Upload protocol text file (.txt)",
    type=["txt"],
    help="If provided, this will override the default protocol text."
)

# -----------------------------
# Generate patient data
# -----------------------------
patients_df = generate_synthetic_patients(n=n_patients, seed=seed)

st.subheader("👥 Synthetic Patient Dataset")
st.caption("These records are fully synthetic and generated on the fly for PoC purposes.")
st.dataframe(patients_df, use_container_width=True, height=400)

st.markdown("**Basic summary statistics**")
st.write(patients_df.describe(include="all"))

# -----------------------------
# Protocol criteria input
# -----------------------------
st.markdown("---")
st.subheader("📋 Mock Trial Protocol Criteria")

default_protocol = (
    "StudyTitle: Example Metabolic Study\n"
    "Age between 40 and 70\n"
    "Diagnosis: Type 2 Diabetes\n"
    "Biomarker A >= 1.8\n"
    "Biomarker B <= 60\n"
    "Exclude smokers"
)

protocol_text = default_protocol

# If user uploaded a .txt file, use its contents instead of the default
if uploaded_protocol is not None:
    try:
        protocol_text = uploaded_protocol.read().decode("utf-8", errors="ignore")
    except Exception:
        st.warning("Could not read uploaded protocol as UTF-8. Using default text instead.")
        protocol_text = default_protocol

protocol_text = st.text_area(
    "Enter or edit mock protocol text",
    value=protocol_text,
    height=180,
    help=(
        "This represents the free-text eligibility description from a clinical trial protocol.\n"
        "The app will parse this into structured criteria below."
    ),
)

st.markdown("**Current protocol text:**")
st.code(protocol_text, language="text")

# -----------------------------
# Parse protocol text into structured criteria
# -----------------------------
criteria = parse_protocol_text(protocol_text)

st.markdown("**Parsed structured criteria (NLP-lite):**")
st.json(criteria)

st.info(
    "✅ The protocol text is now being parsed into a structured representation.\n\n"
    "Next step will be to use these criteria to score and rank each patient based on how well they match."
)

st.info(
    "✅ You can now define trial eligibility as free text.\n\n"
    "In the next step, we'll add a parser that turns this into structured criteria "
    "and then score patients against it."
)
