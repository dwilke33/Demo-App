import streamlit as st
import pandas as pd
import numpy as np
import re
import json

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
        norm_map = {"≥": ">=", "≤": "<=", "<=": "<=", ">=": ">=", ">": ">", "<": "<", "=": "="}
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
# Simple rule-based scoring
# -----------------------------

def compare(value, op, threshold):
    """Compare a numeric value to a threshold using a simple operator."""
    if pd.isna(value):
        return None  # treat missing as unknown
    if op == ">=":
        return value >= threshold
    if op == "<=":
        return value <= threshold
    if op == ">":
        return value > threshold
    if op == "<":
        return value < threshold
    if op == "=":
        return abs(value - threshold) < 1e-6
    return None


def score_patients(df: pd.DataFrame, criteria: dict, weight_cfg=None):
    """
    Simple, transparent scoring:
    - Age match: up to 20 points
    - Diagnosis match: 25 points
    - Biomarkers: up to 35 points (shared)
    - Non-smoker when smokers are excluded: 20 points
    """
    if weight_cfg is None:
        weight_cfg = {
            "age": 20,
            "diagnosis": 25,
            "biomarkers_total": 35,   # split equally across listed biomarkers
            "exclude_smokers": 20,
        }

    scores = []
    explanations = []

    n_bio = max(1, len(criteria.get("biomarkers", [])))
    bio_each = weight_cfg["biomarkers_total"] / n_bio

    for _, row in df.iterrows():
        s = 0.0
        reasons = []

        # ---- Age ----
        age_ok = True
        if criteria.get("age_min") is not None:
            if row["Age"] >= criteria["age_min"]:
                reasons.append(f"+ age >= {criteria['age_min']}")
            else:
                reasons.append(f"- age < {criteria['age_min']}")
                age_ok = False
        if criteria.get("age_max") is not None:
            if row["Age"] <= criteria["age_max"]:
                reasons.append(f"+ age <= {criteria['age_max']}")
            else:
                reasons.append(f"- age > {criteria['age_max']}")
                age_ok = False
        if (criteria.get("age_min") is not None) or (criteria.get("age_max") is not None):
            if age_ok:
                s += weight_cfg["age"]

        # ---- Diagnosis ----
        diag_req = criteria.get("diagnosis_required")
        if diag_req:
            if diag_req.lower() in str(row["Diagnosis"]).lower():
                s += weight_cfg["diagnosis"]
                reasons.append(f"+ diagnosis matches ({diag_req})")
            else:
                reasons.append(f"- diagnosis mismatch (needs {diag_req})")

        # ---- Biomarkers ----
        for bio in criteria.get("biomarkers", []):
            col = f"Biomarker_{bio['name']}"
            val = row.get(col, np.nan)
            comp = compare(val, bio["op"], bio["value"])
            if comp is None:
                reasons.append(f"~ {col} missing")
            elif comp:
                s += bio_each
                reasons.append(f"+ {col} {bio['op']} {bio['value']}")
            else:
                reasons.append(f"- {col} not {bio['op']} {bio['value']} (val={val})")

        # ---- Exclude smokers ----
        if criteria.get("exclude_smokers", False):
            if row["Smoker"] == 0:
                s += weight_cfg["exclude_smokers"]
                reasons.append("+ non-smoker")
            else:
                reasons.append("- smoker (excluded)")

        # Clamp to [0, 100]
        s = float(np.clip(s, 0, 100))
        scores.append(s)
        explanations.append(reasons)

        # You could treat very low scores as effectively "not eligible",
        # but for PoC we'll just show the full range 0–100.

    out = df.copy()
    out["MatchScore"] = np.round(scores, 1)
    out["Explanation"] = [json.dumps(r, ensure_ascii=False) for r in explanations]
    out.sort_values("MatchScore", ascending=False, inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


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
This app demonstrates a **simple AI-assisted clinical trial patient matching PoC**
using synthetic EHR-style data.

**What it does now**
1. Generates a synthetic patient dataset
2. Accepts mock protocol criteria as free text (or uploaded .txt file)
3. Parses the protocol into structured criteria (NLP-lite)
4. Applies a transparent, rule-based scoring system to rank patients by match
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
st.dataframe(patients_df, use_container_width=True, height=350)

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
        "The app parses this into structured criteria and uses it to score patients."
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

# -----------------------------
# Score and rank patients
# -----------------------------
st.markdown("---")
st.subheader("🧮 Ranked Patient Matches")

# Let user choose minimum score threshold
min_score = st.slider(
    "Minimum MatchScore to display",
    min_value=0,
    max_value=100,
    value=60,
    step=5,
    help="Filter out patients whose match score is below this threshold."
)

scored_df = score_patients(patients_df, criteria)

# Apply threshold
view = scored_df[scored_df["MatchScore"] >= min_score].copy()

# Filters
c1, c2, c3 = st.columns(3)
with c1:
    gender_filter = st.multiselect(
        "Filter by Gender",
        options=sorted(scored_df["Gender"].dropna().unique().tolist())
    )
with c2:
    diag_filter = st.multiselect(
        "Filter by Diagnosis",
        options=sorted(scored_df["Diagnosis"].dropna().unique().tolist())
    )
with c3:
    loc_filter = st.multiselect(
        "Filter by Location",
        options=sorted(scored_df["Location"].dropna().unique().tolist())
    )

if gender_filter:
    view = view[view["Gender"].isin(gender_filter)]
if diag_filter:
    view = view[view["Diagnosis"].isin(diag_filter)]
if loc_filter:
    view = view[view["Location"].isin(loc_filter)]

st.caption("Patients sorted by MatchScore (highest first).")
st.dataframe(view, use_container_width=True, height=350)

# Simple score distribution chart (all patients)
st.markdown("**MatchScore distribution (all patients)**")
st.bar_chart(scored_df["MatchScore"].value_counts().sort_index())

# Download scored results
csv_data = scored_df.to_csv(index=False)
st.download_button(
    "⬇️ Download all scored results (CSV)",
    data=csv_data,
    file_name="scored_patient_matches.csv",
    mime="text/csv",
)

st.info(
    "✅ You now have an end-to-end PoC: synthetic patients → protocol text → parsed criteria → "
    "rule-based AI scoring → ranked patient list.\n\n"
    "You can adjust the protocol text and immediately see how eligibility and scores change."
)
