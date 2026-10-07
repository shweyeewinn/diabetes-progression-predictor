import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.datasets import load_diabetes

st.set_page_config(page_title="Diabetes Progression", page_icon="🩺", layout="wide")

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "diabetes_pipeline.pkl"

# column, label, unit, step, help text  (order must match training data)
FEATURES = [
    ("age", "Age", "years", 1.0, "Patient age"),
    ("sex", "Sex", "", 1.0, "Coding inferred from the data (dataset stores sex as 1/2 without labels)"),
    ("bmi", "BMI", "kg/m²", 0.1, "Body mass index"),
    ("bp", "Blood pressure", "mmHg", 1.0, "Average blood pressure"),
    ("s1", "S1 · Total cholesterol", "mg/dL", 1.0, "Total serum cholesterol (TC)"),
    ("s2", "S2 · LDL", "mg/dL", 0.1, "Low-density lipoproteins, 'bad' cholesterol"),
    ("s3", "S3 · HDL", "mg/dL", 1.0, "High-density lipoproteins, 'good' cholesterol"),
    ("s4", "S4 · TC / HDL ratio", "ratio", 0.01, "Total cholesterol divided by HDL (TCH)"),
    ("s5", "S5 · Triglycerides (log)", "log", 0.01, "Log of serum triglycerides (LTG)"),
    ("s6", "S6 · Blood sugar", "mg/dL", 1.0, "Blood glucose (GLU)"),
]

# The dataset stores sex as 1 or 2 without documenting which is which.
# Inferred from the data: group 1 has much higher HDL (54 vs 45 mg/dL, p < 1e-16)
# and lower blood pressure - both well-known female patterns - so 1 = Female, 2 = Male.
SEX_CODES = {"Female": 1, "Male": 2}

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
html, body, .stApp, p, h1, h2, h3, h4, label, button, input {font-family: 'Nunito', sans-serif !important;}
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2.5rem; max-width: 1150px;}
.stMarkdown p.hero-title {font-size: 2.6rem !important; font-weight: 800; margin: 0; line-height: 1.2;}
.stMarkdown p.hero-sub {color: #6B6580; font-size: 1.1rem !important; margin: .3rem 0 1.6rem;}
.pill {display: inline-block; background: #EDE9FE; color: #5B4BD8; padding: 4px 12px;
       border-radius: 999px; font-size: .8rem; font-weight: 700; letter-spacing: .5px;}
.stMarkdown p.score {font-size: 3.2rem !important; font-weight: 800; color: #5B4BD8; margin: .3rem 0 0; line-height: 1.1;}
.stMarkdown p.score-sub {color: #6B6580; margin: 0 0 1rem;}
</style>
""", unsafe_allow_html=True)

# ---------- Data & model ----------
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_reference():
    X, y = load_diabetes(return_X_y=True, as_frame=True, scaled=False)
    return X, y

model = load_model()
X_ref, y_ref = load_reference()
low_cut, high_cut = y_ref.quantile([1/3, 2/3])

# ---------- Header ----------
st.markdown('<p class="hero-title">Diabetes Progression Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Enter real patient measurements to estimate disease progression one year later.</p>',
            unsafe_allow_html=True)

left, right = st.columns([1.5, 1], gap="large")

# ---------- Inputs (ranges come from the real dataset) ----------
values = {}
with left:
    st.markdown("#### Patient measurements")
    c1, c2 = st.columns(2, gap="medium")
    for i, (col, label, unit, step, help_text) in enumerate(FEATURES):
        target = c1 if i < 5 else c2
        lo, med, hi = X_ref[col].min(), X_ref[col].median(), X_ref[col].max()
        if col == "sex":
            choice = target.radio(label, list(SEX_CODES), horizontal=True, help=help_text)
            values[col] = SEX_CODES[choice]  # the model still receives 1 or 2
        else:
            text = f"{label} ({unit})" if unit not in ("", "ratio", "log") else label
            values[col] = target.slider(text, float(lo), float(hi), float(med), step, help=help_text)

# ---------- Prediction (updates live) ----------
row = pd.DataFrame([values])[[f[0] for f in FEATURES]]
score = float(model.predict(row)[0])
percentile = (y_ref < score).mean()

if score < low_cut:
    level, show = "Lower progression", st.success
elif score < high_cut:
    level, show = "Moderate progression", st.warning
else:
    level, show = "Higher progression", st.error

with right:
    with st.container(border=True):
        st.markdown('<span class="pill">PREDICTION</span>', unsafe_allow_html=True)
        st.markdown(f'<p class="score">{score:.0f}</p>'
                    f'<p class="score-sub">disease-progression score (dataset range '
                    f'{y_ref.min():.0f}–{y_ref.max():.0f})</p>', unsafe_allow_html=True)
        st.progress(min(max(percentile, 0.0), 1.0),
                    text=f"Higher than {percentile:.0%} of patients in the dataset")
        show(f"**{level}** compared with the 442 patients in the dataset")
        st.caption(f"Bands are the dataset's thirds: lower < {low_cut:.0f} ≤ moderate < {high_cut:.0f} ≤ higher.")

st.info("Educational demo trained on scikit-learn's diabetes dataset (442 patients). "
        "It is not a medical tool and should not be used for diagnosis.")
