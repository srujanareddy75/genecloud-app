import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="GeneCloud", page_icon="🧬")
st.title("🧬 GeneCloud: Colon Cancer Diagnosis")
st.caption("Upload gene expression data. The cloud model predicts Tumor or Normal.")


@st.cache_resource
def load():
    b = joblib.load("model_bundle.joblib")
    return b["model"], b["genes"]


model, genes = load()

src = st.radio("Data source", ["Sample patients", "Upload my CSV"])
if src == "Sample patients":
    df = pd.read_csv("sample_patients.csv")
else:
    f = st.file_uploader("Upload CSV with the 2000 gene columns", type="csv")
    if f is None:
        st.stop()
    df = pd.read_csv(f)

missing = [g for g in genes if g not in df.columns]
if missing:
    st.error(f"{len(missing)} gene columns are missing, for example: {missing[:3]}")
    st.stop()

proba = model.predict_proba(df[genes])[:, 1]
out = pd.DataFrame({
    "Patient": range(1, len(df) + 1),
    "Prediction": ["Tumor" if p >= 0.5 else "Normal" for p in proba],
    "Tumor probability %": (proba * 100).round(1),
})
if "NSP" in df.columns:
    out["Actual"] = df["NSP"].map({1: "Tumor", 2: "Normal"})

st.subheader("Results")
st.dataframe(out)
st.bar_chart(out.set_index("Patient")["Tumor probability %"])
st.warning("Learning project, not a medical diagnosis.")
