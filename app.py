import pickle
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Heart Disease Classifier", layout="wide")

# Load artifacts
@st.cache_resource
def load_artifacts():
    with open("preprocessing_scaler.pkl", "rb") as f:
        arts = pickle.load(f)
    with open("catboost_best_model.pkl", "rb") as f:
        model = pickle.load(f)
    scaler = arts["scaler"]
    selected_features = arts["selected_features"]
    feature_order = arts["feature_order"]
    return scaler, selected_features, feature_order, model

scaler, selected_features, feature_order, model = load_artifacts()

# The 8 user-facing features
visible_features = [
    "Age",
    "Gender",
    "Heart_rate",
    "Systolic_blood_pressure",
    "Diastolic_blood_pressure",
    "Blood_sugar",
    "CK-MB",
    "Troponin"
]

# Validation rules: min, max for each feature
validation_rules = {
    "Age": (0, 120),
    "Gender": (0, 1),  # 0 = Female, 1 = Male
    "Heart_rate": (30, 220),
    "Systolic_blood_pressure": (70, 250),
    "Diastolic_blood_pressure": (40, 150),
    "Blood_sugar": (0, 30),
    "CK-MB": (0, 100),
    "Troponin": (0, 50)
}

def preprocess_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    # Fill engineered features as NaN if missing
    for col in feature_order:
        if col not in df_raw.columns:
            df_raw[col] = np.nan
    X = df_raw.reindex(columns=feature_order)
    X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_order, index=df_raw.index)
    X_rfe = X_scaled_df[selected_features]
    return X_rfe

def predict_df(df_raw: pd.DataFrame):
    X_rfe = preprocess_df(df_raw)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_rfe)
        pos_proba = proba[:, 1] if proba.shape[1] == 2 else np.max(proba, axis=1)
        y_pred = (pos_proba >= 0.5).astype(int)
        return y_pred, pos_proba
    else:
        y_pred = model.predict(X_rfe)
        return y_pred, None

# UI
st.title("💓 Heart Disease Classifier")

with st.expander("ℹ️ Instructions", expanded=True):
    st.markdown("""
    **How to use:**
    - Enter values for the fields below.
    - The app uses a tuned CatBoost model trained on medical data.
    - Prediction will show whether heart disease is likely.
    """)

tab1, tab2 = st.tabs(["🧍 Single Prediction", "Batch (CSV)"])

# ---- Single Prediction ----
with tab1:
    st.subheader("Enter patient details")
    cols = st.columns(3)
    single_input = {}
    valid_input = True
    for i, feat in enumerate(visible_features):
        min_val, max_val = validation_rules[feat]
        with cols[i % 3]:
            val = st.number_input(f"{feat}", min_value=float(min_val), max_value=float(max_val), value=float(min_val))
            single_input[feat] = val
            # Check range validity
            if val < min_val or val > max_val:
                st.error(f"{feat} must be between {min_val} and {max_val}")
                valid_input = False

    if st.button("Predict"):
        if valid_input:
            # Fill missing engineered features with NaN
            for col in feature_order:
                if col not in single_input:
                    single_input[col] = np.nan
            df_single = pd.DataFrame([single_input])
            try:
                y_pred, proba = predict_df(df_single)
                if y_pred[0] == 1:
                    st.error(f"🚨 Heart Disease Detected")
                else:
                    st.success(f"✅ Heart Disease Not Detected")
                if proba is not None:
                    st.write(f"Confidence level: **{float(proba[0]) * 100:.2f}%**")
            except Exception as e:
                st.error(str(e))
        else:
            st.warning("Please correct invalid inputs above.")

# ---- Batch Prediction ----
with tab2:
    st.subheader("Upload CSV")
    st.markdown(f"CSV must contain at least these columns: {', '.join(visible_features)}")
    file = st.file_uploader("Choose a CSV file", type=["csv"])
    if file is not None:
        try:
            df_raw = pd.read_csv(file)
            # Fill missing engineered features
            for col in feature_order:
                if col not in df_raw.columns:
                    df_raw[col] = np.nan
            st.write("Preview:", df_raw.head())
            y_pred, proba = predict_df(df_raw)
            out = df_raw.copy()
            out["prediction"] = ["Heart Disease" if p == 1 else "No Heart Disease" for p in y_pred]
            if proba is not None:
                out["confidence"] = (proba * 100).round(2)
            st.success("Predictions complete.")
            st.dataframe(out.head(20))
            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download predictions as CSV",
                data=csv_bytes,
                file_name="predictions.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(str(e))
