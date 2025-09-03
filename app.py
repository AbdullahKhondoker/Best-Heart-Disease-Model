import pickle
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Heart Disease Classifier", layout="wide")

# ---------- Load artifacts ----------
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

# Build a mapping of training means (raw space) for safe imputation
# StandardScaler.mean_ aligns with feature_order
train_means = {feat: scaler.mean_[i] for i, feat in enumerate(feature_order)}

# The 8 user-facing features
visible_features = [
    "Age",
    "Gender",
    "Heart_rate",
    "Systolic_blood_pressure",
    "Diastolic_blood_pressure",
    "Blood_sugar",
    "CK-MB",
    "Troponin",
]

# Validation rules (min, max)
validation_rules = {
    "Age": (0, 120),
    "Gender": (0, 1),  # 0 = Female, 1 = Male
    "Heart_rate": (30, 220),
    "Systolic_blood_pressure": (70, 250),
    "Diastolic_blood_pressure": (40, 150),
    "Blood_sugar": (0, 30),
    "CK-MB": (0, 100),
    "Troponin": (0, 50),
}

# ---------- Preprocess ----------
def align_and_impute(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all training columns exist, coerce numeric, and fill missing values with training means.
    Returns raw (unscaled) DataFrame with columns in feature_order.
    """
    # Ensure all training columns exist
    for col in feature_order:
        if col not in df_raw.columns:
            df_raw[col] = np.nan

    # Reindex to exact training order
    X = df_raw.reindex(columns=feature_order)

    # Coerce to numeric and fill NaNs with training means (raw space)
    X = X.apply(pd.to_numeric, errors="coerce")
    for col in feature_order:
        X[col] = X[col].fillna(train_means.get(col, 0.0))

    return X

def preprocess_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    X = align_and_impute(df_raw)
    X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_order, index=df_raw.index)
    X_rfe = X_scaled_df[selected_features]
    return X_rfe

def predict_df(df_raw: pd.DataFrame):
    X_rfe = preprocess_df(df_raw)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_rfe)
        # Find positive-class index robustly
        if hasattr(model, "classes_") and 1 in list(model.classes_):
            pos_idx = list(model.classes_).index(1)
        else:
            pos_idx = proba.shape[1] - 1  # fallback
        pos_proba = proba[:, pos_idx]
        y_pred = (pos_proba >= 0.5).astype(int)  # fixed 0.5 threshold as requested
        return y_pred, pos_proba
    else:
        y_pred = model.predict(X_rfe)
        return y_pred, None

# ---------- UI ----------
st.title("💓 CardioRisk AI :Heart Disease Risk Prediction")

with st.expander("ℹ️ Instructions", expanded=True):
    st.markdown("""
**How to use:**
- Enter the fields below (others are handled internally).
- The app applies the same preprocessing (scaling + feature selection) used in training.
- Output shows whether heart disease is detected and the confidence level.
""")

tab1, tab2 = st.tabs(["🧍 Single Prediction", "🗂️ Batch (CSV)"])

# ---- Single Prediction ----
with tab1:
    st.subheader("Enter patient details")

    cols = st.columns(3)
    single_input = {}
    valid_input = True

    # Age
    with cols[0]:
        single_input["Age"] = st.number_input("Age", min_value=0.0, max_value=120.0, value=30.0, step=1.0)

    # Gender (radio for clarity)
    with cols[1]:
        gender_label = st.radio("Gender", options=["Female (0)", "Male (1)"], index=1, horizontal=True)
        single_input["Gender"] = 1.0 if "Male" in gender_label else 0.0

    # Heart rate
    with cols[2]:
        single_input["Heart_rate"] = st.number_input("Heart_rate", min_value=30.0, max_value=220.0, value=72.0, step=1.0)

    # Systolic / Diastolic
    with cols[0]:
        single_input["Systolic_blood_pressure"] = st.number_input(
            "Systolic_blood_pressure", min_value=70.0, max_value=250.0, value=120.0, step=1.0
        )
    with cols[1]:
        single_input["Diastolic_blood_pressure"] = st.number_input(
            "Diastolic_blood_pressure", min_value=40.0, max_value=150.0, value=80.0, step=1.0
        )

    # Blood sugar
    with cols[2]:
        single_input["Blood_sugar"] = st.number_input("Blood_sugar", min_value=0.0, max_value=30.0, value=5.5, step=0.1)

    # CK-MB
    with cols[0]:
        single_input["CK-MB"] = st.number_input("CK-MB", min_value=0.0, max_value=100.0, value=1.0, step=0.1)

    # Troponin
    with cols[1]:
        single_input["Troponin"] = st.number_input("Troponin", min_value=0.0, max_value=50.0, value=0.01, step=0.01)

    # Cross-field validation: SBP >= DBP
    if single_input["Systolic_blood_pressure"] < single_input["Diastolic_blood_pressure"]:
        st.error("Systolic_blood_pressure should be greater than or equal to Diastolic_blood_pressure.")
        valid_input = False

    # Buttons
    if st.button("Predict"):
        # Range checks (redundant with number_input limits, but kept for clarity)
        for feat, (mn, mx) in validation_rules.items():
            val = single_input.get(feat)
            if val is None or val < mn or val > mx:
                st.error(f"{feat} must be between {mn} and {mx}.")
                valid_input = False

        if valid_input:
            # Add hidden/engineered columns (they’ll be imputed with training means)
            for col in feature_order:
                if col not in single_input:
                    single_input[col] = np.nan
            df_single = pd.DataFrame([single_input])

            try:
                y_pred, proba = predict_df(df_single)
                if y_pred[0] == 1:
                    st.error("🚨 **Heart Disease Detected**")
                else:
                    st.success("✅ **Heart Disease Not Detected**")
                if proba is not None:
                    st.write(f"Confidence level: **{float(proba[0]) * 100:.2f}%**")
            except Exception as e:
                st.error(str(e))
        else:
            st.warning("Please correct invalid inputs above.")

# ---- Batch Prediction ----
with tab2:
    st.subheader("Upload CSV")
    st.markdown(f"CSV must include at least: {', '.join(visible_features)}")
    file = st.file_uploader("Choose a CSV file", type=["csv"])
    if file is not None:
        try:
            df_raw = pd.read_csv(file)

            # Coerce to numeric and fill visible missing
            for col in visible_features:
                if col not in df_raw.columns:
                    df_raw[col] = np.nan

            # Align & impute ALL required training columns
            X_ready = align_and_impute(df_raw)

            # Predict
            y_pred, proba = predict_df(X_ready)

            out = df_raw.copy()
            out["prediction"] = ["Heart Disease" if p == 1 else "No Heart Disease" for p in y_pred]
            if proba is not None:
                out["confidence"] = (proba * 100).round(2)

            st.success("Predictions complete.")
            st.dataframe(out.head(20))

            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions as CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(str(e))
