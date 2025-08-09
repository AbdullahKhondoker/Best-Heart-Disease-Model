import pickle
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Heart Disease Classifier", layout="wide")

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

def preprocess_df(df_raw: pd.DataFrame) -> pd.DataFrame:
  
    missing = [c for c in feature_order if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df_raw.reindex(columns=feature_order)

    X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_order, index=df_raw.index)

    X_rfe = X_scaled_df[selected_features]
    return X_rfe

def predict_df(df_raw: pd.DataFrame, threshold: float = 0.5):
    X_rfe = preprocess_df(df_raw)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_rfe)
        if proba.shape[1] == 2:

            pos_proba = proba[:, 1]
            y_pred = (pos_proba >= threshold).astype(int)
            return y_pred, pos_proba
        else:

            y_pred = np.argmax(proba, axis=1)
            return y_pred, np.max(proba, axis=1)
    else:
        y_pred = model.predict(X_rfe)
        return y_pred, None


st.title("💓 Heart Disease Classifier (CatBoost, RFE + Scaler)")

with st.expander("ℹ️ Instructions", expanded=True):
    st.markdown("""
**How to use:**
- **Single Prediction**: Enter values for each feature (same names as your training data).
- **Batch Prediction**: Upload a CSV with **exactly these columns** (in any order):  
  `""" + ", ".join(feature_order) + """`
- The app will align columns to training order, apply the **same scaler**, then **RFE selection**, and finally predict with the **tuned CatBoost** model.
""")

tab1, tab2 = st.tabs(["🧍 Single Prediction", "Batch (CSV)"])

threshold = st.sidebar.slider("Decision threshold (for class 1)", 0.0, 1.0, 0.5, 0.01)

with tab1:
    st.subheader("Enter feature values")
    cols = st.columns(3)
    single_input = {}
    for i, feat in enumerate(feature_order):
        with cols[i % 3]:

            single_input[feat] = st.number_input(f"{feat}", value=0.0, format="%.4f")
    if st.button("Predict (Single)"):
        df_single = pd.DataFrame([single_input])
        try:
            y_pred, proba = predict_df(df_single, threshold=threshold)
            st.success(f"Predicted class: **{int(y_pred[0])}**")
            if proba is not None:
                st.write(f"Positive class probability: **{float(proba[0]):.4f}**")
        except Exception as e:
            st.error(str(e))


with tab2:
    st.subheader("Upload CSV")
    file = st.file_uploader("Choose a CSV with your feature columns", type=["csv"])
    if file is not None:
        try:
            df_raw = pd.read_csv(file)
            st.write("Preview:", df_raw.head())

            y_pred, proba = predict_df(df_raw, threshold=threshold)
            out = df_raw.copy()
            out["prediction"] = y_pred
            if proba is not None:
                out["prob_positive"] = proba

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
