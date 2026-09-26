from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "finbert_data"

TRAIN_FILE = DATA_DIR / "train.csv"

MAX_FEATURES = 10000
RANDOM_STATE = 42


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Stock News Predictor",
    page_icon="📈",
    layout="centered"
)

st.title("📈 Stock News Predictor")
st.caption("XGBoost + TF-IDF research prototype")


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model():

    df = pd.read_csv(TRAIN_FILE)

    df["title"] = df["title"].fillna("").astype(str)

    # -------------------------
    # Text ONLY
    # -------------------------

    text = df["title"]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=MAX_FEATURES,
        sublinear_tf=True,
        dtype=np.float32
    )

    X = vectorizer.fit_transform(text)

    # -------------------------
    # Labels
    # -------------------------

    labels = sorted(df["label"].unique())

    label_to_id = {
        label: i for i, label in enumerate(labels)
    }

    y = df["label"].map(label_to_id).values

    # -------------------------
    # XGBoost
    # -------------------------

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(labels),

        n_estimators=400,
        max_depth=5,
        learning_rate=0.03,

        subsample=0.85,
        colsample_bytree=0.85,

        min_child_weight=3,
        gamma=0,

        reg_alpha=0,
        reg_lambda=1,

        eval_metric="mlogloss",

        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(
        X,
        y,
        verbose=False
    )

    return model, vectorizer, labels


# ============================================================
# LOAD MODEL
# ============================================================

with st.spinner("Loading XGBoost model..."):
    model, vectorizer, labels = train_model()

st.success("Model ready")


# ============================================================
# INPUT
# ============================================================

st.subheader("Enter News")

headline = st.text_area(
    "News Headline",
    placeholder="Enter a stock-related news headline...",
    height=150
)


# ============================================================
# PREDICT
# ============================================================

if st.button("🚀 Predict", use_container_width=True):

    if not headline.strip():

        st.error("Please enter a news headline.")

    else:

        try:

            # -------------------------
            # NEWS TEXT ONLY
            # -------------------------

            X_input = vectorizer.transform(
                [headline]
            )

            # -------------------------
            # Prediction
            # -------------------------

            probabilities = model.predict_proba(
                X_input
            )[0]

            predicted_id = int(
                np.argmax(probabilities)
            )

            prediction = labels[predicted_id]
            confidence = probabilities[predicted_id]

            # -------------------------
            # RESULT
            # -------------------------

            st.divider()

            st.subheader("Prediction")

            if prediction == "Spike":

                st.success(
                    f"📈 {prediction}"
                )

            elif prediction == "Crash":

                st.error(
                    f"📉 {prediction}"
                )

            else:

                st.info(
                    f"➡️ {prediction}"
                )

            st.metric(
                "Model Confidence",
                f"{confidence * 100:.2f}%"
            )

            st.subheader("Class Probabilities")

            for label, probability in zip(
                labels,
                probabilities
            ):

                st.write(
                    f"**{label}**: "
                    f"{probability * 100:.2f}%"
                )

                st.progress(
                    float(probability)
                )

            st.caption(
                "Research prototype. Predictions are not financial advice."
            )

        except Exception as e:

            st.error(
                f"Prediction error: {e}"
            )