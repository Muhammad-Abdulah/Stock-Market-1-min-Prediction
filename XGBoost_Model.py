from pathlib import Path
import time
import json

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from scipy.sparse import hstack, csr_matrix
from xgboost import XGBClassifier


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "finbert_data"
OUTPUT_DIR = DATA_DIR / "xgboost_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "train.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"

RANDOM_STATE = 42

# TF-IDF settings
MAX_FEATURES = 10000
NGRAM_RANGE = (1, 2)
MIN_DF = 2
MAX_DF = 0.95

# XGBoost settings
N_ESTIMATORS = 400
MAX_DEPTH = 5
LEARNING_RATE = 0.03
SUBSAMPLE = 0.85
COLSAMPLE_BYTREE = 0.85


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 80)
print("MODEL 3 — TF-IDF + MARKET/TIME FEATURES + XGBOOST")
print("=" * 80)

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VALIDATION_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Train      : {train_df.shape}")
print(f"Validation : {val_df.shape}")
print(f"Test       : {test_df.shape}")


# ============================================================
# 3. VALIDATE COLUMNS
# ============================================================

required_columns = [
    "news_datetime",
    "ticker",
    "title",
    "reference_close",
    "label",
]

for column in required_columns:
    if column not in train_df.columns:
        raise ValueError(f"Missing required column: {column}")

print("\nRequired columns verified.")


# ============================================================
# 4. CLEAN BASIC INPUTS
# ============================================================

for df in [train_df, val_df, test_df]:

    df["title"] = df["title"].fillna("").astype(str)
    df["ticker"] = df["ticker"].fillna("UNKNOWN").astype(str)

    df["news_datetime"] = pd.to_datetime(
        df["news_datetime"],
        errors="coerce"
    )

    df["reference_close"] = pd.to_numeric(
        df["reference_close"],
        errors="coerce"
    )

    df["reference_close"] = df["reference_close"].fillna(
        train_df["reference_close"].median()
    )


# ============================================================
# 5. CREATE TIME FEATURES
# ============================================================

def create_time_features(df):

    result = pd.DataFrame(index=df.index)

    dt = df["news_datetime"]

    result["hour"] = dt.dt.hour.fillna(0)
    result["minute"] = dt.dt.minute.fillna(0)

    # Continuous representation of time of day
    result["time_of_day"] = (
        result["hour"] * 60 + result["minute"]
    )

    # Cyclical time features
    result["hour_sin"] = np.sin(
        2 * np.pi * result["time_of_day"] / 1440
    )

    result["hour_cos"] = np.cos(
        2 * np.pi * result["time_of_day"] / 1440
    )

    result["day_of_week"] = (
        dt.dt.dayofweek.fillna(0)
    )

    result["day_sin"] = np.sin(
        2 * np.pi * result["day_of_week"] / 7
    )

    result["day_cos"] = np.cos(
        2 * np.pi * result["day_of_week"] / 7
    )

    # Market-session style flags
    result["is_morning"] = (
        ((result["hour"] >= 9) & (result["hour"] < 12))
        .astype(int)
    )

    result["is_midday"] = (
        ((result["hour"] >= 12) & (result["hour"] < 15))
        .astype(int)
    )

    result["is_afternoon"] = (
        ((result["hour"] >= 15) & (result["hour"] < 20))
        .astype(int)
    )

    return result


train_time = create_time_features(train_df)
val_time = create_time_features(val_df)
test_time = create_time_features(test_df)


# ============================================================
# 6. REFERENCE CLOSE FEATURES
# ============================================================

def create_numeric_features(df):

    result = pd.DataFrame(index=df.index)

    close = df["reference_close"].astype(float)

    result["reference_close"] = close

    # Log price provides a more stable representation
    result["log_reference_close"] = np.log1p(
        np.maximum(close, 0)
    )

    return result


train_numeric = create_numeric_features(train_df)
val_numeric = create_numeric_features(val_df)
test_numeric = create_numeric_features(test_df)


# ============================================================
# 7. COMBINE NUMERICAL FEATURES
# ============================================================

train_extra = pd.concat(
    [train_numeric, train_time],
    axis=1
)

val_extra = pd.concat(
    [val_numeric, val_time],
    axis=1
)

test_extra = pd.concat(
    [test_numeric, test_time],
    axis=1
)

print("\nAdditional numerical features:")
print(list(train_extra.columns))


# ============================================================
# 8. PREPARE TEXT
# ============================================================

def prepare_text(df):

    return (
        "[TICKER] "
        + df["ticker"].str.upper()
        + " [NEWS] "
        + df["title"]
    )


train_text = prepare_text(train_df)
val_text = prepare_text(val_df)
test_text = prepare_text(test_df)


# ============================================================
# 9. TF-IDF
# ============================================================

print("\n" + "=" * 80)
print("BUILDING TF-IDF FEATURES")
print("=" * 80)

vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=NGRAM_RANGE,
    min_df=MIN_DF,
    max_df=MAX_DF,
    max_features=MAX_FEATURES,
    sublinear_tf=True,
    dtype=np.float32,
)

start = time.time()

X_train_text = vectorizer.fit_transform(train_text)

X_val_text = vectorizer.transform(val_text)
X_test_text = vectorizer.transform(test_text)

print(f"Training TF-IDF matrix : {X_train_text.shape}")
print(f"Validation TF-IDF      : {X_val_text.shape}")
print(f"Test TF-IDF            : {X_test_text.shape}")
print(f"Vocabulary size        : {len(vectorizer.vocabulary_)}")
print(f"TF-IDF time             : {time.time() - start:.2f}s")


# ============================================================
# 10. SCALE NUMERICAL FEATURES
# ============================================================

# Convert numeric features to float32 sparse matrices
X_train_extra = csr_matrix(
    train_extra.astype(np.float32).values
)

X_val_extra = csr_matrix(
    val_extra.astype(np.float32).values
)

X_test_extra = csr_matrix(
    test_extra.astype(np.float32).values
)


# ============================================================
# 11. COMBINE TEXT + NUMERICAL FEATURES
# ============================================================

X_train = hstack(
    [X_train_text, X_train_extra],
    format="csr"
)

X_val = hstack(
    [X_val_text, X_val_extra],
    format="csr"
)

X_test = hstack(
    [X_test_text, X_test_extra],
    format="csr"
)

print("\nCombined feature matrices:")
print(f"Train      : {X_train.shape}")
print(f"Validation : {X_val.shape}")
print(f"Test       : {X_test.shape}")


# ============================================================
# 12. LABEL ENCODING
# ============================================================

label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(
    train_df["label"].astype(str)
)

y_val = label_encoder.transform(
    val_df["label"].astype(str)
)

y_test = label_encoder.transform(
    test_df["label"].astype(str)
)

print("\nLabel mapping:")
for index, label in enumerate(label_encoder.classes_):
    print(f"  {index} -> {label}")


# ============================================================
# 13. TRAIN XGBOOST
# ============================================================

print("\n" + "=" * 80)
print("TRAINING XGBOOST")
print("=" * 80)

print("Training started...")

start = time.time()

model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(label_encoder.classes_),

    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    learning_rate=LEARNING_RATE,

    subsample=SUBSAMPLE,
    colsample_bytree=COLSAMPLE_BYTREE,

    min_child_weight=3,
    gamma=0,

    reg_alpha=0.0,
    reg_lambda=1.0,

    eval_metric="mlogloss",

    tree_method="hist",
    random_state=RANDOM_STATE,

    n_jobs=-1,
)

model.fit(
    X_train,
    y_train,
    eval_set=[
        (X_train, y_train),
        (X_val, y_val),
    ],
    verbose=False,
)

runtime = time.time() - start

print("Training completed.")
print(f"Runtime: {runtime:.2f} seconds")


# ============================================================
# 14. EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, X, y, dataset_name):

    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)

    balanced_acc = balanced_accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y,
        predictions,
        average="weighted",
        zero_division=0
    )

    print("\n" + dataset_name.upper())
    print("-" * 50)

    print(f"Accuracy              : {accuracy:.4f}")
    print(f"Balanced Accuracy     : {balanced_acc:.4f}")
    print(f"Macro Precision       : {precision:.4f}")
    print(f"Macro Recall          : {recall:.4f}")
    print(f"Macro F1              : {macro_f1:.4f}")
    print(f"Weighted F1           : {weighted_f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y,
            predictions,
            target_names=label_encoder.classes_,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(y, predictions)
    )

    return {
        "dataset": dataset_name,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }, predictions


# ============================================================
# 15. VALIDATION
# ============================================================

val_metrics, val_predictions = evaluate_model(
    model,
    X_val,
    y_val,
    "Validation Results"
)


# ============================================================
# 16. TEST
# ============================================================

test_metrics, test_predictions = evaluate_model(
    model,
    X_test,
    y_test,
    "Test Results"
)


# ============================================================
# 17. SAVE PREDICTIONS
# ============================================================

val_output = val_df.copy()

val_output["actual_label"] = label_encoder.inverse_transform(
    y_val
)

val_output["predicted_label"] = label_encoder.inverse_transform(
    val_predictions
)

val_output.to_csv(
    OUTPUT_DIR / "validation_predictions.csv",
    index=False
)


test_output = test_df.copy()

test_output["actual_label"] = label_encoder.inverse_transform(
    y_test
)

test_output["predicted_label"] = label_encoder.inverse_transform(
    test_predictions
)

test_output.to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False
)


# ============================================================
# 18. SAVE MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame([
    val_metrics,
    test_metrics,
])

comparison.to_csv(
    OUTPUT_DIR / "xgboost_metrics.csv",
    index=False
)


# ============================================================
# 19. SAVE CONFIGURATION
# ============================================================

config = {
    "model": "XGBoost",
    "tfidf_max_features": MAX_FEATURES,
    "ngram_range": NGRAM_RANGE,
    "min_df": MIN_DF,
    "max_df": MAX_DF,

    "n_estimators": N_ESTIMATORS,
    "max_depth": MAX_DEPTH,
    "learning_rate": LEARNING_RATE,
    "subsample": SUBSAMPLE,
    "colsample_bytree": COLSAMPLE_BYTREE,

    "input_features": [
        "title",
        "ticker",
        "reference_close",
        "news_datetime"
    ],

    "excluded_future_columns": [
        "future_datetime",
        "future_close",
        "return_pct",
        "label"
    ],

    "train_rows": len(train_df),
    "validation_rows": len(val_df),
    "test_rows": len(test_df),

    "runtime_seconds": runtime,
}

with open(
    OUTPUT_DIR / "config.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        config,
        f,
        indent=4,
        default=str
    )


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL XGBOOST SUMMARY")
print("=" * 80)

print(
    f"Validation Accuracy : "
    f"{val_metrics['accuracy']:.4f}"
)

print(
    f"Validation Macro F1 : "
    f"{val_metrics['macro_f1']:.4f}"
)

print(
    f"Test Accuracy       : "
    f"{test_metrics['accuracy']:.4f}"
)

print(
    f"Test Macro F1       : "
    f"{test_metrics['macro_f1']:.4f}"
)

print(
    f"Test Balanced Acc.  : "
    f"{test_metrics['balanced_accuracy']:.4f}"
)

print(f"\nResults saved to:")
print(OUTPUT_DIR)

print("\nDONE.")