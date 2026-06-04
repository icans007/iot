# ============================================================
# mlflow.py — Tahap 9: Tracking Eksperimen Model Terbaik
# ============================================================
# File ini BUKAN untuk eksplorasi ulang.
# Tugasnya hanya: jalankan pipeline dengan parameter terbaik
# yang sudah ditemukan di Notebook (Tahap 7), catat ke MLflow,
# dan simpan artifact model.
# ============================================================

import os
import joblib

import mlflow
import mlflow.sklearn

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)

# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 55)
print("  MLflow Tracking — IoT Vulnerability Detection")
print("=" * 55)

FILE_NAME = "Preprocessed_Balanced_dataset.csv"

if not os.path.exists(FILE_NAME):
    import requests
    URL = (
        "https://data.mendeley.com/public-files/datasets/"
        "7m58kxs742/files/4d57de6a-a140-4607-bf38-146899f1a723/"
        "file_downloaded"
    )
    print(f"[1/4] Mengunduh dataset...")
    response = requests.get(URL)
    with open(FILE_NAME, "wb") as f:
        f.write(response.content)
    print(f"      Dataset berhasil diunduh.")
else:
    print(f"[1/4] Dataset ditemukan: {FILE_NAME}")

df = pd.read_csv(FILE_NAME)
print(f"      Shape: {df.shape}")

# ============================================================
# 2. PREPROCESSING
# ============================================================

print("[2/4] Preprocessing data...")

le = LabelEncoder()
df["Attack_sub_category"] = le.fit_transform(df["Attack_sub_category"])

X = df.drop("Attack_sub_category", axis=1)
y = df["Attack_sub_category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(f"      X_train: {X_train.shape}, X_test: {X_test.shape}")

# ============================================================
# 3. PARAMETER TERBAIK (hasil Tahap 7 di Notebook)
#    Diperoleh dari GridSearchCV:
#    Best Params: {'feature_selection__max_features': 15,
#                  'model__max_depth': 10,
#                  'model__n_estimators': 100}
#    Best CV Accuracy: 0.9909
#    Parameter diisi secara STATIS — tidak ada pencarian ulang
# ============================================================

BEST_PARAMS = {
    # Parameter Feature Selection — hasil GridSearchCV Tahap 7
    "feature_selection_method"      : "SelectFromModel (Embedded)",
    "feature_selection_max_features": 15,   # ← hasil GridSearch: max_features=15

    # Parameter Model — hasil GridSearchCV di Notebook
    "model_n_estimators"            : 100,
    "model_max_depth"               : 10,
    "model_min_samples_split"       : 2,
    "model_min_samples_leaf"        : 1,
    "model_random_state"            : 42,

    # Info tambahan
    "scaler"                        : "StandardScaler",
    "cv_strategy"                   : "StratifiedKFold(n_splits=3)",
    "tuning_method"                 : "GridSearchCV",
    "dataset"                       : "IoT Vulnerability (Preprocessed Balanced)",
}

# ============================================================
# 4. MLFLOW TRACKING
# ============================================================

print("[3/4] Memulai MLflow run...")

# Set tracking URI (simpan lokal di folder mlruns/)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("IoT_Vulnerability_Detection")

with mlflow.start_run(run_name="Pipeline_Embedded_RF_BestParams"):

    # --------------------------------------------------
    # Bangun Pipeline dengan parameter terbaik (statis)
    # Urutan: StandardScaler → SelectFromModel → RandomForest
    # --------------------------------------------------
    pipeline = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "feature_selection",
            SelectFromModel(
                estimator=RandomForestClassifier(
                    random_state=BEST_PARAMS["model_random_state"]
                ),
                max_features=BEST_PARAMS["feature_selection_max_features"]
            )
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators      = BEST_PARAMS["model_n_estimators"],
                max_depth         = BEST_PARAMS["model_max_depth"],
                min_samples_split = BEST_PARAMS["model_min_samples_split"],
                min_samples_leaf  = BEST_PARAMS["model_min_samples_leaf"],
                random_state      = BEST_PARAMS["model_random_state"]
            )
        )
    ])

    # --------------------------------------------------
    # Training
    # --------------------------------------------------
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # --------------------------------------------------
    # Evaluasi metrik
    # --------------------------------------------------
    accuracy  = accuracy_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall    = recall_score(y_test, y_pred, average="macro", zero_division=0)

    # --------------------------------------------------
    # Log parameter ke MLflow
    # --------------------------------------------------
    mlflow.log_params(BEST_PARAMS)

    # --------------------------------------------------
    # Log metrik ke MLflow
    # --------------------------------------------------
    mlflow.log_metrics({
        "accuracy"        : round(accuracy,  4),
        "f1_score_macro"  : round(f1,        4),
        "precision_macro" : round(precision, 4),
        "recall_macro"    : round(recall,    4),
    })

    # --------------------------------------------------
    # Simpan classification report sebagai artifact teks
    # --------------------------------------------------
    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_
    )
    report_path = "classification_report.txt"
    with open(report_path, "w") as f:
        f.write("Classification Report — IoT Vulnerability Detection\n")
        f.write("=" * 55 + "\n")
        f.write(report)
    mlflow.log_artifact(report_path)

    # --------------------------------------------------
    # Simpan pipeline sebagai artifact model MLflow
    # --------------------------------------------------
    mlflow.sklearn.log_model(
        sk_model      = pipeline,
        artifact_path = "pipeline_model",
        registered_model_name = "IoT_Vulnerability_Pipeline"
    )

    # --------------------------------------------------
    # Simpan juga sebagai .pkl untuk Streamlit (Tahap 10)
    # --------------------------------------------------
    joblib.dump(pipeline, "pipeline_terbaik.pkl")
    mlflow.log_artifact("pipeline_terbaik.pkl")

    # --------------------------------------------------
    # Output ringkasan ke terminal
    # --------------------------------------------------
    print()
    print("=" * 55)
    print("  HASIL EVALUASI (Test Set)")
    print("=" * 55)
    print(f"  Accuracy         : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  F1 Score (macro) : {f1:.4f}  ({f1*100:.2f}%)")
    print(f"  Precision (macro): {precision:.4f}")
    print(f"  Recall (macro)   : {recall:.4f}")
    print("=" * 55)
    print()
    print(report)
    print("=" * 55)
    print("  MLflow tracking berhasil!")
    print("  Jalankan: mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print("  Buka    : http://127.0.0.1:5000")
    print("=" * 55)

print("[4/4] Selesai.")