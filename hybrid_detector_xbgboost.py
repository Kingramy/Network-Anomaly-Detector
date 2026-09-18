"""
Definitive Hybrid Anomaly Detection Pipeline:
Parallel Decision-Level Fusion (Late Fusion) combining GPU-accelerated XGBoost
with an Unsupervised PyTorch Autoencoder.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import (
    load_data, load_test_data, get_feature_columns,
    LABEL_COLUMN, ATTACK_TYPE_COLUMN
)
from autoencoder import AutoencoderDetector

DECISION_THRESHOLD_XGB = 0.30

def run_hybrid_pipeline():
    # 1. Load Data
    train_df = load_data()
    feature_columns = get_feature_columns(train_df)
    test_df = load_test_data(feature_columns)

    if test_df is None:
        raise FileNotFoundError("KDDTest+ not found in data/ directory.")

    X_train = train_df[feature_columns]
    y_train = train_df[LABEL_COLUMN]
    X_test = test_df[feature_columns]
    y_test = test_df[LABEL_COLUMN]

    print(f"Dataset: {len(train_df)} train samples, {len(test_df)} test samples, {len(feature_columns)} features.")

    # 2. Train Stage 1: XGBoost (Supervised, GPU-Accelerated)
    print("\n--- Training Stage 1: XGBoost Classifier ---")
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        tree_method="hist",
        device="cuda",
        random_state=42
    )
    xgb.fit(X_train, y_train)

    # 3. Train Stage 2: Deep Autoencoder (Unsupervised, Normal Only)
    print("\n--- Training Stage 2: Deep Autoencoder ---")
    detector = AutoencoderDetector(input_dim=len(feature_columns))
    X_train_scaled = detector.fit_scaler(X_train)
    X_test_scaled = detector.transform(X_test)

    normal_mask = (y_train == 0).values
    detector.train_model(X_train_scaled[normal_mask], epochs=30, batch_size=256, percentile_threshold=96.0)

    # 4. Independent Inferences
    print("\n--- Running Independent Inferences ---")
    xgb_probs = xgb.predict_proba(X_test)[:, 1]
    xgb_preds = (xgb_probs >= DECISION_THRESHOLD_XGB).astype(int)

    ae_errors = detector.compute_reconstruction_errors(X_test_scaled)
    ae_preds = (ae_errors > detector.threshold).astype(int)

    # 5. Late Fusion (Decision-Level OR)
    hybrid_preds = np.maximum(xgb_preds, ae_preds)

    # 6. Evaluation Reports
    print("\n" + "=" * 60)
    print("STAGE 1: XGBOOST ALONE (EXTERNAL TEST SET)")
    print("=" * 60)
    print(classification_report(y_test, xgb_preds, target_names=["normal", "anomaly"], digits=4))

    print("\n" + "=" * 60)
    print("HYBRID ENGINE: PARALLEL LATE FUSION (EXTERNAL TEST SET)")
    print("=" * 60)
    print(classification_report(y_test, hybrid_preds, target_names=["normal", "anomaly"], digits=4))

    # 7. Zero-Day Attack Analysis
    trained_attacks = set(train_df[ATTACK_TYPE_COLUMN].str.strip().str.lower().unique())
    test_analysis = test_df.copy()
    test_analysis["seen_in_train"] = test_analysis[ATTACK_TYPE_COLUMN].str.strip().str.lower().isin(trained_attacks)
    test_analysis["xgb_pred"] = xgb_preds
    test_analysis["ae_pred"] = ae_preds
    test_analysis["hybrid_pred"] = hybrid_preds

    novel = test_analysis[(test_analysis[LABEL_COLUMN] == 1) & (~test_analysis["seen_in_train"])]
    total_novel = len(novel)

    print("\n" + "=" * 60)
    print(f"BREAKDOWN ON 17 UNSEEN NOVEL ATTACKS ({total_novel} samples)")
    print("=" * 60)
    print(f"XGBoost alone caught       : {(novel['xgb_pred'] == 1).sum():>5d} / {total_novel} ({(novel['xgb_pred'] == 1).sum() / total_novel:.2%})")
    print(f"Autoencoder alone caught   : {(novel['ae_pred'] == 1).sum():>5d} / {total_novel} ({(novel['ae_pred'] == 1).sum() / total_novel:.2%})")
    print(f"Parallel Hybrid caught     : {(novel['hybrid_pred'] == 1).sum():>5d} / {total_novel} ({(novel['hybrid_pred'] == 1).sum() / total_novel:.2%})")

if __name__ == "__main__":
    run_hybrid_pipeline()