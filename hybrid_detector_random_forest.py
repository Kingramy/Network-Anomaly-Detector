"""
Hybrid Anomaly Detection Pipeline:
Combines Supervised Random Forest (Known Attacks) with Unsupervised Autoencoder (Zero-Day Attacks).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import (
    load_data, load_test_data, get_feature_columns, get_attack_category,
    LABEL_COLUMN, ATTACK_TYPE_COLUMN
)
from autoencoder import AutoencoderDetector

DECISION_THRESHOLD_RF = 0.30

def run_hybrid_evaluation():
    train_df = load_data()
    feature_columns = get_feature_columns(train_df)
    test_df = load_test_data(feature_columns)

    print(f"Loaded {len(train_df)} training samples across {len(feature_columns)} features.")
    print(f"Loaded {len(test_df)} test samples.")

    X_train = train_df[feature_columns]
    y_train = train_df[LABEL_COLUMN]
    X_test = test_df[feature_columns]
    y_test = test_df[LABEL_COLUMN]

    print("\n--- Training Stage 1: Random Forest ---")
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    print("\n--- Training Stage 2: Deep Autoencoder (Normal Traffic Only) ---")
    detector = AutoencoderDetector(input_dim=len(feature_columns))
    
    X_train_scaled = detector.fit_scaler(X_train)
    X_test_scaled = detector.transform(X_test)

    normal_indices = (y_train == 0).values
    X_train_normal = X_train_scaled[normal_indices]

    detector.train_model(X_train_normal, epochs=30, batch_size=256, percentile_threshold=96.0)

    print("\n--- Running Stage 1 (Random Forest) Inference ---")
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    rf_preds = (rf_probs >= DECISION_THRESHOLD_RF).astype(int)

    print("\n--- Running Stage 2 (Autoencoder) Inference ---")
    ae_errors = detector.compute_reconstruction_errors(X_test_scaled)
    ae_preds = (ae_errors > detector.threshold).astype(int)

    hybrid_preds = np.maximum(rf_preds, ae_preds)

    print("\n" + "=" * 60)
    print("STAGE 1: RANDOM FOREST ALONE (EXTERNAL TEST SET)")
    print("=" * 60)
    print(classification_report(y_test, rf_preds, target_names=["normal", "anomaly"], digits=4))

    print("\n" + "=" * 60)
    print("HYBRID: RANDOM FOREST + AUTOENCODER (EXTERNAL TEST SET)")
    print("=" * 60)
    print(classification_report(y_test, hybrid_preds, target_names=["normal", "anomaly"], digits=4))

    trained_attacks = set(train_df[ATTACK_TYPE_COLUMN].str.strip().str.lower().unique())
    test_analysis = test_df.copy()
    test_analysis["seen_in_train"] = test_analysis[ATTACK_TYPE_COLUMN].str.strip().str.lower().isin(trained_attacks)
    test_analysis["rf_pred"] = rf_preds
    test_analysis["ae_pred"] = ae_preds
    test_analysis["hybrid_pred"] = hybrid_preds

    novel_attacks = test_analysis[(test_analysis[LABEL_COLUMN] == 1) & (~test_analysis["seen_in_train"])]

    rf_novel_caught = (novel_attacks["rf_pred"] == 1).sum()
    ae_novel_caught = (novel_attacks["ae_pred"] == 1).sum()
    hybrid_novel_caught = (novel_attacks["hybrid_pred"] == 1).sum()
    total_novel = len(novel_attacks)

    print("\n" + "=" * 60)
    print(f"BREAKDOWN ON THE 17 UNSEEN NOVEL ATTACKS ({total_novel} total rows)")
    print("=" * 60)
    print(f"Random Forest caught : {rf_novel_caught:>5d} / {total_novel} ({rf_novel_caught / total_novel:.2%})")
    print(f"Autoencoder caught   : {ae_novel_caught:>5d} / {total_novel} ({ae_novel_caught / total_novel:.2%})")
    print(f"Hybrid Engine caught : {hybrid_novel_caught:>5d} / {total_novel} ({hybrid_novel_caught / total_novel:.2%})")

if __name__ == "__main__":
    run_hybrid_evaluation()