"""
Supervised/Boosted anomaly classifier using XGBoost, optimized for intrusion detection
with stacked Autoencoder reconstruction error features.

Run directly: `python src/classifier.py`
"""

import os
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import (
    load_data, load_test_data, get_feature_columns, get_attack_category,
    LABEL_COLUMN, ATTACK_TYPE_COLUMN,
)
from autoencoder import AutoencoderDetector

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "classifier_results.csv")
TEST_SIZE = 0.2
RANDOM_STATE = 42
DECISION_THRESHOLD = 0.30  

CATEGORY_WEIGHTS = {
    "normal": 1.0,
    "DoS": 1.0,
    "Probe": 1.0,
    "R2L": 8.0,
    "U2R": 15.0,
    "Unknown": 1.0,
}

def compute_sample_weights(attack_series: pd.Series) -> pd.Series:
    categories = attack_series.apply(get_attack_category)
    return categories.map(CATEGORY_WEIGHTS).fillna(1.0)

def train_and_evaluate(df: pd.DataFrame, feature_columns: list, use_class_weights: bool = True):
    X = df[feature_columns]
    y = df[LABEL_COLUMN]

    X_train, X_test, y_train, y_test, attack_train, attack_test = train_test_split(
        X, y, df[ATTACK_TYPE_COLUMN], test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # XGBoost setup with native GPU acceleration (GTX 1660 SUPER)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        tree_method='hist',
        device='cuda',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    if use_class_weights:
        sample_weight = compute_sample_weights(attack_train)
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)

    results = X_test.copy()
    results[LABEL_COLUMN] = y_test.values
    results[ATTACK_TYPE_COLUMN] = attack_test.values
    results["predicted"] = y_pred
    results["anomaly_probability"] = y_proba

    return model, results, y_test, y_pred

def get_category_recall(results: pd.DataFrame, category: str) -> tuple:
    results = results.copy()
    results["category"] = results[ATTACK_TYPE_COLUMN].apply(get_attack_category)
    subset = results[results["category"] == category]
    if len(subset) == 0:
        return 0, 0
    return int((subset["predicted"] == 1).sum()), len(subset)

def compare_weighted_vs_unweighted(df: pd.DataFrame, feature_columns: list, test_df):
    print("\n" + "=" * 60)
    print("Training baseline (unweighted) model for comparison...")
    print("=" * 60)
    baseline_model, _, _, _ = train_and_evaluate(df, feature_columns, use_class_weights=False)

    print("\n" + "=" * 60)
    print("Training weighted model (boosts R2L/U2R)...")
    print("=" * 60)
    weighted_model, _, _, _ = train_and_evaluate(df, feature_columns, use_class_weights=True)

    if test_df is None:
        print("\nNo external test file found — skipping comparison.")
        return weighted_model

    X_ext = test_df[feature_columns]
    
    baseline_proba = baseline_model.predict_proba(X_ext)[:, 1]
    baseline_pred = (baseline_proba >= DECISION_THRESHOLD).astype(int)
    
    weighted_proba = weighted_model.predict_proba(X_ext)[:, 1]
    weighted_pred = (weighted_proba >= DECISION_THRESHOLD).astype(int)

    baseline_results = test_df.copy()
    baseline_results["predicted"] = baseline_pred
    weighted_results = test_df.copy()
    weighted_results["predicted"] = weighted_pred

    print("\n" + "=" * 60)
    print("COMPARISON: recall on EXTERNAL test set, before vs after weighting")
    print("=" * 60)
    print(f"  {'Category':10s}  {'Baseline':>12s}  {'Weighted':>12s}  {'Change':>10s}")
    for category in ["DoS", "Probe", "R2L", "U2R"]:
        b_caught, b_total = get_category_recall(baseline_results, category)
        w_caught, w_total = get_category_recall(weighted_results, category)
        if b_total == 0:
            continue
        b_recall = b_caught / b_total
        w_recall = w_caught / w_total
        change = w_recall - b_recall
        sign = "+" if change >= 0 else ""
        print(f"  {category:10s}  {b_recall:>11.1%}  {w_recall:>11.1%}  {sign}{change:.1%}")

    return weighted_model

def print_feature_importance(model, feature_columns: list, top_n: int = 15):
    importances = sorted(zip(feature_columns, model.feature_importances_), key=lambda x: x[1], reverse=True)
    print(f"\nTop {top_n} most important features:")
    for name, score in importances[:top_n]:
        print(f"  {name:30s} {score:.4f}")

def print_attack_category_breakdown(results: pd.DataFrame, label: str = "held-out"):
    results = results.copy()
    results["category"] = results[ATTACK_TYPE_COLUMN].apply(get_attack_category)

    print(f"\n--- Recall by attack category ({label} set) ---")
    for category in ["DoS", "Probe", "R2L", "U2R", "Unknown"]:
        subset = results[results["category"] == category]
        if len(subset) == 0:
            continue
        caught = (subset["predicted"] == 1).sum()
        total = len(subset)
        recall = caught / total if total else 0
        print(f"  {category:10s}  {caught:>5d} / {total:<5d} caught  ({recall:.1%} recall)")

    missed = results[(results[LABEL_COLUMN] == 1) & (results["predicted"] == 0)]
    if len(missed) > 0:
        print(f"\n  Most-missed specific attack types ({label} set):")
        top_missed = missed[ATTACK_TYPE_COLUMN].value_counts().head(5)
        for name, count in top_missed.items():
            print(f"    {name:20s} missed {count} times")

def evaluate_on_external_test(test_df, feature_columns: list, train_df: pd.DataFrame):
    print("\n=== Retraining final XGBoost model on 100% of Training Data for External Evaluation ===")
    
    full_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        tree_method='hist',
        device='cuda',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    sample_weight = compute_sample_weights(train_df[ATTACK_TYPE_COLUMN])
    full_model.fit(train_df[feature_columns], train_df[LABEL_COLUMN], sample_weight=sample_weight)
    
    X_ext = test_df[feature_columns]
    y_ext = test_df[LABEL_COLUMN]
    
    y_proba = full_model.predict_proba(X_ext)[:, 1]
    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)

    print(f"\n=== Evaluation on EXTERNAL test set ({len(test_df)} rows, never seen during training) ===")
    print(classification_report(y_ext, y_pred, target_names=["normal", "anomaly"]))

    cm = confusion_matrix(y_ext, y_pred)
    print("Confusion matrix:")
    print("                 predicted normal   predicted anomaly")
    print(f"actual normal    {cm[0][0]:>15d}   {cm[0][1]:>17d}")
    print(f"actual anomaly   {cm[1][0]:>15d}   {cm[1][1]:>17d}")

    ext_results = test_df.copy()
    ext_results["predicted"] = y_pred
    print_attack_category_breakdown(ext_results, label="EXTERNAL test")

def main():
    df = load_data()
    feature_columns = get_feature_columns(df)
    print(f"Loaded {len(df)} rows of traffic data.")
    print(f"Using {len(feature_columns)} features.")

    test_df = load_test_data(feature_columns)

    _ = compare_weighted_vs_unweighted(df, feature_columns, test_df)

    model, results, y_test, y_pred = train_and_evaluate(df, feature_columns, use_class_weights=True)

    print(f"\n--- Detailed evaluation on held-out test set ({len(y_test)} rows), weighted model ---")
    print(classification_report(y_test, y_pred, target_names=["normal", "anomaly"]))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:")
    print("                 predicted normal   predicted anomaly")
    print(f"actual normal    {cm[0][0]:>15d}   {cm[0][1]:>17d}")
    print(f"actual anomaly   {cm[1][0]:>15d}   {cm[1][1]:>17d}")

    print_feature_importance(model, feature_columns)
    print_attack_category_breakdown(results, label="held-out")

    if test_df is not None:
        evaluate_on_external_test(test_df, feature_columns, train_df=df)
    else:
        print("\nNo external test file found in data/. Add one to validate on truly unseen data.")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    results.sort_values("anomaly_probability", ascending=False).to_csv(OUTPUT_PATH, index=False)
    print(f"\nFull results saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()