import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from data_loader import load_data, load_test_data, get_feature_columns, LABEL_COLUMN, ATTACK_TYPE_COLUMN

# ---------------------------------------------------------
# 1. Robust Denoising Autoencoder
# ---------------------------------------------------------
class RobustAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(RobustAutoencoder, self).__init__()
        # Injecting Dropout to prevent memorization and force structural learning
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2), 
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def get_reconstruction_error(model, data_tensor, device="cuda"):
    model.eval() # Turns off dropout during actual inference
    with torch.no_grad():
        x = data_tensor.to(device)
        reconstructed = model(x)
        # Calculate raw Mean Squared Error per row
        row_mse = torch.mean(torch.pow(x - reconstructed, 2), dim=1)
    return row_mse.cpu().numpy()

# ---------------------------------------------------------
# 2. Master Pipeline
# ---------------------------------------------------------
def run_robust_pipeline():
    print("--- 1. Loading and Scaling Data ---")
    train_df = load_data()
    feature_columns = get_feature_columns(train_df)
    test_df = load_test_data(feature_columns)
    
    # We cast to float32 to satisfy XGBoost's CUDA requirements and remove warnings
    X_train = train_df[feature_columns].values.astype(np.float32)
    y_train = train_df[LABEL_COLUMN].values
    X_test = test_df[feature_columns].values.astype(np.float32)
    y_test = test_df[LABEL_COLUMN].values
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n--- 2. Training Stage 1: XGBoost (Gatekeeper) ---")
    xgb = XGBClassifier(
        n_estimators=400, 
        max_depth=6, 
        learning_rate=0.05, 
        tree_method='hist', 
        device='cuda', 
        random_state=42
    )
    xgb.fit(X_train_scaled, y_train)
    
    print("\n--- 3. Training Stage 2: Robust Autoencoder ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae_model = RobustAutoencoder(input_dim=len(feature_columns)).to(device)
    
    optimizer = optim.Adam(ae_model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    normal_mask = (y_train == 0)
    X_train_normal = torch.tensor(X_train_scaled[normal_mask], dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_train_normal, X_train_normal)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    
    ae_model.train()
    for epoch in range(30):
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            reconstructed = ae_model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
    print("Robust Autoencoder successfully trained.")

    print("\n--- 4. Calibrating Anomaly Threshold ---")
    # Calculate MSE on normal training data
    train_errors = get_reconstruction_error(ae_model, X_train_normal, device)
    
    # We set a highly aggressive threshold at the 95th percentile.
    # We are willing to sacrifice a tiny bit of precision (false alarms) to catch the novel attacks.
    threshold = np.percentile(train_errors, 95.0)
    print(f"Calibrated Reconstruction Threshold: {threshold:.6f}")

    print("\n--- 5. Executing Asymmetric Veto Fusion ---")
    # XGBoost Predictions
    xgb_probs = xgb.predict_proba(X_test_scaled)[:, 1]
    xgb_preds = (xgb_probs >= 0.30).astype(int)
    
    # Autoencoder Predictions
    test_errors = get_reconstruction_error(ae_model, torch.tensor(X_test_scaled, dtype=torch.float32), device)
    ae_preds = (test_errors > threshold).astype(int)
    
    # Fusion (OR Logic)
    hybrid_preds = np.maximum(xgb_preds, ae_preds)
    
    print("\n" + "=" * 60)
    print("HYBRID ENGINE: XGBOOST + ROBUST AUTOENCODER")
    print("=" * 60)
    print(classification_report(y_test, hybrid_preds, target_names=["normal", "anomaly"], digits=4))
    
    trained_attacks = set(train_df[ATTACK_TYPE_COLUMN].str.strip().str.lower().unique())
    novel_mask = (y_test == 1) & (~test_df[ATTACK_TYPE_COLUMN].str.strip().str.lower().isin(trained_attacks))
    total_novel = novel_mask.sum()
    
    xgb_caught = (xgb_preds[novel_mask] == 1).sum()
    ae_caught = (ae_preds[novel_mask] == 1).sum()
    hybrid_caught = (hybrid_preds[novel_mask] == 1).sum()
    
    print(f"\nBREAKDOWN ON 17 UNSEEN NOVEL ATTACKS ({total_novel} total rows)")
    print("-" * 60)
    print(f"XGBoost alone caught       : {xgb_caught:>5d} / {total_novel} ({xgb_caught/total_novel:.2%})")
    print(f"Autoencoder alone caught   : {ae_caught:>5d} / {total_novel} ({ae_caught/total_novel:.2%})")
    print(f"Hybrid Engine caught       : {hybrid_caught:>5d} / {total_novel} ({hybrid_caught/total_novel:.2%})")

if __name__ == "__main__":
    run_robust_pipeline()