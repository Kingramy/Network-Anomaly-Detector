"""
Deep Autoencoder for unsupervised network anomaly detection.
Trained strictly on normal network traffic to detect zero-day anomalies.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class TabularAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 12):
        super(TabularAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, latent_dim),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, input_dim),
            nn.Sigmoid()  
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


class AutoencoderDetector:
    def __init__(self, input_dim: int, latent_dim: int = 12, lr: float = 1e-3, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TabularAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(self.device)
        self.scaler = MinMaxScaler()
        self.threshold = None
        self.lr = lr

    def fit_scaler(self, X: pd.DataFrame) -> np.ndarray:
        return self.scaler.fit_transform(X)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(X)

    def train_model(
        self,
        X_train_normal: np.ndarray,
        epochs: int = 40,
        batch_size: int = 256,
        val_split: float = 0.1,
        percentile_threshold: float = 98.0
    ):
        """
        Trains strictly on normal instances. The anomaly threshold is derived from
        the upper percentile of reconstruction errors on held-out normal validation data.
        """
        n_val = int(len(X_train_normal) * val_split)
        val_data = torch.tensor(X_train_normal[:n_val], dtype=torch.float32)
        train_data = torch.tensor(X_train_normal[n_val:], dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(train_data, train_data),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        print(f"Training Autoencoder on {len(train_data)} normal records (Device: {self.device})...")

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            for batch_x, _ in train_loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                reconstructed = self.model(batch_x)
                loss = criterion(reconstructed, batch_x)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch_x.size(0)

            epoch_loss = running_loss / len(train_data)

            self.model.eval()
            with torch.no_grad():
                val_x = val_data.to(self.device)
                val_rec = self.model(val_x)
                val_loss = criterion(val_rec, val_x).item()

            scheduler.step(val_loss)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1:02d}/{epochs:02d}] - Train Loss: {epoch_loss:.6f} | Val Loss: {val_loss:.6f}")

        self.model.eval()
        with torch.no_grad():
            val_x = val_data.to(self.device)
            val_rec = self.model(val_x)
            val_mse = torch.mean((val_x - val_rec) ** 2, dim=1).cpu().numpy()

        self.threshold = float(np.percentile(val_mse, percentile_threshold))
        print(f"\nCalibrated Reconstruction Anomaly Threshold (at {percentile_threshold}%): {self.threshold:.6f}")

    def compute_reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        tensor_x = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            reconstructed = self.model(tensor_x)
            mse = torch.mean((tensor_x - reconstructed) ** 2, dim=1).cpu().numpy()
        return mse

    def predict(self, X: np.ndarray) -> np.ndarray:
        errors = self.compute_reconstruction_errors(X)
        return (errors > self.threshold).astype(int)