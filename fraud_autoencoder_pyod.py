import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def preprocess(df: pd.DataFrame):
    X = df.drop("Class", axis=1)
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train.to_numpy(), y_test.to_numpy()


def train_autoencoder(X_train, epochs=50, batch_size=4, lr=1e-3):
    device = torch.device("cpu")
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = X_train.shape[1]
    model = Autoencoder(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        epoch_loss /= len(dataset)
    return model


def compute_reconstruction_errors(model, X):
    device = torch.device("cpu")
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        output = model(X_tensor)
        errors = torch.mean((output - X_tensor) ** 2, dim=1)
    return errors.cpu().numpy()


def train_and_evaluate(csv_path: str):
    print("Loading dataset...")
    df = load_data(csv_path)
    print("Dataset shape:", df.shape)
    print("Class distribution:")
    print(df["Class"].value_counts())

    X_train, X_test, y_train, y_test = preprocess(df)

    print("Training autoencoder...")
    model = train_autoencoder(X_train, epochs=50, batch_size=4, lr=1e-3)

    train_errors = compute_reconstruction_errors(model, X_train)
    threshold = np.percentile(train_errors, 95)
    print(f"Anomaly threshold (95th percentile): {threshold:.6f}")

    test_errors = compute_reconstruction_errors(model, X_test)
    y_pred = (test_errors > threshold).astype(int)

    print("\nClassification report (1 = fraud, 0 = normal):")
    print(classification_report(y_test, y_pred, digits=4))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    plt.figure(figsize=(8, 4))
    plt.hist(train_errors, bins=10, alpha=0.7)
    plt.axvline(threshold, linestyle="--")
    plt.title("Reconstruction error on training data")
    plt.xlabel("Error")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    CSV_PATH = "creditcard.csv"
    train_and_evaluate(CSV_PATH)
