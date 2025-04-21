# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:27:00 2025

@author: Chris
"""

import torch
import torch.nn as nn
import torch.optim as optim
from models.dlinear import DLinearTrend
from data_loader import get_data_loader

# Training function
def train_dlinear(seq_len=48, pred_len=48, epochs=10, batch_size=32, lr=0.001, selected_columns=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ Load dataset with selected columns
    train_loader = get_data_loader(seq_len, pred_len, batch_size=batch_size, split="train", selected_columns=selected_columns)
    val_loader = get_data_loader(seq_len, pred_len, batch_size=batch_size, split="val", selected_columns=selected_columns)

    # ✅ Get number of channels from dataset
    num_channels = len(selected_columns) if selected_columns else train_loader.dataset.num_features
    print(f"Training with {num_channels} features.")

    # ✅ Initialize model with multiple channels
    model = DLinearTrend(seq_len, pred_len, channels=num_channels).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        total_loss = 0
        model.train()

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # Shape: [Batch, SeqLen, Channels]

            optimizer.zero_grad()
            pred_trend = model(batch_x)  # Shape: [Batch, PredLen, Channels]
            loss = criterion(pred_trend, batch_y)  # Compute MSE loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred_trend = model(batch_x)
                val_loss += criterion(pred_trend, batch_y).item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "checkpoints/dlinear_trend.pth")

    print("Training complete! Best Validation Loss:", best_val_loss)

# ✅ Run training with selected columns
# =============================================================================
# if __name__ == "__main__":
#     selected_columns = ["0", "1", "3"]  # Example: selecting 3 features
#     train_dlinear(selected_columns=selected_columns)
# =============================================================================
