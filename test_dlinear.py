# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:29:26 2025

@author: Chris
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from models.dlinear import DLinearTrend
from data_loader import get_data_loader

# Load trained model
seq_len = 48
pred_len = 24
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DLinearTrend(seq_len=seq_len, pred_len=pred_len).to(device)
model.load_state_dict(torch.load("checkpoints/dlinear_trend.pth"))
model.eval()

# Load test dataset
test_loader = get_data_loader(seq_len, pred_len, batch_size=1, split="test")

# Get a single test sample
for batch_x, batch_y in test_loader:
    batch_x, batch_y = batch_x.to(device), batch_y.numpy()  # Ground truth trend
    with torch.no_grad():
        predicted_trend = model(batch_x).cpu().numpy()
    break  # Only take the first sample for plotting

# Convert tensors to numpy arrays
input_series = batch_x.cpu().squeeze().numpy()  # Lookback window
true_trend = batch_y.squeeze()  # True future trend
predicted_trend = predicted_trend.squeeze()  # Model prediction

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(range(seq_len), input_series, label="Lookback Window", color='blue')
plt.plot(range(seq_len, seq_len + pred_len), true_trend, label="True Future Trend", color='green', linestyle='dashed')
plt.plot(range(seq_len, seq_len + pred_len), predicted_trend, label="Predicted Trend", color='red', linestyle='dotted')

plt.xlabel("Time Steps")
plt.ylabel("Value")
plt.title("DLinear Trend Prediction on Custom Time Series")
plt.legend()
plt.grid(True)
plt.show()