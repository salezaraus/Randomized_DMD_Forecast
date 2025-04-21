# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 20:21:15 2025

@author: Chris
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from models.dlinear import DLinearTrend
from models.rkdmd import RKDMD
from data_loader import get_data_loader

class StreamingForecast:
    """
    Streaming Forecaster using DLinear for trend and RKDMD for residuals.
    """
    def __init__(self, seq_len=150, pred_len=200, rff_dim=300, gamma=1e-3, m=70, selected_columns=None, train_dlinear=False):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load dataset (only to get number of features)
        self.selected_columns = selected_columns
        self.num_features = len(selected_columns) if selected_columns else 1  # Default to 1 if not specified
        self.gamma = gamma

        # Train DLinear if required
        if train_dlinear:
            from train_dlinear import train_dlinear  # Import training function
            train_dlinear(seq_len, pred_len, epochs=10, batch_size=32, lr=0.001, selected_columns=selected_columns)

        # Load trained DLinear model
        self.dlinear = DLinearTrend(seq_len=self.seq_len, pred_len=self.pred_len, channels=self.num_features).to(self.device)
        self.dlinear.load_state_dict(torch.load("checkpoints/dlinear_trend.pth", map_location=self.device, weights_only=True))
        self.dlinear.eval()

        # Single RKDMD model that handles all features at once
        self.rkdmd = RKDMD(horizon=pred_len, w=seq_len, gamma=self.gamma, rff_dim=rff_dim, rank=rff_dim, m=m, num_features= self.num_features)

        # Store recent data
        self.lookback_window = None

    def update_and_predict(self, new_data_point):
        """
        Streaming prediction: Update lookback window with new multivariate data and make a forecast.
        """
        new_data_point = torch.tensor(new_data_point, dtype=torch.float32).to(self.device) if not isinstance(new_data_point, torch.Tensor) else new_data_point.clone().detach().to(self.device)

        if self.lookback_window is None:
            self.lookback_window = torch.zeros(self.seq_len, self.num_features).to(self.device)  # Initialize empty window

        # Shift window left and add new data point (for all features)
        self.lookback_window[:-1] = self.lookback_window[1:].clone()
        self.lookback_window[-1] = new_data_point  # Ensure shape matches `[num_features]`

        # Ensure lookback window is ready
        if torch.count_nonzero(self.lookback_window) < self.seq_len * self.num_features:
            #print("Waiting for enough data...")
            return None  # Wait until window is filled

        # Step 1: Extract historical trend using moving average
        moving_avg_layer = self.dlinear.decomposition.moving_avg
        with torch.no_grad():
            historical_trend = moving_avg_layer(self.lookback_window.unsqueeze(0)).squeeze()  # Shape: [SeqLen, Channels]

        # Step 2: Predict future trend using DLinear
        with torch.no_grad():
            future_trend = self.dlinear(self.lookback_window.unsqueeze(0)).squeeze()  # Shape: [PredLen, Channels]

        # Step 3: Compute residuals for all features
        residuals = (self.lookback_window - historical_trend).T  # Shape: [d, SeqLen] (d = num_features)
        
        #print(residuals)

        # Step 4: Fit RKDMD with all residuals at once
        self.rkdmd.fit(residuals.detach().cpu().numpy())  # Fit with matrix of shape [d, SeqLen]

        # Step 5: Predict future residuals for all features in one call
        residual_forecast = self.rkdmd.propagate(self.num_features)  # Shape: [d, PredLen] 

        # Step 6: Combine future trend + residual forecasts
        final_forecast = future_trend.detach().cpu().numpy() + residual_forecast.T  # Transpose to [PredLen, d]

        return final_forecast, residual_forecast.T, future_trend.detach().cpu().numpy() #final_forecast  # Shape: [PredLen, Channels]


# Streaming test
if __name__ == "__main__":
    # Configuration ================================================================
    SEQ_LEN = 150
    PRED_LEN = 200
    RFF_DIM = 300
    GAMMA = 1e-10
    M = 20
    TRAIN = False
    SELECTED_COLUMNS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    
    streamer = StreamingForecast(
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        rff_dim=RFF_DIM,
        gamma=GAMMA,
        m=M,
        selected_columns=SELECTED_COLUMNS,
        train_dlinear=TRAIN
    )

    test_loader = get_data_loader(
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        batch_size=1,
        split="test",
        selected_columns=SELECTED_COLUMNS
    )

    all_predictions = []
    true_trends = []
    plt.ion()

    for i, (batch_x, batch_y) in enumerate(test_loader):
        new_point = batch_x.squeeze()[-1]
        result = streamer.update_and_predict(new_point)  # Get full result
        
        if result is not None:  # Check for valid prediction
            prediction, _, _ = result  # Unpack only if not None
            all_predictions.append(prediction)
            true_trends.append(batch_y.squeeze().numpy())

            plt.clf()
            num_features = len(SELECTED_COLUMNS)
            for j in range(num_features):
                plt.subplot(num_features, 1, j + 1)
                plt.plot(batch_y.squeeze().numpy()[:, j], label="GT", color="green")
                # Ensure prediction is a 2D array before indexing
                if prediction.ndim == 2:
                    plt.plot(prediction[:, j], label="Prediction", color="red", linestyle="dashed")
                else:
                    plt.plot(prediction, label="Prediction", color="red", linestyle="dashed")
                plt.xlabel("Time")
                plt.ylabel(f"Feature {SELECTED_COLUMNS[j]}")
                plt.legend()
                plt.title(f"Streaming Forecast for Feature {SELECTED_COLUMNS[j]}")
            
            plt.tight_layout()
            plt.pause(0.01)

    plt.ioff()
    plt.show()

