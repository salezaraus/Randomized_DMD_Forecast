# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 13:33:02 2025

@author: Chris
"""

"""
Complete Forecasting Pipeline
1. Trains DLinear model (if needed)
2. Performs hyperparameter search for RKDMD
3. Saves results to CSV
"""

import os
import itertools
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from models.dlinear import DLinearTrend
from data_loader import get_data_loader

# Configuration ================================================================
# Common parameters
SEQ_LEN = 150
PRED_LEN = 200
SELECTED_COLUMNS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]  # Replace with your features

# DLinear training config
TRAIN_DLINEAR = True  # Set False to skip training
DLINEAR_EPOCHS = 20
DLINEAR_PARAMS = {
    "seq_len": SEQ_LEN,
    "pred_len": PRED_LEN,
    "batch_size": 32,
    "lr": 0.001
}

# RKDMD hyperparameter search grid
RKDMD_GRID = {
    "rff_dim": [20, 50, 300, 500],
    "gamma": [1e-1, 1e-3, 1e-5, 1e-7],
    "m": [5, 10, 20]  # Takens embedding dimension
}

# Paths
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "dlinear_trend.pth")

# Helper Functions =============================================================
def train_dlinear_model():
    """Train and save DLinear model"""
    from train_dlinear import train_dlinear
    
    print("\n" + "="*40)
    print("Training DLinear Model...")
    print(f"Checkpoint will be saved to: {CHECKPOINT_PATH}")
    
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    
    train_dlinear(
        seq_len=DLINEAR_PARAMS["seq_len"],
        pred_len=DLINEAR_PARAMS["pred_len"],
        epochs=DLINEAR_EPOCHS,
        batch_size=DLINEAR_PARAMS["batch_size"],
        lr=DLINEAR_PARAMS["lr"],
        selected_columns=SELECTED_COLUMNS
    )

def evaluate_rkdmd_params(rff_dim, gamma, m):
    """Evaluate one set of RKDMD parameters"""
    from streaming_forecast import StreamingForecast
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize streaming forecaster
    streamer = StreamingForecast(
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        rff_dim=rff_dim,
        gamma=gamma,
        m=m,
        selected_columns=SELECTED_COLUMNS,
        train_dlinear=False  # We already trained this
    )
    
    # Load test data
    test_loader = get_data_loader(
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        batch_size=1,
        split="test",
        selected_columns=SELECTED_COLUMNS
    )
    
    total_mse = 0
    count = 0
    
    # Process test data stream
    for batch_x, batch_y in test_loader:
        new_point = batch_x.squeeze()[-1]
        prediction = streamer.update_and_predict(new_point)
        
        if prediction is not None:
            # Calculate MSE between prediction and ground truth
            target = batch_y.squeeze().numpy()
            mse = np.mean((prediction - target) ** 2)
            total_mse += mse
            count += 1
    
    return total_mse / count if count > 0 else np.inf

# Main Execution ===============================================================
if __name__ == "__main__":
    # Step 1: Train DLinear if needed
    if TRAIN_DLINEAR or not os.path.exists(CHECKPOINT_PATH):
        train_dlinear_model()
    else:
        print(f"Using existing DLinear model at {CHECKPOINT_PATH}")

    # Step 2: Run RKDMD hyperparameter search
    print("\n" + "="*40)
    print("Starting RKDMD Hyperparameter Search")
    
    # Generate parameter combinations
    param_combinations = [
        dict(zip(RKDMD_GRID.keys(), values))
        for values in itertools.product(*RKDMD_GRID.values())
    ]
    
    results = []
    
    # Evaluate each parameter set
    for params in tqdm(param_combinations, desc="Hyperparameter Search"):
        try:
            mse = evaluate_rkdmd_params(**params)
            results.append({
                **params,
                "mse": mse
            })
        except Exception as e:
            print(f"Failed for {params}: {str(e)}")
            results.append({
                **params,
                "mse": np.nan,
                "error": str(e)
            })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_file = "rkdmd_hyperparameter_results.csv"
    results_df.to_csv(results_file, index=False)
    
    print("\n" + "="*40)
    print(f"Search complete! Results saved to {results_file}")
    print("Top 5 configurations:")
    print(results_df.sort_values("mse").head(5))