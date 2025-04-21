# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:41:47 2025

@author: Chris
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class CustomTimeSeriesDataset(Dataset):
    """
    Custom dataset for the given toy time series:
    s = np.sin(np.linspace(0, 100*np.pi, 5000)) + np.random.normal(0, 0.1, 5000)
    """

    def __init__(self, seq_len=100, pred_len=100, num_samples=10000, split="train"):
        super().__init__()

        # Generate the synthetic time series
        np.random.seed(42)
        self.data = np.sin(np.linspace(0, 100 * np.pi, num_samples)) + np.random.normal(0, 0.1, num_samples)

        # Define dataset splits (70% train, 20% val, 10% test)
        train_size = int(0.7 * num_samples)
        val_size = int(0.2 * num_samples)
        test_size = num_samples - train_size - val_size

        if split == "train":
            self.start_idx = 0
            self.end_idx = train_size
        elif split == "val":
            self.start_idx = train_size
            self.end_idx = train_size + val_size
        elif split == "test":
            self.start_idx = train_size + val_size
            self.end_idx = num_samples - pred_len  # Ensure we have enough data for prediction
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return self.end_idx - self.start_idx - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        index += self.start_idx  # Adjust index for dataset split
        x = self.data[index: index + self.seq_len]  # Input sequence
        y = self.data[index + self.seq_len: index + self.seq_len + self.pred_len]  # Future trend (ground truth)

        return torch.tensor(x, dtype=torch.float32).unsqueeze(-1), torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    
    
class TimeSeriesDataset(Dataset):
    """
    Custom dataset for multivariate time series forecasting.
    Allows selecting specific columns from the dataset.
    """

    def __init__(self, seq_len=48, pred_len=48, split="train", selected_columns=None):
        super().__init__()

        # Load dataset
        df = pd.read_csv("data/exp/ETTm1.csv")

        # Drop the "date" column if present
        if "date" in df.columns:
            df = df.drop(columns=["date"])

        # Select specific columns if specified, otherwise use all
        if selected_columns:
            df = df[selected_columns]  # Select only specified columns

        self.data = df.to_numpy()  # Convert DataFrame to NumPy array
        num_samples, num_features = self.data.shape  # Shape: [TimeSteps, Channels]
        print(f"Dataset Loaded: {num_samples} samples, {num_features} selected features.")

        # Define dataset splits (70% train, 20% val, 10% test)
        train_size = int(0.7 * num_samples)
        val_size = int(0.2 * num_samples)
        test_size = num_samples - train_size - val_size

        if split == "train":
            self.start_idx = 0
            self.end_idx = train_size
        elif split == "val":
            self.start_idx = train_size
            self.end_idx = train_size + val_size
        elif split == "test":
            self.start_idx = train_size + val_size
            self.end_idx = num_samples - pred_len  # Ensure enough data for prediction
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_features = num_features  # Store feature count

    def __len__(self):
        return self.end_idx - self.start_idx - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        index += self.start_idx  # Adjust index for dataset split

        # Extract selected features
        x = self.data[index: index + self.seq_len, :]  # Shape: [SeqLen, Channels]
        y = self.data[index + self.seq_len: index + self.seq_len + self.pred_len, :]  # Shape: [PredLen, Channels]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)  # Shape: [SeqLen, Channels] and [PredLen, Channels]

# Function to get DataLoader with column selection
def get_data_loader(seq_len, pred_len, batch_size=32, split="train", selected_columns=None):
    dataset = TimeSeriesDataset(seq_len=seq_len, pred_len=pred_len, split=split, selected_columns=selected_columns)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))
