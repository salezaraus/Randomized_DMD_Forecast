# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:41:47 2025

@author: Chris
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CustomTimeSeriesDataset(Dataset):
    """
    Custom dataset for the given toy time series:
    s = np.sin(np.linspace(0, 100*np.pi, 5000)) + np.random.normal(0, 0.1, 5000)
    """

    def __init__(self, seq_len=48, pred_len=24, num_samples=5000, split="train"):
        super().__init__()

        # Generate the synthetic time series
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

# Function to get DataLoader
def get_data_loader(seq_len, pred_len, batch_size=32, split="train"):
    dataset = CustomTimeSeriesDataset(seq_len=seq_len, pred_len=pred_len, split=split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))
