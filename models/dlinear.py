# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 13:48:58 2025

@author: Chris
"""

import torch
import torch.nn as nn

class moving_avg(nn.Module):
    """Moving average block to extract trends"""
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # Padding on both ends
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """Series decomposition block (trend extraction)"""
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        trend = self.moving_avg(x)
        return trend  # We only return trend

class DLinearTrend(nn.Module):
    """DLinear model that learns trends only (Multivariate Version)"""
    def __init__(self, seq_len, pred_len, channels=1):
        super(DLinearTrend, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels

        kernel_size = 25  # Kernel size for moving average
        self.decomposition = series_decomp(kernel_size)

        # Separate Linear Layers for Each Channel
        self.Linear_Trend = nn.ModuleList([
            nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)
        ])

        # Initialize Weights
        for layer in self.Linear_Trend:
            layer.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

    def forward(self, x):
        # Decompose to extract trend
        trend = self.decomposition(x)  # [Batch, SeqLen, Channels]
        trend = trend.permute(0, 2, 1)  # Change to [Batch, Channels, SeqLen]

        # Apply Per-Channel Linear Layers
        trend_forecast = torch.stack([
            self.Linear_Trend[i](trend[:, i, :]) for i in range(self.channels)
        ], dim=1)  # [Batch, Channels, PredLen]

        trend_forecast = trend_forecast.permute(0, 2, 1)  # Back to [Batch, PredLen, Channels]
        return trend_forecast
