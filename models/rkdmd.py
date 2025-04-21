# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:07:40 2025

@author: Chris
"""

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils.helper import create_hankel_matrix, featurize, RFF, forecast, construct_multivariate_hankel

import matplotlib.pyplot as plt

class RKDMD:
    """
    Placeholder class for Randomized Kernel Dynamic Mode Decomposition (RKDMD).
    This will be used for residual forecasting.
    """

    def __init__(self,  horizon=200, w = 100,
                 gamma=1e-6, rff_dim=400, rank=60, m = 50, num_features = None):
        # Configuration parameters
        self.horizon = horizon
        self.gamma = gamma
        self.rff_dim = rff_dim
        self.rank = rank
        self.window_size = w
        self.m = m
        self.d = num_features * self.m
        # Data initialization
        self.scaler = StandardScaler()
        
        # RFF initialization
        self.z, self.theta = RFF(gamma=gamma, d=self.d, s=rff_dim)
    
    
    def build_matrix(self):
        """Initialize Hankel matrices and normalization"""
        #self.scaler.fit(self.ts.reshape(-1,1))
        
        # Create initial Hankel matrix
        self.H = construct_multivariate_hankel(self.ts, self.m)
        
        self.d = np.shape(self.H)[0]
        
        # Initial state matrices
        self.X = self.H[:, :-1]  # (d, window_size-1)
        self.Y = self.H[:, 1:]   # (d, window_size-1)
        
        
                
        self.Phi_X = featurize(self.X, self.z, self.theta)
        self.Phi_Y = featurize(self.Y, self.z, self.theta)
        
        
    def fit(self, new_residual):
        """
        Incrementally update RKDMD with new residuals while keeping memory of previous state.
        """

        self.ts = new_residual
        # Rebuild Hankel matrices with updated residuals
        self.build_matrix()
    
        # Update Koopman operator
        U, S, VT = np.linalg.svd(self.Phi_X.T, full_matrices=False)
        Sigma = np.diag(S[:self.rank])
        Q = U[:, :self.rank]
    
        self.K_hat = np.linalg.multi_dot([
            np.linalg.pinv(Sigma),
            Q.T,
            self.Phi_Y.T @ self.Phi_X,
            Q,
            np.linalg.pinv(Sigma)
        ])
    
        # Compute updated modes & eigenvalues
        eigenvalues, V_hat = np.linalg.eig(self.K_hat)
        self.modes = np.linalg.multi_dot([
            self.X,
            Q,
            np.linalg.pinv(Sigma),
            np.linalg.inv(V_hat).T
        ])
        self.eigs = eigenvalues[np.abs(eigenvalues) > 1e-16]
    
        return self

    def propagate(self, feat):
        """
        Predict residuals for the next 'steps' time steps using learned Koopman operator.
        """
        
        self.pinv_modes = np.linalg.pinv(self.modes)
        pred = forecast(self.modes, self.eigs, 
                        self.Y[:,-1], self.horizon, 
                        self.pinv_modes)
        
        return pred[-feat:,:]
    