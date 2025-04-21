# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 12:48:13 2025

@author: Chris
"""

import numpy as np
from scipy.linalg import hankel

def construct_multivariate_hankel(data, m):
    """
    Constructs a multivariate Hankel matrix for time series data.
    
    Parameters:
        data (numpy.ndarray): Input time series of shape (n_features, n_time_steps).
        m (int): Number of columns in the Hankel matrix (time delays).
    
    Returns:
        X (numpy.ndarray): Hankel matrix of shape (n_features * (T-m+1), m-1).
        Y (numpy.ndarray): Shifted Hankel matrix of the same shape as X.
    """
    n_features, n_time_steps = data.shape
    #if m > n_time_steps:
    #    raise ValueError("m (number of columns) must be <= number of time steps.")
    
    # Create Hankel matrix blocks
    hankel_matrix = []
    for i in range(n_time_steps - m + 1):
        block = data[:, i:i + m].flatten(order='F')  # Flatten in column-major order
        hankel_matrix.append(block)
    
    H = np.array(hankel_matrix).T  # Shape (n_features * m, T - m + 1)
    
    
    return H

def create_hankel_matrix(time_series, columns=None):
    """
    Create a Hankel matrix from a time series sequence.

    Parameters:
    time_series (list or numpy.array): The time series data.
    columns (int, optional): The number of columns in the Hankel matrix. If not provided,
                             the Hankel matrix will have columns equal to the length of the time series.

    Returns:
    numpy.array: Hankel matrix.
    """

    time_series = np.asarray(time_series)
    n = len(time_series)

    if columns is None or columns > n:
        columns = n

    # Calculate the number of rows based on the number of columns
    rows = n - columns + 1

    # The first column of the Hankel matrix
    c = time_series[:rows]

    # The last row of the Hankel matrix
    r = time_series[rows - 1:]

    hankel_matrix = hankel(c, r)

    return hankel_matrix

def RFF(gamma, d, s):
    # Generate all zi vectors in one batch
    zi = np.sqrt(2 * gamma) * np.random.randn(d, s)  # s Gaussian vectors of dimension d
    
    # Generate all theta_i values in one batch
    theta_i = 2 * np.pi * np.random.rand(s)  # s uniform scalars
    
    return zi, theta_i

# =============================================================================
# def featurize(X, z, theta): 
#     
#     d, B = np.shape(X)
#     s = len(theta)
#     
#     PHI = np.zeros((s,B))
#     
#     for j in range(B):
#         for i in range(s): 
#             PHI[i,j] = np.sqrt(2/s) * np.cos(theta[i] + np.dot(z[:,i], X[:,j]))
#             
#     return PHI
# =============================================================================

def featurize(X, z, theta):
    # X: (d, B), z: (d, s), theta: (s,)
    projections = np.dot(z.T, X)  # (s, B)
    phases = theta[:, None] + projections  # (s, B)
    return np.sqrt(2 / len(theta)) * np.cos(phases)  # (s, B)

def forecast(modes, eigs, y_init, horizon, pinv_modes):
    #amps = np.linalg.pinv(modes) @ y_init
    amps = np.matmul(pinv_modes, y_init)
    
    # Precompute powers of eigenvalues for all time steps
    t_indices = np.arange(1, horizon + 1)
    eig_powers = np.power(eigs[:, None], t_indices)  # Broadcasting to create a matrix of eigenvalues raised to the power of t

    # Compute predictions using matrix operations
    pred = modes @ (eig_powers * amps[:, None])  # Element-wise multiplication and matrix product
    
    # Compute the average of anti-diagonals
    #pred = average_antidiagonals(predictions)
    
    return pred


