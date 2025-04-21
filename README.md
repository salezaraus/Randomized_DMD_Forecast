---
layout: default
---
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

Streaming Forecasting by Decomposing Trend & Learning Residuals

This repository implements a two‑stage forecasting pipeline:

Trend Extraction via DLinear: a lightweight linear model that decomposes the input into trend and remainder.

Residual Correction via Randomized Kernel DMD (RKDMD): a non‑linear, streaming DMD variant to model and predict the residual dynamics.

📖 Methodology Overview

Our combined approach builds on Dynamic Mode Decomposition (DMD) and advances it through kernelization and streaming updates. We perform:

1. Univariate Time Series & DMD Framework

Arrange a single‑variable series $S = (x_1,\dots,x_T)$ into a Hankel matrix of $m$ columns (snapshots).

Split into $X,Y$ pairs of shifted snapshots:


Compute the least‑squares Koopman operator


Extract DMD modes $\Phi_{dmd}$ and eigenvalues $\Lambda$ via SVD truncation to rank $r$.

Forecast future points by

averaging anti‑diagonals of the predicted Hankel trajectory for horizon $\ell$.

2. Kernel DMD (KDMD)

Lift $X,Y$ into feature space via kernel $\psi(\cdot)$ (e.g., RBF).

Compute kernel Gram matrices

implicitly via the kernel trick.

Form

from the eigendecomposition of $\widehat G = U,\Sigma^2,U^T$.

Derive kernel DMD modes $\Phi_{kdmd}$ and forecast non‑linear dynamics as in standard DMD.

3. Incremental Kernel DMD (IKDMD)

Streaming data demands online updates without recomputing full kernel SVD.

Adopt rank‑one kernel SVD updates: append new snapshot, update $\widehat G,\widehat A$ via low‑rank corrections.

Incrementally update $\widehat K$, modes, and eigenvalues, enabling continual forecasting with fixed memory.

4. Randomized Kernel DMD (RKDMD)

Scale to high‑dimensional feature maps by approximating $\psi(\cdot)$ using Random Fourier Features (RFF).

Replace exact SVD with randomized SVD: project $\Psi_x$ onto a Gaussian sketch, QR factor, then small‑scale SVD on the sketch.

Efficiently compute DMD modes and eigenvalues for kernelized streaming forecasts.

🗂️ Code Structure

RK_DLinear/
├── data_loader.py          # Multivariate + synthetic time series DataLoader
├── train_dlinear.py        # Train DLinearTrend model (trend extraction)
├── test_dlinear.py         # Visualize DLinear trend forecasts
├── hyper_tuning.py         # Train DLinear (optional) + RKDMD grid search on residuals
├── streaming_forecast.py   # StreamingForecast class: DLinear trend + RKDMD residuals
├── models/
│   ├── dlinear.py          # DLinearTrend implementation (trend decomposition + forecasting)
│   └── rkdmd.py            # RKDMD: Random Fourier Features + incremental Koopman
├── checkpoints/            # Saved `dlinear_trend.pth` weights
├── requirements.txt
└── README.md               # (this file)

🚀 Installation

git clone https://github.com/YourUsername/RK_DLinear.git
cd RK_DLinear
python3 -m venv venv
source venv/bin/activate       # Windows: venv\\Scripts\\activate
pip install -r requirements.txt

🎯 Quickstart

1. Train Trend Model

python train_dlinear.py --seq_len 150 --pred_len 200 --epochs 20 --batch_size 32 \
    --lr 0.001 --selected_columns '["HUFL","HULL","MUFL","MULL","LUFL","LULL","OT"]'

2. Grid Search RKDMD Residuals

python hyper_tuning.py

3. Real‑Time Streaming Forecast

python streaming_forecast.py

4. Inspect Results

Trend: via test_dlinear.py plots.

Residual Tuning: see rkdmd_hyperparameter_results.csv.

Live Forecast: runtime plot of combined trend+residual forecasts.
