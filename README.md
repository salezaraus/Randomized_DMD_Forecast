RK_DLinear

Streaming Forecasting by Decomposing Trend & Learning Residuals

A two-stage pipeline for accurate and interpretable time series forecasting:

DLinear Trend ExtractionExtract the smooth, long-term trend component using a simple linear decomposition model.

RKDMD Residual ModelingCapture and forecast remaining dynamics via Randomized Kernel Dynamic Mode Decomposition, updated on-the-fly for streaming data.

Methodology Overview

Trend Decomposition (DLinear)

Fits a linear model to separate the core trend from raw observations.

Produces a residual series containing higher-frequency and nonlinear components.

Residual Correction (RKDMD)

Embeds residuals into a nonlinear feature space with Random Fourier Features.

Applies a streaming variant of Kernel DMD to learn underlying modes and eigenvalues.

Updates the model incrementally with each new data point, enabling real-time forecasting.

Combined Forecast

Add the DLinear trend forecast to the RKDMD residual forecast for a final prediction.

Supports single-step and multi-step horizons in a fully streaming setting.

Code Structure

RK_DLinear/
├── data_loader.py         # Datasets: multivariate CSVs & synthetic toy series
├── train_dlinear.py       # Train DLinearTrend model and save checkpoint
├── test_dlinear.py        # Visualize trend forecasts on test data
├── hyper_tuning.py        # Grid-search RKDMD hyperparameters on residual data
├── streaming_forecast.py  # StreamingForecast class: DLinear + RKDMD pipeline
├── models/
│   ├── dlinear.py         # DLinearTrend implementation
│   └── rkdmd.py           # RKDMD implementation (RFF + incremental updates)
├── checkpoints/           # Saved model weights (`dlinear_trend.pth`)
├── requirements.txt       # Python dependencies
└── README.md              # This file

Installation

git clone https://github.com/YourUsername/RK_DLinear.git
cd RK_DLinear
python3 -m venv venv
source venv/bin/activate       # Windows: venv\\Scripts\\activate
pip install -r requirements.txt

Quickstart

Train Trend Model

python train_dlinear.py \
  --seq_len 150 --pred_len 200 \
  --epochs 20 --batch_size 32 --lr 0.001 \
  --selected_columns '["HUFL","HULL","MUFL","MULL","LUFL","LULL","OT"]'

Tune RKDMD Residuals

python hyper_tuning.py

Run Streaming Forecast

python streaming_forecast.py

Visualize Trend Forecast
