### LSTM Stock Predictions
# Overview
This repository contains Jupyter notebooks for predicting stock prices using Long Short-Term Memory (LSTM) neural networks, built with PyTorch and scikit-learn. The project evolved from an initial naive implementation focused on Tesla (TSLA) stock data to a more sophisticated approach using Johnson & Johnson (JNJ) stock data. The goal is to explore time series forecasting for financial data, incorporating advanced feature engineering and modeling techniques.
The initial TSLA model (in LSTM_TSLA.ipynb) relied on basic technical indicators from a pre-processed CSV dataset generated via DataExtraction.ipynb. This served as a proof-of-concept but highlighted limitations in handling non-stationary data and simplistic feature sets. The updated JNJ model (in JNJ_Stock_Prediction_LSTM.ipynb) addresses these shortcomings, demonstrating iterative improvements in methodology.
Key Improvements
From TSLA to JNJ: A More Refined Approach
In the original TSLA implementation, I naively treated stock prices as directly predictable targets using a standard LSTM on raw closing prices and basic technical indicators (e.g., RSI, MACD, SMAs). This approach overlooked the non-stationary nature of financial time series, leading to potential overfitting and poor generalization. The model scaled features with MinMaxScaler and used a simple sequence-to-one prediction, but it didn't account for stationarity, cyclical patterns, or advanced input formatting, resulting in suboptimal performance (e.g., high mean squared errors and visual discrepancies in predictions).
The JNJ model represents a significant upgrade:

Data Source and Scope: We fetch fresh JNJ data via yfinance (from 2020 onward for focus on recent trends) and incorporate external market data (e.g., S&P 500 and VIX indices). This provides broader context beyond the TSLA's isolated dataset.
Stationarity Handling: Instead of predicting raw stock prices (which exhibit trends and volatility clustering), the model now forecasts log price returns (log(close_t / close_{t-1})). Log returns are stationary (mean-reverting), making them more suitable for LSTM training and reducing issues like exploding gradients or non-convergent training. This shift aligns with financial econometrics best practices.
Feature Engineering Enhancements:

Look-Back Returns: We create lagged features for log returns with a 5-day look-back, accounting for the standard trading week (Monday to Friday). These are formatted as sequential inputs for the LSTM, allowing the model to learn temporal dependencies explicitly.
Cyclical Encoding: Date-based features (e.g., month, day of week, quarter) are encoded using cyclical transformations (sin/cos) to handle their periodic nature, preventing the model from treating December as "farther" from January than it actually is in a yearly cycle.
Additional Features: Incorporated several technical indicators using the ta library (e.g., RSI with 14-day window, MACD and its signal/diff components) along with manually engineered features like returns, log returns, 5-day volatility, 5-day momentum, and merged external returns (S&P 500, VIX with log transformations). This results in a total of over 20 input features, normalized with StandardScaler for better gradient flow. Volume and volatility features add context on market liquidity and risk.


Model Architecture and Training:

The LSTM uses dropout for regularization, trained on sequences of 5 timesteps (look-back window) to predict the next log return, with an input size of 16 (adjusted for the feature set).
TimeSeriesSplit for cross-validation respects temporal order, avoiding data leakage.
PyTorch implementation allows finer control over optimization (Adam optimizer, MSE loss) compared to the TensorFlow/Keras setup in TSLA, with 2 LSTM layers and hidden size of 50.


Evaluation: Metrics include MSE, MAE, and visualizations of predictions vs. actuals. Inverse transformation back to prices for interpretability.

These changes make the model more robust to financial data's inherent noise and non-stationarity, though stock prediction remains challenging due to external factors (e.g., news, macroeconomic events).
