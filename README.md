# Tesla Stock Price Forecasting Using LSTM

This repository contains a deep learning model developed to forecast Tesla's stock price using Long Short-Term Memory (LSTM) networks. The model leverages various technical indicators, such as RSI, MACD, SMAs, EMAs, and VWAP, to predict the next day's closing price for Tesla stock. The project is implemented using TensorFlow and is optimized with KerasTuner for hyperparameter tuning.

## Project Overview

The goal of this project is to build a predictive model using LSTM to forecast stock prices with improved accuracy by incorporating key technical indicators. The model is trained and tested on Tesla's stock price data, with the objective to predict the closing price for the following day.

### Key Components:

- **Data Preprocessing**: Stock data from Yahoo Finance is preprocessed by adding technical indicators like RSI, MACD, SMAs, EMAs, VWAP, etc.
- **Feature Selection**: Random Forest is used to evaluate feature importance, which helps in selecting relevant features for training the LSTM.
- **Model Architecture**: A simple unidirectional LSTM model is used with a dense output layer.
- **Evaluation**: Model performance is evaluated using metrics like R², Mean Squared Error (MSE), and Mean Absolute Percentage Error (MAPE).
- **Hyperparameter Tuning**: KerasTuner is used for optimizing model parameters, including LSTM units, dropout rates, and learning rates.

## Getting Started

### Prerequisites

To run this project locally, you'll need the following Python libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `tensorflow`
- `keras-tuner`

You can install all the dependencies via the `requirements.txt` file:

```bash
pip install -r requirements.txt
