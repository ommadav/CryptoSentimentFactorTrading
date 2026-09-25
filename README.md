# Solution to MATLAB and Simulink Challenge project <'239'> <'Sentiment Analysis in Cryptocurrency Trading'>
## Crypto Sentiment Factor Trading System

## 📌 Project Overview

The **Crypto Sentiment Factor Trading System** is a MATLAB-based machine learning research project designed to analyze Bitcoin market data together with cryptocurrency sentiment information and generate short-term **BUY, HOLD, and SELL** signals.

The system performs data preprocessing, feature engineering, machine learning model comparison, time-series validation, historical backtesting, distribution-shift analysis, and interactive signal generation through a MATLAB dashboard.

> **Academic Research Prototype:** This project uses historical data for research and demonstration. It is not a live trading system and does not provide financial or investment advice.

---

## 🎯 Objectives

- Analyze historical Bitcoin market data.
- Integrate cryptocurrency sentiment information.
- Perform data preprocessing and exploratory analysis.
- Engineer technical, momentum, volatility, volume, and sentiment features.
- Predict one-day-ahead Bitcoin returns.
- Generate BUY/HOLD/SELL signals.
- Compare multiple machine learning and regression models.
- Perform chronological train/validation/test evaluation.
- Perform historical backtesting.
- Analyze model stability and distribution shift.
- Develop an interactive MATLAB prototype dashboard.

---

## 🧠 System Pipeline

```text
Bitcoin Market Data
        +
Sentiment Data
        ↓
Data Preprocessing
        ↓
Data Alignment
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Target Generation
        ↓
Model Training
        ↓
Model Comparison
        ↓
Validation
        ↓
Final Out-of-Sample Testing
        ↓
Historical Backtesting
        ↓
Distribution-Shift Analysis
        ↓
MATLAB Interactive Prototype
        ↓
BUY / HOLD / SELL Signal
```

---

## 🛠️ Technologies Used

- MATLAB
- MATLAB Statistics and Machine Learning Toolbox
- Historical Bitcoin market data
- Cryptocurrency sentiment data
- Regression models
- Ensemble machine learning models
- MATLAB UI components

---

## 🤖 Models Evaluated

The project evaluates:

1. **AR(5) Ridge Regression**
2. **Multivariate Ridge Regression**
3. **Bagged Regression Trees**
4. **Random Forest Regression**
5. **LSBoost Regression**

### AR(5) Ridge Regression

The primary prototype uses an AR(5) regression model with ridge regularization.

```text
Return(t-5)
Return(t-4)
Return(t-3)
Return(t-2)
Return(t-1)
       ↓
  AR(5) Ridge
       ↓
Predicted Return(t)
       ↓
BUY / HOLD / SELL
```

---

## 📂 Project Structure

```text
CryptoSentimentFactorTrading/
│
├── README.md
├── .gitignore
│
├── src/
│   ├── main_pipeline.m
│   ├── step1_data_loading_and_eda.m
│   ├── step2_sentiment_classification.m
│   ├── step3_feature_engineering.m
│   ├── step4_model_comparison.m
│   ├── step5_final_master_backtest.m
│   ├── step6_final_results_analysis.m
│   ├── step7_multivariate_ridge_factor_model.m
│   ├── step8_unified_final_model_comparison.m
│   ├── step9_distribution_shift_diagnostics.m
│   ├── step9b_prediction_stability_diagnostics.m
│   ├── step10_reduced_factor_experiment.m
│   ├── prototype_trading_app.m
│   │
│   └── utils/
│       ├── dataUtils.m
│       ├── featureUtils.m
│       ├── modelUtils.m
│       ├── backtestUtils.m
│       ├── plotUtils.m
│       └── sentimentUtils.m
│
├── data/
│   ├── raw/
│   │   ├── btc_1d_data_2018_to_2025.csv
│   │   └── sentiment_data_consolidated.csv
│   │
│   └── processed/
│
└── results/
    ├── figures/
    ├── tables/
    └── models/
```

---

# 📊 Dataset

## Bitcoin Dataset

The Bitcoin dataset contains historical market information including:

- Open price
- High price
- Low price
- Close price
- Trading volume
- Number of trades
- Additional market-volume information

Expected file:

```text
data/raw/btc_1d_data_2018_to_2025.csv
```

## Sentiment Dataset

The sentiment dataset contains daily cryptocurrency sentiment information.

Expected file:

```text
data/raw/sentiment_data_consolidated.csv
```

The Bitcoin and sentiment datasets are aligned using their common dates before model development.

---

# ⚙️ Feature Engineering

The system generates features from multiple categories.

### Price and Return Features

- Daily returns
- Log returns
- Lagged returns
- Multi-day momentum

### Technical Indicators

- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- RSI
- MACD
- MACD Signal
- MACD Histogram

### Volatility Features

- Rolling volatility
- ATR
- Normalized ATR
- Daily price range

### Volume Features

- Volume change
- Relative volume
- Volume moving averages

### Sentiment Features

- Sentiment
- Sentiment lags
- Sentiment moving averages
- Sentiment change
- Sentiment momentum
- Sentiment-price interactions

---

# 📅 Model Evaluation

The project uses **chronological train/validation/test splitting** instead of random shuffling because the data represents a time series.

The final AR(5) evaluation uses:

```text
Training:
06-Jan-2018 → 10-Apr-2020

Validation:
11-Apr-2020 → 04-Oct-2020

Testing:
05-Oct-2020 → 31-Mar-2021
```

The test period is kept separate from model and threshold selection.

---

# 📈 Evaluation Metrics

## Prediction Metrics

- RMSE
- MAE
- MSE
- R²
- Directional Accuracy
- Prediction Correlation

## Trading Metrics

- Total Return
- Final Capital
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Win Rate
- Profit Factor
- Number of Trades

The strategies are also compared with a **Buy & Hold** benchmark.

---

# 🔬 Distribution-Shift Analysis

The project includes diagnostics to investigate changes between validation and test periods.

The analysis includes:

- Feature distribution shift
- Sentiment regime shift
- Prediction distribution shift
- Actual-return distribution shift
- Feature multicollinearity
- Signal distribution
- Validation/test trading behavior

---

# 🖥️ MATLAB Interactive Prototype

The project includes an interactive MATLAB dashboard for demonstrating the AR(5) model.

Main file:

```text
src/prototype_trading_app.m
```

Run it using:

```matlab
addpath(genpath('src'));
prototype_trading_app
```

## Prototype Dashboard

![Crypto Sentiment Trading System MATLAB Prototype](results/figures/prototype_dashboard.png)

### Dashboard Components

#### Market Inputs

- Current Price
- Previous Close
- Volume
- Sentiment
- RSI

#### AI Trading Signal

The dashboard displays:

- Predicted Market Condition
- BUY / HOLD / SELL signal
- Signal Strength
- Predicted Return
- Validation Threshold

#### Final Model Comparison

The dashboard displays:

- AR(5) model
- Prediction horizon
- Test period
- Test RMSE
- Strategy Return
- Sharpe Ratio
- Buy & Hold Return
- Buy & Hold Sharpe
- Maximum Drawdown
- Transaction Cost

#### Visualizations

- Bitcoin Price History
- AR(5) Prediction Strength
- Final Test Performance

---

# 🔄 Interactive Signal Generation

The user can change:

```text
Current Price
Previous Close
Volume
Sentiment
RSI
```

and select:

```text
GENERATE TRADING SIGNAL
```

The system calculates the input return and generates a new AR(5) prediction.

```text
Current Price
      +
Previous Close
      ↓
Current Return
      ↓
AR(5) Ridge Regression
      ↓
Predicted Return
      ↓
Threshold Comparison
      ↓
BUY / HOLD / SELL
```

> **Important:** Signal Strength is a relative prediction-magnitude indicator. It is **not** a statistical probability or confidence estimate.

---

# 📊 Final Test Results

The final AR(5) test evaluation produced:

| Metric | Result |
|---|---:|
| RMSE | 0.042473 |
| MAE | 0.030789 |
| R² | -0.073428 |
| Directional Accuracy | 50.56% |
| Strategy Return | +11.4028% |
| Sharpe Ratio | 0.6646 |
| Sortino Ratio | 0.7480 |
| Maximum Drawdown | -51.69% |
| Win Rate | 49.02% |
| Profit Factor | 1.1086 |
| Number of Trades | 128 |

### Buy & Hold Benchmark

| Metric | Result |
|---|---:|
| Return | +443.55% |
| Sharpe Ratio | 4.8243 |
| Maximum Drawdown | -25.17% |

---

# 📌 Final Findings

The experiments did not provide evidence that the engineered multivariate sentiment-factor models consistently improved out-of-sample Bitcoin trading performance over the simpler AR(5) baseline or the Buy & Hold benchmark.

The project demonstrates the importance of:

- Chronological validation
- Untouched test data
- Avoiding data leakage
- Out-of-sample evaluation
- Distribution-shift analysis
- Prediction-stability analysis
- Benchmark comparison

The results also demonstrate that development-period behavior does not necessarily generalize to unseen cryptocurrency market conditions.

---

# ⚠️ Limitations

- The system uses historical data.
- Cryptocurrency markets are highly non-stationary.
- Sentiment data covers a limited period.
- Market regimes can change significantly.
- Transaction costs and real-world market effects are simplified.
- The model does not guarantee future performance.
- The prototype is not connected to a live cryptocurrency exchange.
- The prototype is intended for academic demonstration.

---

# 🚀 Future Work

Possible future improvements include:

- Larger multi-cycle cryptocurrency datasets
- Real-time sentiment APIs
- Advanced NLP and Transformer-based sentiment models
- XGBoost and LightGBM
- CNN-LSTM and Transformer time-series models
- Walk-forward model retraining
- Dynamic risk management
- Position sizing
- More realistic slippage and transaction-cost modeling
- Multi-cryptocurrency analysis
- Real-time deployment

---

# ▶️ How to Run

### 1. Open MATLAB

Open the repository folder in MATLAB.

### 2. Add the source directory

```matlab
addpath(genpath('src'));
```

### 3. Run the prototype

```matlab
prototype_trading_app
```

### 4. Run the complete pipeline

```matlab
main_pipeline
```

Individual scripts can also be executed separately when their required input files are available.

---

# 📁 Results

Generated outputs are stored in:

```text
results/figures/
results/tables/
results/models/
```

These directories contain:

- Model comparison tables
- Prediction results
- Backtesting results
- Performance summaries
- Diagnostic figures
- Saved MATLAB models

---

# 👥 Team Members

| Name | Roll Number |
|---|---|
| Om Madav | 24108A0066 |
| Aniket Lonkar | 24108A0063 |
| Tejas Dhanvi | 24108A0072 |
| Sagar Talpade | 25108A2008 |

---

# 🎓 Academic Project

This project was developed as an academic research project to study the application of:

- Machine Learning
- Time-Series Prediction
- Sentiment Analysis
- Feature Engineering
- Cryptocurrency Data Analysis
- Historical Backtesting
- Model Evaluation

---
## Generative AI Acknowledgment

Generative AI tools, including ChatGPT, were used as a supporting tool during the development of this project.

The assistance included:
- Brainstorming and refining the project workflow.
- Assistance with MATLAB code development and debugging.
- Reviewing data-processing and model-evaluation logic.
- Assistance with documentation and README preparation.
- Helping identify implementation issues and improve code robustness.

All AI-assisted code and suggestions were reviewed, tested, modified where necessary, and integrated by the project team. The team is responsible for understanding, verifying, and explaining the final implementation, experimental results, and conclusions.

No AI-generated data or fabricated experimental results were used. All reported results were obtained from the project's actual MATLAB experiments and datasets.

# ⚠️ Disclaimer

**This project is an academic research prototype based on historical data. It is not a live trading system and does not provide financial or investment advice. Historical model performance does not guarantee future results.**
