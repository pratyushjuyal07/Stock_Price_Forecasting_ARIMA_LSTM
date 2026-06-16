# Hybrid ARIMA + LSTM Framework for Stock Price Forecasting

This repository implements a decoupled hybrid time series forecasting model combining the linear statistical capacity of **ARIMA** with the non-linear sequence modeling of an **LSTM** neural network. The validation is executed using a rigorous **1-Step-Ahead Rolling (Walk-Forward) Loop** over a 10-year historical dataset of the NIFTY50 index (`^NSEI`).

---

## 1. Objective
Raw stock prices contain both linear patterns (macro trends) and non-linear patterns (volatility spikes, market noise). 
* Traditional models like ARIMA cannot handle non-linear volatility.
* Deep learning models like LSTMs often struggle with raw, non-stationary price values.
* **The Solution:** Use ARIMA to capture the baseline linear trend, extract its prediction errors (residuals), and train an LSTM to forecast those errors. Combining them yields a more robust, adaptive prediction.

---

## 2. Methodology & Workflow Steps

### Step 1: Data Split (80/20)
We download 10 years of NIFTY50 data and partition it using a strict chronological 80/20 split. The last 20% (~493 trading days) is strictly kept separate for out-of-sample rolling validation.

### Step 2: Stationarity Testing (ADF)
We apply the Augmented Dickey-Fuller (ADF) test on the training data. The raw price series is non-stationary, but its first-difference is highly stationary ($p < 0.01$), setting our differencing term to **$d=1$**.

### Step 3: Seasonal Decomposition
We compute an additive seasonal decomposition over a 252-day business year cycle to visually break down the index into its underlying trend, seasonality, and unstructured noise components.

### Step 4: Parameter Isolation (ACF / PACF)
By tracking autocorrelation and partial autocorrelation plots of the differenced series, we observe sharp cutoffs after the first lag. This determines our statistical order baseline: **ARIMA(1, 1, 1)**.

### Step 5: Rolling Walk-Forward ARIMA
To eliminate the classic "flat line" error of long-term forecasting, we build a rolling window loop. The model forecasts exactly 1 day ahead, updates its history with the true price at market close, and repeats for all 493 test days.

### Step 6: Residual Extraction & Scaling
We calculate the out-of-sample linear errors: $\epsilon_t = \text{Actual Price}_t - \text{ARIMA Predicted}_t$. These residuals are normalized into a $[-1, 1]$ window and formatted into 10-day lookback sequences.

### Step 7: Deep Recurrent LSTM Network
A sequential network architecture consisting of two stacked LSTM layers (64 and 32 units) and two Dense layers is trained specifically to model the underlying patterns inside the ARIMA residuals.

### Step 8: Hybrid Model Synthesis
The final output is generated daily by adding the standalone rolling ARIMA forecast and the unscaled LSTM error adjustment prediction:
$$\text{Final Forecast}_{t+1} = \text{ARIMA Forecast}_{t+1} + \text{LSTM Residual Prediction}_{t+1}$$

### Step 9: Strategy Backtesting
We simulate a Long/Short trading strategy over the evaluation window. If the hybrid model predicts tomorrow's price will be higher than today's price, the strategy goes Long ($+1$); otherwise, it takes a Short/Cash position ($-1$).

---

## 3. Empirical Observations & Performance Metrics

The out-of-sample performance over the 493 trading days generated the following results:

| Evaluation Category | Value / Performance Metric |
| :--- | :--- |
| **Total Trading Days Evaluated** | 493 Days |
| **Correct Directional Predictions** | 258 Days |
| **True Directional Win Rate** | **52.33%** |
| **Buy & Hold NIFTY50 Return** | **2.23%** |
| **Hybrid ARIMA+LSTM Strategy Return** | **17.40%** |

* **The Visual Illusion:** The final forecast line appears to match actual prices almost perfectly. In reality, a close-up inspection reveals a 1-day lag effect—the model uses today’s true price as a mathematical anchor to estimate tomorrow.
* **Equity Curve Output:** The strategy significantly outperforms the passive benchmark during sideways and volatile market phases where buying and holding yields minimal returns.

---

## 4. Quantitative Inferences

### The Profit vs. Win-Rate Paradox
A directional win rate of **52.33%** is close to a coin flip, which confirms that daily market noise remains highly unpredictable. However, the model still achieved a **17.40%** return due to **asymmetric payoff magnitudes**: its losses on incorrect days were small, while its correct choices successfully captured major multi-day market corrections and rallies.

### Real-World Trading Friction
The simulation assumes zero friction. In a real-world setting over 493 days, executing trades almost daily would attract high portfolio turnover costs:
* Securities Transaction Tax (STT), exchange charges, and GST.
* Brokerage fees.
* Execution slippage.

Even a tiny average friction fee of **0.05% per trade** across hundreds of position adjustments would severely reduce or completely erase the active outperformance ($17.40\%$).

---

## 5. Conclusion
The hybrid ARIMA + LSTM framework provides a sophisticated, multi-layered approach to tracking asset prices. While walk-forward loops generate visually impressive tracking lines and low error metrics (RMSE/MAPE), true model validation requires a strict directional backtest. Combining statistical frameworks with deep learning works exceptionally well for capturing large macro-level momentum shifts, but real-world trading friction must be factored into execution to generate sustainable net alpha.
