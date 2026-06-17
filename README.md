# Hybrid ARIMA + LSTM Framework for NIFTY50 Stock Price Forecasting

Implemented a decoupled hybrid time-series forecasting model that combines the linear statistical capacity of **ARIMA** with the non-linear sequence modelling of an **LSTM** neural network. The validation is executed using a rigorous **1-Step-Ahead Rolling (Walk-Forward) Loop** over a 10-year historical dataset of the NIFTY50 index (`^NSEI`).

---

## 1. Objective
Raw stock prices contain both linear patterns (macro trends) and non-linear patterns (volatility spikes, market noise). 
* Traditional models like ARIMA cannot handle non-linear volatility.
* Deep learning models like LSTMs often struggle with raw, non-stationary price values.
* **The Solution:** Use ARIMA to capture the baseline linear trend, extract its prediction errors (residuals), and train an LSTM to forecast those errors. Combining them yields a more robust, adaptive prediction.

---

## 2. Methodology & Workflow Steps

### Step 1: Data Split (80/20)
Downloaded 10 years of NIFTY50 data and partitioned it into training and test sets using a strict chronological 80/20 split. The last 20% (~493 trading days) is strictly kept separate for out-of-sample rolling validation.

### Step 2: Stationarity Testing (ADF)
Applied the Augmented Dickey-Fuller (ADF) test on the training data. The raw price series is non-stationary, but its first difference is highly stationary ($p < 0.01$), setting the differencing term to **$d=1$**.

### Step 3: Seasonal Decomposition
Computed an additive seasonal decomposition over a 252-day business year cycle to visually break down the index into its underlying trend, seasonality, and unstructured noise components.

<img width="1389" height="989" alt="image" src="https://github.com/user-attachments/assets/db74f131-3cac-484f-bd8f-644b6d32699a" />


### Step 4: Parameter Isolation (ACF / PACF)
By tracking autocorrelation and partial autocorrelation plots of the differenced series, sharp cutoffs are observed after the first lag. This determines the final statistical order baseline: **ARIMA(1, 1, 1)**.

<img width="1590" height="490" alt="image" src="https://github.com/user-attachments/assets/08694370-e401-4349-b97f-6b5fb60e89fd" />


### Step 5: Rolling Walk-Forward ARIMA
To eliminate the classic "flat line" error of long-term forecasting, a rolling window loop was built. The model forecasts exactly 1 day ahead, updates its history with the true price at market close, and repeats for all 493 test days.

### Step 6: Residual Extraction & Scaling
Calculated the out-of-sample linear errors. These residuals are normalized into a $[-1, 1]$ window and formatted into 10-day lookback sequences.

<img width="1255" height="547" alt="image" src="https://github.com/user-attachments/assets/e258efd4-4da2-4add-843c-27d13abcfa28" />


### Step 7: Deep Recurrent LSTM Network
A sequential network architecture consisting of two stacked LSTM layers (64 and 32 units) and two Dense layers is trained specifically to model the underlying patterns inside the ARIMA residuals.

### Step 8: Hybrid Model Synthesis
The final output is generated daily by adding the standalone rolling ARIMA forecast and the unscaled LSTM error adjustment prediction:

<img width="1255" height="548" alt="image" src="https://github.com/user-attachments/assets/70460ce9-a042-441b-8b0f-9a1486c43f06" />


### Step 9: Strategy Backtesting
Simulated a Long/Short trading strategy over the evaluation window. If the hybrid model predicts tomorrow's price will be higher than today's price, the strategy goes Long ($+1$); otherwise, it takes a Short/Cash position ($-1$).

<img width="1242" height="548" alt="image" src="https://github.com/user-attachments/assets/7c353056-e7f8-41c5-948a-a8e33929f4bd" />


---

## 3. Empirical Observations & Performance Metrics

RMSE (Root Mean Squared Error): 201.80
MAE  (Mean Absolute Error):     148.18
MAPE (Mean Absolute Percentage): 0.61%

The out-of-sample performance over the 493 trading days generated the following results:

| Evaluation Category | Value / Performance Metric |
| :--- | :--- |
| **Total Trading Days Evaluated** | 493 Days |
| **Correct Directional Predictions** | 258 Days |
| **True Directional Win Rate** | **52.33%** |
| **Buy & Hold NIFTY50 Return** | **2.23%** |
| **Hybrid ARIMA+LSTM Strategy Return** | **17.40%** |

* **The Visual Illusion:** The final forecast line appears to match actual prices almost perfectly. In reality, a close-up inspection reveals a 1-day lag effect: the model uses today’s true price as a mathematical anchor to estimate tomorrow.
* **Equity Curve Output:** The strategy significantly outperforms the passive benchmark during sideways and volatile market phases where buying and holding yield minimal returns.

---

## 4. Quantitative Inferences

### The Profit vs. Win-Rate Paradox
A directional win rate of **52.33%** is close to a coin flip, which confirms that daily market noise remains highly unpredictable. However, the model still achieved a **17.40%** return due to **asymmetric payoff magnitudes**: its losses on incorrect days were small, while its correct choices successfully captured major multi-day market corrections and rallies.

### Real-World Trading Friction
The simulation assumes zero friction. In a real-world setting, over 493 days, executing trades almost daily would attract high portfolio turnover costs:
* Securities Transaction Tax (STT), exchange charges, and GST.
* Brokerage fees.
* Execution slippage.

Even a tiny average friction fee of **0.05% per trade** across hundreds of position adjustments would severely reduce or completely erase the active outperformance ($17.40\%$).

---

## 5. Conclusion
The hybrid ARIMA + LSTM framework provides a sophisticated, multi-layered approach to tracking asset prices. While walk-forward loops generate visually impressive tracking lines and low error metrics (RMSE/MAPE), true model validation requires a strict directional backtest. Combining statistical frameworks with deep learning works exceptionally well for capturing large macro-level momentum shifts, but real-world trading friction must be factored into execution to generate sustainable returns.
