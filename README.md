# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
#### Name: SHYAM S
#### Reg.No: 212223240156

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("/content/index_1.csv", parse_dates=['datetime'])

data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

daily_sales = data['money'].resample('D').sum()

plt.figure(figsize=(12, 6))
plt.plot(daily_sales, label='Daily Total Sales')
plt.xlabel('Date')
plt.ylabel('Sales Amount (₹)')
plt.title('Daily Coffee Sales Time Series')
plt.legend()
plt.grid()
plt.show()

plot_acf(daily_sales)
plt.show()

plot_pacf(daily_sales)
plt.show()

train_size = int(len(daily_sales) * 0.8)
train, test = daily_sales[:train_size], daily_sales[train_size:]

sarima_model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,7))
sarima_result = sarima_model.fit()

predictions = sarima_result.predict(start=len(train), end=len(train)+len(test)-1)

rmse = np.sqrt(mean_squared_error(test, predictions))
print("RMSE:", rmse)

plt.plot(test.index, test, label='Actual Sales')
plt.plot(test.index, predictions, color='red', label='Predicted Sales')
plt.xlabel('Date')
plt.ylabel('Sales Amount (₹)')
plt.title('SARIMA Model Predictions for Daily Coffee Sales')
plt.legend()
plt.grid()
plt.show()
```
### OUTPUT:
<img width="1005" height="547" alt="image" src="https://github.com/user-attachments/assets/efe29030-6417-40e1-a481-a988eff3cf3f" />

<img width="568" height="435" alt="image" src="https://github.com/user-attachments/assets/3e024a66-60ca-4e37-a2d1-8f6e897cdf79" />

<img width="571" height="455" alt="image" src="https://github.com/user-attachments/assets/81ee4ad7-449c-41a7-b78e-a401acd2d87f" />

### RESULT:
Thus the program run successfully based on the SARIMA model.
