import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.datasets import sunspots
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load built-in Sunspots dataset
data = sunspots.load_pandas().data
data['YEAR'] = pd.to_datetime(data['YEAR'], format='%Y')
data.set_index('YEAR', inplace=True)
series = data['SUNACTIVITY']

# Plot ACF and PACF
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

plot_acf(series, ax=axes[0], lags=40)
axes[0].set_title('Autocorrelation Function (ACF)')

plot_pacf(series, ax=axes[1], lags=40, method='ywm')
axes[1].set_title('Partial Autocorrelation Function (PACF)')

plt.tight_layout()
plt.show()

# Fit ARIMA model (order can be tuned better; here using simple order)
model = ARIMA(series, order=(2, 1, 2))  # (p,d,q) = (2,1,2)
model_fit = model.fit()

# Print model summary
print(model_fit.summary())

# Predict
start = len(series) - 50  # Last 50 points for visualization
end = len(series) + 20    # Forecast 20 steps ahead

predictions = model_fit.predict(start=start, end=end, typ='levels')

# Plot actual vs predicted
plt.figure(figsize=(12, 6))
plt.plot(series, label='Actual Series')
plt.plot(predictions, label='Predictions', color='red')
plt.title('Actual vs Predicted Sunspots Activity')
plt.xlabel('Year')
plt.ylabel('Sunspot Activity')
plt.legend()
plt.show()
