import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.datasets import sunspots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load built-in Sunspots dataset
data = sunspots.load_pandas().data
data['YEAR'] = pd.to_datetime(data['YEAR'], format='%Y')
data.set_index('YEAR', inplace=True)
series = data['SUNACTIVITY']

# Decompose the time series
decomposition = seasonal_decompose(series, model='additive', period=11)  # Approximate sunspot cycle ~11 years

# Plot the decomposed components
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

axes[0].plot(series)
axes[0].set_title('Original Time Series')

axes[1].plot(decomposition.trend)
axes[1].set_title('Trend Component')

axes[2].plot(decomposition.seasonal)
axes[2].set_title('Seasonal Component')

axes[3].plot(decomposition.resid)
axes[3].set_title('Residual Component')

plt.tight_layout()
plt.show()

# Plot ACF and PACF
fig, ax = plt.subplots(2, 1, figsize=(12, 8))

plot_acf(series, ax=ax[0], lags=40)
ax[0].set_title('Autocorrelation Function (ACF)')

plot_pacf(series, ax=ax[1], lags=40, method='ywm')
ax[1].set_title('Partial Autocorrelation Function (PACF)')

plt.tight_layout()
plt.show()
