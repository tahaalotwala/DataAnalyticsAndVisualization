library(stats)     # For decomposition
library(forecast)  # For ACF, PACF and nice plotting

data("AirPassengers")
series <- AirPassengers

decomposition <- decompose(series)  
plot(decomposition)

par(mfrow = c(2, 1))  # 2 rows, 1 column for ACF and PACF

acf(series, lag.max = 40, main = "Autocorrelation Function (ACF)")
pacf(series, lag.max = 40, main = "Partial Autocorrelation Function (PACF)")
