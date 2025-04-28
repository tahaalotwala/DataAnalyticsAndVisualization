import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

def mse(y_test, y_pred) : 
  return np.mean((y_test - y_pred) ** 2)

# n_features = 1 for SLR, 2 for MLR
X, y = datasets.make_regression(n_samples = 100, n_features = 2, noise = 20, random_state = 4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 1234)

# Plot only for SLR
# fig = plt.figure(figsize=(8, 6))
# plt.scatter(X[:, 0], y, color = "b", marker = "o", s = 30)
# plt.show()

model = LinearRegression(lr=0.1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"MSE : {mse(y_test, y_pred)}")
