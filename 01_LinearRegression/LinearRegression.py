import numpy as np

class LinearRegression : 
  def __init__(self, lr = 0.001, epochs = 1000) : 
    self.lr = lr
    self.epochs = epochs
    self.weights = None
    self.bias = None

  def fit(self, X, y) :
    n_samples, n_features = X.shape
    self.weights = np.zeros(n_features)
    self.bias = 0

    for _ in range(self.epochs) : 
      y_pred = np.dot(X, self.weights) + self.bias

      dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
      db = (1 / n_samples) * np.sum(y_pred - y)

      self.weights = self.weights - self.lr * dw
      self.bias = self.bias - self.lr * db

    print("Weights : ", self.weights);
    print("Bias : ", self.bias);


  def predict(self, X) : 
    return np.dot(X, self.weights) + self.bias
