import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def sigmoid(x) :
  return 1 / ( 1 + np.exp(-x))

class LogisticRegression : 
  def __init__(self, lr = 0.01, epochs = 1000) : 
    self.lr = lr
    self.epochs = epochs
    self.weights = None
    self.bias = 0

  def fit(self, X, y) : 
    n_samples, n_features = X.shape
    self.weights = np.zeros(n_features)
    self.bias = 0
    
    for _ in range(self.epochs) : 
      linear_pred = np.dot(X, self.weights) + self.bias
      y_pred = sigmoid(linear_pred)
      
      dw = (1 / n_samples) * np.dot(X.T, (y_pred - y));
      db = (1 / n_samples) * np.sum(y_pred - y)

      self.weights = self.weights - self.lr * dw
      self.bias = self.bias - self.lr * db

    print(f"Weights : {self.weights}")
    print(f"Bias : {self.bias}")

  def predict(self, X) : 
    linear_pred = np.dot(X, self.weights) + self.bias
    y_pred = sigmoid(linear_pred)
    return [0 if y <= 0.5 else 1 for y in y_pred]

data = load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train ,y_test = train_test_split(X, y, train_size=0.8, random_state=1234)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Classification report : ");
print(classification_report(y_test, y_pred))

c = -5
x = []
while(c <= 5) : 
  x.append(c)
  c += 0.01

y = [sigmoid(xi) for xi in x]
plt.scatter(x, y) 
plt.plot([0, 0], [0, 1], c="b")
plt.plot([-5, 5], [0.5, 0.5], c="r")
plt.show()
      