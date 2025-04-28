from sklearn import datasets;
import numpy as np;
import matplotlib.pyplot as plt;

X, y = datasets.make_regression(n_features = 1, n_samples = 100, noise = 20, random_state = 10);

x = np.array(X[:, 0]);
y = np.array(y);

# x = np.array([3, 9, 5, 3]);
# y = np.array([8, 6, 4, 2]);

sy = sum(y);
sx = sum(x);
sxy = sum(x * y);
sx2 = sum(x ** 2);
sy2 = sum(y ** 2);
n = 100

a = (sy * sx2 - sx * sxy) / (n * sx2 - sx ** 2);
b = (n * sxy - sx * sy) / (n * sx2 - sx ** 2);

print(f"a : {a}, b : {b}");

plt.scatter(x, y);
plt.plot([xi for xi in range(-3, 4)], [a + b * xi for xi in range(-3, 4)], color = "g", linewidth = 5);
plt.show();