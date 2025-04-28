import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
print(iris.feature_names)

fig, axes = plt.subplots(5, 1, sharex = False)

# 1. Line plot
axes[0].plot(X[:, 0])
axes[0].set_xlabel("Sepal length")

# 2. Scatter plot
axes[1].scatter(X[:, 0], X[:, 1])

# 3. Bar plot
axes[2].bar([0, 1, 2], [sum(y==0), sum(y==1), sum(y==2)])

# 4. Histogram
axes[3].hist(X[:, 0], bins=20)

# 5. Box plot
axes[4].boxplot(X[:, 0])

plt.show()