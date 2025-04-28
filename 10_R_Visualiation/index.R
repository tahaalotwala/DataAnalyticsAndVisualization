print(iris)

# 1. Line plot
plot(iris$Sepal.Length, type = "l", main = "Line Plot of Sepal Length", 
     xlab = "Sample Index", ylab = "Sepal Length (cm)")

# 2. Scatter plot
plot(iris$Sepal.Length, iris$Sepal.Width, main = "Scatter Plot: Sepal Length vs Sepal Width", 
     xlab = "Sepal Length (cm)", ylab = "Sepal Width (cm)")

# 3. Bar plot
barplot(table(iris$Species), main = "Bar Plot of Class Counts", 
        xlab = "Species", ylab = "Count")

# 4. Histogram
hist(iris$Sepal.Length, breaks = 20, main = "Histogram of Sepal Length", 
     xlab = "Sepal Length (cm)", ylab = "Frequency")

# 5. Box plot
boxplot(iris$Sepal.Length, main = "Box Plot of Sepal Length", 
        ylab = "Sepal Length (cm)")
