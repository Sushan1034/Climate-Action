# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate some random data for demonstration purposes
np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # Independent variable
y = 4 + 3 * X + np.random.randn(100, 1)  # Dependent variable with noise

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Make predictions
X_new = np.array([[0], [2]])
y_pred = model.predict(X_new)

# Plot the original data points and the linear regression line
plt.scatter(X, y, alpha=0.6, label='Original Data')
plt.plot(X_new, y_pred, 'r-', label='Linear Regression Prediction')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
