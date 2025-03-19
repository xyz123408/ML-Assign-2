import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Example data for simple linear regression
# Replace this with your own dataset if needed
# x: input feature (e.g., years of experience, etc.)
# y: target variable (e.g., salary, etc.)
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)  # Input feature (reshape for a single feature)
y = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5])  # Target variable

# Split the data into training and test sets (80% training, 20% testing)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(x_train, y_train)

# Get the coefficients (intercept and slope)
b0 = model.intercept_
b1 = model.coef_[0]

# Make predictions on the test data
y_pred = model.predict(x_test)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print the model's coefficients and RMSE
print(f"Intercept (B0): {b0}")
print(f"Slope (B1): {b1}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Plot the original data points and the regression line
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, model.predict(x), color='red', label='Regression line')
plt.title("Simple Linear Regression")
plt.xlabel("Input (x)")
plt.ylabel("Target (y)")
plt.legend()
plt.show()
