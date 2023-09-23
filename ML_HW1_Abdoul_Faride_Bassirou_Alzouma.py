import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Generate synthetic data with different feature scales
np.random.seed(0)

# Daily temperature (feature x1) with a range of [20, 40] degrees Celsius
temperature = 20 + 20 * np.random.rand(100, 1)

# Ice cream sales (feature x2) with a range of [0, 200] units
ice_cream_sales = 200 * np.random.rand(100, 1)

# Combine temperature and ice cream sales into a single feature matrix X
X = np.hstack((temperature, ice_cream_sales))

# Daily profit (target variable y)
# Assume that daily profit depends on the temperature, ice cream sales,
# and some random noise (normal distribution)
daily_profit = 1000 + 30 * temperature + 5 * ice_cream_sales + np.random.randn(100, 1)

# Now, you have synthetic data to analyze and build a predictive model for daily profit based on temperature and ice cream sales.

# Perform mean normalization (standardization) on input features

#!!! create a function and scale our X and return X_scaled
#!!! do this for whole method like max, mean and z-score 
#!!! compare the result

def z_score_func(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def max_func(X):
    X_scaled = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    return X_scaled

def mean_func(X):
    X_scaled = (X - X.mean(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    return X_scaled

z_score_X = z_score_func(X)
max_function_X = max_func(X)
mean_func_X = mean_func(X)


# Split the data into training and testing sets
X_train_Z_score, X_test_Z_score, y_train, y_test = train_test_split(z_score_X, daily_profit, test_size=0.2, random_state=42)
X_train_max, X_test_max, y_train, y_test = train_test_split(max_function_X, daily_profit, test_size=0.2, random_state=42)
X_train_mean, X_test_mean, y_train, y_test = train_test_split(mean_func_X, daily_profit, test_size=0.2, random_state=42)

# Rest of your code for linear regression, gradient descent, and evaluation
# Initialize the weights and bias terms closer to zero
n_features = X_train_Z_score.shape[1]
w = np.zeros((n_features, 1))  # Match the shape of the weights to (n_features, 1)
b = 0

# Use a smaller learning rate for smoother convergence
alpha = 0.01  # Learning rate
num_iterations = 1000

# Implement gradient descent
m = len(X_train_Z_score)  # Number of training data points

for iteration in range(num_iterations):
    # Compute predictions
    y_pred = np.dot(X_train_Z_score, w) + b

    # Compute gradients
    w_gradient = (1 / m) * np.dot(X_train_Z_score.T, (y_pred - y_train))
    b_gradient = (1 / m) * np.sum(y_pred - y_train)

    # Update weights and bias
    w -= alpha * w_gradient.reshape(-1, 1)  # Reshape to match the shape of w
    b -= alpha * b_gradient

# Make predictions on the test data
y_pred_Z_score = np.dot(X_test_Z_score, w) + b
y_pred_mean = np.dot(X_test_mean, w) + b
y_pred_max = np.dot(X_test_max, w) + b

# Evaluate the model using Mean Squared Error (MSE)
mse_Z_score = np.mean((y_pred_Z_score - y_test) ** 2)
mse_mean = np.mean((y_pred_mean - y_test) ** 2)
mse_max = np.mean((y_pred_max - y_test) ** 2)
print("Mean Squared Error for Z_score Method:", mse_Z_score)
print("Mean Squared Error for Mean Method:", mse_mean)
print("Mean Squared Error for Max Method:", mse_max)

# Plot one feature (e.g., x1) against y_test
plt.scatter(X_test_Z_score[:, 0], X_test_Z_score[:, 1], label='Test Data Points', alpha=0.7)
plt.scatter(X_test_max[:, 0], X_test_max[:, 1], label='Test Data Points', alpha=0.7)
plt.scatter(X_test_mean[:, 0], X_test_mean[:, 1], label='Test Data Points', alpha=0.7)

plt.xlabel('Feature x1')
plt.ylabel('Feature x2')
plt.legend()
plt.title('Linear Regression with Gradient Descent - Synthetic Data and 3 Methods')
plt.show()
# Print the learned coefficients
print("Learned Coefficients (Weights):")
print(w)
print("Bias (Intercept):", b)