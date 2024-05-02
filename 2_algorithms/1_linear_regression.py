import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the linear regression model.

        Parameters:
        - learning_rate (float): The learning rate used in gradient descent.
        - num_iterations (int): The number of iterations for gradient descent.

        Attributes:
        - weights (ndarray): Coefficients (weights) for each feature.
        - bias (float): Intercept term.
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Train the linear regression model using gradient descent.

        Parameters:
        - X (ndarray): Input features of shape (num_samples, num_features).
        - y (ndarray): Target labels of shape (num_samples,).

        Returns:
        - None (updates model weights and bias in-place).
        """
        # Initialize weights and bias
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Gradient descent for optimization
        for _ in range(self.num_iterations):
            # Compute predicted values
            y_pred = np.dot(X, self.weights) + self.bias

            # Compute gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        Predict target values for input features X.

        Parameters:
        - X (ndarray): Input features of shape (num_samples, num_features).

        Returns:
        - ndarray: Predicted target values of shape (num_samples,).
        """
        # Compute predicted values
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred

    def mean_squared_error(self, y_true, y_pred):
        """
        Calculate the mean squared error (MSE) between true and predicted values.

        Parameters:
        - y_true (ndarray): True target values of shape (num_samples,).
        - y_pred (ndarray): Predicted target values of shape (num_samples,).

        Returns:
        - float: Mean squared error (MSE) between true and predicted values.
        """
        # Calculate mean squared error
        mse = np.mean((y_true - y_pred) ** 2)
        return mse


# Example usage:
if __name__ == "__main__":
    # Sample dataset
    X_train = np.array([[1], [2], [3], [4], [5]])
    y_train = np.array([2, 4, 5, 4, 5])
    X_test = np.array([[6], [7]])
    y_test = np.array([6, 7])

    # Initialize and train linear regression model
    model = LinearRegression(learning_rate=0.01, num_iterations=1000)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Evaluate model performance
    train_mse = model.mean_squared_error(y_train, y_pred_train)
    test_mse = model.mean_squared_error(y_test, y_pred_test)

    print(f"Training MSE: {train_mse:.2f}")
    print(f"Test MSE: {test_mse:.2f}")
