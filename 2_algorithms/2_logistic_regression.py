import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the logistic regression model.

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

    def sigmoid(self, z):
        """
        Compute the sigmoid function.

        Parameters:
        - z (float or ndarray): Input value(s) to the sigmoid function.

        Returns:
        - float or ndarray: Output of the sigmoid function.
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent.

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
            # Compute predicted probabilities
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            # Compute gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        Predict binary classes (0 or 1) for input features X.

        Parameters:
        - X (ndarray): Input features of shape (num_samples, num_features).

        Returns:
        - ndarray: Predicted binary labels (0 or 1) of shape (num_samples,).
        """
        # Compute predicted probabilities
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)

        # Convert probabilities to binary predictions
        y_pred_class = [1 if p >= 0.5 else 0 for p in y_pred]
        return np.array(y_pred_class)

    def accuracy(self, y_true, y_pred):
        """
        Calculate accuracy of model predictions.

        Parameters:
        - y_true (ndarray): True labels of shape (num_samples,).
        - y_pred (ndarray): Predicted labels of shape (num_samples,).

        Returns:
        - float: Accuracy score (proportion of correct predictions).
        """
        # Calculate accuracy
        accuracy = np.mean(y_true == y_pred)
        return accuracy


# Example usage:
if __name__ == "__main__":
    # Sample dataset
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y_train = np.array([0, 0, 1, 1, 1])
    X_test = np.array([[1, 1], [2, 2], [5, 5]])
    y_test = np.array([0, 0, 1])

    # Initialize and train logistic regression model
    model = LogisticRegression(learning_rate=0.01, num_iterations=1000)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Evaluate model performance
    train_accuracy = model.accuracy(y_train, y_pred_train)
    test_accuracy = model.accuracy(y_test, y_pred_test)

    print(f"Training accuracy: {train_accuracy:.2f}")
    print(f"Test accuracy: {test_accuracy:.2f}")
