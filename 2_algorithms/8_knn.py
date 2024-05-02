import numpy as np
from collections import Counter


class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]

        # Get indices of the k-nearest examples
        k_indices = np.argsort(distances)[: self.k]

        # Get the labels of the k-nearest examples
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Return the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


# Example usage:
# Generate synthetic data
np.random.seed(0)
X = np.random.rand(20, 2)  # 20 samples, 2 features
y = np.array([0] * 10 + [1] * 10)  # Binary classification labels

# Split data into training and test sets
X_train, X_test = X[:15], X[15:]
y_train, y_test = y[:15], y[15:]

# Initialize and fit KNN classifier
knn = KNNClassifier(k=3)
knn.fit(X_train, y_train)

# Predict classes for test data
y_pred = knn.predict(X_test)

print("Predictions for test data:", y_pred)
