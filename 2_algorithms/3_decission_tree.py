import numpy as np


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  # Index of feature to split on
        self.threshold = threshold  # Threshold value for binary split
        self.left = left  # Left subtree (Node)
        self.right = right  # Right subtree (Node)
        self.value = value  # Value if leaf node (prediction)


class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth  # Maximum depth of the tree
        self.tree = None  # Root of the decision tree

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        # Stopping criteria: if pure or max_depth reached
        if depth == self.max_depth or np.unique(y).size == 1:
            value = np.bincount(y).argmax()  # Majority class
            return Node(value=value)

        # Find best split (feature, threshold) using Gini impurity
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold
                left_gini = self._gini_impurity(y[left_indices])
                right_gini = self._gini_impurity(y[right_indices])
                total_gini = (
                    left_gini * np.sum(left_indices)
                    + right_gini * np.sum(right_indices)
                ) / len(y)
                gain = self._gini_impurity(y) - total_gini

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        if best_gain == 0:
            value = np.bincount(y).argmax()  # Majority class
            return Node(value=value)

        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold
        left_child = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
        )

    def _gini_impurity(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities**2)
        return gini

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)


# Example usage:
if __name__ == "__main__":
    # Create a synthetic dataset for binary classification
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y = np.array([0, 0, 1, 1, 1])

    # Initialize and train the Decision Tree classifier
    clf = DecisionTreeClassifier(max_depth=2)
    clf.fit(X, y)

    # Make predictions
    X_test = np.array([[2, 3], [4, 5]])
    predictions = clf.predict(X_test)
    print("Predictions:", predictions)
