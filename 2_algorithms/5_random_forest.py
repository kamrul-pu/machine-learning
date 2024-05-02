import numpy as np


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        unique_classes = np.unique(y)

        # Base case: if node has only one class
        if len(unique_classes) == 1:
            return unique_classes[0]

        # Base case: if max depth is reached
        if self.max_depth is not None and depth == self.max_depth:
            return np.bincount(y).argmax()

        # Find best split
        best_feature, best_threshold = self._find_best_split(X, y)

        # Split data based on best feature and threshold
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = X[:, best_feature] > best_threshold

        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return (best_feature, best_threshold, left_subtree, right_subtree)

    def _find_best_split(self, X, y):
        n_samples, n_features = X.shape
        best_gini = float("inf")
        best_feature = None
        best_threshold = None

        for feature in range(n_features):
            unique_values = np.unique(X[:, feature])
            for threshold in unique_values:
                left_mask = X[:, feature] <= threshold
                right_mask = X[:, feature] > threshold

                gini = self._gini_impurity(y[left_mask], y[right_mask])
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _gini_impurity(self, left_y, right_y):
        p_left = len(left_y) / (len(left_y) + len(right_y))
        p_right = len(right_y) / (len(left_y) + len(right_y))

        gini_left = 1 - np.sum((np.bincount(left_y) / len(left_y)) ** 2)
        gini_right = 1 - np.sum((np.bincount(right_y) / len(right_y)) ** 2)

        return p_left * gini_left + p_right * gini_right

    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x, tree):
        if isinstance(tree, np.int64):
            return tree

        feature, threshold, left_subtree, right_subtree = tree
        if x[feature] <= threshold:
            return self._predict_tree(x, left_subtree)
        else:
            return self._predict_tree(x, right_subtree)


class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            tree = DecisionTree(max_depth=self.max_depth)
            indices = np.random.choice(len(X), size=len(X), replace=True)
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(predictions, axis=0).astype(int)


# Example usage:
np.random.seed(42)
X = np.random.rand(100, 2)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

rf = RandomForest(n_estimators=10, max_depth=3)
rf.fit(X, y)

test_data = np.array([[0.5, 0.5], [0.8, 0.2]])
predictions = rf.predict(test_data)
print("Predictions:", predictions)
