import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class SVM:
    def __init__(self, C=1.0, tol=1e-3, max_iter=100):
        self.C = C  # Regularization parameter
        self.tol = tol  # Tolerance for stopping criteria
        self.max_iter = max_iter  # Maximum number of iterations
        self.alpha = None  # Lagrange multipliers (dual variables)
        self.b = 0  # Intercept
        self.support_vectors = None  # Support vectors
        self.support_vector_labels = None  # Labels of support vectors

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize alpha (Lagrange multipliers) and b (intercept)
        self.alpha = np.zeros(n_samples)
        self.b = 0

        # Training loop
        for _ in range(self.max_iter):
            alpha_prev = np.copy(self.alpha)

            for i in range(n_samples):
                # Select random index j != i
                j = np.random.choice(np.delete(np.arange(n_samples), i))

                # Compute kernel values and error for pair (i, j)
                K_ij = np.dot(X[i], X[j])
                E_i = self._decision_function(X[i]) - y[i]
                E_j = self._decision_function(X[j]) - y[j]

                # Compute bounds L and H for alpha[j] using current alpha values
                L, H = self._compute_bounds(self.alpha[j], self.alpha[i], y[i], y[j])

                if L == H:
                    continue

                # Compute eta (2*K_ij - K_ii - K_jj)
                eta = 2 * K_ij - np.dot(X[i], X[i]) - np.dot(X[j], X[j])

                if eta >= 0:
                    continue

                # Update alpha[j]
                self.alpha[j] -= y[j] * (E_i - E_j) / eta

                # Clip alpha[j] using bounds L and H
                self.alpha[j] = np.clip(self.alpha[j], L, H)

                # Update alpha[i]
                self.alpha[i] += y[i] * y[j] * (alpha_prev[j] - self.alpha[j])

            # Check convergence
            if np.linalg.norm(self.alpha - alpha_prev) < self.tol:
                break

        # Extract support vectors and their labels
        support_vector_indices = np.where(self.alpha > 0)[0]
        self.support_vectors = X[support_vector_indices]
        self.support_vector_labels = y[support_vector_indices]

        # Compute intercept b
        self.b = np.mean(
            self.support_vector_labels - self._decision_function(self.support_vectors)
        )

    def predict(self, X):
        return np.sign(self._decision_function(X) + self.b)

    def _decision_function(self, X):
        # Compute the decision function f(x) = sum(alpha_i * y_i * K(x_i, x)) + b
        return np.dot(
            self.alpha * self.support_vector_labels, np.dot(self.support_vectors, X.T)
        )

    def _compute_bounds(self, alpha_j, alpha_i, y_i, y_j):
        if y_i != y_j:
            L = max(0, alpha_j - alpha_i)
            H = min(self.C, self.C + alpha_j - alpha_i)
        else:
            L = max(0, alpha_i + alpha_j - self.C)
            H = min(self.C, alpha_i + alpha_j)
        return L, H


# Example usage:
if __name__ == "__main__":
    # Generate synthetic data
    X, y = make_classification(
        n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42
    )
    y = np.where(y == 0, -1, 1)  # Convert labels to -1 and 1

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the SVM classifier
    svm = SVM(C=1.0, tol=1e-3, max_iter=100)
    svm.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred_train = svm.predict(X_train_scaled)
    y_pred_test = svm.predict(X_test_scaled)

    # Evaluate accuracy
    train_accuracy = np.mean(y_pred_train == y_train)
    test_accuracy = np.mean(y_pred_test == y_test)
    print(f"Train Accuracy: {train_accuracy:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")
