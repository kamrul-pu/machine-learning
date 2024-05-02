import numpy as np


class GaussianNaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.mean = np.zeros((self.n_classes, n_features))
        self.var = np.zeros((self.n_classes, n_features))
        self.priors = np.zeros(self.n_classes)

        # Compute mean, variance, and priors for each class
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[i, :] = X_c.mean(axis=0)
            self.var[i, :] = X_c.var(axis=0)
            self.priors[i] = len(X_c) / float(n_samples)

    def predict(self, X):
        # Calculate posterior probabilities for each class
        posteriors = np.zeros((X.shape[0], self.n_classes))

        for i, c in enumerate(self.classes):
            prior = np.log(self.priors[i])
            posterior = np.sum(
                np.log(self.pdf(X, self.mean[i, :], self.var[i, :])), axis=1
            )
            posteriors[:, i] = prior + posterior

        # Return class with the highest posterior probability
        return self.classes[np.argmax(posteriors, axis=1)]

    def pdf(self, X, mean, var):
        # Compute Gaussian probability density function
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((X - mean) ** 2) / (2 * var))


# Example usage:
# Generate synthetic data
np.random.seed(0)
X = np.random.randn(100, 2)
y = np.random.choice([0, 1], size=100)

# Initialize and fit Gaussian Naive Bayes model
model = GaussianNaiveBayes()
model.fit(X, y)

# Predict classes for new data
X_new = np.array([[0, 0], [1, 1]])
y_pred = model.predict(X_new)

print("Predictions for new data:", y_pred)
