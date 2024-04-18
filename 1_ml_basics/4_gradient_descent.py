"""Gradient Descent."""

from typing import List

import numpy as np


def gradient_descent(x: List[int], y: List[int]) -> None:
    # Number of data points
    n: int = len(x)

    # Initialize slope (m) and intercept (b) to zero
    m_curr = b_curr = 0

    # Number of iterations for gradient descent
    iterations = 10000

    # Learning rate for gradient descent
    learning_rate: float = 0.08

    # Gradient descent loop
    for i in range(iterations):
        # Predicted values of y (y = mx + b)
        y_predicted = m_curr * x + b_curr

        # Calculate mean squared error (cost function)
        cost = (1 / n) * sum(
            [(val - y_pred) ** 2 for val, y_pred in zip(y, y_predicted)]
        )

        # Partial derivatives of the cost function with respect to m and b
        md = -(2 / n) * sum(x * (y - y_predicted))  # Derivative w.r.t. m
        bd = -(2 / n) * sum(y - y_predicted)  # Derivative w.r.t. b

        # Update m and b using gradient descent formula
        m_curr = m_curr - learning_rate * md  # Update m
        b_curr = b_curr - learning_rate * bd  # Update b

        # Print current values of m, b, cost, and iteration number
        print(f"m {m_curr} b {b_curr}, cost {cost} iteration {i}")


if __name__ == "__main__":
    # Input data points (x and y arrays)
    x: List[int] = np.array([1, 2, 3, 4, 5])
    y: List[int] = np.array([5, 7, 9, 11, 13])

    # Perform gradient descent to fit a linear regression model
    gradient_descent(x=x, y=y)
