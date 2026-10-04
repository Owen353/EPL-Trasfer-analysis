"""Linear and logistic regression implemented from scratch with NumPy.

Conventions (same as COMP 432 lab 1):
  X_tilde: bias-augmented data matrix of shape (N, D + 1), with a column of 1s on the right.
  theta:   parameter vector (w_1, ..., w_D, b) of shape (D + 1,).
"""
import numpy as np


def add_bias(X):
    """Append a column of 1s so the bias becomes just another weight."""
    X = np.asarray(X, dtype=float).reshape(len(X), -1)
    return np.column_stack([X, np.ones(len(X))])


class Standardizer:
    """Rescale each feature to mean 0, std 1.

    Raw spending is in the hundreds of millions; without rescaling the loss surface
    is a long, narrow valley and gradient descent barely moves.
    """

    def fit(self, X):
        X = np.asarray(X, dtype=float).reshape(len(X), -1)
        self.mean, self.std = X.mean(axis=0), X.std(axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).reshape(len(X), -1)
        return (X - self.mean) / self.std


# ---------- Linear regression ----------

def linear_predict(X_tilde, theta):
    return X_tilde @ theta


def mse(X_tilde, y, theta):
    return np.mean((linear_predict(X_tilde, theta) - y) ** 2)


def linear_regression_gd(X_tilde, y, theta_init=None, lr=0.1, num_steps=10_000, eps=1e-9):
    """Minimize mean squared error by gradient descent. Returns (theta, steps_taken, loss_history)."""
    theta = np.zeros(X_tilde.shape[1]) if theta_init is None else np.asarray(theta_init, float)
    history = []
    for i in range(num_steps):
        r = X_tilde @ theta - y
        grad = 2 / len(y) * X_tilde.T @ r
        theta = theta - lr * grad
        history.append(np.mean(r ** 2))
        if np.all(np.abs(grad) < eps):
            break
    return theta, i + 1, np.array(history)


def linear_regression_direct(X_tilde, y):
    """Normal equation. np.linalg.solve is more stable than forming the inverse."""
    return np.linalg.solve(X_tilde.T @ X_tilde, X_tilde.T @ y)


def r_squared(y, y_hat):
    return 1 - np.sum((y - y_hat) ** 2) / np.sum((y - np.mean(y)) ** 2)


# ---------- Logistic regression ----------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logistic_predict(X_tilde, theta):
    return sigmoid(X_tilde @ theta)


def binary_cross_entropy(X_tilde, y, theta):
    p = np.clip(logistic_predict(X_tilde, theta), 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def logistic_regression_grad(X_tilde, y, theta):
    r = logistic_predict(X_tilde, theta) - y
    return X_tilde.T @ r / len(y)


def logistic_regression_gd(X_tilde, y, theta_init=None, lr=0.5, num_steps=20_000, eps=1e-7):
    """Minimize binary cross-entropy by gradient descent. Returns (theta, steps_taken, loss_history)."""
    theta = np.zeros(X_tilde.shape[1]) if theta_init is None else np.asarray(theta_init, float)
    history = []
    for i in range(num_steps):
        grad = logistic_regression_grad(X_tilde, y, theta)
        theta = theta - lr * grad
        history.append(binary_cross_entropy(X_tilde, y, theta))
        if np.all(np.abs(grad) < eps):
            break
    return theta, i + 1, np.array(history)
