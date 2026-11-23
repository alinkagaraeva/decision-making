import numpy as np
import weakref
from scipy import sparse
from scipy.sparse.linalg import svds

from interfaces import LossFunction, LossFunctionClosedFormMixin, LinearRegressionInterface, AbstractOptimizer
from descents import AnalyticSolutionOptimizer


def mean_gradient(X, residual):
    if sparse.issparse(X):
        return np.asarray(X.T @ residual).ravel() / X.shape[0]
    return np.einsum('...nd,...n->...d', X, residual) / X.shape[-2]


class MSELoss(LossFunction, LossFunctionClosedFormMixin):
    def __init__(self, analytic_solution_func=None):
        self.analytic_solution_func = analytic_solution_func or self._plain_analytic_solution

    def loss(self, X, y, w):
        residual = X @ w - y
        return float(np.mean(residual ** 2))

    def gradient(self, X, y, w):
        return 2 * mean_gradient(X, X @ w - y)

    def analytic_solution(self, X, y):
        return self.analytic_solution_func(X, y)

    @classmethod
    def _plain_analytic_solution(cls, X, y):
        if sparse.issparse(X):
            X = X.toarray()
        gram = X.T @ X
        if np.linalg.cond(gram) > 1 / np.finfo(float).eps:
            return np.linalg.lstsq(X, y, rcond=None)[0]
        return np.linalg.solve(gram, X.T @ y)

    @classmethod
    def _svd_analytic_solution(cls, X, y):
        if sparse.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=float)
        size = min(X.shape)
        if size == 1:
            norm_squared = np.sum(X ** 2)
            return np.zeros(X.shape[1]) if norm_squared == 0 else X.T @ y / norm_squared
        if not np.any(X):
            return np.zeros(X.shape[1])

        U, singular_values, Vt = svds(X, k=size - 1, tol=0, solver='arpack', rng=42)
        threshold = np.finfo(float).eps * max(X.shape) * singular_values.max()
        nonzero = singular_values > threshold
        solution = Vt[nonzero].T @ ((U[:, nonzero].T @ y) / singular_values[nonzero])

        residual = X - (U * singular_values) @ Vt
        column = residual[:, np.argmax(np.linalg.norm(residual, axis=0))]
        column_norm = np.linalg.norm(column)
        if column_norm > 10 * threshold:
            last_u = column / column_norm
            last_row = last_u @ residual
            last_sigma = np.linalg.norm(last_row)
            solution += last_row * (last_u @ y) / last_sigma ** 2
        return solution


class L2Regularization(LossFunction):
    def __init__(self, core_loss, mu_rate=1.0, analytic_solution_func=None):
        self.core_loss = core_loss
        self.mu_rate = mu_rate
        self.analytic_solution_func = analytic_solution_func

    def loss(self, X, y, w):
        return self.core_loss.loss(X, y, w) + self.mu_rate / 2 * np.sum(w ** 2)

    def gradient(self, X, y, w):
        return self.core_loss.gradient(X, y, w) + self.mu_rate * w


class LogCoshLoss(LossFunction):
    def loss(self, X, y, w):
        residual = X @ w - y
        return float(np.mean(np.logaddexp(residual, -residual) - np.log(2)))

    def gradient(self, X, y, w):
        return mean_gradient(X, np.tanh(X @ w - y))


class HuberLoss(LossFunction):
    def __init__(self, delta=1.0):
        if delta <= 0:
            raise ValueError('delta должна быть положительной')
        self.delta = delta

    def loss(self, X, y, w):
        residual = X @ w - y
        absolute_residual = np.abs(residual)
        values = np.where(
            absolute_residual <= self.delta,
            residual ** 2 / 2,
            self.delta * (absolute_residual - self.delta / 2)
        )
        return float(np.mean(values))

    def gradient(self, X, y, w):
        residual = np.clip(X @ w - y, -self.delta, self.delta)
        return mean_gradient(X, residual)


class CustomLinearRegression(LinearRegressionInterface):
    def __init__(self, optimizer: AbstractOptimizer, loss_function=None):
        self.optimizer = optimizer
        self.optimizer.set_model(weakref.proxy(self))
        self.loss_function = loss_function if loss_function is not None else MSELoss()
        self.loss_history = []
        self.w = None
        self.X_train = None
        self.y_train = None

    def predict(self, X):
        if self.w is None:
            raise ValueError('Сначала обучите модель')
        return np.asarray(X @ self.w)

    def compute_gradients(self, X_batch=None, y_batch=None):
        X_batch = self.X_train if X_batch is None else X_batch
        y_batch = self.y_train if y_batch is None else y_batch
        return self.loss_function.gradient(X_batch, y_batch, self.w)

    def compute_loss(self, X_batch=None, y_batch=None):
        X_batch = self.X_train if X_batch is None else X_batch
        y_batch = self.y_train if y_batch is None else y_batch
        return self.loss_function.loss(X_batch, y_batch, self.w)

    def fit(self, X, y):
        X = X.tocsr().astype(float, copy=False) if sparse.issparse(X) else np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2 or X.shape[0] != y.size or y.size == 0:
            raise ValueError('Проверьте размеры X и y')
        self.X_train, self.y_train = X, y
        self.w = np.zeros(X.shape[1])
        self.loss_history = []
        self.optimizer.set_model(weakref.proxy(self))
        self.optimizer.optimize()
        return self
