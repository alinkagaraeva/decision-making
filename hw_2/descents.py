import numpy as np
from abc import ABC, abstractmethod
from scipy import sparse
from interfaces import LearningRateSchedule, AbstractOptimizer, LinearRegressionInterface


class ConstantLR(LearningRateSchedule):
    def __init__(self, lr):
        self.lr = lr

    def get_lr(self, iteration):
        return self.lr


class TimeDecayLR(LearningRateSchedule):
    def __init__(self, lambda_=1.0):
        self.s0 = 1
        self.p = 0.5
        self.lambda_ = lambda_

    def get_lr(self, iteration):
        return self.lambda_ * (self.s0 / (self.s0 + iteration)) ** self.p


class BaseDescent(AbstractOptimizer, ABC):
    def __init__(self, lr_schedule=None, tolerance=1e-6, max_iter=1000):
        self.lr_schedule = lr_schedule if lr_schedule is not None else TimeDecayLR()
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.iteration = 0
        self.model: LinearRegressionInterface = None
        self.stop_reason = None

    @abstractmethod
    def _update_weights(self):
        pass

    def _reset_state(self):
        self.iteration = 0
        self.stop_reason = None

    def _step(self):
        delta = self._update_weights()
        self.iteration += 1
        return delta

    def optimize(self):
        self._reset_state()
        self.model.loss_history = [self.model.compute_loss()]
        self.model.weight_norm_history = [float(np.dot(self.model.w, self.model.w))]
        self.stop_reason = 'max_iter'
        for _ in range(self.max_iter):
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                delta = self._step()
                self.model.loss_history.append(self.model.compute_loss())
                self.model.weight_norm_history.append(float(np.dot(self.model.w, self.model.w)))
                if not np.isfinite(delta).all() or not np.isfinite(self.model.loss_history[-1]):
                    self.stop_reason = 'nonfinite'
                    break
                if np.dot(delta, delta) < self.tolerance:
                    self.stop_reason = 'tolerance'
                    break


class VanillaGradientDescent(BaseDescent):
    def _update_weights(self):
        gradient = self.model.compute_gradients()
        delta = -self.lr_schedule.get_lr(self.iteration) * gradient
        self.model.w += delta
        return delta


class StochasticGradientDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        if batch_size <= 0:
            raise ValueError('batch_size должен быть положительным')
        self.batch_size = batch_size

    def _update_weights(self):
        indices = np.random.randint(self.model.X_train.shape[0], size=self.batch_size)
        gradient = self.model.compute_gradients(self.model.X_train[indices], self.model.y_train[indices])
        delta = -self.lr_schedule.get_lr(self.iteration) * gradient
        self.model.w += delta
        return delta


class SAGDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        if batch_size <= 0:
            raise ValueError('batch_size должен быть положительным')
        self.batch_size = batch_size
        self.grad_memory = None
        self.grad_sum = None
        self.avg_grad = None

    def _reset_state(self):
        super()._reset_state()
        self.grad_memory = None
        self.grad_sum = None
        self.avg_grad = None

    def _update_weights(self):
        X_train = self.model.X_train
        y_train = self.model.y_train
        num_objects, num_features = X_train.shape
        if self.grad_memory is None:
            self.grad_memory = np.zeros((num_objects, num_features))
            self.grad_sum = np.zeros(num_features)

        indices = np.unique(np.random.randint(num_objects, size=self.batch_size))
        X_batch = X_train[indices]
        if sparse.issparse(X_batch):
            X_batch = X_batch.toarray()
        gradients = self.model.compute_gradients(X_batch[:, None, :], y_train[indices, None])
        self.grad_sum += np.sum(gradients - self.grad_memory[indices], axis=0)
        self.grad_memory[indices] = gradients
        self.avg_grad = self.grad_sum / num_objects
        delta = -self.lr_schedule.get_lr(self.iteration) * self.avg_grad
        self.model.w += delta
        return delta


class MomentumDescent(BaseDescent):
    def __init__(self, *args, beta=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.velocity = None

    def _reset_state(self):
        super()._reset_state()
        self.velocity = None

    def _update_weights(self):
        if self.velocity is None:
            self.velocity = np.zeros_like(self.model.w)
        gradient = self.model.compute_gradients()
        self.velocity = self.beta * self.velocity + self.lr_schedule.get_lr(self.iteration) * gradient
        delta = -self.velocity
        self.model.w += delta
        return delta


class Adam(BaseDescent):
    def __init__(self, *args, beta1=0.9, beta2=0.999, eps=1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None

    def _reset_state(self):
        super()._reset_state()
        self.m = None
        self.v = None

    def _update_weights(self):
        if self.m is None:
            self.m = np.zeros_like(self.model.w)
            self.v = np.zeros_like(self.model.w)
        gradient = self.model.compute_gradients()
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient ** 2
        step = self.iteration + 1
        m_hat = self.m / (1 - self.beta1 ** step)
        v_hat = self.v / (1 - self.beta2 ** step)
        delta = -self.lr_schedule.get_lr(self.iteration) * m_hat / (np.sqrt(v_hat) + self.eps)
        self.model.w += delta
        return delta


class AnalyticSolutionOptimizer(AbstractOptimizer):
    def __init__(self):
        self.model = None

    def optimize(self):
        self.model.loss_history = [self.model.compute_loss()]
        self.model.w = self.model.loss_function.analytic_solution(self.model.X_train, self.model.y_train)
        self.model.loss_history.append(self.model.compute_loss())
