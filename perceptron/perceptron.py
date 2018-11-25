import math
import numpy as np
from utils import Sigmoid, ReLU
from utils import SquareLoss, CrossEntropy

class Perceptron():
    def __init__(self, n_iterations=1500, activation_function=Sigmoid, loss=SquareLoss, learning_rate=0.01):
        self.n_itertions = n_iterations
        self.activation_function = activation_function()
        self.learning_rate = learning_rate
        self.loss = loss()

    def fit(self, X, y):
        n_sample, n_features = np.shape(X)
        _, n_outputs = np.shape(y)

        # 初始化权重
        limit = 1 / math.sqrt(n_features)
        self.W = np.random.uniform(-limit, limit, (n_features, n_outputs))
        self.W0 = np.zeros((1, n_outputs))

        for i in range(self.n_itertions):
            linear_output = X.dot(self.W) + self.W0
            y_pred = self.activation_function(linear_output)
            error_gradient = self.loss.gradient(y, y_pred) *  \
                            self.activation_function.gradient(linear_output)

            grad_w = X.T.dot(error_gradient)
            grad_w0 = np.sum(error_gradient, axis=0, keepdims=True)

            self.W  -= self.learning_rate * grad_w
            self.W0 -= self.learning_rate * grad_w0

    def fit(self, X):
        y_pred = self.activation_function(X.dot(self.W) + self.W0)
        return y_pred
