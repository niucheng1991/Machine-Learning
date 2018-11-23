#-*- coding: utf-8 -*-
import sys
import numpy as np
import math

class Logistic_Regression(object):
    def __init__(self, n_iter=1000, config=None):
        self.n_iter = n_iter
        self.sigmoid = Sigmoid()
        self.config = config
        if self.config is None:
            self.config = {}
            self.config.setdefault('learning_rate', 1e-2)
            self.config.setdefault('optimizaType', 'gradient_descent')

    def _initialize_paramters(self, X):
        n_features = np.shape(X)[1]
        limit = 1 / math.sqrt(n_features)
        self.param = np.random.uniform(-limit, limit, (n_features,))

    def fit(self, X, y):
        # 初始化权重
        self._initialize_paramters(X)

        m_sample = np.shape(X)[0]

        for i in range(self.n_iter):
            if self.config['optimizeType'] == 'gradient_descent':
                y_pred = self.sigmoid(X.dot(self.param))
                self.param -= self.config['learning_rate'] * -(y - y_pred) .dot(X)

            elif self.config['optimizeType'] == 'stocgradient_descent':
                pass
            else :
                print("No support optimization type")

    def predict(self, X):
        y_pred = np.round(self.sigmoid(X.dot(self.param))).astype(int)
        return y_pred