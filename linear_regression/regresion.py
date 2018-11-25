#-*- coding: utf-8 -*-
import math
import numpy as np

# 回归基类
class Regression(object):

    def __init__(self, n_iter, learning_rate):
        self.n_iter = n_iter
        self.learning_rate = learning_rate

    def initialize_weights(self, n_features):
        """
        随机初始化权重参数[-1 / N, 1 / N]
        """
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, (n_features))

    def fit(self, X, y):

        # 插入偏置值列
        X = np.insert(X, 0, 1, axis=1)

        # 训练误差集合
        self.training_errors = []

        # 初始化权重
        self.initialize_weights(n_features=X.shape[1])

        # 迭代训练
        for i in range(self.n_iter):

            # 计算预测值
            y_pred = X.dot(self.W)

            # 计算损失值
            mse = np.mean(0.5 * (y - y_pred)**2 + self.regularization(self.w))

            # 保存每轮迭代mse值
            self.training_errors.append(mse)

            # 计算损失值梯度
            grad_w = -(y - y_pred).dot(X) + self.regularization.grad(self.w)

            # 更新权重
            self.w = sgd(self.w, grad_w, self.learning_rate)

        def predict(self, X):
            X = np.insert(X, 0, 1, axis=1)
            y_pred = X.dot(self.w)
            return y_pred


# 线性回归
class Linear_Regression(Regression):

    """
    n_iter: 迭代次数
    learning_rate: 学习速率
    gradient_descent: True: 使用梯度下降法   False:使用正规方程法
    """
    def __init__(self, n_iter=120, learning_rate=0.001, gradient_descent=True):
        self.gradient_descent = gradient_descent
        self.regularization = lambda x:0
        self.regularization.grad = lambda x:0
        super(Linear_Regression, self).__init__(n_iter, learning_rate=learning_rate)

    def fit(self, X, y):
        # 使用梯度下降法
        if(self.gradient_descent):
            super(Linear_Regression, self).fit(X, y)
        else:
            X = np.insert(X, 0, 1, axis=1)
            # 使用正规方程求加闭式解
            self.w = normal_equation(X, y)

# Lasso回归
class Lasso_Regression(Regression):
    def __init__(self, degree, reg_factor, n_iter=1000, learning_rate=0.0011):
        self.degree = degree
        self.regularization = l1_regularization(alpha=reg_factor)
        super(Lasso_Regression, self).__init__(n_iter=n_iter, learning_rate=learning_rate)

    def fit(self, X, y):
        X = normalize(polynomial_features(X, degree=self.degree))
        super(Lasso_Regression, self).fit(X, y)

    def predict(self, X):
        X = normalize(polynomial_features(X, degree=self.degree))
        return super(Lasso_Regression, self).predict(X)


# Ridge回归
class Ridge_Regression(Regression):
    def __init__(self, degree, reg_factor, n_iter=1000, learning_rate=0.001):
        self.degree = degree
        self.regularization = l2_regularization(alpha=reg_factor)
        super(Ridge_Regression, self).__init__(n_iter=n_iter, learning_rate=learning_rate)

    def fit(self, X, y):
        X = normalize(polynomial_features(X, degree=self.degree))
        super(Ridge_Regression, self).fit(X, y)

    def predict(self, X):
        return super(Ridge_Regression, self).predict(X)



# 局部加权线性回归 (没测试)
class LocallyWeight_Regression(Regression):
    def __init__(self, k=0.01):
        self.k = k

    def fit(self, point_X, X, y):

        m_sample = shape(X)[0]













