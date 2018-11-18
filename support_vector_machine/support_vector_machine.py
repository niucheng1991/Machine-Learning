import cvxopt
import numpy as np

class SVM(object):
    def __init__(self, C=1, kernel=rbf_kernel, power=4, gamma=None, coef=4):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.power = power
        self.coef = coef
        self.lagr_multipliers = None
        self.support_vectors = None
        self.support_vectors_labels = None
        self.intercept = None

    def fit(self, X, y):
        n_samples, n_features = np.shape(X)

        if not self.gamma:
            self.gamma = 1 / n_features

        self.kernel = self.kernel(
            power = self.power,
            gamma = self.gamma,
            coef = self.coef
        )

        # kernel矩阵
        kernel_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i, j] = self.kernel(X[i], X[j])

        # 定义二次优化问题
        P = cvxopt.matrix(np.outer(y, y) * kernel_matrix, tc='d')
        q = cvxopt.matrix(np.outer(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples), tc='d')
        b = cvxopt.matrix(0, tc='d')

        if not self.C:
            G = cvxopt.matrix(np.id)


        minimization = cvxopt.solvers.qp(P, q, G, h, A, b)

        # 拉格朗日乘子
        lagr_mult = np.ravel(minimization['x'])

        # 提取支持向量
        idx = lagr_mult > 1e-7

        # 获取大于零值的乘子
        self.lagr_multipliers = lagr_mult[idx]

        # 获取大于零的乘子对应的支持向量
        self.support_vectors = X[idx]

        # 获取支持向量的标签值
        self.support_vectors_labels = y[idx]

        # 用第一个支持向量对象的标签值计算偏置值
        self.intercept = self.support_vectors_labels[0]
        for i in range(len(self.lagr_multipliers)):
            self.intercept -= self.lagr_multipliers[i] * self.support_vectors_labels[i] * \
                self.kernel(self.support_vectors[i], self.support_vectors[0])



        def predict(self, X):
            y_pred = []

            for sampel in X:
                prediction = 0
                for i in range(len(self.lagr_multipliers)):
                    prediction += self.lagr_multipliers[i] * 