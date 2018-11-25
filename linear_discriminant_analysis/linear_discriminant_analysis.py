import numpy as np
import math
from utils import calculate_covariance_matrix, normalize, standardize

# 二分类LDA
class TwoClassLDA():
    def __init__(self):
        self.w = None

    def fit(self, X, y):
        # 将数据集分两类
        X1 = X[y == 0]
        X2 = X[y == 1]

        # 计算各分类的散列矩阵
        X1_conv_mat = calculate_covariance_matrix(X1)
        X2_conf_mat = calculate_covariance_matrix(X2)

        # 散列矩阵
        SW = X1_conv_mat + X2_conf_mat

        # 均值
        X1_mean = X1.mean(0)
        X2_mean = X2.mean(0)

        diff_mean = np.atleast_1d(X1_mean - X2_mean)

        # 最佳方向w : w = SW的逆矩阵 * (diff_mean)
        self.w = np.linalg.pinv(SW).dot(diff_mean).dot(diff_mean)

    def predict(self, X):
        y_pred = []
        for sample in X:
            h = sample.dot(self.w)
            y = 1 * (h < 0)
            y_pred.append(y)
        return y_pred

# 多分类LDA
class MultiClassLDA():
    def _calculate_scatter_matrix(self, X, y):
        # 计算SW
        n_features = np.shape(X)[1]
        n_class = np.unique(y)
        SW = np.empty((n_features, n_features))

        for _class in n_class:
            class_X = X[y == _class]
            SW += calculate_covariance_matrix(class_X)

        SB = np.empty(n_features, n_features)
        tatol_mean = np.mean(X, axis=0)
        for _class in n_class:
            class_X = X[y==_class]
            class_mean = np.mean(class_X, axis=0)
            SB += len(class_X) * (class_mean - tatol_mean)

        return SW, SB

    def transform(self, X, y, n_components):
        SW, SB = self._calculate_scatter_matrix(X, y)

        eigmat = np.linalg.inv(SW).dot(SB)

        # 求解特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(eigmat)

        # 降序排序特征值
        idx = np.argsort(eigenvalues)[::-1]

        # 获取前n_component个特征值
        eigenvalues = eigenvalues[idx][:n_components]

        # 获取前n_component个特征值
        eigenvectors = eigenvalues[:, idx][:, :n_components]

        X_transformed = x.dot(eigenvectors)

        return X_transformed