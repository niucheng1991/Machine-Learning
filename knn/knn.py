#-*- coding: utf-8 -*-
import numpy as np

class KNN():
    def __init__(self, k=6):
        self.k = k

    def predict(self, X_test, X_train, y_train):
        y_pred = np.empty(X_test.shape[0])

        m_sample = np.shape(X_train[0])
        distance = []

        for i, test_sample in enumerate(X_test):
            for train_sample in X_train:
                # 计算两个样本的欧式距离
                distance.append(euclidean_distance(test_sample, train_sample))

            # 排序并获取排序好的前K个下标序号
            idx = np.argsort(distance)[:self.k]

            # K个进邻目标标签值
            k_nearest_neighbors = np.array([y_train[j] for j in idx])

            # 投票最多的进邻值
            counts = np.bincount(k_nearest_neighbors.astype('int'))
            y_pred[i] = np.argmax(counts)
            distance = []

        return y_pred
