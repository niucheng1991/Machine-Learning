#-*- coding: utf-8 -*-
import numpy as np
from utils.distance import euclidean_distance

class KMeans():
    def __init__(self, k, n_iters=1000):
        self.k = k
        self.n_iters = n_iters

    # 初始化聚类重心点
    def _init_random_centroids(self, X):
        n_feature = np.shape(X)[1]
        n_samples = np.shape(X)[0]
        centroids = np.zeros((self.K, n_feature))
        for i in range(self.K):
            centroids[i] = X[np.random.choice(range(n_samples))]

        return centroids

    # 计算聚类重心点
    def _calculate_centroids(self, clusters, X):
        n_feature = np.shape(X)[1]
        centroids = np.zeros((self.k, n_feature))
        for i, cluster in enumerate(clusters):
            centroid = np.mean(X[cluster], axis=0)
            centroids[i] = centroid
        return centroids

    def _closest_centroid(self, sample, centroids):
        closest_i = 0
        closes_dist = float("inf")
        for i, centroid in enumerate(centroids):
            distance = euclidean_distance(sample, centroid)
            if distance < closes_dist:
                closes_dist = distance
                closest_i = i
        return closest_i

    def _create_cluster(self, centroids, X):
        clusters = [[] for _ in range(self.k)]
        for sample_i, sample in enumerate(X):
            centroids_i = selfr._closest_centroid(sample, centroids)
            clusters[centroids_i].append(sample_i)
        return clusters

    def _get_cluster_labels(self, clusters, X):
        y_pred = np.zeros((np.shape(X)[0]))
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i
        return y_pred

    def predict(self, X):
        # 随机初始化K个聚类中心
        cluster_centroids = self._init_random_centroids(X)

        for i in range(self.n_iters):
            # 聚类
            clusters = self._create_cluster(cluster_centroids, X)

            # 保存上次的聚类重心点
            prev_centroids = cluster_centroids

            # 重新计算新的聚类重心点
            cluster_centroids = self._calculate_centroids(clusters, X)

            # 如果当前聚类中心点均值保存不变,跳出循环
            diff_centroids = cluster_centroids - prev_centroids
            if not diff_centroids.any():
                break

        return self._get_cluster_labels(clusters, X)









