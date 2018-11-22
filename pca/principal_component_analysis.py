import numpy as np

class PCA():
    def calculate_covariance_matrix(self, X, Y=None):
        if Y is None:
            Y = X
        n_samples = np.shape(X)[0]
        covariance_matrix = (1 / (n_samples - 1)) * (X - X.mean(axis=0).T.dot(Y - Y.mean(axis=0)))

    def zero_mean(self, X):
        mean = np.mean(X, axis=0)     # 求各列特征均值mkdi
        x_new = X - mean              # 减去相应的均值
        return x_new

    def calculate_percent(self, eigenvalues, n_percent):
        sortArray = np.sort(eigenvalues)
        sortArray = sortArray[-1::-1]
        arraySum = sum(sortArray)
        tmp = 0
        n_components = 0
        for i in sortArray:
            imp += i
            n_components += 1
            if tmp >= arraySum * n_percent:
                return n_components
        return n_components

    def transfrom(self, X, n_percent=0.90):
        print(X.shape)
        x_new = self.zero_mean(X)
        covariance_matrix = np.cov(x_new, rowvar=0)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        n_components = self.calculate_percent(eigenvalues, n_percent)

        eigenval_indice = np.argsort(eigenvalues)
        n_eigenval_indice = eigenval_indice[-1:-(n_components + 1):-1]
        n_eigenvectors = eigenvectors[:, n_eigenval_indice]
        X_transformed = x_new.dot(n_eigenvectors)

        return X_transformed
