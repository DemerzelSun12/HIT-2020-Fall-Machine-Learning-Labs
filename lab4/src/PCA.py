import numpy as np


class PCA(object):

    def __init__(self, data, k):
        self.data = data
        self.k = k
        self.rows, self.columns = self.data.shape

    def pca(self):
        data_mean = np.sum(self.data, axis=0) / self.rows
        central_data = self.data - data_mean
        cov_matrix = central_data.T.dot(central_data)
        eigen_values, eigen_vector = np.linalg.eig(cov_matrix)
        eigen_value_sort = np.argsort(eigen_values)
        eigen_result = eigen_vector[:, eigen_value_sort[:-(self.k + 1):-1]]
        return central_data, eigen_result, data_mean
