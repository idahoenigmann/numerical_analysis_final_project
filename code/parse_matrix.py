import numpy as np
import math


def is_square(matrix):
    return matrix.shape[0] == matrix.shape[1]


def is_symmetrical(matrix, epsilon=1e-8):
    return is_square(matrix) and np.all(np.abs(matrix - matrix.T) < epsilon)


def is_positive_definite(matrix):
    return np.all(np.linalg.eigvals(matrix) > 0)


def cholesky(matrix):
    if not is_square(matrix):
        raise Exception("matrix must be square")
    if not is_symmetrical(matrix):
        raise Exception("matrix must be symmetrical")
    if not is_positive_definite(matrix):
        raise Exception("matrix must be positive definite")

    n = matrix.shape[0]
    l = np.empty((n, n))

    for k in range(n):
        s = np.sum(l[k, j] * l[k, j] for j in range(k-1))
        l[k, k] = math.sqrt(matrix[k, k] - s)

        for i in range(k+1, n):
            s = np.sum(l[i, j] * l[k, j] for j in range(k-1))
            l[i, k] = (matrix[i, k] - s) / l[k, k]

    return l


class SkylineMatrix:
    values = list()
    branch_lengths = list()
    dimension = 0

    def __init__(self, matrix):
        self.dimension = len(matrix)

        for i in range(self.dimension):
            j = i                                       # up branch
            while j >= 0 and (not matrix[j, i] == 0):
                self.values.append(matrix[j, i])
                j = j-1
            up_branch_length = i - j

            j = i - 1                                   # left branch
            while j >= 0 and (not matrix[i, j] == 0):
                self.values.append(matrix[i, j])
                j = j-1
            self.branch_lengths.append((up_branch_length, i - j))

    def __str__(self):
        return self.to_matrix().__str__()

    def __repr__(self):
        return self.__str__()

    def to_matrix(self):
        matrix = np.empty((self.dimension, self.dimension))

        k = 0
        for i in range(self.dimension):
            for j in range(self.branch_lengths[i][0]):      # up branch
                matrix[i-j, i] = self.values[k]
                k = k+1
            for j in range(self.branch_lengths[i][1] - 1):  # left branch
                matrix[i, i-j-1] = self.values[k]
                k = k+1

        return matrix
