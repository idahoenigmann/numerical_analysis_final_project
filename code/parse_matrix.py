import numpy as np
import math


def is_square(matrix):
    return matrix.shape[0] == matrix.shape[1]


def is_symmetrical(matrix, epsilon=1e-8):
    return is_square(matrix) and np.all(np.abs(matrix - matrix.T) < epsilon)


def is_positive_definite(matrix):
    return np.all(np.linalg.eigvals(matrix) > 0)


def generate_rand_SPD_matrix(dimension, decimal_presision=2):
    matrix = np.empty((dimension, dimension))

    while not (is_symmetrical(matrix) and is_positive_definite(matrix)):
        matrix = np.random.rand(dimension, dimension)

        for i in range(dimension):
            for j in range(dimension):
                scale = np.random.randint(-100, 100)
                matrix[i, j] = matrix[i, j] * scale

        matrix = 0.5 * (matrix + matrix.T) + np.identity(dimension)
        matrix = matrix.round(decimal_presision)

    return np.matrix(matrix)


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
        s = np.dot(l[k][:k], l[k][:k])
        l[k, k] = np.sqrt(matrix[k, k] - s)

        for i in range(k+1, n):
            s = np.dot(l[i][:k], l[k][:k])
            l[i, k] = (matrix[i, k] - s) / l[k, k]

    return l


class SkylineMatrix:
    values = list()

    def __init__(self, matrix):
        for i in range(len(matrix)):
            up_branch = (np.concatenate(matrix[:, i][:i + 1]).ravel().tolist())[0]
            up_branch.reverse()
            while len(up_branch) > 0 and up_branch[-1] == 0:
                up_branch.pop(-1)

            left_branch = []
            if len(matrix[i][:i]) > 0:
                left_branch = np.concatenate(matrix[i,:].ravel().tolist()).tolist()[0:i]
            left_branch.reverse()
            while len(left_branch) > 0 and left_branch[-1] == 0:
                left_branch.pop(-1)

            self.values.append([up_branch, left_branch])

    def __str__(self):
        return self.to_matrix().__str__()

    def __repr__(self):
        return self.__str__()

    def to_matrix(self):
        matrix = np.empty((len(self.values), len(self.values)))

        k = 0
        for i in range(len(self.values)):
            up_branch, left_branch = self.values[i]

            for j in range(len(up_branch)):
                matrix[i - j, i] = up_branch[j]
            for k in range(len(left_branch)):
                matrix[i, i - k - 1] = left_branch[k]

        return matrix


class SPDSkylineMatrix(SkylineMatrix):
    def __init__(self, matrix):
        if not is_symmetrical(matrix):
            raise Exception("matrix must be symmetric")
        if not is_positive_definite(matrix):
            raise Exception("matrix must be positive definite")

        for i in range(len(matrix)):
            branch = matrix[:, i][:i + 1].tolist()
            if type(matrix) == np.matrix:
                branch = (np.concatenate(matrix[:, i][:i + 1]).ravel().tolist())[0]
            branch.reverse()
            while len(branch) > 0 and branch[-1] == 0:
                branch.pop(-1)

            self.values.append(branch)

    def __getitem__(self, item):
        row_idx, col_idx = item
        row_idx, col_idx = min(row_idx, col_idx), max(row_idx, col_idx)

        if row_idx < 0 or col_idx > len(self.values):
            raise Exception("index out of bounds")

        col = self.values[col_idx]

        if col_idx - row_idx < len(col):
            return col[col_idx - row_idx]
        else:
            return 0

    def to_matrix(self):
        matrix = np.empty((len(self.values), len(self.values)))

        for i in range(len(self.values)):
            for j in range(len(self.values[i])):
                matrix[i - j, i] = self.values[i][j]
                matrix[i, i - j] = self.values[i][j]

        return matrix

    def cholesky(self):
        l = np.empty((len(self.values), len(self.values)))

        for k in range(len(self.values)):
            s = np.dot(l[k][:len(self.values[k]) - 1], l[k][:len(self.values[k]) - 1])
            l[k, k] = np.sqrt(self[k, k] - s)

            for i in range(k + 1, len(self.values)):
                s = np.dot(l[i][:len(self.values[k]) - 1], l[k][:len(self.values[k]) - 1])
                l[i, k] = (self[i, k] - s) / l[k, k]

        return l