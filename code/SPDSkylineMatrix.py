import numpy as np
import matrix_utils
from SkylineMatrix import SkylineMatrix


class SPDSkylineMatrix(SkylineMatrix):
    def __init__(self, matrix):
        if not matrix_utils.is_symmetrical(matrix):
            raise Exception("matrix must be symmetric")
        if not matrix_utils.is_positive_definite(matrix):
            raise Exception("matrix must be positive definite")

        for i in range(len(matrix)):
            branch = matrix[:, i][:i + 1].T.tolist()
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
        matrix = np.zeros((len(self.values), len(self.values)))

        for i in range(len(self.values)):
            for j in range(len(self.values[i])):
                matrix[i - j, i] = self.values[i][j]
                matrix[i, i - j] = self.values[i][j]

        return matrix

    def cholesky(self):
        l = np.zeros((len(self.values), len(self.values)))

        max_width = np.max(list(len(lst) for lst in self.values))

        for k in range(len(self.values)):
            start_idx = k - len(self.values[k]) + 1
            s = np.dot(l[k][start_idx:k], l[k][start_idx:k])
            l[k, k] = np.sqrt(self[k, k] - s)

            for i in range(k + 1, min(k + max_width, len(self.values))):
                if k > i - len(self.values[i]):
                    s = np.dot(l[i][start_idx:k], l[k][start_idx:k])
                    l[i, k] = (self[i, k] - s) / l[k, k]

        return l
