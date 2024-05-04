import numpy as np


class SkylineMatrix:
    values = list()

    def __init__(self, matrix):
        for i in range(len(matrix)):
            up_branch = (matrix[:, i][:i + 1]).T.tolist()[0]
            up_branch.reverse()
            while len(up_branch) > 0 and up_branch[-1] == 0:
                up_branch.pop(-1)

            left_branch = [] if len(matrix[i][:i]) == 0 else (matrix[i, :][:i]).tolist()[0]
            left_branch.reverse()
            while len(left_branch) > 0 and left_branch[-1] == 0:
                left_branch.pop(-1)

            self.values.append([up_branch, left_branch])

    def __str__(self):
        return self.to_matrix().__str__()

    def __repr__(self):
        return self.__str__()

    def to_matrix(self):
        matrix = np.zeros((len(self.values), len(self.values)))

        for i in range(len(self.values)):
            up_branch, left_branch = self.values[i]

            for j in range(len(up_branch)):
                matrix[i - j, i] = up_branch[j]
            for k in range(len(left_branch)):
                matrix[i, i - k - 1] = left_branch[k]

        return matrix
