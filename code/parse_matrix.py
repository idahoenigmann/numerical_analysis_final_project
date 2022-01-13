import numpy as np

def is_square(matrix):
    return False

def is_symmetrical(matrix):
    return False


def is_positive_definite(matrix):
    return False


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
