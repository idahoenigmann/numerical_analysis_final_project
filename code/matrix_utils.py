import numpy as np


def is_square(matrix):
    return matrix.shape[0] == matrix.shape[1]


def is_symmetrical(matrix, epsilon=1e-8):
    return is_square(matrix) and np.all(np.abs(matrix - matrix.T) < epsilon)


def is_positive_definite(matrix, epsilon=1e-3):
    return np.all(np.linalg.eigvals(matrix) > epsilon)


def print_matrix(matrix, decimal_precision=2):
    n = len(matrix)

    for i in range(n):
        for j in range(n):
            if round(matrix[i, j], decimal_precision) != 0:
                print("{0:+.{1}f}".format(matrix[i, j], decimal_precision), end="  ")
            else:
                print(" " * len("{0:+.{1}f}".format(0, decimal_precision)), end="  ")
        print()


def generate_rand_spd_matrix(dimension, decimal_precision=-1):
    matrix = np.zeros((dimension, dimension))

    while not is_positive_definite(matrix):
        matrix = np.random.rand(dimension, dimension)

        matrix = 0.5 * (matrix + matrix.T)
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals = np.diag(np.abs(eigvals))
        matrix = np.dot(np.dot(eigvecs, eigvals), eigvecs.T)

        if decimal_precision >= 0:
            matrix = matrix.round(decimal_precision)

    return np.matrix(matrix)


def generate_rand_spd_skyline_matrix(dimension, decimal_presision=-1, avg_branch_len=4):
    matrix = np.zeros((dimension, dimension))

    while not is_positive_definite(matrix):
        matrix = generate_rand_spd_matrix(dimension, decimal_presision)

        for i in range(1, dimension):
            branch_len = min(i, round(abs(np.random.normal() * avg_branch_len) + 1))
            for j in range(i - branch_len):
                matrix[i, j] = 0.
                matrix[j, i] = 0.

    return matrix


def cholesky(matrix):
    if not is_square(matrix):
        raise Exception("matrix must be square")
    if not is_symmetrical(matrix):
        raise Exception("matrix must be symmetrical")
    if not is_positive_definite(matrix):
        raise Exception("matrix must be positive definite")

    n = matrix.shape[0]
    l = np.zeros((n, n))

    for k in range(n):
        s = 0
        for j in range(k):
            s += l[k, j] * l[k, j]

        l[k, k] = np.sqrt(matrix[k, k] - s)

        for i in range(k+1, n):
            s = 0
            for j in range(k):
                s += l[i, j] * l[k, j]
            l[i, k] = (matrix[i, k] - s) / l[k, k]

    return l
