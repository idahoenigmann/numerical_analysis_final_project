import numpy as np
import timeit
import sys
import matrix_utils
from SPDSkylineMatrix import SPDSkylineMatrix


def measure_time_cholesky(matrix):
    # ssm = SPDSkylineMatrix(matrix)
    conversion_factor = 1e9         # seconds to nanoseconds

    """return round(timeit.timeit(ssm.cholesky, number=1) * conversion_factor),\
           round(timeit.timeit(lambda: matrix_utils.cholesky(matrix), number=1) * conversion_factor),\
           round(timeit.timeit(lambda: np.linalg.cholesky(matrix), number=1) * conversion_factor)"""

    return round(timeit.timeit(lambda: np.linalg.cholesky(matrix), number=1) * conversion_factor)


if __name__ == "__main__":
    step = 10
    b = 191
    n = int(sys.argv[1]) * step if len(sys.argv) > 1 else 10

    random_matrix = matrix_utils.generate_rand_spd_skyline_matrix(n, 3)

    matrix = SPDSkylineMatrix(random_matrix)

    matrix_utils.print_matrix(random_matrix)

    l = matrix.cholesky()

    matrix_utils.print_matrix(l)

    print(np.max(random_matrix - np.matmul(l, l.T)))

