import numpy as np
import timeit
import sys
import matrix_utils
from SPDSkylineMatrix import SPDSkylineMatrix


def measure_time_cholesky(matrix):
    ssm = SPDSkylineMatrix(matrix)
    conversion_factor = 1e9         # seconds to nanoseconds

    return round(timeit.timeit(ssm.cholesky, number=1) * conversion_factor), \
           round(timeit.timeit(lambda: matrix_utils.cholesky(matrix), number=1) * conversion_factor),\
           round(timeit.timeit(lambda: np.linalg.cholesky(matrix), number=1) * conversion_factor)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
        b = 10

        random_matrix = matrix_utils.generate_rand_spd_skyline_matrix(n, b)

        time_cholesky_skyline, time_cholesky, time_numpy = measure_time_cholesky(random_matrix)

        print("{}, {}, {}, {}".format(n, time_cholesky_skyline, time_cholesky, time_numpy))

    else:
        n, b = 10, 2

        random_matrix = matrix_utils.generate_rand_spd_skyline_matrix(n, b)

        matrix = SPDSkylineMatrix(random_matrix)

        print("random input matrix:")
        matrix_utils.print_matrix(random_matrix)
        print("-" * 50)

        l = matrix.cholesky()

        print("cholesky decomposition:")
        matrix_utils.print_matrix(l)
        print("-" * 50)

        print("l * l.T:")
        matrix_utils.print_matrix(np.matmul(l, l.T))
        print("-" * 50)

        print("maximum error (matrix - l * l.T): {}".format(abs(np.max(random_matrix - np.matmul(l, l.T)))))
        print("-" * 50)

        print("time measurement:")

        time_cholesky_skyline, time_cholesky, time_numpy = measure_time_cholesky(random_matrix)

        print("cholesky_skyline: {}ns; cholesky: {}ns; numpy: {}ns".format(time_cholesky_skyline, time_cholesky, time_numpy))
        print("-" * 50)
