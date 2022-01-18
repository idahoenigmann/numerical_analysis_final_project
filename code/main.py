import numpy as np
import timeit
import sys
import matrix_utils
from SPDSkylineMatrix import SPDSkylineMatrix


def measure_time_cholesky(matrix):
    ssm = SPDSkylineMatrix(matrix)
    conversion_factor = 1e9         # seconds to nanoseconds

    return round(timeit.timeit(ssm.cholesky, number=1) * conversion_factor),\
           round(timeit.timeit(lambda: matrix_utils.cholesky(matrix), number=1) * conversion_factor),\
           round(timeit.timeit(lambda: np.linalg.cholesky(matrix), number=1) * conversion_factor)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    random_matrix = matrix_utils.generate_rand_spd_skyline_matrix(n, 5)

    time_ssm, time_nor, time_linalg = measure_time_cholesky(random_matrix)
    print("{:}, {:}, {:}, {:}".format(n, time_ssm, time_nor, time_linalg))
