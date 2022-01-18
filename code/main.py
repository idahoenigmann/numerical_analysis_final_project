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
    # generate matrix
    """for b in range(1, 10):
        random_matrix = matrix_utils.generate_rand_spd_skyline_matrix(n, b)
        np.savetxt("matrix_" + str(b) + ".csv", random_matrix, delimiter=",")"""

    random_matrix = np.loadtxt("matrix_7.csv", delimiter=",")

    # n = int(sys.argv[1]) * 10 if len(sys.argv) > 1 else 500
    for n in range(100, 5001, 100):
        random_matrix = random_matrix[:n, :n]

        # matrix_utils.print_matrix(random_matrix)

        time_ssm, time_nor, time_linalg = measure_time_cholesky(random_matrix)
        print("{:}, {:}, {:}, {:}".format(n, time_ssm, time_nor, time_linalg))


