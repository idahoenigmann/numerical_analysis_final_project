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
    n = int(sys.argv[1]) * step if len(sys.argv) > 1 else 2000

    if False:
        # generate matrix

        random_matrix = matrix_utils.generate_rand_spd_skyline_matrix(n, b, 3)
        ssm = SPDSkylineMatrix(random_matrix)
        avg_b = round(np.max(list(len(lst) for lst in ssm.values)))
        print(avg_b)
        np.savetxt("matrix/matrix_" + str(avg_b) + ".csv", random_matrix, delimiter=",")
    else:
        random_matrix = np.loadtxt("matrix/matrix_" + str(b) + ".csv", delimiter=",")

        # for n in range(step, n + step, step):
        random_matrix = random_matrix[:n, :n]

        # matrix_utils.print_matrix(random_matrix)

        time_linalg = measure_time_cholesky(random_matrix)
        print("{:}, {:}".format(n, time_linalg))
