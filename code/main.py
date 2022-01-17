import numpy as np
import parse_matrix
import time
import sys

def measure_time_cholesky(matrix):
    ssm = parse_matrix.SPDSkylineMatrix(matrix)

    t1 = time.perf_counter_ns()
    ssm.cholesky()
    t2 = time.perf_counter_ns()
    parse_matrix.cholesky(matrix)
    t3 = time.perf_counter_ns()
    np.linalg.cholesky(matrix)
    t4 = time.perf_counter_ns()

    return t2 - t1, t3 - t2, t4 - t3


if __name__ == "__main__":
    np.set_printoptions(precision=1)
    n = 10
    if (len(sys.argv) > 1):
      n = int(sys.argv[1])

    matrix = parse_matrix.generate_rand_spd_skyline_matrix(n)

    # parse_matrix.print_matrix(matrix)

    # time_ssm, time_nor, time_linalg = measure_time_cholesky(matrix)
    # print("{:}, {:}, {:}, {:}".format(n, time_ssm, time_nor, time_linalg))

    ssm = parse_matrix.SPDSkylineMatrix(matrix)
    l = ssm.cholesky()
    print(np.max(matrix - np.matmul(l, l.T)))
