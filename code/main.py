import numpy as np
import parse_matrix
import time
import matplotlib.pyplot as plt


def measure_time_cholesky(dim):
    matrix = parse_matrix.generate_rand_SPD_skyline_matrix(dim)
    ssm = parse_matrix.SPDSkylineMatrix(matrix)

    t1 = time.perf_counter_ns()
    l1 = ssm.cholesky()
    t2 = time.perf_counter_ns()
    l2 = parse_matrix.cholesky(matrix)
    t3 = time.perf_counter_ns()
    l3 = np.linalg.cholesky(matrix)
    t4 = time.perf_counter_ns()

    time_ssm = t2 - t1
    time_nor = t3 - t2
    time_linalg = t4 - t3

    return time_ssm, time_nor, time_linalg


if __name__ == "__main__":
    np.set_printoptions(precision=1)
    n = 20
    t_ssm = list()
    t_nor = list()
    t_linalg = list()

    for n in range(1, n + 1):
        time_ssm, time_nor, time_linalg = measure_time_cholesky(n)
        t_ssm.append(time_ssm)
        t_nor.append(time_nor)
        t_linalg.append(time_linalg)

    plt.plot(list(range(1, n + 1)), t_ssm)
    plt.plot(list(range(1, n + 1)), t_nor)
    plt.plot(list(range(1, n + 1)), t_linalg)

    plt.show()
