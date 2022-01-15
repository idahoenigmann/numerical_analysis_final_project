import numpy as np
import parse_matrix

if __name__ == "__main__":
    np.set_printoptions(precision=1)
    n = 15

    matrix = parse_matrix.generate_rand_SPD_skyline_matrix(n)
    parse_matrix.print_matrix(matrix)

    ssm = parse_matrix.SPDSkylineMatrix(matrix)

    l1 = ssm.cholesky()
    l2 = parse_matrix.cholesky(matrix)
    l3 = np.linalg.cholesky(matrix)

    m1 = np.matmul(l1, l1.T)
    m2 = np.matmul(l2, l2.T)
    m3 = np.matmul(l3, l3.T)

    parse_matrix.print_matrix(l1)

    print("absolute error: {}".format(np.max(m1 - matrix)))