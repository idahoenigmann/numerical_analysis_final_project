import numpy as np
import parse_matrix

if __name__ == "__main__":
    n = 5
    matrix = parse_matrix.generate_rand_SPD_matrix(n, 3)

    print(matrix)

    ssm = parse_matrix.SPDSkylineMatrix(matrix)

    l1 = ssm.cholesky()
    l2 = parse_matrix.cholesky(matrix)
    l3 = np.linalg.cholesky(matrix)

    m1 = np.matmul(l1, l1.T)
    m2 = np.matmul(l2, l2.T)
    m3 = np.matmul(l3, l3.T)

    print((l3 - l1).round(3))
