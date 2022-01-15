import numpy as np
import parse_matrix

if __name__ == "__main__":
    np.set_printoptions(precision=5)

    n = 5
    matrix = parse_matrix.generate_rand_SPD_matrix(n, 5)
    #matrix = np.matrix([[1, 0, 0, 0, 1], [0, 1, 0, 2, 2], [0, 0, 1, 3, 3], [1, 2, 3, 14, 18], [0, 4, 5, 29, 48]])
    #matrix = 0.5 * (matrix + matrix.T) + np.identity(n)

    print(matrix)

    ssm = parse_matrix.SPDSkylineMatrix(matrix)

    l1 = ssm.cholesky()
    l2 = parse_matrix.cholesky(matrix)
    l3 = np.linalg.cholesky(matrix)

    m1 = np.matmul(l1, l1.T)
    m2 = np.matmul(l2, l2.T)
    m3 = np.matmul(l3, l3.T)

    #print(l1)
    #print(l2)
    #print(l3)

    print(l1 - l3)
    print(l2 - l3)