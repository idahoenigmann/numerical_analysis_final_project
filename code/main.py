import numpy as np
import parse_matrix

if __name__ == "__main__":
    n = 5
    matrix = parse_matrix.generate_rand_SPD_matrix(n)

    print(matrix)

    ssm = parse_matrix.SPDSkylineMatrix(matrix)

    l1 = ssm.cholesky()
    print(l1 * l1.T)
