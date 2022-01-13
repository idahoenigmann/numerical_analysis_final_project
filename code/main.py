import numpy as np
import parse_matrix

if __name__ == "__main__":
    matrix = np.matrix([[5, 1], [1, 3]])
    print(matrix)

    l = parse_matrix.cholesky(matrix)

    #print(l)
    print(np.dot(l, l.T))

    """
    print(parse_matrix.is_square(matrix))
    print(parse_matrix.is_symmetrical(matrix))
    print(parse_matrix.is_positive_definite(matrix))

    skyline_representation = parse_matrix.SkylineMatrix(matrix)
    print(skyline_representation)"""
