import numpy as np
import parse_matrix

if __name__ == "__main__":
    matrix = np.matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    print(matrix)

    skyline_representation = parse_matrix.SkylineMatrix(matrix)
    print(skyline_representation)
