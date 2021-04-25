import numpy as np


def rotation_transformation(data, theta=0, axis='x'):
    if axis == 'x':
        rotation_matrix = [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]]
    elif axis == 'y':
        rotation_matrix = [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
    elif axis == 'z':
        rotation_matrix = [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    else:
        print("wrong input")
        return data
    return np.dot(rotation_matrix, data)


def generate_data(number, noise, height):
    tt1 = (3 * np.pi / 2) * (1 + 2 * np.random.rand(1, number))
    x = tt1 * np.cos(tt1)
    y = height * np.random.rand(1, number)
    z = tt1 * np.sin(tt1)
    X = np.concatenate((x, y, z))
    X += noise * np.random.randn(3, number)
    X = rotation_transformation(X, 30, 'z')
    return X.T
