import numpy as np


def sigmoid_function(z):
    """
    计算Sigmoid函数值
    :param z: Sigmoid函数参数
    :return: 计算得到的值
    """
    return 1.0 / (1.0 + np.exp(z))


class NewtonIteration(object):

    def __init__(self, X, Y, w, penalty_coefficient, learning_rate=0, deviation=1e-6):
        self.X = X
        self.Y = Y
        self.w = w
        self.penalty_coefficient = penalty_coefficient
        self.learning_rate = learning_rate
        self.deviation = deviation
        self.data_number = len(X)
        self.feature_number = len(X[0])

    def __calculate_once_derivative__(self, w):
        result = np.zeros(self.feature_number)
        for i in range(self.data_number):
            result += (self.X[i] * (self.Y[i] - (1.0 - sigmoid_function(w.dot(self.X[i])))))
        return -1 * result + self.learning_rate * w

    def __calculate_second_derivative__(self, w):
        result = np.eye(self.feature_number) * self.learning_rate
        for i in range(self.data_number):
            diff = sigmoid_function(w.dot(self.X[i]))
            result += self.X[i] * np.transpose([self.X[i]]).dot(diff).dot(1 - diff)
        return np.linalg.pinv(result)

    def newton_iteration(self):
        pre_w = self.w
        number = 0
        while True:
            number += 1
            # print(number)
            grad = self.__calculate_once_derivative__(pre_w)
            # print(np.linalg.norm(grad))
            if np.linalg.norm(grad) < self.deviation:
                break
            post_w = pre_w - self.__calculate_second_derivative__(pre_w).dot(grad)
            pre_w = post_w
        return pre_w
