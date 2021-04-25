import numpy as np


def sigmoid_function(z):
    """
    计算Sigmoid函数值
    :param z: Sigmoid函数参数
    :return: 计算得到的值
    """
    return 1.0 / (1.0 + np.exp(z))


class GradientDescent(object):

    def __init__(self, X, Y, w, penalty_coefficient, learning_rate=0.1, deviation=1e-6):
        self.X = X
        self.Y = Y
        self.w = w
        self.penalty_coefficient = penalty_coefficient
        self.learning_rate = learning_rate
        self.deviation = deviation
        self.data_number = len(X)
        self.feature_number = len(X[0])

    def __calculate_loss__(self, w):
        ans = 0
        for i in range(self.data_number):
            ans += (-self.Y[i] * w.dot(self.X[i]) + np.log(1 + np.exp(w.dot(self.X[i]))))
        return (ans + 0.5 * self.penalty_coefficient * w.dot(w)) / self.data_number

    def __calculate_gradient__(self, w):
        grad = np.zeros(self.feature_number)
        for i in range(self.data_number):
            grad += (self.X[i] * (self.Y[i] - (1.0 - sigmoid_function(w.dot(self.X[i])))))
        return (-1 * grad + self.penalty_coefficient * w) / self.data_number

    def gradient_descent(self):
        loss0 = self.__calculate_loss__(self.w)
        k = 0
        pre_w = self.w
        while True:
            post_w = pre_w - self.learning_rate * self.__calculate_gradient__(pre_w)
            loss1 = self.__calculate_loss__(post_w)
            if np.abs(loss1 - loss0) < self.deviation:
                break
            else:
                k = k + 1
                # print(k)
                # print("loss:", loss1)
                if loss1 > loss0:
                    self.learning_rate *= 0.5
                loss0 = loss1
                pre_w = post_w
        return pre_w
