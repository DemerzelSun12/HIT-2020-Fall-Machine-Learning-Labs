import numpy as np


def sigmoid_function(z):
    """
    计算Sigmoid函数值
    :param z: Sigmoid函数参数
    :return: 计算得到的值
    """
    return 1.0 / (1.0 + np.exp(-z))


def calculate_test_accuracy(test_X, test_Y, w):
    """
    计算训练得到的模型在测试集上的准确率
    :param test_X: 训练集的特征
    :param test_Y: 训练集的标签
    :param w: 模型参数
    :return: 预测准确率
    """
    number_test = len(test_X)
    correct_number = 0
    for i in range(number_test):
        if sigmoid_function(w.dot(test_X[i])) > 0.5 and test_Y[i] == 1:
            correct_number += 1
        elif sigmoid_function(w.dot(test_X[i])) < 0.5 and test_Y[i] == 0:
            correct_number += 1
    # print(correct_number)
    # print(number_test)
    return (1.0 * correct_number) / number_test
