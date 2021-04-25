import numpy as np


def split_train_test_data(X, Y, train_rate):
    """
    将数据集划分为训练集与测试集
    :param X: 数据集的特征
    :param Y: 数据集的标签
    :param train_rate: 训练集的比例
    :return: 训练集的特征；训练集的标签；测试集的特征；测试集的标签
    """
    number = len(X)
    number_train = int(number * train_rate)
    number_test = number - number_train
    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    for i in range(number):
        if number_test > 0:
            if number_train == 0 or np.random.randint(2) == 0:
                number_test -= 1
                test_X.append(X[i])
                test_Y.append(Y[i])
            else:
                number_train -= 1
                train_X.append(X[i])
                train_Y.append(Y[i])
        else:
            number_train -= 1
            train_X.append(X[i])
            train_Y.append(Y[i])
    return np.array(train_X), np.array(train_Y), np.array(test_X), np.array(test_Y)
