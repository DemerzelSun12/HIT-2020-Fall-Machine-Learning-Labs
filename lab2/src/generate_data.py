import random

import numpy as np


def generate_data(number, pos_mean_1, pos_mean_2, neg_mean_1, neg_mean_2, number_pos, cov11, cov22, cov12,
                  cov21):
    """
    生成数据集，不分训练集和测试集
    :param number: 生成数据总数量
    :param pos_mean_1: 正例特征1的均值
    :param pos_mean_2: 正例特征2的均值
    :param neg_mean_1: 反例特征1的均值
    :param neg_mean_2: 反例特征2的均值
    :param number_pos: 正例数量，要求0 < number_pos < number
    :param cov11: 第一个特征的方差
    :param cov22: 第二个特征的方差
    :param cov12: 第一个特征与第二个特征的协方差
    :param cov21: 第二个特征与第一个特征的协方差
    :return: X:生成数据集的标签, Y:生成数据集的分类
    """
    assert (0 < number_pos < number)
    X = []
    Y = []
    number_neg = number - number_pos

    while True:
        if number_pos == 0 and number_neg == 0:
            break
        elif number_pos == 0:
            number_neg -= 1
            x1_temp, x2_temp = np.random.multivariate_normal([pos_mean_1, pos_mean_2], [[cov11, cov12], [cov21, cov22]],
                                                             1).T
            X.append([x1_temp[0], x2_temp[0]])
            Y.append(0)
        elif number_neg == 0:
            number_pos -= 1
            x1_temp, x2_temp = np.random.multivariate_normal([neg_mean_1, neg_mean_2], [[cov11, cov12], [cov21, cov22]],
                                                             1).T
            X.append([x1_temp[0], x2_temp[0]])
            Y.append(1)
        else:
            if random.randint(0, 1) == 0:
                number_neg -= 1
                x1_temp, x2_temp = np.random.multivariate_normal([pos_mean_1, pos_mean_2],
                                                                 [[cov11, cov12], [cov21, cov22]],
                                                                 1).T
                X.append([x1_temp[0], x2_temp[0]])
                Y.append(0)
            else:
                number_pos -= 1
                x1_temp, x2_temp = np.random.multivariate_normal([neg_mean_1, neg_mean_2],
                                                                 [[cov11, cov12], [cov21, cov22]],
                                                                 1).T
                X.append([x1_temp[0], x2_temp[0]])
                Y.append(1)
    return X, np.array(Y)
