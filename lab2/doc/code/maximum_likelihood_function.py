import numpy as np


def calculate_likelihood_function(train_X, train_Y, w):
    """
    计算极大条件似然函数
    :param train_X: 训练集数据特征矩阵
    :param train_Y: 训练集数据标签
    :param w: 权重w向量
    :return: 极大条件似然函数
    """
    number = np.size(train_X, axis=0)
    post_probability = np.zeros((number, 1))
    sum_ln = 0
    for i in range(number):
        post_probability[i] = w.dot(train_X[i].T)
        sum_ln += np.log(np.exp(post_probability[i]) + 1)
    return train_Y.dot(post_probability) - sum_ln
