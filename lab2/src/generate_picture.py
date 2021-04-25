import matplotlib.pyplot as plt


def generate_2_dimension_plot(X, Y):
    """
    画出二维分布的图像
    :param X: 数据集的二维特征
    :param Y: 数据集的标签
    :return: 无
    """
    feature1_X = []
    feature2_X = []
    feature1_Y = []
    feature2_Y = []
    number = len(X)
    for i in range(number):
        if Y[i] == 1:
            feature1_X.append(X[i][1])
            feature1_Y.append(X[i][2])
        else:
            feature2_X.append(X[i][1])
            feature2_Y.append(X[i][2])
    plt.scatter(feature1_X, feature1_Y, facecolor="none", color="b", label="positive dot")
    plt.scatter(feature2_X, feature2_Y, marker="x", color="r", label="negative dot")
