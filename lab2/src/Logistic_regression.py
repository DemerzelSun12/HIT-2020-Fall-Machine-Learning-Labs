from src import gradient_descent_solution
from src import newton_iteration_solution
from src import read_bank_note
from src.generate_data import *
from src.split_train_test_data import *
from src.generate_picture import *
from src.calculate_test_accuracy import *

import numpy as np
import matplotlib.pyplot as plt


def test_own_data_cov0():
    print("-----------start, the covariance matrix is diagonal-------------")
    penalty_coefficient = 30
    data_number = 100
    number_pos = 50
    pos_mean_1 = -0.5
    pos_mean_2 = -0.5
    neg_mean_1 = 0.5
    neg_mean_2 = 0.5
    train_rate = 0.7
    X, Y = generate_data(data_number, pos_mean_1, pos_mean_2, neg_mean_1, neg_mean_2, number_pos, 0.5, 0.5, 0.0, 0.0)
    X = np.c_[np.ones(len(X)), X]
    train_X, train_Y, test_X, test_Y = split_train_test_data(X, Y, train_rate)
    # print(np.shape(train_X))
    # print(train_X)
    # print(train_Y)
    # print(test_X)
    # print(test_Y)
    rows_number, column_number = np.shape(train_X)
    gradient_descent_without_penalty = gradient_descent_solution.GradientDescent(train_X, train_Y,
                                                                                 np.zeros(column_number),
                                                                                 penalty_coefficient=0)
    gradient_descent_without_penalty_result = gradient_descent_without_penalty.gradient_descent()

    # print(gradient_descent_without_penalty_result)
    gradient_descent_with_penalty = gradient_descent_solution.GradientDescent(train_X, train_Y,
                                                                              np.zeros(column_number),
                                                                              penalty_coefficient=penalty_coefficient)
    gradient_descent_with_penalty_result = gradient_descent_with_penalty.gradient_descent()
    # print("finish grad")
    newton_iteration = newton_iteration_solution.NewtonIteration(train_X, train_Y, np.zeros(column_number),
                                                                 penalty_coefficient=penalty_coefficient)
    newton_iteration_result = newton_iteration.newton_iteration()
    # print("------the result of three methods------")
    # print("gradient descent without penalty: ", gradient_descent_without_penalty_result)
    # print("gradient descent without penalty: ", gradient_descent_with_penalty_result)
    # print("newton iteration: ", newton_iteration_result)

    x_length = np.linspace(-3, 3)
    # y_length_grad_without = -(
    #         gradient_descent_without_penalty_result[0] + gradient_descent_without_penalty_result[1] * x_length) / \
    #                         gradient_descent_without_penalty_result[2]
    # y_length_grad_with = -(
    #         gradient_descent_with_penalty_result[0] + gradient_descent_with_penalty_result[1] * x_length) / \
    #                      gradient_descent_with_penalty_result[2]

    # y_length_grad_without = np.linspace(-3, 3)
    # y_length_grad_with = np.linspace(-3, 3)

    y_length_grad_without = -(gradient_descent_without_penalty_result / gradient_descent_without_penalty_result[2])[0:2]
    y_predict_1 = np.poly1d(y_length_grad_without[::-1])
    y_length_1 = y_predict_1(x_length)
    y_length_grad_with = -(gradient_descent_with_penalty_result / gradient_descent_with_penalty_result[2])[0:2]
    y_predict_2 = np.poly1d(y_length_grad_with[::-1])
    y_length_2 = y_predict_2(x_length)
    y_length_newton = -(newton_iteration_result / newton_iteration_result[2])[0:2]
    y_predict_3 = np.poly1d(y_length_newton[::-1])
    y_length_3 = y_predict_3(x_length)

    plt.title("Train")
    plt.plot(x_length, y_length_1, label="gradient descent without penalty")
    plt.plot(x_length, y_length_2, label="gradient descent with penalty")
    plt.plot(x_length, y_length_3, label="newton iteration")
    generate_2_dimension_plot(train_X, train_Y)
    plt.legend()
    plt.show()

    plt.title("Test")
    plt.plot(x_length, y_length_1, label="gradient descent without penalty")
    plt.plot(x_length, y_length_2, label="gradient descent with penalty")
    plt.plot(x_length, y_length_3, label="newton iteration")
    generate_2_dimension_plot(test_X, test_Y)
    plt.legend()
    plt.show()

    print("The accuracy on train data set:")
    print("Gradient descent without penalty's accuracy: ",
          calculate_test_accuracy(train_X, train_Y, gradient_descent_without_penalty_result))
    print("Gradient descent with penalty's accuracy: ",
          calculate_test_accuracy(train_X, train_Y, gradient_descent_with_penalty_result))
    print("Newton iteration: ", calculate_test_accuracy(train_X, train_Y, newton_iteration_result))

    print()
    print("The accuracy on test data set:")
    print("Gradient descent without penalty's accuracy: ",
          calculate_test_accuracy(test_X, test_Y, gradient_descent_without_penalty_result))
    print("Gradient descent with penalty's accuracy: ",
          calculate_test_accuracy(test_X, test_Y, gradient_descent_with_penalty_result))
    print("Newton iteration: ", calculate_test_accuracy(test_X, test_Y, newton_iteration_result))
    print()


def test_own_data_cov1():
    print("-----------start, the covariance matrix is not diagonal-------------")
    penalty_coefficient = 30
    data_number = 100
    number_pos = 50
    pos_mean_1 = -0.5
    pos_mean_2 = -0.5
    neg_mean_1 = 0.5
    neg_mean_2 = 0.5
    train_rate = 0.7
    X, Y = generate_data(data_number, pos_mean_1, pos_mean_2, neg_mean_1, neg_mean_2, number_pos, 0.5, 0.5, 0.3, 0.3)
    X = np.c_[np.ones(len(X)), X]
    train_X, train_Y, test_X, test_Y = split_train_test_data(X, Y, train_rate)
    # print(np.shape(train_X))
    # print(train_X)
    # print(train_Y)
    # print(test_X)
    # print(test_Y)
    rows_number, column_number = np.shape(train_X)
    gradient_descent_without_penalty = gradient_descent_solution.GradientDescent(train_X, train_Y,
                                                                                 np.zeros(column_number),
                                                                                 penalty_coefficient=0)
    gradient_descent_without_penalty_result = gradient_descent_without_penalty.gradient_descent()

    # print(gradient_descent_without_penalty_result)
    gradient_descent_with_penalty = gradient_descent_solution.GradientDescent(train_X, train_Y,
                                                                              np.zeros(column_number),
                                                                              penalty_coefficient=penalty_coefficient)
    gradient_descent_with_penalty_result = gradient_descent_with_penalty.gradient_descent()
    # print("finish grad")
    newton_iteration = newton_iteration_solution.NewtonIteration(train_X, train_Y, np.zeros(column_number),
                                                                 penalty_coefficient=penalty_coefficient)
    newton_iteration_result = newton_iteration.newton_iteration()
    # print("------the result of three methods------")
    # print("gradient descent without penalty: ", gradient_descent_without_penalty_result)
    # print("gradient descent without penalty: ", gradient_descent_with_penalty_result)
    # print("newton iteration: ", newton_iteration_result)

    x_length = np.linspace(-3, 3)
    # y_length_grad_without = -(
    #         gradient_descent_without_penalty_result[0] + gradient_descent_without_penalty_result[1] * x_length) / \
    #                         gradient_descent_without_penalty_result[2]
    # y_length_grad_with = -(
    #         gradient_descent_with_penalty_result[0] + gradient_descent_with_penalty_result[1] * x_length) / \
    #                      gradient_descent_with_penalty_result[2]

    # y_length_grad_without = np.linspace(-3, 3)
    # y_length_grad_with = np.linspace(-3, 3)

    y_length_grad_without = -(gradient_descent_without_penalty_result / gradient_descent_without_penalty_result[2])[0:2]
    y_predict_1 = np.poly1d(y_length_grad_without[::-1])
    y_length_1 = y_predict_1(x_length)
    y_length_grad_with = -(gradient_descent_with_penalty_result / gradient_descent_with_penalty_result[2])[0:2]
    y_predict_2 = np.poly1d(y_length_grad_with[::-1])
    y_length_2 = y_predict_2(x_length)
    y_length_newton = -(newton_iteration_result / newton_iteration_result[2])[0:2]
    y_predict_3 = np.poly1d(y_length_newton[::-1])
    y_length_3 = y_predict_3(x_length)

    plt.title("Train")
    plt.plot(x_length, y_length_1, label="gradient descent without penalty")
    plt.plot(x_length, y_length_2, label="gradient descent with penalty")
    plt.plot(x_length, y_length_3, label="newton iteration")
    generate_2_dimension_plot(train_X, train_Y)
    plt.legend()
    plt.show()

    plt.title("Test")
    plt.plot(x_length, y_length_1, label="gradient descent without penalty")
    plt.plot(x_length, y_length_2, label="gradient descent with penalty")
    plt.plot(x_length, y_length_3, label="newton iteration")
    generate_2_dimension_plot(test_X, test_Y)

    plt.legend()
    plt.show()

    print("The accuracy on train data set:")
    print("Gradient descent without penalty's accuracy: ",
          calculate_test_accuracy(train_X, train_Y, gradient_descent_without_penalty_result))
    print("Gradient descent with penalty's accuracy: ",
          calculate_test_accuracy(train_X, train_Y, gradient_descent_with_penalty_result))
    print("Newton iteration: ", calculate_test_accuracy(train_X, train_Y, newton_iteration_result))

    print()
    print("The accuracy on test data set:")
    print("Gradient descent without penalty's accuracy: ",
          calculate_test_accuracy(test_X, test_Y, gradient_descent_without_penalty_result))
    print("Gradient descent with penalty's accuracy: ",
          calculate_test_accuracy(test_X, test_Y, gradient_descent_with_penalty_result))
    print("Newton iteration: ", calculate_test_accuracy(test_X, test_Y, newton_iteration_result))
    print()


def test_bank_note():
    print("-----------start, the bank note authentication data set-------------")
    penalty_coefficient = 10
    train_rate = 0.7
    bank_note_data = read_bank_note.ReadBankNote()
    X, Y = bank_note_data.generate_data()
    train_X, train_Y, test_X, test_Y = split_train_test_data(X, Y, train_rate)
    rows_number, column_number = np.shape(train_X)
    gradient_descent_without_penalty = gradient_descent_solution.GradientDescent(train_X, train_Y,
                                                                                 np.zeros(column_number),
                                                                                 penalty_coefficient=0)
    gradient_descent_without_penalty_result = gradient_descent_without_penalty.gradient_descent()

    # print(gradient_descent_without_penalty_result)
    gradient_descent_with_penalty = gradient_descent_solution.GradientDescent(train_X, train_Y,
                                                                              np.zeros(column_number),
                                                                              penalty_coefficient=penalty_coefficient)
    gradient_descent_with_penalty_result = gradient_descent_with_penalty.gradient_descent()
    # print("finish grad")
    newton_iteration = newton_iteration_solution.NewtonIteration(train_X, train_Y, np.zeros(column_number),
                                                                 penalty_coefficient=penalty_coefficient)
    newton_iteration_result = newton_iteration.newton_iteration()
    # print("------the result of three methods------")
    # print("gradient descent without penalty: ", gradient_descent_without_penalty_result)
    # print("gradient descent without penalty: ", gradient_descent_with_penalty_result)
    # print("newton iteration: ", newton_iteration_result)

    print("The accuracy on train data set:")
    print("Gradient descent without penalty's accuracy: ",
          calculate_test_accuracy(train_X, train_Y, gradient_descent_without_penalty_result))
    print("Gradient descent with penalty's accuracy: ",
          calculate_test_accuracy(train_X, train_Y, gradient_descent_with_penalty_result))
    print("Newton iteration: ", calculate_test_accuracy(train_X, train_Y, newton_iteration_result))

    print()
    print("The accuracy on test data set:")
    print("Gradient descent without penalty's accuracy: ",
          calculate_test_accuracy(test_X, test_Y, gradient_descent_without_penalty_result))
    print("Gradient descent with penalty's accuracy: ",
          calculate_test_accuracy(test_X, test_Y, gradient_descent_with_penalty_result))
    print("Newton iteration: ", calculate_test_accuracy(test_X, test_Y, newton_iteration_result))
    print()


def main():
    test_own_data_cov0()
    test_own_data_cov1()
    test_bank_note()


if __name__ == '__main__':
    main()
