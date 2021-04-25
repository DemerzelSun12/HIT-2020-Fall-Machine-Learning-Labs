import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os


def read_image_data():
    file_dir_path = '../data/'
    file_list = os.listdir(file_dir_path)
    image_data = []
    plt.figure(figsize=(50, 50))
    i = 1
    for file in file_list:
        file_path = os.path.join(file_dir_path, file)
        print("open figure " + file_path)
        plt.subplot(3, 3, i)
        with open(file_path) as f:
            img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
            print(img.shape)
            img = cv.resize(img, (50, 50), interpolation=cv.INTER_NEAREST)
            print(img.shape)
            # cv.imshow("sdg", img)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
            # height, width = img.shape
            # img_temp = img.reshape(height * width)
            # data.append(img_temp)
            data = np.asarray(img)
            image_data.append(data)
            plt.imshow(img, cmap=plt.cm.gray)
            # cv.imshow("sg", img)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
        i += 1
    plt.show()
    return np.asarray(image_data)


def calculate_noise_ratio(img_1, img_2):
    noise = np.mean(np.square(img_1 / 255. - img_2 / 255.))
    if noise < 1e-10:
        return 100
    return 20 * np.log10(1 / np.sqrt(noise))
