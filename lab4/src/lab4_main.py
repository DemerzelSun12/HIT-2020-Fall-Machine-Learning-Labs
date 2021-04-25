from src.PCA import *
from src.generate_data import *
from src.generate_picture import *
from src.dimensionality_reduction_image import *


def test_pca(data):
    generate_3_dimension_picture(data)
    central_data, eig_vector, data_mean = PCA(data, 2).pca()
    pca_data = np.dot(central_data, eig_vector)
    generate_2_dimension_picture(pca_data)


def test_image_data_set():
    data = read_image_data()
    image_number, image_feature = data[0].shape
    print(data.shape)
    central_data = []
    eig_vector = []
    data_mean = []
    pca_data = []
    rebuild_data = []
    for i in range(len(data)):
        central_data_i, eig_vector_i, data_mean_i = PCA(data[i], 8).pca()
        central_data.append(central_data_i)
        eig_vector.append(eig_vector_i)
        data_mean.append(data_mean_i)
        print(eig_vector[i])
        eig_vector_i = np.real(eig_vector_i)
        pca_data.append(np.dot(central_data_i, eig_vector_i))
        # print(pca_data)
        rebuild_data.append(np.dot(pca_data[i], eig_vector[i].T) + data_mean[i])
    plt.figure(figsize=(50, 50))
    for i in range(len(data)):
        plt.subplot(3, 3, i + 1)
        plt.imshow(rebuild_data[i], cmap=plt.cm.gray)
    plt.show()

    print("the signal to noise ratio of the image after PCA:")
    for i in range(len(data)):
        ratio = calculate_noise_ratio(data[i], rebuild_data[i])
        print('The noise ratio of image ' + str(i) + ' is ' + str(ratio))


def test_single_picture():
    data = read_image_data()
    image_number, image_feature = data.shape
    print(data.shape)
    central_data, eig_vector, data_mean = PCA(data, 20).pca()
    print(eig_vector)
    eig_vector = np.real(eig_vector)
    pca_data = np.dot(central_data, eig_vector)
    rebuild_data = np.dot(pca_data, eig_vector.T) + data_mean
    plt.figure(figsize=(50, 50))
    plt.imshow(rebuild_data)
    plt.show()


def main():
    data_1 = generate_data(1000, 0, 100)
    # print(np.shape(data_1))
    data_2 = generate_data(2000, 0, 10)
    data_3 = generate_data(2000, 1, 10)

    test_pca(data_1)
    test_pca(data_2)
    test_pca(data_3)

    test_image_data_set()


if __name__ == '__main__':
    main()
