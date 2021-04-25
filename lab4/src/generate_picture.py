import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_3_dimension_picture(data):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=data[:, 2], label='Swiss Roll dots')
    ax.view_init(elev=15, azim=20)
    ax.legend(loc='upper right')
    plt.show()
    plt.savefig('../figure/fig.png', bbox_inches='tight')


def generate_2_dimension_picture(data):
    plt.scatter(data[:, 0].tolist(), data[:, 1].tolist(), c=data[:, 0].tolist(), label='Swiss Roll dots after PCA')
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig('../figure/fig1.png', bbox_inches='tight')

