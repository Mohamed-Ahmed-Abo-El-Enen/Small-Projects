import tensorflow_datasets as tfds
from utils.config import Config
import utils.config as config_utils


def get_mnist_dataset():
    X_train, y_train = tfds.as_numpy(tfds.load('mnist', split='train', batch_size=-1, shuffle_files=True,
                                               as_supervised=True))
    X_train = scale_images(X_train)

    config_utils.recalculate_dim(X_train.shape[1], X_train.shape[2], X_train.shape[3])

    print(X_train.shape, y_train.shape)
    return X_train, y_train


def scale_images(X_dataset):
    return (X_dataset - 127.5) / 127.5


def flatten_x_dataset(X_dataset):
    return X_dataset.reshape(-1,Config.IMG_FLATTEN_SHAPE)


def reconstruct_x_dataset_img(X_dataset):
    return X_dataset.reshape(-1, Config.IMG_H, Config.IMG_W, Config.IMG_C)