import numpy as np
from keras.datasets import cifar10
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_cifar10_data():
    """
    Load CIFAR-10 dataset from Keras datasets.

    Returns:
        tuple: (X_train, y_train, X_test, y_test)
            - X_train: (50000, 32, 32, 3) training images
            - y_train: (50000, 1) training labels
            - X_test: (10000, 32, 32, 3) test images
            - y_test: (10000, 1) test labels
    """
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    return X_train, y_train, X_test, y_test


def preprocess_data(X_train, y_train, X_test, y_test):
    """
    Preprocess CIFAR-10 data using z-score normalization.

    Research shows z-score (per-channel mean/std) normalization outperforms
    simple division by 255. This is the standard preprocessing for achieving
    >85% accuracy on CIFAR-10.

    Args:
        X_train: Training images array
        y_train: Training labels array
        X_test: Test images array
        y_test: Test labels array

    Returns:
        tuple: Preprocessed (X_train, y_train, X_test, y_test)
            - Images normalized using z-score
            - Labels one-hot encoded (10 classes)
    """
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))

    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return X_train, y_train, X_test, y_test


def create_data_generator():
    """
    Create ImageDataGenerator for data augmentation.

    Configuration based on research-backed best practices for CIFAR-10:
    - Horizontal flip: highest-yield augmentation (+5-10% accuracy)
    - Width/height shifts: ±10% to simulate position changes
    - NO vertical flip (upside-down objects not in test distribution)
    - NO rotation (optional, but can hurt on CIFAR-10)
    - NO zoom (destroys features in tiny 32×32 images)

    Returns:
        ImageDataGenerator: Configured for CIFAR-10 augmentation
    """
    return ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )


def get_class_names():
    """
    Return CIFAR-10 class names in order.

    Returns:
        list: 10 class names
    """
    return [
        'airplane',
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck'
    ]
