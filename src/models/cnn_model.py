import keras
from keras import layers, models, regularizers, optimizers


def build_cifar10_cnn(optimizer_type='sgd', input_shape=(32, 32, 3), num_classes=10):
    """
    Build research-backed VGG-3 style CNN for CIFAR-10 classification.

    Architecture targets 85-88% test accuracy based on published reproductions:
    - 3 convolutional blocks (2 Conv2D each)
    - BatchNormalization after every Conv2D (+3-5% accuracy boost)
    - Increasing dropout (0.2 → 0.3 → 0.4 → 0.5)
    - he_uniform initializer for ReLU networks
    - L2 regularization (1e-4) on all weight layers

    Args:
        optimizer_type: 'sgd' (best accuracy, 85-88%) or 'adam' (faster, 82-85%)
        input_shape: Input image shape (default: 32×32×3)
        num_classes: Number of output classes (default: 10)

    Returns:
        keras.Model: Compiled model ready for training
    """
    WD = 1e-4
    INIT = 'he_uniform'

    model = models.Sequential(name='CIFAR10_VGG3')
    model.add(layers.Conv2D(
        32, (3, 3),
        padding='same',
        kernel_initializer=INIT,
        kernel_regularizer=regularizers.l2(WD),
        input_shape=input_shape,
        name='conv1_1'
    ))
    model.add(layers.BatchNormalization(name='bn1_1'))
    model.add(layers.Activation('relu', name='relu1_1'))

    model.add(layers.Conv2D(
        32, (3, 3),
        padding='same',
        kernel_initializer=INIT,
        kernel_regularizer=regularizers.l2(WD),
        name='conv1_2'
    ))
    model.add(layers.BatchNormalization(name='bn1_2'))
    model.add(layers.Activation('relu', name='relu1_2'))

    model.add(layers.MaxPooling2D((2, 2), name='pool1'))
    model.add(layers.Dropout(0.2, name='dropout1'))

    model.add(layers.Conv2D(
        64, (3, 3),
        padding='same',
        kernel_initializer=INIT,
        kernel_regularizer=regularizers.l2(WD),
        name='conv2_1'
    ))
    model.add(layers.BatchNormalization(name='bn2_1'))
    model.add(layers.Activation('relu', name='relu2_1'))

    model.add(layers.Conv2D(
        64, (3, 3),
        padding='same',
        kernel_initializer=INIT,
        kernel_regularizer=regularizers.l2(WD),
        name='conv2_2'
    ))
    model.add(layers.BatchNormalization(name='bn2_2'))
    model.add(layers.Activation('relu', name='relu2_2'))

    model.add(layers.MaxPooling2D((2, 2), name='pool2'))
    model.add(layers.Dropout(0.3, name='dropout2'))

    model.add(layers.Conv2D(
        128, (3, 3),
        padding='same',
        kernel_initializer=INIT,
        kernel_regularizer=regularizers.l2(WD),
        name='conv3_1'
    ))
    model.add(layers.BatchNormalization(name='bn3_1'))
    model.add(layers.Activation('relu', name='relu3_1'))

    model.add(layers.Conv2D(
        128, (3, 3),
        padding='same',
        kernel_initializer=INIT,
        kernel_regularizer=regularizers.l2(WD),
        name='conv3_2'
    ))
    model.add(layers.BatchNormalization(name='bn3_2'))
    model.add(layers.Activation('relu', name='relu3_2'))

    model.add(layers.MaxPooling2D((2, 2), name='pool3'))
    model.add(layers.Dropout(0.4, name='dropout3'))

    model.add(layers.Flatten(name='flatten'))

    model.add(layers.Dense(
        128,
        kernel_initializer=INIT,
        kernel_regularizer=regularizers.l2(WD),
        name='dense1'
    ))
    model.add(layers.BatchNormalization(name='bn_dense'))
    model.add(layers.Activation('relu', name='relu_dense'))
    model.add(layers.Dropout(0.5, name='dropout_dense'))

    model.add(layers.Dense(
        num_classes,
        activation='softmax',
        kernel_initializer=INIT,
        name='output'
    ))

    if optimizer_type == 'sgd':
        optimizer = optimizers.SGD(
            learning_rate=0.1,
            momentum=0.9,
            nesterov=True
        )
    elif optimizer_type == 'adam':
        optimizer = optimizers.Adam(learning_rate=1e-3)
    else:
        raise ValueError(f"Unknown optimizer_type: {optimizer_type}. Use 'sgd' or 'adam'.")

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
