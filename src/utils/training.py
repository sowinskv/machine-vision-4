from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau


def step_decay_schedule(epoch):
    """
    Step decay learning rate schedule for SGD optimizer.

    Research-backed schedule used in Keras cifar10_resnet.py and
    >90% CIFAR-10 reproductions:
    - Epochs 0-79: LR = 0.1
    - Epochs 80-119: LR = 0.01 (×0.1)
    - Epochs 120-159: LR = 0.001 (×0.1)
    - Epochs 160+: LR = 0.0005

    Args:
        epoch: Current epoch number

    Returns:
        float: Learning rate for this epoch
    """
    if epoch < 80:
        return 0.1
    elif epoch < 120:
        return 0.01
    elif epoch < 160:
        return 0.001
    else:
        return 0.0005


def create_callbacks_sgd(model_save_path):
    """
    Create callbacks for SGD training.

    SGD approach uses fixed step-decay schedule + ModelCheckpoint.
    Expected accuracy: 85-88% in 150 epochs.

    Args:
        model_save_path: Path to save best model

    Returns:
        list: Callback objects
    """
    return [
        ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        LearningRateScheduler(step_decay_schedule, verbose=1)
    ]


def create_callbacks_adam(model_save_path):
    """
    Create callbacks for Adam training.

    Adam approach uses adaptive ReduceLROnPlateau + ModelCheckpoint.
    Expected accuracy: 82-85% in 80-100 epochs.

    Args:
        model_save_path: Path to save best model

    Returns:
        list: Callback objects
    """
    return [
        ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )
    ]


def train_model(model, train_generator, validation_data, epochs, callbacks, steps_per_epoch):
    """
    Train model with data augmentation.

    Args:
        model: Compiled Keras model
        train_generator: Data generator from ImageDataGenerator.flow()
        validation_data: Tuple of (X_val, y_val)
        epochs: Number of training epochs
        callbacks: List of callback objects
        steps_per_epoch: Number of batches per epoch

    Returns:
        History: Keras History object containing training metrics
    """
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=1
    )
    return history


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test set.

    Args:
        model: Trained Keras model
        X_test: Test images
        y_test: Test labels (one-hot encoded)

    Returns:
        dict: Test metrics (loss and accuracy)
    """
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    return {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy)
    }
