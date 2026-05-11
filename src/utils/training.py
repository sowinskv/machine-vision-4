from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau


def step_decay_schedule(epoch):
    if epoch < 80:
        return 0.1
    elif epoch < 120:
        return 0.01
    elif epoch < 160:
        return 0.001
    else:
        return 0.0005


def create_callbacks_sgd(model_save_path):
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
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    return {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy)
    }
