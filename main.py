import os
import numpy as np

from src.models.cnn_model import build_cifar10_cnn
from src.utils.data_loader import (
    load_cifar10_data,
    preprocess_data,
    create_data_generator,
    get_class_names
)
from src.utils.training import (
    create_callbacks_sgd,
    create_callbacks_adam,
    train_model,
    evaluate_model
)
from src.visualization.plots import (
    plot_model_architecture,
    plot_training_history,
    plot_confusion_matrix
)
from src.visualization.metrics import (
    generate_classification_report,
    save_metrics_to_json
)


def main():
    print("=" * 70)
    print(" " * 15 + "CIFAR-10 CNN image classification")
    print("=" * 70)
    print("\ntarget: >80% test accuracy")
    print("expected: 85-88% with VGG-3 + BatchNorm + augmentation")
    print("=" * 70)

    OPTIMIZER_TYPE = 'sgd'
    BATCH_SIZE = 64
    EPOCHS = 150 if OPTIMIZER_TYPE == 'sgd' else 100
    MODEL_PATH = 'outputs/models/cifar10_model.keras'
    ARCHITECTURE_PATH = 'outputs/plots/model_architecture.png'
    HISTORY_PATH = 'outputs/plots/training_history.png'
    CONFUSION_PATH = 'outputs/plots/confusion_matrix.png'
    METRICS_PATH = 'outputs/reports/metrics.json'

    print(f"\n[config] optimizer: {OPTIMIZER_TYPE.upper()}")
    print(f"[config] epochs: {EPOCHS}")
    print(f"[config] batch size: {BATCH_SIZE}")
    print()

    print("\n" + "=" * 70)
    print("[1/7] loading CIFAR-10 data")
    print("=" * 70)

    X_train, y_train, X_test, y_test = load_cifar10_data()
    print(f"✓ raw data loaded")
    print(f"  training samples: {X_train.shape[0]:,}")
    print(f"  test samples: {X_test.shape[0]:,}")
    print(f"  image shape: {X_train.shape[1:]}")

    print("\n[preprocessing] applying z-score normalization...")
    X_train, y_train, X_test, y_test = preprocess_data(
        X_train, y_train, X_test, y_test
    )
    print(f"✓ z-score normalization applied (per-channel mean/std)")
    print(f"✓ labels one-hot encoded (10 classes)")

    print("\n" + "=" * 70)
    print("[2/7] building CNN model")
    print("=" * 70)

    model = build_cifar10_cnn(optimizer_type=OPTIMIZER_TYPE)
    print(f"✓ VGG-3 model built with {OPTIMIZER_TYPE.upper()} optimizer")
    print(f"  total layers: {len(model.layers)}")
    print(f"  total parameters: {model.count_params():,}")

    print("\nmodel architecture:")
    print("-" * 70)
    model.summary()
    print("-" * 70)

    print("\n" + "=" * 70)
    print("[3/7] generating architecture diagram")
    print("=" * 70)

    plot_model_architecture(model, ARCHITECTURE_PATH)

    print("\n" + "=" * 70)
    print("[4/7] setting up data augmentation")
    print("=" * 70)

    datagen = create_data_generator()
    datagen.fit(X_train)
    train_generator = datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
    steps_per_epoch = len(X_train) // BATCH_SIZE

    print("✓ data augmentation configured:")
    print("  - horizontal flip: True")
    print("  - width shift: ±10%")
    print("  - height shift: ±10%")
    print(f"  - steps per epoch: {steps_per_epoch}")

    print("\n" + "=" * 70)
    print("[5/7] training model")
    print("=" * 70)

    if OPTIMIZER_TYPE == 'sgd':
        callbacks = create_callbacks_sgd(MODEL_PATH)
        print("✓ SGD callbacks: LearningRateScheduler + ModelCheckpoint")
    else:
        callbacks = create_callbacks_adam(MODEL_PATH)
        print("✓ Adam callbacks: ReduceLROnPlateau + ModelCheckpoint")

    print(f"\ntraining for {EPOCHS} epochs...")
    print("-" * 70)

    history = train_model(
        model=model,
        train_generator=train_generator,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch
    )

    print("-" * 70)
    print("✓ training complete")

    print("\n" + "=" * 70)
    print("[6/7] evaluating model on test set")
    print("=" * 70)

    test_metrics = evaluate_model(model, X_test, y_test)
    test_acc = test_metrics['test_accuracy']
    test_loss = test_metrics['test_loss']

    print(f"\n{'test results':^70}")
    print("=" * 70)
    print(f"  test accuracy: {test_acc * 100:.2f}%")
    print(f"  test loss: {test_loss:.4f}")
    print("=" * 70)

    if test_acc >= 0.80:
        points = 14
        grade_msg = "🎉 excellent - 14 points!"
    elif test_acc >= 0.78:
        points = 12
        grade_msg = "✓ very good - 12 points"
    elif test_acc >= 0.70:
        points = 9
        grade_msg = "✓ good - 9 points"
    elif test_acc >= 0.60:
        points = 7
        grade_msg = "acceptable - 7 points"
    else:
        points = 0
        grade_msg = "below threshold - 0 points"

    print(f"\n  grade: {grade_msg}\n")

    print("=" * 70)
    print("[7/7] generating visualizations and reports")
    print("=" * 70)

    print("\n[visualization 1/2] training history...")
    plot_training_history(history, HISTORY_PATH)

    print("[visualization 2/2] confusion matrix...")
    y_pred = model.predict(X_test, verbose=0)
    class_names = get_class_names()
    plot_confusion_matrix(y_test, y_pred, class_names, CONFUSION_PATH)

    print("\n[report] generating classification metrics...")
    classification_rep = generate_classification_report(
        y_test, y_pred, class_names
    )

    all_metrics = {
        'test_metrics': test_metrics,
        'grade': {
            'accuracy': float(test_acc),
            'points': points,
            'message': grade_msg
        },
        'classification_report': classification_rep,
        'training_config': {
            'optimizer': OPTIMIZER_TYPE,
            'batch_size': BATCH_SIZE,
            'epochs_trained': len(history.history['loss']),
            'total_epochs': EPOCHS
        }
    }
    save_metrics_to_json(all_metrics, METRICS_PATH)

    print("\n" + "=" * 70)
    print(" " * 25 + "training complete")
    print("=" * 70)
    print(f"\n  final test accuracy: {test_acc * 100:.2f}%")
    print(f"  grade: {grade_msg}")
    print("\n" + "=" * 70)
    print("outputs generated:")
    print("=" * 70)
    print(f"  ✓ model: {MODEL_PATH}")
    print(f"  ✓ architecture: {ARCHITECTURE_PATH}")
    print(f"  ✓ training history: {HISTORY_PATH}")
    print(f"  ✓ confusion matrix: {CONFUSION_PATH}")
    print(f"  ✓ metrics report: {METRICS_PATH}")
    print("=" * 70)
    print("\n🎓 ready for submission!\n")


if __name__ == "__main__":
    main()
