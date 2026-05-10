#!/usr/bin/env python3
"""
CIFAR-10 CNN Image Classification
University ML Project

Target: >80% test accuracy for 14 points
Expected: 85-88% with research-backed VGG-3 architecture
"""

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
    print(" " * 15 + "CIFAR-10 CNN IMAGE CLASSIFICATION")
    print("=" * 70)
    print("\nTarget: >80% test accuracy (14 points)")
    print("Expected: 85-88% with VGG-3 + BatchNorm + Augmentation")
    print("=" * 70)

    OPTIMIZER_TYPE = 'sgd'
    BATCH_SIZE = 64
    EPOCHS = 150 if OPTIMIZER_TYPE == 'sgd' else 100
    MODEL_PATH = 'outputs/models/cifar10_model.keras'
    ARCHITECTURE_PATH = 'outputs/plots/model_architecture.png'
    HISTORY_PATH = 'outputs/plots/training_history.png'
    CONFUSION_PATH = 'outputs/plots/confusion_matrix.png'
    METRICS_PATH = 'outputs/reports/metrics.json'

    print(f"\n[CONFIG] Optimizer: {OPTIMIZER_TYPE.upper()}")
    print(f"[CONFIG] Epochs: {EPOCHS}")
    print(f"[CONFIG] Batch size: {BATCH_SIZE}")
    print()

    print("\n" + "=" * 70)
    print("[1/7] LOADING CIFAR-10 DATA")
    print("=" * 70)

    X_train, y_train, X_test, y_test = load_cifar10_data()
    print(f"✓ Raw data loaded")
    print(f"  Training samples: {X_train.shape[0]:,}")
    print(f"  Test samples: {X_test.shape[0]:,}")
    print(f"  Image shape: {X_train.shape[1:]}")

    print("\n[PREPROCESSING] Applying z-score normalization...")
    X_train, y_train, X_test, y_test = preprocess_data(
        X_train, y_train, X_test, y_test
    )
    print(f"✓ Z-score normalization applied (per-channel mean/std)")
    print(f"✓ Labels one-hot encoded (10 classes)")

    print("\n" + "=" * 70)
    print("[2/7] BUILDING CNN MODEL")
    print("=" * 70)

    model = build_cifar10_cnn(optimizer_type=OPTIMIZER_TYPE)
    print(f"✓ VGG-3 model built with {OPTIMIZER_TYPE.upper()} optimizer")
    print(f"  Total layers: {len(model.layers)}")
    print(f"  Total parameters: {model.count_params():,}")

    print("\nModel Architecture:")
    print("-" * 70)
    model.summary()
    print("-" * 70)

    print("\n" + "=" * 70)
    print("[3/7] GENERATING ARCHITECTURE DIAGRAM")
    print("=" * 70)

    plot_model_architecture(model, ARCHITECTURE_PATH)

    print("\n" + "=" * 70)
    print("[4/7] SETTING UP DATA AUGMENTATION")
    print("=" * 70)

    datagen = create_data_generator()
    datagen.fit(X_train)
    train_generator = datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
    steps_per_epoch = len(X_train) // BATCH_SIZE

    print("✓ Data augmentation configured:")
    print("  - Horizontal flip: True")
    print("  - Width shift: ±10%")
    print("  - Height shift: ±10%")
    print(f"  - Steps per epoch: {steps_per_epoch}")

    print("\n" + "=" * 70)
    print("[5/7] TRAINING MODEL")
    print("=" * 70)

    if OPTIMIZER_TYPE == 'sgd':
        callbacks = create_callbacks_sgd(MODEL_PATH)
        print("✓ SGD callbacks: LearningRateScheduler + ModelCheckpoint")
    else:
        callbacks = create_callbacks_adam(MODEL_PATH)
        print("✓ Adam callbacks: ReduceLROnPlateau + ModelCheckpoint")

    print(f"\nTraining for {EPOCHS} epochs...")
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
    print("✓ Training complete")

    print("\n" + "=" * 70)
    print("[6/7] EVALUATING MODEL ON TEST SET")
    print("=" * 70)

    test_metrics = evaluate_model(model, X_test, y_test)
    test_acc = test_metrics['test_accuracy']
    test_loss = test_metrics['test_loss']

    print(f"\n{'TEST RESULTS':^70}")
    print("=" * 70)
    print(f"  Test Accuracy: {test_acc * 100:.2f}%")
    print(f"  Test Loss: {test_loss:.4f}")
    print("=" * 70)

    # Determine grade
    if test_acc >= 0.80:
        points = 14
        grade_msg = "🎉 EXCELLENT - 14 points!"
    elif test_acc >= 0.78:
        points = 12
        grade_msg = "✓ Very Good - 12 points"
    elif test_acc >= 0.70:
        points = 9
        grade_msg = "✓ Good - 9 points"
    elif test_acc >= 0.60:
        points = 7
        grade_msg = "Acceptable - 7 points"
    else:
        points = 0
        grade_msg = "Below threshold - 0 points"

    print(f"\n  Grade: {grade_msg}\n")

    print("=" * 70)
    print("[7/7] GENERATING VISUALIZATIONS AND REPORTS")
    print("=" * 70)

    print("\n[Visualization 1/2] Training history...")
    plot_training_history(history, HISTORY_PATH)

    print("[Visualization 2/2] Confusion matrix...")
    y_pred = model.predict(X_test, verbose=0)
    class_names = get_class_names()
    plot_confusion_matrix(y_test, y_pred, class_names, CONFUSION_PATH)

    print("\n[Report] Generating classification metrics...")
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
    print(" " * 25 + "TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n  Final Test Accuracy: {test_acc * 100:.2f}%")
    print(f"  Grade: {grade_msg}")
    print("\n" + "=" * 70)
    print("OUTPUTS GENERATED:")
    print("=" * 70)
    print(f"  ✓ Model: {MODEL_PATH}")
    print(f"  ✓ Architecture: {ARCHITECTURE_PATH}")
    print(f"  ✓ Training history: {HISTORY_PATH}")
    print(f"  ✓ Confusion matrix: {CONFUSION_PATH}")
    print(f"  ✓ Metrics report: {METRICS_PATH}")
    print("=" * 70)
    print("\n🎓 Ready for submission!\n")


if __name__ == "__main__":
    main()
