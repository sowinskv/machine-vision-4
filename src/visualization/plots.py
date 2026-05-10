import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix


def plot_model_architecture(model, save_path):
    """
    Generate and save model architecture diagram.

    Args:
        model: Keras model
        save_path: Path to save the diagram (e.g., 'model_architecture.png')
    """
    plot_model(
        model,
        to_file=save_path,
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB',  # Top to bottom
        dpi=150,
        show_layer_activations=True
    )
    print(f"✓ Model architecture saved to {save_path}")


def plot_training_history(history, save_path):
    """
    Plot training and validation accuracy/loss curves.

    Creates 2 subplots showing accuracy and loss progression over epochs.

    Args:
        history: Keras History object from model.fit()
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Loss plot
    axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Training history saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    Generate and save confusion matrix heatmap.

    Args:
        y_true: True labels (one-hot encoded or integer)
        y_pred: Predicted labels (one-hot encoded or probabilities)
        class_names: List of class names
        save_path: Path to save the plot
    """
    # Convert one-hot to class indices if needed
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'},
        square=True
    )
    plt.title('Confusion Matrix - CIFAR-10 CNN', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved to {save_path}")
