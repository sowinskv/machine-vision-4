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
    try:
        plot_model(
            model,
            to_file=save_path,
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            dpi=150,
            show_layer_activations=True
        )
        print(f"✓ Model architecture saved to {save_path}")
    except ImportError as e:
        print(f"⚠ Warning: Could not generate architecture diagram")
        print(f"  Install graphviz: brew install graphviz")
        print(f"  Skipping architecture visualization...")


def plot_training_history(history, save_path):
    """
    Plot training and validation accuracy/loss curves.

    Creates 2 subplots showing accuracy and loss progression over epochs.

    Args:
        history: Keras History object from model.fit()
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history.history['accuracy']) + 1)

    axes[0].plot(epochs, history.history['accuracy'],
                 label='Train', linewidth=1.2, color='#3266cc', alpha=0.8)
    axes[0].plot(epochs, history.history['val_accuracy'],
                 label='Validation', linewidth=1.2, color='#dc3912', alpha=0.8)
    axes[0].set_xlabel('Epoch', fontsize=9)
    axes[0].set_ylabel('Accuracy', fontsize=9)
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].spines[['top', 'right']].set_visible(False)
    axes[0].spines[['left', 'bottom']].set_linewidth(0.5)
    axes[0].tick_params(labelsize=8)
    axes[0].grid(True, alpha=0.2, linewidth=0.5)

    axes[1].plot(epochs, history.history['loss'],
                 label='Train', linewidth=1.2, color='#3266cc', alpha=0.8)
    axes[1].plot(epochs, history.history['val_loss'],
                 label='Validation', linewidth=1.2, color='#dc3912', alpha=0.8)
    axes[1].set_xlabel('Epoch', fontsize=9)
    axes[1].set_ylabel('Loss', fontsize=9)
    axes[1].legend(fontsize=8, frameon=False)
    axes[1].spines[['top', 'right']].set_visible(False)
    axes[1].spines[['left', 'bottom']].set_linewidth(0.5)
    axes[1].tick_params(labelsize=8)
    axes[1].grid(True, alpha=0.2, linewidth=0.5)

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
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.3,
        linecolor='white',
        annot_kws={'size': 8},
        cbar_kws={'shrink': 0.8}
    )
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True', fontsize=10)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved to {save_path}")
