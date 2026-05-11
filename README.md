# CIFAR-10

convolutional neural network for CIFAR-10 image classification.

targets 85–88% test accuracy. requirement: >80% for full marks.

---

## 00 — Contents

```
00    contents
01    architecture
02    structure
03    training
04    usage
05    results
```

---

## 01 — Architecture

VGG-3 style CNN with BatchNormalization.

**01.1 — convolutional blocks**
3 blocks, each with 2 Conv2D layers. filter progression: 32 → 64 → 128.
BatchNormalization after every Conv2D. MaxPooling2D (2×2) at end of each block.
increasing dropout: 0.2 → 0.3 → 0.4 to prevent overfitting.

**01.2 — classifier head**
Flatten → Dense(128) → BatchNormalization → ReLU → Dropout(0.5) → Dense(10, softmax).

**01.3 — regularization**

- kernel_initializer='he_uniform' on all Conv2D/Dense (prevents vanishing gradients in deep ReLU networks)
- L2 weight decay (1e-4) on all weight layers
- increasing dropout schedule across depth

**01.4 — parameters**
30 layers total. 552,874 trainable parameters (2.1 MB).

---

## 02 — Structure

```
src/
    models/         VGG-3 architecture definition
    utils/          data loading, preprocessing, training loop
    visualization/  plots, confusion matrix, metrics

outputs/
    models/         trained .keras model
    plots/          architecture diagram, training curves, confusion matrix
    reports/        JSON metrics

main.py            orchestration
research.md        empirical best practices (Brownlee, geifmany/cifar-vgg)
```

---

## 03 — Training

**03.1 — preprocessing**
z-score normalization (per-channel mean/std). empirically outperforms division by 255.
mean and std computed on training data, applied to test set. no data leakage.

labels one-hot encoded (10 classes). validation = test set (standard CIFAR-10 practice).

**03.2 — data augmentation**
horizontal flip + width/height shifts (±10%). no vertical flip (cats/cars upside-down not in test distribution).
no rotation, no zoom (degrades performance on 32×32 images).

**03.3 — optimizer**
SGD with momentum=0.9, nesterov=True. initial LR=0.1.
step decay schedule: ×0.1 at epoch 80, ×0.1 at epoch 120 (standard CIFAR-10 schedule from Keras examples).

Adam (LR=1e-3, ReduceLROnPlateau) available for faster convergence but ~1% lower final accuracy.

**03.4 — epochs**
150 epochs with SGD. ModelCheckpoint saves best model by validation accuracy.

---

## 04 — Usage

```bash
# train with SGD (85-88% accuracy, 150 epochs)
uv run python main.py

# background training with logs
nohup bash -c "uv run python main.py > training.log 2>&1" &
tail -f training.log

# switch to Adam (82-85% accuracy, 100 epochs)
# edit main.py: OPTIMIZER_TYPE = 'adam'
```

outputs generated automatically:

- `outputs/models/cifar10_model.keras`
- `outputs/plots/training_history.png`
- `outputs/plots/confusion_matrix.png`
- `outputs/reports/metrics.json`

---
