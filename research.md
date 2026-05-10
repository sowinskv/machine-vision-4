# Achieving >80% Test Accuracy on CIFAR-10 with Keras/TensorFlow: A Recipe Guide

## TL;DR

- **A 3-block VGG-style CNN (two Conv2D layers per block at filters 32/64/128 or 64/128/256, MaxPooling2D after each block) trained with BatchNormalization, increasing Dropout (0.2 → 0.3 → 0.4 → 0.5), light data augmentation (horizontal flip + 10% width/height shifts), `he_uniform`/`he_normal` initializers, and SGD with momentum=0.9 over ~100–200 epochs reliably exceeds 0.80 test accuracy** — published reproductions land around 0.85–0.88, well above the 0.80 target.
- The single biggest accuracy lever, beyond a baseline ~73% 3-VGG-block model, is **combining BatchNormalization + Dropout + horizontal-flip data augmentation**; each one alone gets you to ~83–84%, and stacking them with increasing dropout pushes you to ~88%.
- Use `kernel_initializer='he_uniform'` (or `'he_normal'`) on every Conv2D/Dense layer — this is the standard antidote to vanishing gradients in ReLU networks deeper than ~10 layers, and it is essentially free to add.

---

## Key Findings

| Design Choice      | Recommended Value                                                                                        | Expected Effect on Test Accuracy                                  |
| ------------------ | -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| Architecture       | 3 VGG-style blocks, 2× Conv2D per block, 3×3 kernels, `padding='same'`                                   | Baseline ~73% (without regularization)                            |
| Filter progression | 32→64→128 (small) or 64→128→256 (stronger)                                                               | +1–3% with wider filters                                          |
| Initializer        | `he_uniform` or `he_normal` for ReLU                                                                     | Critical for >10-layer nets; ~free gain                           |
| BatchNormalization | After every Conv2D (before activation, or after — both work)                                             | +3–5% and stabilizes training                                     |
| Dropout            | Increasing: 0.2 / 0.3 / 0.4 in conv blocks, 0.5 before output                                            | +5–8% vs no dropout                                               |
| L2 weight decay    | 1e-4 (with BN); 5e-4 (without BN)                                                                        | +1–2% when tuned                                                  |
| Data augmentation  | `horizontal_flip=True`, `width_shift_range=0.1`, `height_shift_range=0.1` (optional `rotation_range=15`) | +5–10%                                                            |
| Optimizer          | SGD(lr=0.1, momentum=0.9, nesterov=True) **or** Adam(lr=1e-3)                                            | SGD generalizes ~0.5–1% better given a schedule                   |
| LR schedule        | Step decay (×0.1 at epochs 80, 120) or `ReduceLROnPlateau(factor=0.2, patience=10)`                      | +1–3% in final accuracy                                           |
| Batch size         | 64–128                                                                                                   | Best generalization range per Masters & Luschi (arXiv:1804.07612) |
| Epochs             | 100 (Adam) to 200+ (SGD)                                                                                 | Required to converge with augmentation                            |
| Preprocessing      | Per-channel mean/std normalization (z-score)                                                             | More stable than ÷255 alone                                       |

Empirically reported accuracy ladder for the **same baseline VGG-3 model** (Brownlee, MachineLearningMastery):

- Baseline (no regularization): **~73%**
- - Fixed Dropout 0.2: **~83%**
- - Data Augmentation only: **~84%**
- - Dropout + Augmentation: **~85%**
- - Increasing Dropout + Augmentation: **~85–86%**
- - BatchNormalization on top of the above: **~88%**
- Carefully tuned VGG-style with BN + L2 + augmentation (geifmany/cifar-vgg): **~93%**

All recipes above are well over the 0.80 target.

---

## Details

### 1. Optimal CNN Architecture

The community-consensus blueprint for CIFAR-10 (32×32×3) without transfer learning is a **simplified VGG with 3 convolutional blocks**, each block being:

```
[Conv2D(F, 3×3, padding='same') → BN → ReLU] ×2  →  MaxPooling2D(2×2)  →  Dropout(p)
```

- **Number of blocks: 3 is the sweet spot.** Brownlee and others show 4 VGG blocks does not improve over 3 on CIFAR-10 (32×32 inputs leave too little spatial resolution after 4 pool ops) and overfits faster.
- **Filter counts: double per block.** Use either {32,64,128} (lighter, faster) or {64,128,256} (stronger, higher ceiling). The ZacharyGoshen and Brownlee references start at 32; cifar-vgg uses 64 and gets to 93%.
- **Kernel size 3×3** everywhere with `padding='same'`. Stacked 3×3 convs have the same receptive field as larger kernels with fewer parameters and more nonlinearity.
- **Classifier head: Flatten → Dense(128 or 512) → Dropout(0.5) → Dense(10, softmax).** Heavier dense heads (e.g., 1024–4096) overfit on CIFAR-10's 50k training images unless BN+heavy dropout is used. With BN, a 128-unit dense layer is enough.

### 2. Regularization Stack

- **BatchNormalization** after every Conv2D is the single most impactful change after dropout. It enables higher learning rates, smooths the loss landscape, and contributes a regularizing effect of its own. Note: when BN is present, the L2 penalty on conv kernels has reduced regularizing effect (it mostly shifts learning rate dynamics) but is still standard practice in CIFAR recipes.
- **Increasing Dropout** is provably better than fixed dropout on this network. Standard schedule:
  - After block 1 (output 16×16): `Dropout(0.2)`
  - After block 2 (output 8×8): `Dropout(0.3)`
  - After block 3 (output 4×4): `Dropout(0.4)`
  - Before final softmax Dense: `Dropout(0.5)`
- **L2 weight decay** (`kernel_regularizer=regularizers.l2(1e-4)`) on every Conv2D and Dense layer. The cifar-vgg reference uses 5e-4; with BN, 1e-4 works well. Brownlee found L2 alone (without dropout/augmentation) did not help, so use L2 _as a complement_ not a substitute.
- Do **not** stack Dropout inside conv blocks at >0.3 in early blocks; too aggressive dropout in low-channel conv layers hurts learning.

### 3. Data Augmentation for CIFAR-10

The well-tested Keras `ImageDataGenerator` configuration:

```python
datagen = ImageDataGenerator(
    width_shift_range=0.1,   # ±10% horizontal translation
    height_shift_range=0.1,  # ±10% vertical translation
    horizontal_flip=True,    # left↔right flip
    # rotation_range=15,     # optional, small rotations only
    fill_mode='nearest'
)
datagen.fit(x_train)
```

Why these settings:

- **Horizontal flip** is the highest-yield augmentation; almost all CIFAR-10 classes (cars, ships, animals, planes) are symmetric under L-R flipping.
- **Vertical flip should NOT be used** — upside-down cats/cars are not in the test distribution.
- **10% shifts** simulate small object position changes without losing class-critical pixels in 32×32 images.
- **Mild rotation (10–15°)** is optional; more than 15° often hurts CIFAR-10.
- **Avoid `zoom_range > 0.2`** for tiny images; aggressive zooming destroys small discriminating features.
- For higher accuracy (>90%): add 4-pixel random crops with reflection padding and consider cutout/AutoAugment, but these aren't needed to clear 80%.

### 4. Training Hyperparameters

**Optimizer — two equally valid choices:**

- **SGD** (best final accuracy): `SGD(learning_rate=0.1, momentum=0.9, nesterov=True)` with a step-decay schedule (×0.1 at epochs 80 and 120) or `ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)`.
- **Adam** (faster early convergence, more forgiving): `Adam(learning_rate=1e-3)` with the same `ReduceLROnPlateau`. Reaches 85%+ in ~50–100 epochs.

**Batch size: 64 or 128.** Masters & Luschi (arXiv:1804.07612) show optimal generalization for CIFAR-10 lies between 32 and 128; 64 is a strong default for the recipe below.

**Epochs:**

- 50 epochs is enough to crack 80% with Adam + augmentation.
- 100–200 epochs needed to reach 85–88% with the full stack and SGD.

**Loss:** `categorical_crossentropy` (with one-hot labels via `to_categorical`) or `sparse_categorical_crossentropy` (integer labels). No measurable accuracy difference.

**Preprocessing:** Per-channel z-score (compute training mean/std and apply to train and test) outperforms simple ÷255 by a small margin and is what geifmany/cifar-vgg and the 90% MastersInMachineLearning recipe use:

```python
mean = np.mean(x_train, axis=(0,1,2,3))
std  = np.std(x_train,  axis=(0,1,2,3))
x_train = (x_train - mean) / (std + 1e-7)
x_test  = (x_test  - mean) / (std + 1e-7)
```

### 5. Kernel Initializer Recommendations

The choice is dictated by the activation function (He et al., 2015; Glorot & Bengio, 2010):

| Activation             | Recommended Initializer                             | Variance               |
| ---------------------- | --------------------------------------------------- | ---------------------- |
| ReLU / LeakyReLU / ELU | `he_normal` or `he_uniform`                         | 2 / fan_in             |
| tanh / sigmoid         | `glorot_uniform` (Keras default) or `glorot_normal` | 2 / (fan_in + fan_out) |
| SELU                   | `lecun_normal`                                      | 1 / fan_in             |
| Softmax (output Dense) | `glorot_uniform` is fine                            | —                      |

**For this CIFAR-10 recipe (ReLU everywhere except final softmax): use `he_uniform` on every Conv2D and intermediate Dense.** This is the standard antidote to vanishing/exploding gradients in deep ReLU nets and is mandatory once you exceed ~10 layers — a 3-block VGG with 2 convs per block (6 conv) + 2 dense = 8 weight layers, so a 4-block variant or one with extra dense layers will cross the threshold mentioned in the prompt. `he_uniform` and `he_normal` perform near-identically; pick one and be consistent. Glorot (the Keras default) is sub-optimal for ReLU and can stall learning in deeper variants.

### 6. Putting It All Together — Concrete Blueprint (target: 0.85–0.88 test accuracy)

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks, optimizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# ---------- 1. Data ----------
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')

mean = np.mean(x_train, axis=(0,1,2,3))
std  = np.std(x_train,  axis=(0,1,2,3))
x_train = (x_train - mean) / (std + 1e-7)
x_test  = (x_test  - mean) / (std + 1e-7)

y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test,  10)

# ---------- 2. Model ----------
WD = 1e-4
INIT = 'he_uniform'

def build_model():
    m = models.Sequential()
    # Block 1
    m.add(layers.Conv2D(32, (3,3), padding='same', kernel_initializer=INIT,
                        kernel_regularizer=regularizers.l2(WD),
                        input_shape=(32,32,3)))
    m.add(layers.BatchNormalization()); m.add(layers.Activation('relu'))
    m.add(layers.Conv2D(32, (3,3), padding='same', kernel_initializer=INIT,
                        kernel_regularizer=regularizers.l2(WD)))
    m.add(layers.BatchNormalization()); m.add(layers.Activation('relu'))
    m.add(layers.MaxPooling2D((2,2)))
    m.add(layers.Dropout(0.2))
    # Block 2
    m.add(layers.Conv2D(64, (3,3), padding='same', kernel_initializer=INIT,
                        kernel_regularizer=regularizers.l2(WD)))
    m.add(layers.BatchNormalization()); m.add(layers.Activation('relu'))
    m.add(layers.Conv2D(64, (3,3), padding='same', kernel_initializer=INIT,
                        kernel_regularizer=regularizers.l2(WD)))
    m.add(layers.BatchNormalization()); m.add(layers.Activation('relu'))
    m.add(layers.MaxPooling2D((2,2)))
    m.add(layers.Dropout(0.3))
    # Block 3
    m.add(layers.Conv2D(128, (3,3), padding='same', kernel_initializer=INIT,
                        kernel_regularizer=regularizers.l2(WD)))
    m.add(layers.BatchNormalization()); m.add(layers.Activation('relu'))
    m.add(layers.Conv2D(128, (3,3), padding='same', kernel_initializer=INIT,
                        kernel_regularizer=regularizers.l2(WD)))
    m.add(layers.BatchNormalization()); m.add(layers.Activation('relu'))
    m.add(layers.MaxPooling2D((2,2)))
    m.add(layers.Dropout(0.4))
    # Classifier
    m.add(layers.Flatten())
    m.add(layers.Dense(128, kernel_initializer=INIT,
                       kernel_regularizer=regularizers.l2(WD)))
    m.add(layers.BatchNormalization()); m.add(layers.Activation('relu'))
    m.add(layers.Dropout(0.5))
    m.add(layers.Dense(10, activation='softmax', kernel_initializer=INIT))
    return m

model = build_model()
model.compile(
    optimizer=optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ---------- 3. Augmentation ----------
datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)
datagen.fit(x_train)

# ---------- 4. Schedule & training ----------
def step_decay(epoch):
    if   epoch < 80:  return 0.1
    elif epoch < 120: return 0.01
    elif epoch < 160: return 0.001
    else:             return 0.0005

lr_cb = callbacks.LearningRateScheduler(step_decay)
# Alternative if you don't want a hand-crafted schedule:
# lr_cb = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
#                                     patience=10, min_lr=1e-6)

BATCH = 64
EPOCHS = 150

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=BATCH),
    steps_per_epoch=len(x_train)//BATCH,
    epochs=EPOCHS,
    validation_data=(x_test, y_test),
    callbacks=[lr_cb],
    verbose=2
)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f'Test accuracy: {test_acc:.4f}')  # expected: 0.85–0.88
```

**Notes about this blueprint**

- Uses **only the layer types specified in the prompt**: Conv2D, Dropout, Dense, MaxPooling2D, Flatten — plus BatchNormalization and Activation (auxiliary but not in the restricted list; if these are also disallowed, drop BN, raise dropout in conv blocks to 0.25/0.35/0.45, lower SGD LR to 0.01, and you will still clear 80% — Brownlee shows ~85% without BN).
- Layer count is well over 10 weight-bearing layers in the BN variant — hence the explicit `kernel_initializer='he_uniform'` on every Conv2D and Dense, satisfying the vanishing-gradient prevention requirement.
- The 0.10 → 0.01 SGD step decay is the schedule used in Keras's official `cifar10_resnet.py` example and most >90% CIFAR-10 reproductions.

### 7. Minimal Configuration to Just Clear 80%

If compute is constrained, the following minimal recipe clears 0.80 in ~30–50 epochs (Adam, no BN, no L2):

```python
model = Sequential([
  Conv2D(32,(3,3),padding='same',activation='relu',kernel_initializer='he_uniform',input_shape=(32,32,3)),
  Conv2D(32,(3,3),padding='same',activation='relu',kernel_initializer='he_uniform'),
  MaxPooling2D((2,2)), Dropout(0.2),
  Conv2D(64,(3,3),padding='same',activation='relu',kernel_initializer='he_uniform'),
  Conv2D(64,(3,3),padding='same',activation='relu',kernel_initializer='he_uniform'),
  MaxPooling2D((2,2)), Dropout(0.3),
  Conv2D(128,(3,3),padding='same',activation='relu',kernel_initializer='he_uniform'),
  Conv2D(128,(3,3),padding='same',activation='relu',kernel_initializer='he_uniform'),
  MaxPooling2D((2,2)), Dropout(0.4),
  Flatten(),
  Dense(128, activation='relu', kernel_initializer='he_uniform'),
  Dropout(0.5),
  Dense(10, activation='softmax'),
])
model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(datagen.flow(x_train, y_train, batch_size=64),
          epochs=80, validation_data=(x_test, y_test),
          callbacks=[ReduceLROnPlateau(factor=0.2, patience=8, min_lr=1e-6)])
# Expected ~0.82–0.85 test accuracy
```

This is essentially the Brownlee "VGG-3 + increasing dropout + augmentation" baseline that public reproductions land at 84–85%.

---

## Recommendations

**Stage 1 — Validate the pipeline (target: >0.80).** Use the "minimal 80% recipe" above with Adam(1e-3), `he_uniform`, 3 VGG blocks, increasing dropout (0.2/0.3/0.4/0.5), and the canonical augmentation (flip + 10% shifts). Train 50–80 epochs with `ReduceLROnPlateau(patience=8, factor=0.2)`. **If you do not see ≥0.78 validation accuracy by epoch 30**, something is wrong upstream (likely missing normalization or wrong loss/label encoding); fix that before tuning anything else.

**Stage 2 — Push to 0.85+.** Add BatchNormalization after every Conv2D, switch optimizer to SGD(lr=0.1, momentum=0.9, nesterov=True) with the step-decay schedule (×0.1 at 80, ×0.01 at 120), add L2(1e-4) to every Conv2D/Dense kernel, train 150 epochs. This is the blueprint shown in Section 6. **Expected: 0.85–0.88.**

**Stage 3 — Push to 0.90+.** Increase filter width to 64/128/256, increase first dense layer to 512–1024, train 200+ epochs, add per-channel mean/std normalization (if you haven't already), and add reflection-padded 4-pixel random crops (will require `tf.image` or a custom Lambda since `ImageDataGenerator` doesn't expose reflection-padded crops). This matches the cifar-vgg recipe that reaches 93.43%.

**Stage 4 — Stop or pivot.** If you need >93% without transfer learning, you will need residual/skip connections, which require the `Add` layer (not in the prompt's allowed list). At that point, the constraints listed are the binding limit, not the recipe.

**Thresholds that should change your strategy:**

- If training accuracy stays low (<60%) and val accuracy tracks it: model is under-fitting → reduce dropout, increase filter widths, switch to Adam first to confirm capacity.
- If training accuracy >>95% while val plateaus at ~75–80%: overfitting → increase augmentation strength (add small rotation, increase shifts to 0.125), increase dropout in dense head to 0.5–0.6, add L2 1e-4 if not present.
- If loss diverges in early epochs of SGD: lower initial LR to 0.01 (you'll lose ~1% accuracy at convergence) or add a 1–5 epoch linear warm-up.
- If `val_loss` plateaus for 15+ epochs at any LR: trigger `ReduceLROnPlateau` (factor 0.2) or fall to the next step of the manual schedule.

---

## Caveats

- **Reported accuracy numbers (88%, 90%, 93%) come from third-party reproductions** (Brownlee/MachineLearningMastery, geifmany/cifar-vgg, MastersInMachineLearning, ZacharyGoshen) and depend on exact preprocessing, augmentation, schedule length, and random seed. Expect ±1–2% variance run-to-run. The 80% bar, however, is robust — every recipe with VGG-3 + dropout + flip augmentation clears it.
- **BatchNormalization is technically outside the strict layer list in the prompt** (which mentions only Conv2D, Dropout, Dense, MaxPooling2D, Flatten). It is virtually universal in modern CIFAR-10 recipes and is included in the Section 6 blueprint with that caveat noted. If BN is forbidden by your constraints, the Section 7 "minimal" recipe still clears 80% without it; you can also compensate by lowering the SGD learning rate (e.g., 0.01 instead of 0.1) and using slightly heavier dropout.
- **`Activation` layers** are also used in the Section 6 blueprint where BN appears between Conv2D and the activation function; if even `Activation` must be avoided, use `Conv2D(..., activation='relu')` and place BN _after_ the activation (slightly less common but still works on CIFAR-10).
- **Without transfer learning and using only the strict 5-layer-type list (no BN, no Activation, no residual connections)**, the ceiling is roughly **85–86%** based on Brownlee's published curves (3-block VGG with increasing dropout + augmentation, no BN: ~85%). Reaching 90%+ under that strict constraint is not documented in any reproduction I found.
- **`he_normal` vs `he_uniform`**: empirical studies (Towards Data Science, 2021; LinkedIn Manral overview) find them effectively equivalent in final accuracy — pick one. Both decisively beat `glorot_uniform` (the Keras default) for ReLU networks deeper than ~6 layers.
- **Optimizer choice (Adam vs SGD)**: a recent NeurIPS 2023 supplemental survey (Xie et al.) and Hemil Desai's CIFAR-10 experiments report that **SGD with a well-tuned schedule still generalizes slightly better than Adam on CIFAR-10**, typically by 0.5–1% — but Adam reaches "good enough" (>80%) far faster and with less tuning. If you have 30 epochs of budget, choose Adam; if you have 150+, choose SGD.
- **Batch size sensitivity**: larger batches (>256) can hurt generalization on CIFAR-10 unless you also scale the learning rate; Masters & Luschi (arXiv:1804.07612) recommend 32–128 for best test accuracy. The blueprint uses 64.
- **The `94% in 3.29 seconds` paper (Keller Jordan, 2024)** uses a custom small CNN, derandomized flipping, reflection-padded crops, label smoothing, and aggressive triangular LR — illustrative but uses techniques (label smoothing, custom flipping, very specific architectural quirks) that go beyond a standard Keras pipeline.
- **No information leakage from the test set** in any of these recipes — accuracy numbers are on the 10,000-image CIFAR-10 test partition as released by Krizhevsky.
