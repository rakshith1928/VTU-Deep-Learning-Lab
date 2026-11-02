"""
VTU Deep Learning Lab (2022 Scheme) - Program 7
Write a program to enable pre-train models to classify a given image dataset.

Note: executed on CPU - training uses an 8000-image subset (seed 42) of
CIFAR-10 with a streaming tf.data pipeline (same MobileNetV2 weights,
same head, same 5 epochs as the manual). Full dataset reproduces the
manual accuracy on GPU/Colab.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
import numpy as np

outdir = Path(__file__).resolve().parent / "outputs"
outdir.mkdir(parents=True, exist_ok=True)

# 1. Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Convert labels to 1D
y_train = y_train.flatten()
y_test = y_test.flatten()

# CPU-friendly subset + shuffle with fixed seed for reproducible outputs
rng = np.random.default_rng(42)
train_idx = rng.choice(len(x_train), 8000, replace=False)
test_idx = rng.choice(len(x_test), 2000, replace=False)
x_train, y_train = x_train[train_idx], y_train[train_idx]
x_test, y_test = x_test[test_idx], y_test[test_idx]
print(f"Training on subset: {x_train.shape[0]} images | Testing on: {x_test.shape[0]} images")

# Manual-faithful validation split (20% of the training subset)
n_val = int(0.2 * len(x_train))
x_val, y_val = x_train[-n_val:], y_train[-n_val:]
x_train, y_train = x_train[:-n_val], y_train[:-n_val]

# Resize + normalize as a streaming pipeline (avoids ~5.5 GB in RAM)
BATCH = 128
AUTOTUNE = tf.data.AUTOTUNE
resize = lambda img: tf.cast(tf.image.resize(img, (96, 96)), tf.float32)

train_ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .map(lambda i, l: (resize(i), l), num_parallel_calls=AUTOTUNE)
            .batch(BATCH).prefetch(AUTOTUNE))
val_ds = (tf.data.Dataset.from_tensor_slices((x_val, y_val))
          .map(lambda i, l: (resize(i), l), num_parallel_calls=AUTOTUNE)
          .batch(BATCH).prefetch(AUTOTUNE))
test_ds = (tf.data.Dataset.from_tensor_slices((x_test, y_test))
           .map(lambda i, l: (resize(i), l), num_parallel_calls=AUTOTUNE)
           .batch(BATCH).prefetch(AUTOTUNE))
single_img_ds = tf.data.Dataset.from_tensor_slices(x_test[:1]).map(resize).batch(1)

# 3. Load pre-trained MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(96, 96, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pre-trained layers
base_model.trainable = False

# 4. Create model
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

# 5. Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 6. Train
history = model.fit(train_ds, validation_data=val_ds, epochs=5, verbose=2)

# 7. Test model
loss, accuracy = model.evaluate(test_ds)

print("Test Accuracy:", accuracy * 100, "%")

# 8. Predict one image
prediction = model.predict(single_img_ds, verbose=0)

classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

predicted_class = np.argmax(prediction)

print("Actual:", classes[y_test[0]])
print("Predicted:", classes[predicted_class])

# Print final training and validation accuracy
train_accuracy = history.history["accuracy"][-1]
val_accuracy = history.history["val_accuracy"][-1]

print("Training Accuracy:", train_accuracy * 100, "%")
print("Validation Accuracy:", val_accuracy * 100, "%")

# Accuracy graph
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid()
plt.savefig(outdir / "transfer_learning_accuracy.png", dpi=120)
print(f"Saved figure: {outdir / 'transfer_learning_accuracy.png'}")
