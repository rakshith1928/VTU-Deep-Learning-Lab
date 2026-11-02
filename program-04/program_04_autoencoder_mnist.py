"""
VTU Deep Learning Lab (2022 Scheme) - Program 4
Build and demonstrate an autoencoder network using neural layers for
data compression on image dataset.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, Model

outdir = Path(__file__).resolve().parent / "outputs"
outdir.mkdir(parents=True, exist_ok=True)

# Load dataset
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# Normalize and flatten
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# Input
input_img = layers.Input(shape=(784,))

# Encoder - compression
encoded = layers.Dense(32, activation="relu")(input_img)

# Decoder - reconstruction
decoded = layers.Dense(784, activation="sigmoid")(encoded)

# Autoencoder
model = Model(input_img, decoded)

# Compile
model.compile(optimizer="adam", loss="binary_crossentropy")

# Train
model.fit(x_train, x_train, epochs=5, batch_size=256, verbose=2)

# Encoder model
encoder = Model(input_img, encoded)

# Compressed images
encoded_imgs = encoder.predict(x_test, verbose=0)

# Reconstructed images
decoded_imgs = model.predict(x_test, verbose=0)

# Display original and reconstructed images
n = 5
plt.figure(figsize=(10, 4))
for i in range(n):
    # Original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # Reconstructed
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')

plt.tight_layout()
plt.savefig(outdir / "autoencoder_reconstruction.png", dpi=120)
print(f"Saved figure: {outdir / 'autoencoder_reconstruction.png'}")

# Display sizes
print("Original size:", x_test.shape)
print("Compressed size:", encoded_imgs.shape)
