"""
VTU Deep Learning Lab (2022 Scheme) - Program 5
Design and implement a deep learning network for classification of
textual documents.
"""

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense

try:  # keras legacy location (manual)
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except ImportError:  # keras 3 location
    from tensorflow.keras.utils import pad_sequences

# 1. Load IMDB dataset (Movie Reviews)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)

# 2. Make all reviews the same length
x_train = pad_sequences(x_train, maxlen=200)
x_test = pad_sequences(x_test, maxlen=200)

# 3. Create deep learning model
model = Sequential([
    Embedding(10000, 32),
    GlobalAveragePooling1D(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 4. Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 5. Train model
model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    verbose=2
)

# 6. Test model
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print("Test Accuracy:", accuracy)

# 7. Predict first 5 reviews
predictions = model.predict(x_test[:5], verbose=0)

print("\nReview Classification:")
for i in range(5):
    if predictions[i][0] >= 0.5:
        result = "Positive"
    else:
        result = "Negative"

    print("Review", i + 1, ":", result)
    print("Probability:", predictions[i][0])
