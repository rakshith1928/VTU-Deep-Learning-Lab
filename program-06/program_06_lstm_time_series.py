"""
VTU Deep Learning Lab (2022 Scheme) - Program 6
Design and implement a deep learning network for forecasting
time series data.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

outdir = Path(__file__).resolve().parent / "outputs"
outdir.mkdir(parents=True, exist_ok=True)

# 1. Generate synthetic time series data
np.random.seed(42)  # reproducible noise for the executed outputs
time = np.arange(0, 200, 0.1)
data = np.sin(time) + 0.1 * np.random.randn(len(time))

plt.plot(time, data)
plt.title("Synthetic Time Series Data")
plt.savefig(outdir / "fig1_raw_series.png", dpi=120)
print(f"Saved figure: {outdir / 'fig1_raw_series.png'}")

# 2. Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data.reshape(-1, 1))

# 3. Create sequences
def create_sequences(dataset, seq_length):
    X, y = [], []
    for i in range(len(dataset) - seq_length):
        X.append(dataset[i:i+seq_length, 0])
        y.append(dataset[i+seq_length, 0])
    return np.array(X), np.array(y)

SEQ_LEN = 20
X, y = create_sequences(data_scaled, SEQ_LEN)

# 4. Reshape for LSTM (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# 5. Train-test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 6. Build LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(SEQ_LEN, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# 7. Train model
history = model.fit(X_train, y_train, epochs=10, batch_size=32,
                    validation_data=(X_test, y_test), verbose=2)

# 8. Plot training loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("LSTM Training Loss")
plt.savefig(outdir / "fig2_loss.png", dpi=120)
print(f"Saved figure: {outdir / 'fig2_loss.png'}")

# 9. Forecast
predictions = model.predict(X_test, verbose=0)
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# 10. Plot predictions vs actual
plt.figure()
plt.plot(y_test_actual, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.title("Time Series Forecast: Actual vs Predicted")
plt.savefig(outdir / "fig3_forecast.png", dpi=120)
print(f"Saved figure: {outdir / 'fig3_forecast.png'}")
