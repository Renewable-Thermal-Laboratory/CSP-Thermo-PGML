import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, LeakyReLU
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# Load data
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Load y scaler
scaler_y_min = np.load("scaler_y_min.npy")
scaler_y_max = np.load("scaler_y_max.npy")

def scale_y(y):
    return (y - scaler_y_min) / (scaler_y_max - scaler_y_min + 1e-8)

def inverse_scale_y(y_scaled):
    return y_scaled * (scaler_y_max - scaler_y_min) + scaler_y_min

y_train_scaled = scale_y(y_train)
y_val_scaled = scale_y(y_val)
y_test_scaled = scale_y(y_test)

# Hyperparameters from Bayesian optimization
LSTM_UNITS_1 = 113
LSTM_UNITS_2 = 217
LSTM_UNITS_3 = 171
L2_REG = 0.001
BATCH_SIZE = 32

# Build model
model = Sequential()
model.add(LSTM(
    LSTM_UNITS_1,
    return_sequences=True,
    input_shape=(X_train.shape[1], X_train.shape[2]),
    kernel_regularizer=l2(L2_REG)
))
model.add(LeakyReLU(negative_slope=0.3))
model.add(LSTM(
    LSTM_UNITS_2,
    return_sequences=True,
    kernel_regularizer=l2(L2_REG)
))
model.add(LeakyReLU(negative_slope=0.3))
model.add(LSTM(
    LSTM_UNITS_3,
    kernel_regularizer=l2(L2_REG)
))
model.add(LeakyReLU(negative_slope=0.3))
model.add(Dense(10))

model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train_scaled,
    epochs=100,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val_scaled),
    callbacks=[early_stop]
)

# Save model in Keras format
model.save("solar_lstm_model.keras")

# Evaluate on test set
y_pred_scaled = model.predict(X_test, verbose=0)
y_pred = inverse_scale_y(y_pred_scaled)
y_test_orig = y_test

mse = mean_squared_error(y_test_orig, y_pred)
mae = mean_absolute_error(y_test_orig, y_pred)
r2 = r2_score(y_test_orig, y_pred)

print(f"Test MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test R²: {r2:.4f}")
