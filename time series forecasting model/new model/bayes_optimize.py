import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, LeakyReLU
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from bayes_opt import BayesianOptimization

# Load training data
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

# Scale y
from sklearn.preprocessing import MinMaxScaler
scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train)

def train_lstm_model(lstm_units_1, lstm_units_2, lstm_units_3, l2_reg):
    """
    Black-box function for Bayesian optimization.
    """
    lstm_units_1 = int(lstm_units_1)
    lstm_units_2 = int(lstm_units_2)
    lstm_units_3 = int(lstm_units_3)
    l2_reg = float(l2_reg)
    
    model = Sequential()
    model.add(LSTM(
        lstm_units_1,
        return_sequences=True,
        input_shape=(X_train.shape[1], X_train.shape[2]),
        kernel_regularizer=l2(l2_reg)))
    model.add(LeakyReLU(alpha=0.3))

    model.add(LSTM(
        lstm_units_2,
        return_sequences=True,
        kernel_regularizer=l2(l2_reg)))
    model.add(LeakyReLU(alpha=0.3))

    model.add(LSTM(
        lstm_units_3,
        kernel_regularizer=l2(l2_reg)))
    model.add(LeakyReLU(alpha=0.3))

    model.add(Dense(10))
    
    model.compile(optimizer='adam', loss='mse')
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train,
        y_train_scaled,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        verbose=0,
        callbacks=[early_stop]
    )
    
    val_loss = min(history.history["val_loss"])
    
    print(f"Tested config: {lstm_units_1}-{lstm_units_2}-{lstm_units_3} | L2={l2_reg:.6f} | Val Loss={val_loss}")
    
    return -val_loss  # we NEGATE it because Bayesian optimization maximizes the function

pbounds = {
    "lstm_units_1": (64, 1024),
    "lstm_units_2": (64, 512),
    "lstm_units_3": (64, 256),
    "l2_reg": (1e-5, 1e-2)
}

optimizer = BayesianOptimization(
    f=train_lstm_model,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(
    init_points=5,
    n_iter=15,
)

print("Best configuration found:")
print(optimizer.max)
