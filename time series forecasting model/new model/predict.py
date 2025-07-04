import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model

TEMP_COLUMNS = ['TC1_tip','TC2','TC3','TC4','TC5','TC6','TC7','TC8','TC9','TC10']
STATIC_COLUMNS = ['h','flux','abs','surf']
TIME_COLUMN = ['Time']

SEQ_LENGTH = 5

scaler_X_min = np.load("scaler_X_min.npy")
scaler_X_max = np.load("scaler_X_max.npy")
scaler_y_min = np.load("scaler_y_min.npy")
scaler_y_max = np.load("scaler_y_max.npy")

def scale_X(X):
    return (X - scaler_X_min) / (scaler_X_max - scaler_X_min + 1e-8)

def inverse_scale_y(y_scaled):
    return y_scaled * (scaler_y_max - scaler_y_min) + scaler_y_min

def prepare_middle_sequence(df, seq_length=SEQ_LENGTH):
    n_rows = len(df)
    if n_rows < seq_length + 1:
        raise ValueError(f"Data too short: only {n_rows} rows")
    middle_idx = np.random.randint(seq_length, n_rows - 1)
    seq_start = middle_idx - seq_length
    seq_end = middle_idx
    seq_df = df.iloc[seq_start:seq_end]
    temps_seq = seq_df[TEMP_COLUMNS].values
    static_time_seq = seq_df[STATIC_COLUMNS + TIME_COLUMN].values
    last_static_time = static_time_seq[-1]
    static_repeated = np.tile(last_static_time, (seq_length, 1))
    X_seq = np.hstack([temps_seq, static_repeated])
    target_row = df.iloc[middle_idx]
    y_true = target_row[TEMP_COLUMNS].values
    return X_seq, y_true

def plot_depth_profile(actual, predicted, filename, save_path=None):
    depths = np.linspace(0, -0.016, len(actual))
    tc_numbers = list(range(10, 0, -1))
    actual_ordered = actual[::-1]
    predicted_ordered = predicted[::-1]
    plt.figure(figsize=(7,6))
    plt.plot(actual_ordered, depths, 'o-', label='Actual', color='blue', markersize=6)
    plt.plot(predicted_ordered, depths, 'x--', label='Predicted', color='red', markersize=8)
    for i, (act, pred, tc) in enumerate(zip(actual_ordered, predicted_ordered, tc_numbers)):
        plt.text(act, depths[i], f'TC{tc}', ha='right', va='center', fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.1", facecolor='lightblue', alpha=0.7))
        plt.text(pred, depths[i], f'TC{tc}', ha='left', va='center', fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.1", facecolor='lightcoral', alpha=0.7))
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Depth (m)")
    plt.title(f"Vertical Profile: {filename}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def main():
    model = load_model("solar_lstm_model.keras")
    graphs_dir = "./graphs"
    os.makedirs(graphs_dir, exist_ok=True)
    test_files_dir = "./data/test"
    test_files = [os.path.join(test_files_dir, f) for f in os.listdir(test_files_dir) if f.endswith(".csv")]
    for file in test_files:
        try:
            df = pd.read_csv(file)
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            X_seq, y_true = prepare_middle_sequence(df, SEQ_LENGTH)
            X_seq_scaled = scale_X(X_seq)
            X_seq_scaled = np.expand_dims(X_seq_scaled, axis=0)
            y_pred_scaled = model.predict(X_seq_scaled, verbose=0)
            y_pred = inverse_scale_y(y_pred_scaled)[0]
            residuals = np.abs(y_true - y_pred)
            avg_residual = residuals.mean()
            print(f"\nFile: {os.path.basename(file)}")
            print(f"Average residual: {avg_residual:.3f} °C")
            for i, tc in enumerate(TEMP_COLUMNS):
                print(f"{tc:<10} Pred: {y_pred[i]:.2f} °C  Actual: {y_true[i]:.2f} °C  Residual: {residuals[i]:.2f} °C")
            save_path = os.path.join(graphs_dir, f"{os.path.basename(file).replace('.csv', '')}.png")
            plot_depth_profile(y_true, y_pred, os.path.basename(file), save_path=save_path)
            print(f"Saved graph to {save_path}")
        except Exception as e:
            print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    main()
