import torch
import joblib
import numpy as np
import pandas as pd
import warnings
from train_model import TempLSTM  
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def load_parameters():
    model = TempLSTM(input_size=14, output_size=10)  
    model.load_state_dict(torch.load("models/temp_lstm_final.pt"))
    model.eval()
    
    # Load scalers
    scaler = joblib.load("models/scaler.save")
    param_scaler = joblib.load("models/scaler_params.save")
    
    return model, scaler, param_scaler

def predict_next_step(past_sequence, static_params):
    """
    Predict next timestep temperatures
    Args:
        past_sequence: 2D array [10 timesteps, 10 sensors]
        static_params: List [h, flux, abs, surf]
    Returns:
        Predicted temperatures for next timestep [10,] in °C
    """
    model, scaler, param_scaler = load_parameters()
    
    thermal_cols = ["TC1_tip", "TC2", "TC3", "TC4", "TC5", 
                   "TC6", "TC7", "TC8", "TC9", "TC10"]
    
    thermal_df = pd.DataFrame(past_sequence, columns=thermal_cols)
    thermal_df = thermal_df.fillna(method='ffill').fillna(0)  
    seq_scaled = scaler.transform(thermal_df)
    
    param_cols = ["h", "flux", "abs", "surf"]
    params_scaled = param_scaler.transform([static_params])[0]
    
    X = [list(row) + params_scaled.tolist() for row in seq_scaled]
    X = torch.tensor([X], dtype=torch.float32)  
    
    with torch.no_grad():
        pred = model(X).numpy()
    return scaler.inverse_transform(pred)[0]
