import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import mean_absolute_error
from plot_residual_errors import plot_sensor_errors
import matplotlib.pyplot as plt
import joblib
import os
from dataset_builder import TempSequenceDataset

# Configuration
SEQUENCE_LENGTH = 10
BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 3e-4
PATIENCE = 15
VALIDATION_SPLIT = 0.15
WEIGHT_DECAY = 1e-5
EPSILON = 1e-6

# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Enhanced Model Architecture
class HighPrecisionTempModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        
        # Main LSTM Encoder
        self.encoder = nn.LSTM(input_size, 512, num_layers=3, 
                              batch_first=True, dropout=0.1)
        
        # Attention Mechanism
        self.attention = nn.MultiheadAttention(512, num_heads=8, batch_first=True)
        
        # Sensor-specific decoders
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.SiLU(),
                nn.Linear(256, 128),
                nn.SiLU(),
                nn.Linear(128, 1)
            ) for _ in range(output_size)
        ])
        
        # Temperature stabilization
        self.stabilizer = nn.Sequential(
            nn.Linear(output_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        # Encoder
        enc_out, _ = self.encoder(x)
        
        # Attention
        attn_out, _ = self.attention(enc_out, enc_out, enc_out)
        
        # Sensor-specific predictions
        predictions = []
        for decoder in self.decoders:
            predictions.append(decoder(attn_out[:, -1, :]))
        
        combined = torch.cat(predictions, dim=1)
        
        # Stabilization
        stabilized = self.stabilizer(combined)
        
        return combined + 0.1 * stabilized  # Residual connection

# Physics-Informed Loss
class PrecisionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_loss = nn.L1Loss()
        self.temp_range = (20, 50)  # Expected temperature range
        
    def forward(self, preds, targets):
        # Base MAE
        loss = self.base_loss(preds, targets)
        
        # Temperature bounds penalty
        lower_penalty = torch.relu(self.temp_range[0] - preds).mean()
        upper_penalty = torch.relu(preds - self.temp_range[1]).mean()
        range_penalty = (lower_penalty + upper_penalty) * 0.3
        
        # Thermal gradient penalty (TC1 should be warmer than TC10)
        gradient_penalty = torch.relu(preds[:, -1] - preds[:, 0]).mean() * 0.2
        
        return loss + range_penalty + gradient_penalty