import torch
import torch.nn as nn

class TempLSTM(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, 256, batch_first=True)
        self.drop1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(256, 128, batch_first=True)
        self.drop2 = nn.Dropout(0.3)
        self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = self.drop2(out)
        attn_out, _ = self.attn(out, out, out)
        final = attn_out[:, -1, :]  # Take last timestep
        return self.fc(final)
