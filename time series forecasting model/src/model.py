import torch
import torch.nn as nn

class TempLSTM(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, 256, batch_first=True)
        self.drop1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(256, 128, batch_first=True)
        self.drop2 = nn.Dropout(0.2)
        self.lstm3 = nn.LSTM(128, 64, batch_first=True)
        self.drop3 = nn.Dropout(0.2)
        self.fc = nn.Linear(64, output_size)
        self.act = nn.LeakyReLU(0.3)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.act(self.drop1(out))
        out, _ = self.lstm2(out)
        out = self.act(self.drop2(out))
        out, _ = self.lstm3(out)
        out = self.act(self.drop3(out))
        out = out[:, -1, :]  # use the last time step
        return self.fc(out)