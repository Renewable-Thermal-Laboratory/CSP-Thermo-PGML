import torch
import torch.nn as nn

class ImprovedTempModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        
        # Enhanced architecture
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(256, 128),  # 128*2 for bidirectional
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_size)
        )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.1)

    def forward(self, x):
        # LSTM layer
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, 256]
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)  # [batch, seq_len, 1]
        context = torch.sum(attention_weights * lstm_out, dim=1)  # [batch, 256]
        
        # Final prediction
        return self.fc(context)