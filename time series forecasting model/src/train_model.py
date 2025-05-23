import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_builder import TempSequenceDataset
import os

BATCH_SIZE = 32
EPOCHS = 20
SEQUENCE_LENGTH = 10
LEARNING_RATE = 0.001

dataset = TempSequenceDataset(
    data_dir="data/processed",
    sequence_length=SEQUENCE_LENGTH
)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

input_size = dataset[0][0].shape[1]  # num_temps + 4
output_size = dataset[0][1].shape[0]  # num_temps only

print(f"Loaded dataset: {len(dataset)} samples")
print(f"Input shape: [batch, {SEQUENCE_LENGTH}, {input_size}]")
print(f"Target shape: [batch, {output_size}]")

class TempPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])  # [batch, hidden] → [batch, output_size]

model = TempPredictor(input_size=input_size, output_size=output_size)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

for epoch in range(EPOCHS):
    total_loss = 0
    for X, y in loader:
        preds = model(X)
        loss = criterion(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {total_loss / len(loader):.6f}")

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/temp_lstm.pt")
print("✅ Training complete. Model saved to models/temp_lstm.pt")
