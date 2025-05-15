from dataset_builder import TempSequenceDataset
from torch.utils.data import DataLoader

DATA_DIR = "data/processed"
SEQUENCE_LENGTH = 10
BATCH_SIZE = 32

dataset = TempSequenceDataset(data_dir=DATA_DIR, sequence_length=SEQUENCE_LENGTH)
print(f"Loaded dataset with {len(dataset)} samples")

if len(dataset) == 0:
    raise RuntimeError("Dataset is empty. Check processed files and formatting.")

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

for X, y in loader:
    print("Batch shapes:")
    print(f"X: {X.shape} → (batch, seq_len, input_features)")
    print(f"y: {y.shape} → (batch, output_features)")
    break
