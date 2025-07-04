import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import glob
import shutil
import random

# CONFIG
SEQUENCE_LENGTH = 10
SPLIT_RATIO = (0.7, 0.15, 0.15)  # train, val, test

DATA_DIR = "./processed_H6"
TRAIN_DIR = "./data/train"
VAL_DIR = "./data/val"
TEST_DIR = "./data/test"

TEMP_COLUMNS = ['TC1_tip','TC2','TC3','TC4','TC5','TC6','TC7','TC8','TC9','TC10']
STATIC_COLUMNS = ['h','flux','abs','surf']
TIME_COLUMN = ['Time']

# Clean and recreate folders
for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)

# Split files
all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
random.shuffle(all_files)

num_files = len(all_files)
train_end = int(SPLIT_RATIO[0] * num_files)
val_end = train_end + int(SPLIT_RATIO[1] * num_files)

train_files = all_files[:train_end]
val_files = all_files[train_end:val_end]
test_files = all_files[val_end:]

for f in train_files:
    shutil.copy(f, TRAIN_DIR)
for f in val_files:
    shutil.copy(f, VAL_DIR)
for f in test_files:
    shutil.copy(f, TEST_DIR)

print(f"Split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

def prepare_sequences(file_list):
    X_list, y_list = [], []
    scaler_X = MinMaxScaler()
    
    for filepath in file_list:
        df = pd.read_csv(filepath)
        
        temp_static_time = df[TEMP_COLUMNS + STATIC_COLUMNS + TIME_COLUMN].values
        temp_static_time_scaled = scaler_X.fit_transform(temp_static_time)
        
        temps_scaled = temp_static_time_scaled[:, :10]
        static_time_scaled = temp_static_time_scaled[:, 10:]
        
        for i in range(len(df) - SEQUENCE_LENGTH):
            seq_X = temps_scaled[i:i+SEQUENCE_LENGTH]
            static_part = static_time_scaled[i+SEQUENCE_LENGTH-1]
            static_repeated = np.tile(static_part, (SEQUENCE_LENGTH, 1))
            seq_input = np.hstack((seq_X, static_repeated))
            
            target = df[TEMP_COLUMNS].iloc[i+SEQUENCE_LENGTH].values
            X_list.append(seq_input)
            y_list.append(target)
    
    X_all = np.array(X_list)
    y_all = np.array(y_list)
    return X_all, y_all, scaler_X

# Prepare splits
splits = {
    "train": TRAIN_DIR,
    "val": VAL_DIR,
    "test": TEST_DIR,
}

scaler_X = None
y_all_concat = []

for split, dir_path in splits.items():
    file_list = glob.glob(os.path.join(dir_path, "*.csv"))
    X, y, scaler_X = prepare_sequences(file_list)
    np.save(f"X_{split}.npy", X)
    np.save(f"y_{split}.npy", y)
    y_all_concat.append(y)
    print(f"{split} shape: {X.shape}")

# Fit scaler_y on real y values
y_all_concat_flat = np.concatenate(y_all_concat, axis=0)
scaler_y = MinMaxScaler()
scaler_y.fit(y_all_concat_flat)

np.save("scaler_X_min.npy", scaler_X.data_min_)
np.save("scaler_X_max.npy", scaler_X.data_max_)
np.save("scaler_y_min.npy", scaler_y.data_min_)
np.save("scaler_y_max.npy", scaler_y.data_max_)

print("Data preparation complete.")
