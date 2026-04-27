import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Constants
WINDOW_SIZE = 60 
INPUT_FEATURES = 1 
OUTPUT_TARGETS = 4

# Data preprocessing
# ------------------------------------------------------------------------

# 1. Load Data
df = pd.read_csv('dataset_drive_cycle.csv')

# 2. Normalize (LSTMs perform best with scaled data)
scaler_input = MinMaxScaler()
scaler_target = MinMaxScaler()

X = scaler_input.fit_transform(df[['load']])
y = scaler_target.fit_transform(df[['t_stator', 't_rotor1', 't_rotor2', 't_housing']])

# 3. Sequence Generation (The Sliding Window)
def create_sequences(data_x, data_y, window_size):
    x_seq, y_seq = [], []
    for i in range(len(data_x) - window_size):
        x_seq.append(data_x[i : i + window_size])
        y_seq.append(data_y[i + window_size])
    return np.array(x_seq), np.array(y_seq)

X_final, y_final = create_sequences(X, y, WINDOW_SIZE)

# ------------------------------------------------------------------------





