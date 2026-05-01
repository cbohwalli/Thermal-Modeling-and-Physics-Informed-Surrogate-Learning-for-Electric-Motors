import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Constants
WINDOW_SIZE = 60 
INPUT_FEATURES = 1 
OUTPUT_TARGETS = 4

# Data preprocessing
# ------------------------------------------------------------------------

# 1. Load Data
df = pd.read_csv('drive_cycle_dataset.csv')

cycle_ids = df['drive_cycle_number'].unique()
np.random.shuffle(cycle_ids) 

# Split cycles (80% train, 20% validation)
split_idx = int(0.8 * len(cycle_ids))
train_ids = cycle_ids[:split_idx]
val_ids = cycle_ids[split_idx:]

train_df = df[df['drive_cycle_number'].isin(train_ids)]
val_df = df[df['drive_cycle_number'].isin(val_ids)]

# 2. Normalization
scaler_input = MinMaxScaler()
scaler_target = MinMaxScaler()

# Fit on training data only to avoid data leakage
scaler_input.fit(train_df[['load']])
scaler_target.fit(train_df[['t_stator', 't_rotor_1', 't_rotor_2', 't_housing']])

# Transform both
train_x = scaler_input.transform(train_df[['load']])
train_y = scaler_target.transform(train_df[['t_stator', 't_rotor_1', 't_rotor_2', 't_housing']])

val_x = scaler_input.transform(val_df[['load']])
val_y = scaler_target.transform(val_df[['t_stator', 't_rotor_1', 't_rotor_2', 't_housing']])

# 3. Cycle-Aware Sequence Generation
def create_sequences_by_cycle(df_cycles, x_data, y_data, window_size):
    x_seq, y_seq = [], []
    for cycle_id in df_cycles['drive_cycle_number'].unique():
        # Get indices for this cycle
        indices = np.where(df_cycles['drive_cycle_number'] == cycle_id)[0]
        # Get start/end in the transformed array
        start, end = indices[0], indices[-1] + 1
        
        cycle_x = x_data[start:end]
        cycle_y = y_data[start:end]
        
        for i in range(len(cycle_x) - window_size):
            x_seq.append(cycle_x[i : i + window_size])
            y_seq.append(cycle_y[i + window_size])
            
    return np.array(x_seq), np.array(y_seq)

WINDOW_SIZE = 60
X_train, y_train = create_sequences_by_cycle(train_df, train_x, train_y, WINDOW_SIZE)
X_val, y_val = create_sequences_by_cycle(val_df, val_x, val_y, WINDOW_SIZE)

# ------------------------------------------------------------------------

# Initialize the Sequential model
model = Sequential([
    LSTM(32, input_shape=(WINDOW_SIZE, INPUT_FEATURES)),
    Dense(OUTPUT_TARGETS)
])

model.compile(optimizer='adam', loss='mse')

history = model.fit(
    X_train, y_train,
    epochs=20,            
    batch_size=32,        
    validation_data=(X_val, y_val),
    verbose=1
)

predictions_scaled = model.predict(X_val)

# Convert scaled predictions back to actual temperature values
predictions = scaler_target.inverse_transform(predictions_scaled)

# Reverse the scaling for Ground Truth
ground_truth = scaler_target.inverse_transform(y_val)

# visualization

# Setup the Plotting Layout
fig, axes = plt.subplots(4, 1, figsize=(12, 12))
labels = ['Stator', 'Rotor 1', 'Rotor 2', 'Housing']

# Plot the data
for i in range(4):
    axes[i].plot(ground_truth[:, i], label='Actual', color='blue', alpha=0.6)
    axes[i].plot(predictions[:, i], label='Predicted', color='red', linestyle='--')
    axes[i].set_title(f'Comparison: {labels[i]}')
    axes[i].legend()
    axes[i].set_ylabel('Temp')

plt.tight_layout()
plt.savefig('ground_truth_vs_predictions_v0.png')
print("Plot saved as ground_truth_vs_predictions_v0.png")

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss, label='Training Loss', color='blue')
plt.plot(epochs, val_loss, label='Validation Loss', color='orange', linestyle='--')

plt.title('Model Training Progress')
plt.xlabel('Epochs')
plt.ylabel('Loss (Mean Squared Error)')
plt.legend()
plt.grid(True)
plt.savefig('training_loss_curve_v0.png')
print("Plot saved as training_loss_curve_v0.png")