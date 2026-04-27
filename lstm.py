import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Constants
WINDOW_SIZE = 60 
INPUT_FEATURES = 1 
OUTPUT_TARGETS = 4

# Data preprocessing
# ------------------------------------------------------------------------

# 1. Load Data
df = pd.read_csv('drive_cycle_dataset.csv')

# 2. Normalize (LSTMs perform best with scaled data)
scaler_input = MinMaxScaler()
scaler_target = MinMaxScaler()

X = scaler_input.fit_transform(df[['load']])
y = scaler_target.fit_transform(df[['t_stator', 't_rotor_1', 't_rotor_2', 't_housing']])

# 3. Sequence Generation (The Sliding Window)
def create_sequences(data_x, data_y, window_size):
    x_seq, y_seq = [], []
    for i in range(len(data_x) - window_size):
        x_seq.append(data_x[i : i + window_size])
        y_seq.append(data_y[i + window_size])
    return np.array(x_seq), np.array(y_seq)

X_final, y_final = create_sequences(X, y, WINDOW_SIZE)

# ------------------------------------------------------------------------

# Initialize the Sequential model
model = Sequential([
    LSTM(32, input_shape=(WINDOW_SIZE, INPUT_FEATURES)),
    
    Dense(OUTPUT_TARGETS)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Split into train/test (80% training, 20% validation)
split = int(0.8 * len(X_final))
X_train, X_test = X_final[:split], X_final[split:]
y_train, y_test = y_final[:split], y_final[split:]

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=1,            
    batch_size=32,        
    validation_data=(X_test, y_test),
    verbose=1
)

# Predict on test data
predictions_scaled = model.predict(X_test)

# Convert scaled predictions back to actual temperature values
predictions = scaler_target.inverse_transform(predictions_scaled)



# visualization of prediction vs ground truth
import matplotlib.pyplot as plt

# 1. Reverse the scaling for Ground Truth
# This makes sure the units match your predictions
ground_truth = scaler_target.inverse_transform(y_test)

# 2. Setup the Plotting Layout
# We create 4 rows (one for each temperature variable)
fig, axes = plt.subplots(4, 1, figsize=(12, 12))
labels = ['Stator', 'Rotor 1', 'Rotor 2', 'Housing']

# 3. Plot the data
for i in range(4):
    axes[i].plot(ground_truth[:, i], label='Actual', color='blue', alpha=0.6)
    axes[i].plot(predictions[:, i], label='Predicted', color='red', linestyle='--')
    axes[i].set_title(f'Comparison: {labels[i]}')
    axes[i].legend()
    axes[i].set_ylabel('Temp')

plt.tight_layout() # This keeps the titles from overlapping
plt.savefig('ground_truth_vs_predictions.png')
print("Plot saved as ground_truth_vs_predictions.png")

# visualization of training curve

# Extract data from the history object
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
plt.savefig('training_loss_curve.png')
print("Plot saved as training_loss_curve.png")