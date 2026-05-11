import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from src.visualisation.plotting import visualise_results
from src.data.preprocessing import preprocess_data

# Constants
WINDOW_SIZE = 60 
INPUT_FEATURES = 5 
OUTPUT_TARGETS = 4

feature_cols = ['load', 't_stator', 't_rotor_1', 't_rotor_2', 't_housing']
target_cols = ['t_stator', 't_rotor_1', 't_rotor_2', 't_housing']

# Data preprocessing
# ------------------------------------------------------------------------

# 1. Load Data
df = pd.read_csv('drive_cycle_dataset.csv')

training_split = 80 # 80% training data 20% validation data
train_x, train_y, val_x, val_y, scaler_input, scaler_target = preprocess_data(df, training_split, feature_cols, target_cols)

# Adding noice
noise_level = 0.60  # 60% of the normalized range (0 to 1)

# Generate Gaussian noise matching the shape of your target data
train_noise = np.random.normal(0, noise_level, size=train_y.shape)

# Add the noise to the ground truth
train_y = train_y + train_noise

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

# visualisation
experiment_number = '5'
visualise_results(ground_truth, predictions, history, experiment_number)