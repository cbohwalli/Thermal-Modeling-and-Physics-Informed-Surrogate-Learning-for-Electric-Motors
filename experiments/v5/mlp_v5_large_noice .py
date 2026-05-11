import pandas as pd
import numpy as np
from tensorflow.keras import layers, models
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
val_noise = np.random.normal(0, noise_level, size=val_y.shape)

# Add the noise to the ground truth
train_y = train_y + train_noise

# 3. Cycle-Aware Step-Wise Data Generation
def create_step_pairs_by_cycle(df_cycles, x_data, y_data):
    x_pairs, y_pairs = [], []
    
    for cycle_id in df_cycles['drive_cycle_number'].unique():
        # Get indices for this specific cycle
        indices = np.where(df_cycles['drive_cycle_number'] == cycle_id)[0]
        start, end = indices[0], indices[-1] + 1
        
        cycle_x = x_data[start:end]
        cycle_y = y_data[start:end]
        
        for i in range(len(cycle_x) - 1): 
            x_pairs.append(cycle_x[i])      # Current state Features
            y_pairs.append(cycle_y[i + 1])  # Next state Targets
            
    return np.array(x_pairs), np.array(y_pairs)

X_train, y_train = create_step_pairs_by_cycle(train_df, train_x, train_y)
X_val, y_val = create_step_pairs_by_cycle(val_df, val_x, val_y)

model = models.Sequential([
        layers.Dense(64, activation='relu'),
        
        layers.Dense(OUTPUT_TARGETS)
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