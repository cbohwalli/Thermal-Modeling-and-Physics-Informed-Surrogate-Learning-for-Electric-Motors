import pandas as pd
from tensorflow.keras import layers, models
from src.visualisation.plotting import visualise_results
from src.data.preprocessing import preprocess_data

# Constants
WINDOW_SIZE = 60 
INPUT_FEATURES = 16 
OUTPUT_TARGETS = 4


feature_cols = ['load', 't_stator', 't_rotor_1', 't_rotor_2', 't_housing',
                'R_stator_rotor1', 'R_stator_rotor2', 'R_stator_housing', 'R_stator_coolant', 'R_rotor1_housing', 
                'R_rotor2_housing', 'R_housing_ambient', 'C_stator', 'C_rotor_1', 'C_rotor_2', 'C_housing'
                ]
target_cols = ['t_stator', 't_rotor_1', 't_rotor_2', 't_housing']

# Data preprocessing
# ------------------------------------------------------------------------

# 1. Load Data
df = pd.read_csv('drive_cycle_dataset_v6.csv')

training_split = 80 # 80% training data 20% validation data
X_train, y_train, X_val, y_val, scaler_input, scaler_target = preprocess_data(df, training_split, feature_cols, target_cols)

model = models.Sequential([
        layers.Dense(64, activation='relu'),
        
        layers.Dense(OUTPUT_TARGETS)
    ])

model.compile(optimizer='adam', loss='mse')

history = model.fit(
    X_train, y_train,
    epochs=30,            
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
experiment_number = '6'
visualise_results(ground_truth, predictions, history, experiment_number)