import pandas as pd
from src.visualisation.plotting import visualise_results
from src.data.preprocessing import split_and_normalize
from src.data.preprocessing import create_sequences_by_cycle
from src.models.build_mlp import build_mlp
from src.models.train_model import train_model

# Constants
WINDOW_SIZE = 60 
INPUT_FEATURES = 5 
OUTPUT_TARGETS = 4

feature_cols = ['load', 't_stator', 't_rotor_1', 't_rotor_2', 't_housing']
target_cols = ['t_stator', 't_rotor_1', 't_rotor_2', 't_housing']

# Data preprocessing pipeline

# 1. Load Data
df = pd.read_csv('drive_cycle_dataset.csv')

# 2. Split and normalize data
training_split = 80 # 80% training data 20% validation data
train_x, train_y, val_x, val_y, scaler_target = split_and_normalize(df, training_split, feature_cols, target_cols)

# 3. Adapt format of data for input and output of the model (input: [window_size, feature_cols] output: [target_cols])
X_train, y_train = create_sequences_by_cycle(df, train_x, train_y, WINDOW_SIZE)
X_val, y_val = create_sequences_by_cycle(df, val_x, val_y, WINDOW_SIZE)

model = build_mlp(WINDOW_SIZE, INPUT_FEATURES, OUTPUT_TARGETS)
model, history = train_model(model, (X_train, y_train), (val_x, val_y))


predictions_scaled = model.predict(X_val)

# Convert scaled predictions back to actual temperature values
predictions = scaler_target.inverse_transform(predictions_scaled)

# Reverse the scaling for Ground Truth
ground_truth = scaler_target.inverse_transform(y_val)

# visualisation
experiment_number = '3'
visualise_results(ground_truth, predictions, history, experiment_number)