import pandas as pd
from src.visualisation.plotting import visualise_results
from src.data.preprocessing import split_and_normalize
from src.data.preprocessing import create_sequences_by_cycle
from src.models.build_lstm import build_lstm
from src.models.train_model import train_model
from src.models.model_evaluation import model_evaluation

# Constants
WINDOW_SIZE = 60 
INPUT_FEATURES = 5 
OUTPUT_TARGETS = 4

feature_cols = ['load', 't_stator', 't_rotor_1', 't_rotor_2', 't_housing']
target_cols = ['t_stator', 't_rotor_1', 't_rotor_2', 't_housing']

# Data preprocessing pipeline

# 1. Load Data
dataset_filepath = 'data/drive_cycle_dataset.csv'
df = pd.read_csv(dataset_filepath)

# 2. Split and normalize data
training_split = 0.8 # 80% training data 20% validation data
train_x, train_y, val_x, val_y, scaler_target = split_and_normalize(df, training_split, feature_cols, target_cols)

# 3. Adapt format of data for input and output of the model (input: [window_size, feature_cols] output: [target_cols])
X_train, y_train = create_sequences_by_cycle(df, train_x, train_y, WINDOW_SIZE)
X_val, y_val = create_sequences_by_cycle(df, val_x, val_y, WINDOW_SIZE)

# ------------------------------------------------------------------------

model = build_lstm(WINDOW_SIZE, INPUT_FEATURES, OUTPUT_TARGETS)
model, history = train_model(model, (X_train, y_train), (val_x, val_y))

predictions, ground_truth = model_evaluation(model, X_val, y_val, scaler_target)

# visualisation
experiment_number = '2'
visualise_results(ground_truth, predictions, history, experiment_number)