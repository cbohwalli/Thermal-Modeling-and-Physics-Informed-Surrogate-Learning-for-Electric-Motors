import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_sequences_by_cycle(df_cycles, x_data, y_data, window_size):
    x_seq, y_seq = [], []
    for cycle_id in df_cycles['drive_cycle_number'].unique():
        # Get indices for this cycle
        indices = np.where(df_cycles['drive_cycle_number'] == cycle_id)[0]
        # Get start/end in the transformed array
        start, end = indices[0], indices[-1] + 1
        
        cycle_x = x_data[start:end]
        cycle_y = y_data[start:end]

        if(window_size == 1):
            for i in range(len(cycle_x) - 1): 
                x_seq.append(cycle_x[i])      # Current state Features
                y_seq.append(cycle_y[i + 1])  # Next state Targets
        elif(window_size > 1):
            for i in range(len(cycle_x) - window_size):
                x_seq.append(cycle_x[i : i + window_size])
                y_seq.append(cycle_y[i + window_size])

    X = np.array(x_seq)
    Y = np.array(y_seq)

    # If Window=1, X is 2D (Samples, Features). 
    # We must expand it to 3D (Samples, 1, Features) to match the model.
    if window_size == 1 and X.ndim == 2:
        X = np.expand_dims(X, axis=1)

    return X, Y

def split_and_normalize(df, training_split, inputs, outputs):
    cycle_ids = df['drive_cycle_number'].unique()
    np.random.shuffle(cycle_ids) 

    # Split cycles (training_split% train, 100-training_split% validation)
    split_idx = int(training_split * len(cycle_ids))
    train_ids = cycle_ids[:split_idx]
    val_ids = cycle_ids[split_idx:]

    train_df = df[df['drive_cycle_number'].isin(train_ids)]
    val_df = df[df['drive_cycle_number'].isin(val_ids)]

    # 2. Normalization
    scaler_input = MinMaxScaler()
    scaler_target = MinMaxScaler()

    # Fit on training data only to avoid data leakage
    scaler_input.fit(train_df[inputs])
    scaler_target.fit(train_df[outputs])

    # Transform both
    train_x = scaler_input.transform(train_df[inputs])
    train_y = scaler_target.transform(train_df[outputs])

    val_x = scaler_input.transform(val_df[inputs])
    val_y = scaler_target.transform(val_df[outputs])

    return train_x, train_y, val_x, val_y, scaler_target

def add_measurement_noice(noise_level, train_y):
    # Generate Gaussian noise matching the shape of your target data
    train_noise = np.random.normal(0, noise_level, size=train_y.shape)

    # Add the noise to the ground truth
    train_y = train_y + train_noise

    return train_y