from tensorflow.keras import layers, models

def build_mlp(WINDOW_SIZE, INPUT_FEATURES, OUTPUT_TARGETS):
    model = models.Sequential([
        # Flattening the time-series window into a single vector
        layers.Flatten(input_shape=(WINDOW_SIZE, INPUT_FEATURES)),
       
        layers.Dense(64, activation='relu'),
        
        layers.Dense(OUTPUT_TARGETS)
    ])

    model.compile(optimizer='adam', loss='mse')

    return model