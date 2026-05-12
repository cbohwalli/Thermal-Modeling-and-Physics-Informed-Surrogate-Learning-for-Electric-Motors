from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def build_lstm(WINDOW_SIZE, INPUT_FEATURES, OUTPUT_TARGETS):
  
    model = Sequential([
        LSTM(32, input_shape=(WINDOW_SIZE, INPUT_FEATURES)),
        Dense(OUTPUT_TARGETS)
    ])

    model.compile(optimizer='adam', loss='mse')

    return model