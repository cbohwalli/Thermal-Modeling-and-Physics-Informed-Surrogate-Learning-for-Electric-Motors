def train_model(model, train_data, val_data, epochs=20, batch_size=32):
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )
    return model, history