def model_evaluation(model, X_val, y_val, scaler_target):
    predictions_scaled = model.predict(X_val)

    # Convert scaled predictions back to actual temperature values
    predictions = scaler_target.inverse_transform(predictions_scaled)

    # Reverse the scaling for Ground Truth
    ground_truth = scaler_target.inverse_transform(y_val)

    return predictions, ground_truth