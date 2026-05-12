import matplotlib.pyplot as plt

def visualise_results(ground_truth, predictions, history, experiment_number):

    # Setup the Plotting Layout
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    labels = ['Stator', 'Rotor 1', 'Rotor 2', 'Housing']

    # Plot the data
    for i in range(4):
        axes[i].plot(ground_truth[:, i], label='Actual', color='blue', alpha=0.6)
        axes[i].plot(predictions[:, i], label='Predicted', color='red', linestyle='--')
        axes[i].set_title(f'Comparison: {labels[i]}')
        axes[i].legend()
        axes[i].set_ylabel('Temp')

    plt.tight_layout()
    plt.savefig(f'results/ground_truth_vs_predictions_v{experiment_number}.png')
    print("Ground truth vs predictions plot saved")

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, label='Training Loss', color='blue')
    plt.plot(epochs, val_loss, label='Validation Loss', color='orange', linestyle='--')

    plt.title('Model Training Progress')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Mean Squared Error)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/training_loss_curve_v{experiment_number}.png')
    print("Training loss plot saved")