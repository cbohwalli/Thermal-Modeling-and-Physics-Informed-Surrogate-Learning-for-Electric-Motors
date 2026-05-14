# Thermal Modeling of an Axial Flux Motor using ML Surrogate Models

This project investigates machine learning surrogate models for approximating the thermal dynamics of an axial flux motor. A simplified Lumped Parameter Thermal Network (LPTN) is used as the physics-based model to generate synthetic thermal data, which is then used to train neural networks.

The goal is to determine whether lightweight machine learning models can replace computationally expensive thermal simulations while maintaining predictive accuracy.

## Project Overview

The thermal system consists of:

- Housing
- Stator
- Rotor 1
- Rotor 2

The LPTN model generates temperature trajectories based on load profiles, and multiple neural network architectures are evaluated as surrogate models.

Models explored:

- LSTM
- MLP
- Variants with state inputs
- Models trained under noisy conditions
- Models trained across varying system parameters

## Main Findings

- Including the current system state (`T(t)`) significantly improves performance.
- The system behaves approximately as a Markov process:
  
  ```text
  (load(t), T(t)) → T(t+1)
  ```

- Simple MLP models performed similarly to or better than LSTMs.
- Historical sequence windows provided little additional benefit.
- Under high noise conditions, MLPs generalized better than LSTMs.

## Project Structure

```text
.
├── data/              # Generated datasets
├── experiments/       # Experimental model versions
├── results/           # Training curves and predictions
├── src
│   ├── data/          # Dataset generation and preprocessing
│   ├── models/        # Model architectures and training
│   ├── physics/       # LPTN implementation
│   └── visualisation/ # Plotting utilities
└── README.md
```

## Running Experiments

Example:

```bash
python3 -m experiments.v6.mlp_v6
```

Generate datasets:

```bash
python3 -m src.data.generate_dataset
```

## Results

Example outputs include:

- Training loss curves
- Ground truth vs prediction plots
- Validation MSE comparisons

Stored in:

```bash
results/
```

## Conclusion

For this deterministic thermal system, a simple feedforward MLP was sufficient to accurately learn the state transition dynamics and often outperformed more complex sequence models.