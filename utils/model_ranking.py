import os
import tensorflow as tf
import keras_tuner as kt

# Load the tuner
TUNER_DIR = "random_search_results/dg_hyperparameter_search"
tuner = kt.RandomSearch(
    lambda hp: None,  # Placeholder, since we only need to load results
    objective="val_loss",
    max_trials=5000,
    executions_per_trial=1,
    directory="random_search_results",
    project_name="dg_hyperparameter_search"
)

tuner.reload()

# Retrieve the top 3 best hyperparameter sets
num_top_models = 3
top_hps = tuner.get_best_hyperparameters(num_trials=num_top_models)

# Print the architecture details of the top models
for i, hp in enumerate(top_hps):
    print(f"\nModel {i+1} Best Hyperparameters:")
    print(f"  Number of Layers: {hp.get('num_layers')}")
    for layer_idx in range(1, hp.get("num_layers") + 1):
        print(f"  Layer {layer_idx}: {hp.get(f'units_{layer_idx}')} units, Activation: {hp.get(f'activation_{layer_idx}')}")
    print(f"  Learning Rate: {hp.get('learning_rate')}")
