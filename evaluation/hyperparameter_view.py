import pickle
import os

# Path to hyperparameter files
tuning_results_dir = "tuning_results"

# Get all hyperparameter files
hyperparam_files = sorted([f for f in os.listdir(tuning_results_dir) if f.startswith("hyperparams_trial_") and f.endswith(".pkl")])

# Display hyperparameters for each model
for hp_file in hyperparam_files:
    with open(os.path.join(tuning_results_dir, hp_file), "rb") as f:
        hyperparams = pickle.load(f)
    print(f"Hyperparameters for {hp_file}:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")
    print("-" * 50)
