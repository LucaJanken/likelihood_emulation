import sys
import os

# Ensure the project root is on sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import necessary modules
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from keras_tuner import RandomSearch
from training.data_loader import generate_lhs_data, normalize_data
from training.network import build_model  # Uses build_model(hp=None) or build_model(hp=...) if tuning

# 1. Generate Training Data (in-domain)
n_train_samples = 2000
params_train, targets_train = generate_lhs_data(
    n_samples=n_train_samples,
    x_range=(-2, 2),
    y_range=(-2, 2)
)

# 2. Generate Extrapolation Data (out-of-domain) for validation
n_val_samples = 500
params_val, targets_val = generate_lhs_data(
    n_samples=n_val_samples,
    x_range=(-3, 3),
    y_range=(-3, 3)
)

# 3. Normalize data
params_train_scaled, targets_train_scaled, param_scaler, target_scaler = normalize_data(
    params_train, targets_train
)
params_val_scaled = param_scaler.transform(params_val)
targets_val_scaled = target_scaler.transform(targets_val)

# 4. Set up the RandomSearch tuner (minimize out-of-domain validation MSE)
tuner = RandomSearch(
    hypermodel=build_model,   # Directly use build_model from network.py
    objective='val_loss',
    max_trials=20,
    executions_per_trial=1,
    overwrite=True,
    directory="hyperparam_tuning",
    project_name="extrapolation_search"
)

# 5. Run the hyperparameter search
tuner.search(
    x=params_train_scaled,
    y=targets_train_scaled,
    epochs=500,
    validation_data=(params_val_scaled, targets_val_scaled),
    batch_size=64,
    verbose=2,
    callbacks=[keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True
    )]
)

# 6. Retrieve the best hyperparameters and train final model (optional)
best_hp = tuner.get_best_hyperparameters()[0]
print("Best hyperparameters:", best_hp.values)

best_model = tuner.hypermodel.build(best_hp)
best_model.fit(
    x=params_train_scaled,
    y=targets_train_scaled,
    epochs=500,
    validation_data=(params_val_scaled, targets_val_scaled),
    batch_size=64,
    verbose=2,
    callbacks=[keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True
    )]
)

# 7. Save the best model and scalers
os.makedirs("models", exist_ok=True)
best_model.save(os.path.join("models", "best_extrapolation_model.h5"))

os.makedirs("data", exist_ok=True)
with open(os.path.join("data", "param_scaler.pkl"), "wb") as f:
    pickle.dump(param_scaler, f)
with open(os.path.join("data", "target_scaler.pkl"), "wb") as f:
    pickle.dump(target_scaler, f)

print("Hyperparameter search complete. Best model saved.")
