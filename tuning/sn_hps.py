import sys
import os
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pyDOE import lhs
import matplotlib.pyplot as plt
import pickle

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Set paths
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# Choose scaler type: 'standard' for StandardScaler, 'minmax' for MinMaxScaler
scaler_type = "standard"  # or change to "minmax" if desired

def get_scaler(scaler_type):
    if scaler_type == "standard":
        return StandardScaler()
    elif scaler_type == "minmax":
        return MinMaxScaler(feature_range=(0, 1))
    else:
        raise ValueError("Invalid scaler_type. Choose 'standard' or 'minmax'.")

# ------------------------------------------------------
# Define the sterile neutrino chi² function to emulate
# ------------------------------------------------------
# According to the supervisor's description:
# chi² = (N_eff * m0)^2, with best-fit at (0,0)
def chi_squared_sterile_neutrino(N_eff, m0):
    return (N_eff * m0) ** 2

# --------------------------------------------
# Generate training data via Latin Hypercube Sampling
# --------------------------------------------
# Training data in [0,3] for both N_eff and m0
N_eff_min, N_eff_max = 0, 3
m0_min, m0_max = 0, 3

num_samples = 2000
lhs_samples = lhs(2, samples=num_samples)
N_eff_samples = lhs_samples[:, 0] * (N_eff_max - N_eff_min) + N_eff_min
m0_samples = lhs_samples[:, 1] * (m0_max - m0_min) + m0_min
params = np.column_stack((N_eff_samples, m0_samples))
data = np.array([chi_squared_sterile_neutrino(N_eff, m0) for N_eff, m0 in params])
targets = data.reshape(-1, 1)

# --------------------------
# Scale the training parameters and targets
# --------------------------
param_scaler = get_scaler(scaler_type)
target_scaler = get_scaler(scaler_type)

params_scaled = param_scaler.fit_transform(params)
targets_scaled = target_scaler.fit_transform(targets)

# Save scalers for later use
with open(os.path.join(data_dir, "sn_hps_param_scaler.pkl"), "wb") as f:
    pickle.dump(param_scaler, f)
with open(os.path.join(data_dir, "sn_hps_target_scaler.pkl"), "wb") as f:
    pickle.dump(target_scaler, f)

# -------------------------------------------------------
# Create validation data to assess extrapolation ability
# --------------------------------------------
# Validation data is sampled over [0,5] for both parameters,
# covering both the training domain and out-of-domain extrapolation.
val_N_eff_min, val_N_eff_max = 0, 5
val_m0_min, val_m0_max = 0, 5
num_val_samples = 500  # Adjust as needed

lhs_val_samples = lhs(2, samples=num_val_samples)
N_eff_val = lhs_val_samples[:, 0] * (val_N_eff_max - val_N_eff_min) + val_N_eff_min
m0_val    = lhs_val_samples[:, 1] * (val_m0_max - val_m0_min) + val_m0_min
val_params = np.column_stack((N_eff_val, m0_val))
val_data = np.array([chi_squared_sterile_neutrino(N_eff, m0) for N_eff, m0 in val_params])
val_targets = val_data.reshape(-1, 1)

# Scale the validation data using the same scalers as for training
val_params_scaled = param_scaler.transform(val_params)
val_targets_scaled = target_scaler.transform(val_targets)

# -------------------------------------------------------
# Define helper for inverse scaling in the TensorFlow graph
# -------------------------------------------------------
def inverse_transform_tf(scaled_tensor, scaler):
    if isinstance(scaler, StandardScaler):
        means = tf.constant(scaler.mean_, dtype=tf.float32)
        stds  = tf.constant(scaler.scale_, dtype=tf.float32)
        return scaled_tensor * stds + means
    elif isinstance(scaler, MinMaxScaler):
        min_val = tf.constant(scaler.data_min_, dtype=tf.float32)
        max_val = tf.constant(scaler.data_max_, dtype=tf.float32)
        return scaled_tensor * (max_val - min_val) + min_val

# Prepare target scaling constants for use in the Lambda layer
if isinstance(target_scaler, StandardScaler):
    target_mean = tf.constant(target_scaler.mean_[0], dtype=tf.float32)
    target_std  = tf.constant(target_scaler.scale_[0], dtype=tf.float32)
elif isinstance(target_scaler, MinMaxScaler):
    target_min = tf.constant(target_scaler.data_min_[0], dtype=tf.float32)
    target_max = tf.constant(target_scaler.data_max_[0], dtype=tf.float32)

# -------------------------
# Build the neural network for tuning
# -------------------------
# The network maps scaled (N_eff, m0) inputs to 6 Gaussian parameters:
# [p0, x0, y0, cxx, cxy, cyy]
def compute_scaled_values(args):
    scaled_xy, pred_params = args
    # Inverse transform inputs to original scale
    xy_unscaled = inverse_transform_tf(scaled_xy, param_scaler)
    N_eff = xy_unscaled[:, 0:1]
    m0    = xy_unscaled[:, 1:2]
    # Split predicted parameters
    p0, x0, y0, cxx, cxy, cyy = tf.split(pred_params, num_or_size_splits=6, axis=1)
    dx = N_eff - x0
    dy = m0 - y0
    mahalanobis_dist = cxx * dx**2 + 2 * cxy * dx * dy + cyy * dy**2
    unscaled_values = p0 + 0.5 * mahalanobis_dist
    # Rescale to target scale
    if isinstance(target_scaler, StandardScaler):
        return (unscaled_values - target_mean) / target_std
    elif isinstance(target_scaler, MinMaxScaler):
        return (unscaled_values - target_min) / (target_max - target_min)

def build_model(hp):
    inputs_scaled = layers.Input(shape=(2,), name="scaled_xy")
    
    # Hyperparameters: number of layers, units, activation, and learning rate.
    num_layers = hp.Int("num_layers", min_value=2, max_value=6, default=3)
    activation_options = ["relu", "tanh", "sigmoid"]
    
    hidden = layers.Dense(
        hp.Int("units_1", min_value=32, max_value=512, step=32, default=128),
        activation=hp.Choice("activation_1", values=activation_options, default="relu")
    )(inputs_scaled)
    
    for i in range(2, num_layers + 1):
        hidden = layers.Dense(
            hp.Int(f"units_{i}", min_value=32, max_value=512, step=32, default=128),
            activation=hp.Choice(f"activation_{i}", values=activation_options, default="relu")
        )(hidden)
    
    pred_params = layers.Dense(6, name="gaussian_params")(hidden)
    scaled_values = layers.Lambda(compute_scaled_values, name="predicted_values")(
        [inputs_scaled, pred_params]
    )
    
    model = models.Model(inputs=inputs_scaled, outputs=scaled_values)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Float("learning_rate", min_value=1e-6, max_value=1e-2, sampling="log", default=1e-3)
        ),
        loss="mse"
    )
    return model

# ---------------------------------
# Hyperparameter tuning with Hyperband
# ---------------------------------
tuner = kt.Hyperband(
    build_model,
    objective="val_loss",
    max_epochs=500,
    factor=3,
    directory="hyperband_results",
    project_name="sn_hyperparameter_search"
)

early_stopping = EarlyStopping(monitor="val_loss", patience=100, restore_best_weights=True)

# Use the custom validation data (which spans [0,5]) to assess extrapolation performance.
tuner.search(
    params_scaled, targets_scaled,
    epochs=500,
    batch_size=64,
    validation_data=(val_params_scaled, val_targets_scaled),
    callbacks=[early_stopping]
)

# Retrieve the best hyperparameters and build the best model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)

# Train the best model further
history = best_model.fit(
    params_scaled, targets_scaled,
    epochs=500,
    batch_size=64,
    validation_data=(val_params_scaled, val_targets_scaled),
    verbose=2,
    callbacks=[early_stopping]
)

# -------------------------------
# Save the best model and training history
# -------------------------------
os.makedirs("models", exist_ok=True)
best_model.save(os.path.join("models", "sn_hps_trained_model.h5"))

with open(os.path.join(data_dir, "sn_hps_training_history.pkl"), "wb") as f:
    pickle.dump(history.history, f)
