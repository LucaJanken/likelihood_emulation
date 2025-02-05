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

# Add the project root to sys.path if not already added
if project_root not in sys.path:
    sys.path.append(project_root)

# Set paths
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# Choose scaler type: 'standard' for StandardScaler, 'minmax' for MinMaxScaler
scaler_type = "standard"

def get_scaler(scaler_type):
    if scaler_type == "standard":
        return StandardScaler()
    elif scaler_type == "minmax":
        return MinMaxScaler(feature_range=(0, 1))
    else:
        raise ValueError("Invalid scaler_type. Choose 'standard' or 'minmax'.")

# Define function
def neg_log_gaussian_2D(x, y, bestfit_value, bestfit_point, C_inv_xx, C_inv_yy, C_inv_xy):
    x_0, y_0 = bestfit_point
    dx = x - x_0
    dy = y - y_0
    mahalanobis_dist = C_inv_xx * dx**2 + 2 * C_inv_xy * dx * dy + C_inv_yy * dy**2
    return bestfit_value + 0.5 * mahalanobis_dist

# Function parameters
bestfit_value = 0.0
bestfit_point = np.array([0.0, 0.0])
sigma = np.sqrt(0.1)
C_inv_xx = 1 / sigma**2
C_inv_yy = 1 / sigma**2
C_inv_xy = 0.0

# Generate samples
x_min, x_max = -3, 3
y_min, y_max = -3, 3
circle_radius = 3
num_samples = 2000
valid_samples = []

while len(valid_samples) < num_samples:
    lhs_samples = lhs(2, samples=num_samples)
    x_samples = lhs_samples[:, 0] * (x_max - x_min) + x_min
    y_samples = lhs_samples[:, 1] * (y_max - y_min) + y_min
    distances = np.sqrt(x_samples**2 + y_samples**2)
    inside_circle = distances <= circle_radius
    valid_samples.extend(zip(x_samples[inside_circle], y_samples[inside_circle]))
    valid_samples = valid_samples[:num_samples]

params = np.array(valid_samples)
data = np.array([neg_log_gaussian_2D(x, y, bestfit_value, bestfit_point, C_inv_xx, C_inv_yy, C_inv_xy) for x, y in params])
targets = data.reshape(-1, 1)

# Scale parameters
param_scaler = get_scaler(scaler_type)
target_scaler = get_scaler(scaler_type)
params_scaled = param_scaler.fit_transform(params)
targets_scaled = target_scaler.fit_transform(targets)

with open(os.path.join(data_dir, "hps_param_scaler.pkl"), "wb") as f:
    pickle.dump(param_scaler, f)

with open(os.path.join(data_dir, "hps_target_scaler.pkl"), "wb") as f:
    pickle.dump(target_scaler, f)

# Define inverse scaling function
def inverse_transform_tf(scaled_tensor, scaler):
    if isinstance(scaler, StandardScaler):
        means = tf.constant(scaler.mean_, dtype=tf.float32)
        stds = tf.constant(scaler.scale_, dtype=tf.float32)
        return scaled_tensor * stds + means
    elif isinstance(scaler, MinMaxScaler):
        min_val = tf.constant(scaler.data_min_, dtype=tf.float32)
        max_val = tf.constant(scaler.data_max_, dtype=tf.float32)
        return scaled_tensor * (max_val - min_val) + min_val

# Prepare TF constants
if isinstance(target_scaler, StandardScaler):
    target_mean = tf.constant(target_scaler.mean_[0], dtype=tf.float32)
    target_std = tf.constant(target_scaler.scale_[0], dtype=tf.float32)
elif isinstance(target_scaler, MinMaxScaler):
    target_min = tf.constant(target_scaler.data_min_[0], dtype=tf.float32)
    target_max = tf.constant(target_scaler.data_max_[0], dtype=tf.float32)

# Custom function to compute scaled values
def compute_scaled_values(args):
    scaled_xy, pred_params = args
    xy_unscaled = inverse_transform_tf(scaled_xy, param_scaler)
    x = xy_unscaled[:, 0:1]
    y = xy_unscaled[:, 1:2]
    p0, x0, y0, cxx, cxy, cyy = tf.split(pred_params, 6, axis=1)
    dx = x - x0
    dy = y - y0
    mahalanobis_dist = cxx * dx**2 + 2 * cxy * dx * dy + cyy * dy**2
    unscaled_values = p0 + 0.5 * mahalanobis_dist
    if isinstance(target_scaler, StandardScaler):
        return (unscaled_values - target_mean) / target_std
    elif isinstance(target_scaler, MinMaxScaler):
        return (unscaled_values - target_min) / (target_max - target_min)

# Function to build a model for tuning
def build_model(hp):
    inputs_scaled = layers.Input(shape=(2,), name="scaled_xy")
    num_layers = hp.Int("num_layers", min_value=2, max_value=5)
    hidden = layers.Dense(
        hp.Int("units_1", min_value=32, max_value=512, step=32),
        activation=hp.Choice("activation_1", values=["relu", "tanh", "sigmoid"])
    )(inputs_scaled)

    for i in range(2, num_layers + 1):
        hidden = layers.Dense(
            hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
            activation=hp.Choice(f"activation_{i}", values=["relu", "tanh", "sigmoid"])
        )(hidden)

    pred_params = layers.Dense(6, name="gaussian_params")(hidden)
    scaled_values = layers.Lambda(compute_scaled_values, name="predicted_values")([inputs_scaled, pred_params])

    model = models.Model(inputs=inputs_scaled, outputs=scaled_values)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Float("learning_rate", min_value=1e-5, max_value=1e-2, sampling="log")
        ),
        loss="mse"
    )

    return model

# Perform hyperparameter tuning
tuner = kt.Hyperband(
    build_model,
    objective="val_loss",
    max_epochs=50,
    factor=3,
    directory="hyperband_results",
    project_name="hyperparameter_search"
)

early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

tuner.search(
    params_scaled, targets_scaled,
    epochs=50,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stopping]
)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)

history = best_model.fit(
    params_scaled,
    targets_scaled,
    epochs=500,
    batch_size=64,
    validation_split=0.1,
    verbose=2,
    callbacks=[early_stopping]
)

# Save best model
os.makedirs("models", exist_ok=True)
best_model.save(os.path.join("models", "hps_trained_model.h5"))

# Save best training history
with open(os.path.join(data_dir, "hps_training_history.pkl"), "wb") as f:
    pickle.dump(history.history, f)
