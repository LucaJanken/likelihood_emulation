import sys
import os

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the project root to sys.path if not already added
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pyDOE import lhs
import matplotlib.pyplot as plt
import pickle

# Set paths
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# Choose scaler type: 'standard' for StandardScaler, 'minmax' for MinMaxScaler
scaler_type = "standard"  # Change this to 'minmax' to switch to MinMaxScaler

# Function to select the scaler dynamically
def get_scaler(scaler_type):
    if scaler_type == "standard":
        return StandardScaler()
    elif scaler_type == "minmax":
        return MinMaxScaler(feature_range=(0, 1))
    else:
        raise ValueError("Invalid scaler_type. Choose 'standard' or 'minmax'.")

# Define an isotropic -log 2D Gaussian function
def neg_log_gaussian_2D(x, y, bestfit_value, bestfit_point, C_inv_xx, C_inv_yy, C_inv_xy):
    x_0, y_0 = bestfit_point
    dx = x - x_0
    dy = y - y_0
    mahalanobis_dist = C_inv_xx * dx**2 + 2 * C_inv_xy * dx * dy + C_inv_yy * dy**2
    return bestfit_value + 0.5 * mahalanobis_dist

# Choose function parameters
bestfit_value = 0.0
bestfit_point = np.array([0.0, 0.0])
sigma = np.sqrt(0.1)
C_inv_xx = 1 / sigma**2
C_inv_yy = 1 / sigma**2
C_inv_xy = 0.0

# Define sample ranges
x_min, x_max = -3, 3
y_min, y_max = -3, 3
circle_radius = 3

# Generate Latin Hypercube samples in [0,1]
num_samples = 2000
valid_samples = []

while len(valid_samples) < num_samples:
    # Generate Latin Hypercube samples in [0,1] and scale to (x,y) range
    lhs_samples = lhs(2, samples=num_samples)  # Generates more than needed
    x_samples = lhs_samples[:, 0] * (x_max - x_min) + x_min
    y_samples = lhs_samples[:, 1] * (y_max - y_min) + y_min
    
    # Compute distance from center
    distances = np.sqrt(x_samples**2 + y_samples**2)

    # Keep only samples within the circle
    inside_circle = distances <= circle_radius
    valid_samples.extend(zip(x_samples[inside_circle], y_samples[inside_circle]))

    # Reduce to the required number of samples
    valid_samples = valid_samples[:num_samples]

# Convert to numpy array
params = np.array(valid_samples)

# Compute "true" function values
data = np.array([
    neg_log_gaussian_2D(x, y, bestfit_value, bestfit_point, C_inv_xx, C_inv_yy, C_inv_xy)
    for x, y in params
])

# Reshape for scaling
targets = data.reshape(-1, 1)

# Instantiate scalers
param_scaler = get_scaler(scaler_type)
target_scaler = get_scaler(scaler_type)

# Scale parameters and target values
params_scaled = param_scaler.fit_transform(params)
targets_scaled = target_scaler.fit_transform(targets)

# Save scalers for later use
with open(os.path.join(data_dir, "param_scaler.pkl"), "wb") as f:
    pickle.dump(param_scaler, f)

with open(os.path.join(data_dir, "target_scaler.pkl"), "wb") as f:
    pickle.dump(target_scaler, f)

# Define an inverse scaling transformation
def inverse_transform_tf(scaled_tensor, scaler):
    if isinstance(scaler, StandardScaler):
        means = tf.constant(scaler.mean_, dtype=tf.float32)
        stds = tf.constant(scaler.scale_, dtype=tf.float32)
        return scaled_tensor * stds + means
    elif isinstance(scaler, MinMaxScaler):
        min_val = tf.constant(scaler.data_min_, dtype=tf.float32)
        max_val = tf.constant(scaler.data_max_, dtype=tf.float32)
        return scaled_tensor * (max_val - min_val) + min_val

# Prepare TF constant for target scaling
if isinstance(target_scaler, StandardScaler):
    target_mean = tf.constant(target_scaler.mean_[0], dtype=tf.float32)
    target_std = tf.constant(target_scaler.scale_[0], dtype=tf.float32)
elif isinstance(target_scaler, MinMaxScaler):
    target_min = tf.constant(target_scaler.data_min_[0], dtype=tf.float32)
    target_max = tf.constant(target_scaler.data_max_[0], dtype=tf.float32)

# Define the input for (x,y) in scaled space
inputs_scaled = layers.Input(shape=(2,), name="scaled_xy")  # [x_scaled, y_scaled]

# Hidden layers to predict 6 Gaussian parameters [p0, x0, y0, cxx, cxy, cyy]
hidden = layers.Dense(128, activation="relu")(inputs_scaled)
#hidden = layers.Dropout(0.2)(hidden)
hidden = layers.Dense(128, activation="relu")(hidden)
#hidden = layers.Dropout(0.2)(hidden)
hidden = layers.Dense(128, activation="relu")(hidden)
#hidden = layers.Dropout(0.2)(hidden)
pred_params = layers.Dense(6, name="gaussian_params")(hidden)

# Define a custom lambda layer to:
# (A) Inverse transform the scaled input -> (x,y) in original space
# (B) Compute the unscaled function values
# (C) Rescale the function values back to [0,1] or standard scale
def compute_scaled_values(args):
    """
    args = [scaled_xy, pred_params]
    scaled_xy: shape (batch_size, 2)
    gaussian_params: shape (batch_size, 6)
    """
    scaled_xy, pred_params = args

    # (A) unscale (x,y)
    xy_unscaled = inverse_transform_tf(scaled_xy, param_scaler)
    x = xy_unscaled[:, 0:1]  # shape (batch_size, 1)
    y = xy_unscaled[:, 1:2]  # shape (batch_size, 1)

    # (B) Compute function values
    p0 = pred_params[:, 0:1]
    x0 = pred_params[:, 1:2]
    y0 = pred_params[:, 2:3]
    cxx = pred_params[:, 3:4]
    cxy = pred_params[:, 4:5]
    cyy = pred_params[:, 5:6]

    dx = x - x0
    dy = y - y0
    mahalanobis_dist = cxx * dx**2 + 2 * cxy * dx * dy + cyy * dy**2
    unscaled_values = p0 + 0.5 * mahalanobis_dist

    # (C) Rescale the function values
    if isinstance(target_scaler, StandardScaler):
        scaled_values = (unscaled_values - target_mean) / target_std
    elif isinstance(target_scaler, MinMaxScaler):
        scaled_values = (unscaled_values - target_min) / (target_max - target_min)

    return scaled_values

scaled_values = layers.Lambda(compute_scaled_values, name="predicted_values")([inputs_scaled, pred_params])

# Define the model
model = models.Model(inputs=inputs_scaled, outputs=scaled_values)

# Inspect the model
model.summary()

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
    loss="mse"
)

# Define early stopping
early_stopping = EarlyStopping(
    monitor="loss",
    patience=20,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    params_scaled,
    targets_scaled,
    epochs=500,
    batch_size=64,
    validation_split=0.1,
    verbose=2,
    callbacks=[early_stopping]
)

# Save the model and history
os.makedirs("models", exist_ok=True)
model.save(os.path.join("models", "trained_model.h5"))

with open(os.path.join(data_dir, "training_history.pkl"), "wb") as f:
    pickle.dump(history.history, f)

# Plot the training history
plt.figure(figsize=(8, 6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.yscale("log")
plt.grid(True)

#Save the plot
os.makedirs("plots", exist_ok=True)  # Ensure directory exists
plt.savefig(os.path.join("plots", "training_loss.png"))
plt.show()



