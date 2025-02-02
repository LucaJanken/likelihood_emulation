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
from sklearn.preprocessing import MinMaxScaler
from pyDOE import lhs
import matplotlib.pyplot as plt
import pickle

###############################################################################
# 1) Data Generation (same as before)
###############################################################################

# Set paths
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

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
x_min, x_max = -2, 2
y_min, y_max = -2, 2

# Generate Latin Hypercube samples in [0,1]
num_samples = 2000
lhs_samples = lhs(2, samples=num_samples)

# Scale LHS samples to the desired (x,y) range
x_samples = lhs_samples[:, 0] * (x_max - x_min) + x_min
y_samples = lhs_samples[:, 1] * (y_max - y_min) + y_min

# Compute "true" function values = chi^2(x,y)
data = np.array([
    neg_log_gaussian_2D(x, y, bestfit_value, bestfit_point, C_inv_xx, C_inv_yy, C_inv_xy)
    for (x, y) in zip(x_samples, y_samples)
])

# Targets and parameters
targets = data.reshape(-1, 1)  # shape (N, 1)
params = np.column_stack((x_samples, y_samples))  # shape (N, 2)

###############################################################################
# 2) Scaling inputs and targets
###############################################################################

param_scaler = MinMaxScaler(feature_range=(0, 1))
params_scaled = param_scaler.fit_transform(params)  # shape (N, 2)

target_scaler = MinMaxScaler(feature_range=(0, 1))
targets_scaled = target_scaler.fit_transform(targets)  # shape (N, 1)

# Save scalers for later use
with open(os.path.join(data_dir, "param_scaler.pkl"), "wb") as f:
    pickle.dump(param_scaler, f)

with open(os.path.join(data_dir, "target_scaler.pkl"), "wb") as f:
    pickle.dump(target_scaler, f)

###############################################################################
# 3) Build a Keras Model that outputs decomposed parameters,
#    then merges with *unscaled* (x,y) to produce scaled chi^2
###############################################################################

# We'll define a helper to invert MinMaxScaler in TensorFlow
def inverse_transform_tf(scaled_tensor, scaler):
    """
    Applies the inverse of MinMaxScaler using TensorFlow ops.
    scaled_tensor: shape (batch_size, d)
    scaler: a fitted MinMaxScaler
    """
    # We store these as tf constants so it can run in the graph
    min_val = tf.constant(scaler.data_min_, dtype=tf.float32)
    max_val = tf.constant(scaler.data_max_, dtype=tf.float32)
    return scaled_tensor * (max_val - min_val) + min_val

# Prepare TF constants for the target scaling
target_min = tf.constant(target_scaler.data_min_[0], dtype=tf.float32)
target_max = tf.constant(target_scaler.data_max_[0], dtype=tf.float32)

# 3.1. Define the input for (x,y) in scaled space
inputs_scaled = layers.Input(shape=(2,), name='scaled_xy')  # shape (None, 2)

# 3.2. Hidden layers to predict 6 parameters:
#      [p0, x0, y0, cxx, cxy, cyy]
hidden = layers.Dense(128, activation='relu')(inputs_scaled)
hidden = layers.Dense(128, activation='relu')(hidden)
hidden = layers.Dense(128, activation='relu')(hidden)
pred_params = layers.Dense(6, name='gaussian_params')(hidden)
# shape (None, 6)

# 3.3. A Lambda layer to:
#     (A) inverse-transform the scaled input -> (x, y) in original space,
#     (B) compute unscaled chi^2,
#     (C) scale chi^2 back to [0, 1].
def compute_chi2(args):
    """
    args = [scaled_xy, pvec]
      scaled_xy: shape (batch_size, 2)
      pvec: shape (batch_size, 6)

    We do:
      1) unscale xy
      2) parse [p0, x0, y0, cxx, cxy, cyy]
      3) compute chi^2_unscaled
      4) scale to [0, 1] using target_scaler
      5) return the scaled chi^2
    """
    scaled_xy, pvec = args
    
    # (A) unscale (x,y)
    xy_unscaled = inverse_transform_tf(scaled_xy, param_scaler)
    x = xy_unscaled[:, 0:1]  # shape (batch_size, 1)
    y = xy_unscaled[:, 1:2]  # shape (batch_size, 1)
    
    # (B) parse predicted parameters
    p0  = pvec[:, 0:1]  # shape (batch_size,1) => baseline chi^2
    x0  = pvec[:, 1:2]
    y0  = pvec[:, 2:3]
    cxx = pvec[:, 3:4]
    cxy = pvec[:, 4:5]
    cyy = pvec[:, 5:6]

    # (C) compute unscaled chi^2
    dx = x - x0
    dy = y - y0
    # Mahalanobis term:
    mahalanobis = cxx*dx*dx + 2.0*cxy*dx*dy + cyy*dy*dy
    chi2_unscaled = p0 + mahalanobis  # shape (batch_size, 1)

    # (D) scale chi^2 back into [0,1]:
    #     scaled_val = (val - min) / (max - min)
    #     but min_val and max_val are for the chi^2 target
    chi2_scaled = (chi2_unscaled - target_min) / (target_max - target_min)

    return chi2_scaled

chi2_scaled = layers.Lambda(compute_chi2, name='chi2_pred')([inputs_scaled, pred_params])
# shape (None, 1)

# 3.4. Build the Functional Model
model = models.Model(inputs=inputs_scaled, outputs=chi2_scaled, name='DecomposedGaussianModel')

# Inspect
model.summary()

###############################################################################
# 4) Compile and train
###############################################################################

# Use 'mse' on the scaled chi^2
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='mse'
)

# Define Early Stopping
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=50,
    restore_best_weights=True
)

history = model.fit(
    params_scaled,        # scaled input
    targets_scaled,       # scaled target
    epochs=500,
    batch_size=64,
    validation_split=0.1,
    verbose=2,
    callbacks=[early_stopping]
)

###############################################################################
# 5) Save trained model and training history
###############################################################################
os.makedirs("models", exist_ok=True)
model.save(os.path.join("models", "trained_model.h5"))

with open(os.path.join(data_dir, "training_history.pkl"), "wb") as f:
    pickle.dump(history.history, f)

###############################################################################
# 6) Plot training history
###############################################################################
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.yscale('log')
plt.grid(True)

# Save the plot
plt.savefig(os.path.join(data_dir, "new_training_loss.png"))
plt.show()

###############################################################################
# 7) Check scaling (same checks as before)
###############################################################################
print("Before scaling (first 5 params):")
print(params[:5])            # Original (x,y)
print("\nAfter scaling (first 5 params):")
print(params_scaled[:5])     # Scaled (x,y)

print("\nBefore scaling targets (first 5):")
print(targets[:5])           # Original chi^2 values
print("\nAfter scaling targets (first 5):")
print(targets_scaled[:5])    # Scaled chi^2
