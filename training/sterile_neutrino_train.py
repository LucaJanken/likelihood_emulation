import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import pickle

# ---------------------------------------------------
# Set seeds for reproducibility
# ---------------------------------------------------
seed = 40
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# -------------------------------
# Setup paths and scaler settings
# -------------------------------
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
# According to the description:
# chi² = (N_eff * m0)², assuming best-fit at (0,0)
def chi_squared_sterile_neutrino(N_eff, m0):
    return (N_eff * m0) ** 2

# ------------------------------------------------------
# Load the MCMC chain and perform flat sampling
# ------------------------------------------------------
# Assume the chain file (saved previously) has columns:
# multiplicity, χ², param_1, param_2
chain_file = os.path.join("mcmc_plots", "mcmc_chain.txt")
if not os.path.exists(chain_file):
    raise FileNotFoundError(f"MCMC chain file not found: {chain_file}")

# Load the chain; skip header lines starting with '#'
chain_data = np.loadtxt(chain_file, comments='#')
# chain_data shape: (num_entries, 4)

def flat_sample_from_chain(chain, n_samples, T=5.0):
    """
    Perform flat sampling from the MCMC chain using reweighting based on the traditional likelihood.
    
    The chain is expected to have rows of the form:
      [multiplicity, χ², param_1, param_2]
    
    The MCMC chain was generated with a likelihood:
      L ∝ exp(-χ²/(2T))
    so that a sample appears with multiplicity proportional to its likelihood.
    
    To recover a flat sample we reweight each entry by the inverse likelihood:
      weight = multiplicity * exp(χ²/(2T))
    
    For T = 5, this becomes:
      weight = multiplicity * exp(χ²/10)
    
    This function returns a resampled chain of n_samples rows.
    """
    multiplicities = chain[:, 0]
    chi2_vals = chain[:, 1]
    
    weights = multiplicities * np.exp(chi2_vals / (2 * T))
    weights /= np.sum(weights)
    
    indices = np.random.choice(len(chain), size=n_samples, replace=False, p=weights)
    return chain[indices]

# Choose the desired number of training samples
n_samples = 500
flat_samples = flat_sample_from_chain(chain_data, n_samples, T=5.0)
# Extract parameters and targets.
# Parameters are in columns 2 and 3; χ² is in column 1.
params = flat_samples[:, 2:4]
targets = flat_samples[:, 1].reshape(-1, 1)

# --------------------------
# Scale the parameters and targets
# --------------------------
param_scaler = get_scaler(scaler_type)
target_scaler = get_scaler(scaler_type)

params_scaled = param_scaler.fit_transform(params)
targets_scaled = target_scaler.fit_transform(targets)

# Save scalers for later use
with open(os.path.join(data_dir, "sn_param_scaler.pkl"), "wb") as f:
    pickle.dump(param_scaler, f)
with open(os.path.join(data_dir, "sn_target_scaler.pkl"), "wb") as f:
    pickle.dump(target_scaler, f)

# -------------------------------------------------------
# Define helper for inverse scaling in the TensorFlow graph
# -------------------------------------------------------
def inverse_transform_tf(scaled_tensor, scaler):
    if isinstance(scaler, StandardScaler):
        means = tf.constant(scaler.mean_, dtype=tf.float32)
        stds = tf.constant(scaler.scale_, dtype=tf.float32)
        return scaled_tensor * stds + means
    elif isinstance(scaler, MinMaxScaler):
        min_val = tf.constant(scaler.data_min_, dtype=tf.float32)
        max_val = tf.constant(scaler.data_max_, dtype=tf.float32)
        return scaled_tensor * (max_val - min_val) + min_val

# Prepare target scaling constants for use in the Lambda layer
if isinstance(target_scaler, StandardScaler):
    target_mean = tf.constant(target_scaler.mean_[0], dtype=tf.float32)
    target_std = tf.constant(target_scaler.scale_[0], dtype=tf.float32)
elif isinstance(target_scaler, MinMaxScaler):
    target_min = tf.constant(target_scaler.data_min_[0], dtype=tf.float32)
    target_max = tf.constant(target_scaler.data_max_[0], dtype=tf.float32)

# -------------------------
# Build the neural network
# -------------------------
# Set this flag to choose the emulation mode:
# If True, use the Gaussian approximation (current approach).
# If False, emulate the χ² value directly.
use_gaussian = True

inputs_scaled = layers.Input(shape=(2,), name="scaled_xy")
hidden = layers.Dense(128, activation="relu")(inputs_scaled)
hidden = layers.Dense(128, activation="relu")(hidden)

if use_gaussian:
    # Gaussian approximation mode:
    # The network maps scaled (N_eff, m0) inputs to 6 Gaussian parameters:
    # [p0, x0, y0, cxx, cxy, cyy]
    pred_params = layers.Dense(6, name="gaussian_params")(hidden)

    # Custom Lambda layer to reconstruct the local Gaussian value
    def compute_scaled_values(args):
        scaled_xy, pred_params = args
        # (A) Inverse transform the scaled inputs to original (N_eff, m0)
        xy_unscaled = inverse_transform_tf(scaled_xy, param_scaler)
        N_eff = xy_unscaled[:, 0:1]  # shape (batch_size, 1)
        m0    = xy_unscaled[:, 1:2]  # shape (batch_size, 1)
        
        # (B) Use the predicted parameters to form a Gaussian:
        # p0: baseline χ²; (x0, y0): center of the local Gaussian
        # (cxx, cxy, cyy) define the curvature matrix.
        p0   = pred_params[:, 0:1]
        x0   = pred_params[:, 1:2]
        y0   = pred_params[:, 2:3]
        cxx  = pred_params[:, 3:4]
        cxy  = pred_params[:, 4:5]
        cyy  = pred_params[:, 5:6]
        
        dx = N_eff - x0
        dy = m0 - y0
        mahalanobis_dist = cxx * dx**2 + 2 * cxy * dx * dy + cyy * dy**2
        unscaled_values = p0 + 0.5 * mahalanobis_dist  # Gaussian approximation
        
        # (C) Rescale the computed values to match the target scaling
        if isinstance(target_scaler, StandardScaler):
            scaled_values = (unscaled_values - target_mean) / target_std
        elif isinstance(target_scaler, MinMaxScaler):
            scaled_values = (unscaled_values - target_min) / (target_max - target_min)
        return scaled_values

    scaled_values = layers.Lambda(compute_scaled_values, name="predicted_values")(
        [inputs_scaled, pred_params]
    )
    model = models.Model(inputs=inputs_scaled, outputs=scaled_values)
else:
    # Direct χ² emulation mode:
    # The network directly maps scaled (N_eff, m0) inputs to a single output.
    predicted_chi2 = layers.Dense(1, name="predicted_chi2")(hidden)
    model = models.Model(inputs=inputs_scaled, outputs=predicted_chi2)

model.summary()

# --------------------------
# Compile and train the model
# --------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="mse"
)

early_stopping = EarlyStopping(monitor="val_loss", patience=100, restore_best_weights=True)
history = model.fit(
    params_scaled,
    targets_scaled,
    epochs=2000,
    batch_size=16,
    validation_split=0.1,
    verbose=2,
    callbacks=[early_stopping]
)

# -------------------------------
# Save model, training history, and plot results
# -------------------------------
os.makedirs("models", exist_ok=True)
model.save(os.path.join("models", "sn_trained_model.h5"))

with open(os.path.join(data_dir, "sn_training_history.pkl"), "wb") as f:
    pickle.dump(history.history, f)

plt.figure(figsize=(8, 6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss (Sterile Neutrino Likelihood)")
plt.legend()
plt.yscale("log")
plt.grid(True)

os.makedirs("plots", exist_ok=True)
plt.savefig(os.path.join("plots", "sn_training_loss.png"))
plt.show()
