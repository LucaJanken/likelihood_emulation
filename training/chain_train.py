import os
import sys
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pyDOE import lhs  # Still used somewhere in your project

# ---------------------------
# Project Setup & Data Paths
# ---------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

chains_dir = "chains/2_param"
filenames = [
    "2024-09-17_123456789__1.txt",
    "2024-09-17_123456789__2.txt",
    "2024-09-17_123456789__3.txt",
    "2024-09-17_123456789__4.txt"
]
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# ------------------------------------
# Choose and Configure Scaler Type
# ------------------------------------
scaler_type = "standard"  # or "minmax"

def get_scaler(scaler_type):
    if scaler_type == "standard":
        return StandardScaler()
    elif scaler_type == "minmax":
        return MinMaxScaler(feature_range=(0, 1))
    else:
        raise ValueError("Invalid scaler_type. Choose 'standard' or 'minmax'.")

# ---------------------------
# Data Loading & Processing
# ---------------------------
multiplicity = []
neg_log_like = []
param1 = []
param2 = []

# Read the first 3000 lines (after skipping 150 header lines) from each file.
for filename in filenames:
    full_path = os.path.join(chains_dir, filename)
    with open(full_path, "r") as file:
        # Skip header
        for _ in range(150):
            next(file)
        count = 0
        for line in file:
            if count >= 3000:
                break
            columns = line.strip().split()
            if len(columns) < 4:
                continue
            try:
                mult = int(columns[0])
                # Scale negative log-likelihoods by factor 5.
                neg_log_lk = float(columns[1]) * 5  
                p1 = float(columns[2])
                p2 = float(columns[3])
            except ValueError:
                continue
            multiplicity.append(mult)
            neg_log_like.append(neg_log_lk)
            param1.append(p1)
            param2.append(p2)
            count += 1

multiplicity = np.array(multiplicity)
neg_log_like = np.array(neg_log_like)
params = np.column_stack((param1, param2))
targets = neg_log_like.reshape(-1, 1)

# ---------------------------
# Scaling of Data
# ---------------------------
param_scaler = get_scaler(scaler_type)
target_scaler = get_scaler(scaler_type)

params_scaled = param_scaler.fit_transform(params)
targets_scaled = target_scaler.fit_transform(targets)

# (Optional) Save scalers for later use
with open(os.path.join(data_dir, "chain_param_scaler.pkl"), "wb") as f:
    pickle.dump(param_scaler, f)
with open(os.path.join(data_dir, "chain_target_scaler.pkl"), "wb") as f:
    pickle.dump(target_scaler, f)

# Save scaler parameters as dictionaries for use in our custom layer.
if scaler_type == "standard":
    param_scaler_params = {
        "mean": param_scaler.mean_.tolist(),
        "scale": param_scaler.scale_.tolist()
    }
    target_scaler_params = {
        "mean": target_scaler.mean_[0].tolist(),
        "scale": target_scaler.scale_[0].tolist()
    }
elif scaler_type == "minmax":
    param_scaler_params = {
        "data_min": param_scaler.data_min_.tolist(),
        "data_max": param_scaler.data_max_.tolist()
    }
    target_scaler_params = {
        "data_min": target_scaler.data_min_[0].tolist(),
        "data_max": target_scaler.data_max_[0].tolist()
    }

# ---------------------------
# Sampling Strategy Function
# ---------------------------
def sample_indices(sampling_method, num_samples):
    """
    Choose indices based on the desired sampling strategy.
    
    Available methods:
      - "multiplicity": sample proportional to multiplicity.
      - "no_multiplicity": uniform random sampling.
      - "flat": weights = multiplicity / exp(-neg_log_like) (counteracts MCMC bias).
      - "no_multiplicity_flat": weights = 1 / exp(-neg_log_like) (ignores multiplicity).
    """
    indices = np.arange(len(multiplicity))
    
    if sampling_method == "multiplicity":
        probabilities = np.exp(multiplicity) / np.sum(np.exp(multiplicity))
    elif sampling_method == "no_multiplicity":
        probabilities = np.ones_like(multiplicity) / len(multiplicity)
    elif sampling_method == "flat":
        max_neg = np.max(neg_log_like)
        safe_neg = neg_log_like - max_neg
        weights = np.exp(multiplicity) * np.exp(safe_neg)
        probabilities = weights / np.sum(weights)
    elif sampling_method == "no_multiplicity_flat":
        max_neg = np.max(neg_log_like)
        safe_neg = neg_log_like - max_neg
        weights = np.exp(safe_neg)
        probabilities = weights / np.sum(weights)
    else:
        raise ValueError("Invalid sampling method selected.")
    
    sampled_indices = np.random.choice(indices, size=num_samples, p=probabilities, replace=False)
    return sampled_indices

# ---------------------------
# Select Training Samples
# ---------------------------
# Choose one of the following sampling methods:
# "multiplicity", "no_multiplicity", "flat", "no_multiplicity_flat"
sampling_method = "flat"  
num_samples = 2000  # Adjust as needed

sampled_idx = sample_indices(sampling_method, num_samples=num_samples)
X_train = params_scaled[sampled_idx]
y_train = targets_scaled[sampled_idx]

# ---------------------------
# Custom Layer for Scaling
# ---------------------------
class ScaledPredictionLayer(tf.keras.layers.Layer):
    def __init__(self, scaler_type, param_scaler_params, target_scaler_params, **kwargs):
        super(ScaledPredictionLayer, self).__init__(**kwargs)
        self.scaler_type = scaler_type
        self.param_scaler_params = param_scaler_params
        self.target_scaler_params = target_scaler_params

    def call(self, inputs):
        # Unpack inputs: scaled parameters and predicted Gaussian parameters.
        scaled_params, pred_params = inputs

        # (A) Inverse transform the scaled parameters.
        if self.scaler_type == "standard":
            means = tf.constant(self.param_scaler_params["mean"], dtype=tf.float32)
            stds = tf.constant(self.param_scaler_params["scale"], dtype=tf.float32)
            params_unscaled = scaled_params * stds + means
        elif self.scaler_type == "minmax":
            data_min = tf.constant(self.param_scaler_params["data_min"], dtype=tf.float32)
            data_max = tf.constant(self.param_scaler_params["data_max"], dtype=tf.float32)
            params_unscaled = scaled_params * (data_max - data_min) + data_min

        p1 = params_unscaled[:, 0:1]
        p2 = params_unscaled[:, 1:2]

        # (B) Compute the unscaled function values using the Gaussian parameters.
        chi0 = pred_params[:, 0:1]
        p1_0 = pred_params[:, 1:2]
        p2_0 = pred_params[:, 2:3]
        c_inv_p1_p1 = pred_params[:, 3:4]
        c_inv_p1_p2 = pred_params[:, 4:5]
        c_inv_p2_p2 = pred_params[:, 5:6]

        dp1 = p1 - p1_0
        dp2 = p2 - p2_0
        mahalanobis = c_inv_p1_p1 * dp1**2 + 2 * c_inv_p1_p2 * dp1 * dp2 + c_inv_p2_p2 * dp2**2
        unscaled_values = chi0 + 0.5 * mahalanobis

        # (C) Rescale the computed function value to match the target scaling.
        if self.scaler_type == "standard":
            target_mean = tf.constant(self.target_scaler_params["mean"], dtype=tf.float32)
            target_std = tf.constant(self.target_scaler_params["scale"], dtype=tf.float32)
            scaled_values = (unscaled_values - target_mean) / target_std
        elif self.scaler_type == "minmax":
            target_min = tf.constant(self.target_scaler_params["data_min"], dtype=tf.float32)
            target_max = tf.constant(self.target_scaler_params["data_max"], dtype=tf.float32)
            scaled_values = (unscaled_values - target_min) / (target_max - target_min)
        return scaled_values

    def get_config(self):
        config = super(ScaledPredictionLayer, self).get_config()
        config.update({
            "scaler_type": self.scaler_type,
            "param_scaler_params": self.param_scaler_params,
            "target_scaler_params": self.target_scaler_params,
        })
        return config

# ---------------------------
# Build the Neural Network Model
# ---------------------------
# Define input layer for the 2 scaled parameters.
inputs_scaled = layers.Input(shape=(2,), name="scaled_params")

# Hidden layers predicting 6 Gaussian parameters.
#hidden = layers.Dense(480, activation="relu")(inputs_scaled)
#hidden = layers.Dense(480, activation="tanh")(hidden)
#hidden = layers.Dense(160, activation="relu")(hidden)
#hidden = layers.Dense(352, activation="sigmoid")(hidden)
#hidden = layers.Dense(288, activation="tanh")(hidden)
#pred_params = layers.Dense(6, name="gaussian_params")(hidden)

# Result from Hyperband
#hidden = layers.Dense(128, activation="relu")(inputs_scaled)
#hidden = layers.Dense(512, activation="relu")(hidden)
#hidden = layers.Dense(480, activation="relu")(hidden)
#hidden = layers.Dense(384, activation="tanh")(hidden)
#hidden = layers.Dense(352, activation="relu")(hidden)
#hidden = layers.Dense(384, activation="tanh")(hidden)
#hidden = layers.Dense(32, activation="sigmoid")(hidden)
#hidden = layers.Dense(192, activation="tanh")(hidden)
#pred_params = layers.Dense(6, name="gaussian_params")(hidden)

# Result from Random Search
hidden = layers.Dense(160, activation="tanh")(inputs_scaled)
hidden = layers.Dense(256, activation="relu")(hidden)
hidden = layers.Dense(384, activation="relu")(hidden)
hidden = layers.Dense(256, activation="sigmoid")(hidden)
hidden = layers.Dense(224, activation="relu")(hidden)
hidden = layers.Dense(32, activation="sigmoid")(hidden)
pred_params = layers.Dense(6, name="gaussian_params")(hidden)

# 1 Gaussian
#hidden = layers.Dense(256, activation="sigmoid")(inputs_scaled)
#hidden = layers.Dense(512, activation="sigmoid")(hidden)
#hidden = layers.Dense(288, activation="sigmoid")(hidden)
#pred_params = layers.Dense(6, name="gaussian_params")(hidden)

# 2 Gaussians
#hidden = layers.Dense(416, activation="tanh")(inputs_scaled)
#hidden = layers.Dense(448, activation="sigmoid")(hidden)
#hidden = layers.Dense(416, activation="sigmoid")(hidden)
#hidden = layers.Dense(256, activation="sigmoid")(hidden)
#hidden = layers.Dense(448, activation="relu")(hidden)
#pred_params = layers.Dense(6, name="gaussian_params")(hidden)

# Compute final scaled prediction using our custom layer.
scaled_prediction = ScaledPredictionLayer(
    scaler_type=scaler_type,
    param_scaler_params=param_scaler_params,
    target_scaler_params=target_scaler_params,
    name="predicted_values"
)([inputs_scaled, pred_params])

model = models.Model(inputs=inputs_scaled, outputs=scaled_prediction)
model.summary()

# ---------------------------
# Define the Custom Loss Function
# ---------------------------
def chi2_underestimate_penalizing_loss(lambda_weight=5.0):
    def loss(y_true, y_pred):
        mse = tf.square(y_true - y_pred)
        penalty = lambda_weight * tf.square(tf.maximum(y_true - y_pred, 0))  # Extra penalty for underestimates
        return tf.reduce_mean(mse + penalty)
    return loss

# ---------------------------
# Compile and Train the Model
# ---------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
    loss=chi2_underestimate_penalizing_loss(lambda_weight=0)
)

early_stopping = EarlyStopping(
    monitor="loss",
    patience=100,
    restore_best_weights=True
)

history = model.fit(
    X_train,       # Training data sampled by the selected strategy.
    y_train,
    epochs=1000,
    batch_size=32,
    validation_split=0.1,
    verbose=2,
    callbacks=[early_stopping]
)

# ---------------------------
# Save the Model, History, and Plot the Training Loss
# ---------------------------
model_dir = "chain_models"
os.makedirs(model_dir, exist_ok=True)
model.save(os.path.join(model_dir, "chain_trained_model.h5"))
with open(os.path.join(data_dir, "chain_training_history.pkl"), "wb") as f:
    pickle.dump(history.history, f)

plt.figure(figsize=(8, 6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.yscale("log")
plt.grid(True)
os.makedirs("plots", exist_ok=True)
plt.savefig(os.path.join("plots", "chain_training_loss.png"))
plt.show()
