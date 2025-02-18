import os
import sys
import pickle
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
        # Skip header lines
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
                neg_log = float(columns[1]) * 5  
                p1_val = float(columns[2])
                p2_val = float(columns[3])
            except ValueError:
                continue
            multiplicity.append(mult)
            neg_log_like.append(neg_log)
            param1.append(p1_val)
            param2.append(p2_val)
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

# Save scalers for later use
with open(os.path.join(data_dir, "chain_hps_param_scaler.pkl"), "wb") as f:
    pickle.dump(param_scaler, f)
with open(os.path.join(data_dir, "chain_hps_target_scaler.pkl"), "wb") as f:
    pickle.dump(target_scaler, f)

# Save scaler parameters as dictionaries for our custom layer.
if scaler_type == "standard":
    param_scaler_params = {"mean": param_scaler.mean_.tolist(), "scale": param_scaler.scale_.tolist()}
    target_scaler_params = {"mean": target_scaler.mean_[0].tolist(), "scale": target_scaler.scale_[0].tolist()}
elif scaler_type == "minmax":
    param_scaler_params = {"data_min": param_scaler.data_min_.tolist(), "data_max": param_scaler.data_max_.tolist()}
    target_scaler_params = {"data_min": target_scaler.data_min_[0].tolist(), "data_max": target_scaler.data_max_[0].tolist()}

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
# Choose one of the sampling methods: "multiplicity", "no_multiplicity", "flat", "no_multiplicity_flat"
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
# Build Model Function for Hyperparameter Tuning
# ---------------------------
def build_model(hp):
    inputs = layers.Input(shape=(2,), name="scaled_params")
    
    # Hyperparameter: number of hidden layers (between 2 and 6)
    num_layers = hp.Int("num_layers", min_value=2, max_value=6, step=1)
    
    # First hidden layer
    x = layers.Dense(
        hp.Int("units_1", min_value=32, max_value=512, step=32),
        activation=hp.Choice("activation_1", values=["relu", "tanh", "sigmoid"])
    )(inputs)
    
    # Additional hidden layers
    for i in range(2, num_layers + 1):
        x = layers.Dense(
            hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
            activation=hp.Choice(f"activation_{i}", values=["relu", "tanh", "sigmoid"])
        )(x)
    
    # Output layer predicting 6 Gaussian parameters
    pred_params = layers.Dense(6, name="gaussian_params")(x)
    
    # Compute the final scaled prediction using our custom layer.
    output = ScaledPredictionLayer(
        scaler_type=scaler_type,
        param_scaler_params=param_scaler_params,
        target_scaler_params=target_scaler_params,
        name="predicted_values"
    )([inputs, pred_params])
    
    model = models.Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Float("learning_rate", min_value=1e-6, max_value=1e-3, sampling="log")
        ),
        loss="mse"
    )
    return model

# ---------------------------
# Hyperparameter Tuning using Keras Tuner (Random Search)
# ---------------------------
tuner = kt.RandomSearch(
    build_model,
    objective="val_loss",
    max_trials=2500,
    directory="random_search_results",
    project_name="chain_random_search"
)

early_stopping = EarlyStopping(monitor="val_loss", patience=100, restore_best_weights=True)

tuner.search(
    X_train, y_train,
    epochs=500,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stopping]
)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best hyperparameters:")
print(best_hps.values)

# Build the best model and train it further
best_model = tuner.hypermodel.build(best_hps)
history = best_model.fit(
    X_train, y_train,
    epochs=1000,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=2
)

# ---------------------------
# Save the Best Model and Training History
# ---------------------------
os.makedirs("models", exist_ok=True)
best_model.save(os.path.join("models", "chain_hps_trained_model.h5"))
with open(os.path.join(data_dir, "chain_hps_training_history.pkl"), "wb") as f:
    pickle.dump(history.history, f)

# Plot training and validation loss
plt.figure(figsize=(8, 6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Random Search Training History")
plt.legend()
plt.yscale("log")
plt.grid(True)
os.makedirs("plots", exist_ok=True)
plt.savefig(os.path.join("plots", "chain_hps_training_loss.png"))
plt.show()
