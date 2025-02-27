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

chains_dir = "chains/6_param"
# Only include the non-corrupted files.
filenames = [
    "2024-09-17_123456789__1.txt",
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
param3 = []
param4 = []
param5 = []
param6 = []

# Read the first 4000 lines (after skipping 150 header lines) from each file.
for filename in filenames:
    full_path = os.path.join(chains_dir, filename)
    with open(full_path, "r") as file:
        # Skip header lines
        for _ in range(150):
            next(file)
        count = 0
        for line in file:
            if count >= 4000:
                break
            columns = line.strip().split()
            if len(columns) < 8:
                continue
            try:
                mult = int(columns[0])
                # Multiply -loglikelihood by 5.
                neg_log = float(columns[1]) * 5  
                p1_val = float(columns[2])
                p2_val = float(columns[3])
                p3_val = float(columns[4])
                p4_val = float(columns[5])
                p5_val = float(columns[6])
                p6_val = float(columns[7])
            except ValueError:
                continue
            multiplicity.append(mult)
            neg_log_like.append(neg_log)
            param1.append(p1_val)
            param2.append(p2_val)
            param3.append(p3_val)
            param4.append(p4_val)
            param5.append(p5_val)
            param6.append(p6_val)
            count += 1

multiplicity = np.array(multiplicity)
neg_log_like = np.array(neg_log_like)
params = np.column_stack((param1, param2, param3, param4, param5, param6))
targets = neg_log_like.reshape(-1, 1)

# ---------------------------
# Scaling of Data
# ---------------------------
param_scaler = get_scaler(scaler_type)
target_scaler = get_scaler(scaler_type)

params_scaled = param_scaler.fit_transform(params)
targets_scaled = target_scaler.fit_transform(targets)

# Save scalers for later use
with open(os.path.join(data_dir, "chain_6D_hps_param_scaler.pkl"), "wb") as f:
    pickle.dump(param_scaler, f)
with open(os.path.join(data_dir, "chain_6D_hps_target_scaler.pkl"), "wb") as f:
    pickle.dump(target_scaler, f)

# Save scaler parameters as dictionaries for the custom layer.
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
# For 6D data we select 5000 samples.
sampling_method = "flat"
num_samples = 5000

sampled_idx = sample_indices(sampling_method, num_samples=num_samples)
X_train = params_scaled[sampled_idx]
y_train = targets_scaled[sampled_idx]

# ---------------------------
# Custom Layer for Scaling & 6D Gaussian Reconstruction
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

        # (B) Extract predicted Gaussian parameters.
        # For 6D we expect 28 outputs:
        #   - chi0: best-fit offset, shape [batch, 1]
        #   - bestfit: best-fit 6D point, shape [batch, 6]
        #   - inv_cov_flat: 21 entries for the symmetric inverse covariance matrix, shape [batch, 21]
        chi0 = pred_params[:, 0:1]
        bestfit = pred_params[:, 1:7]
        inv_cov_flat = pred_params[:, 7:]  # 21 values

        # Compute the difference vector.
        dp = params_unscaled - bestfit  # shape: [batch, 6]

        # Reconstruct the full symmetric 6x6 inverse covariance matrix.
        def create_matrix(x):
            # x is a tensor of shape [21]
            indices = tf.constant([
                [0,0],
                [0,1],
                [0,2],
                [0,3],
                [0,4],
                [0,5],
                [1,1],
                [1,2],
                [1,3],
                [1,4],
                [1,5],
                [2,2],
                [2,3],
                [2,4],
                [2,5],
                [3,3],
                [3,4],
                [3,5],
                [4,4],
                [4,5],
                [5,5]
            ], dtype=tf.int64)
            mat = tf.scatter_nd(indices, x, shape=[6,6])
            # Make the matrix symmetric.
            mat = mat + tf.transpose(mat) - tf.linalg.diag(tf.linalg.diag_part(mat))
            return mat

        c_inv = tf.map_fn(create_matrix, inv_cov_flat, dtype=tf.float32)  # shape: [batch, 6, 6]

        # (C) Compute the Mahalanobis distance: dp^T * c_inv * dp.
        dp_expanded = tf.expand_dims(dp, axis=-1)  # shape: [batch, 6, 1]
        temp = tf.matmul(c_inv, dp_expanded)         # shape: [batch, 6, 1]
        mahalanobis = tf.matmul(tf.transpose(dp_expanded, perm=[0, 2, 1]), temp)
        mahalanobis = tf.squeeze(mahalanobis, axis=[1,2])  # shape: [batch]
        mahalanobis = tf.expand_dims(mahalanobis, axis=-1)  # shape: [batch, 1]

        # (D) Compute the unscaled predicted -loglikelihood.
        unscaled_values = chi0 + 0.5 * mahalanobis

        # (E) Rescale the computed function value to match the target scaling.
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
    inputs = layers.Input(shape=(6,), name="scaled_params")
    
    # Hyperparameter: number of hidden layers (between 2 and 10)
    num_layers = hp.Int("num_layers", min_value=2, max_value=10, step=1)
    
    # First hidden layer.
    x = layers.Dense(
        hp.Int("units_1", min_value=32, max_value=1024, step=32),
        activation=hp.Choice("activation_1", values=["relu", "tanh", "sigmoid"])
    )(inputs)
    
    # Additional hidden layers.
    for i in range(2, num_layers + 1):
        x = layers.Dense(
            hp.Int(f"units_{i}", min_value=32, max_value=1024, step=32),
            activation=hp.Choice(f"activation_{i}", values=["relu", "tanh", "sigmoid"])
        )(x)
    
    # Output layer predicting 28 Gaussian parameters.
    pred_params = layers.Dense(28, name="gaussian_params")(x)
    
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
    project_name="chain_hps_6D_random_search"
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

# Build the best model and train it further.
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
os.makedirs("chain_models", exist_ok=True)
best_model.save(os.path.join("models", "chain_hps_6D_trained_model.h5"))
with open(os.path.join(data_dir, "chain_hps_6D_training_history.pkl"), "wb") as f:
    pickle.dump(history.history, f)

# Plot training and validation loss.
plt.figure(figsize=(8, 6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("6D Random Search Training History")
plt.legend()
plt.yscale("log")
plt.grid(True)
os.makedirs("plots", exist_ok=True)
plt.savefig(os.path.join("plots", "chain_hps_6D_training_loss.png"))
plt.show()
