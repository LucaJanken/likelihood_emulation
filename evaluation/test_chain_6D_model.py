import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

# -------------------------------
# Custom Layer Definition for 6D
# -------------------------------
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
        #   - chi0: best-fit offset (shape: [batch, 1])
        #   - bestfit: best-fit 6D point (shape: [batch, 6])
        #   - inv_cov_flat: 21 independent entries for the symmetric inverse covariance matrix (shape: [batch, 21])
        chi0 = pred_params[:, 0:1]
        bestfit = pred_params[:, 1:7]
        inv_cov_flat = pred_params[:, 7:]

        # Compute difference vector.
        dp = params_unscaled - bestfit  # shape: [batch, 6]

        # Reconstruct the full symmetric 6x6 inverse covariance matrix.
        def create_matrix(x):
            indices = tf.constant([
                [0, 0],
                [0, 1],
                [0, 2],
                [0, 3],
                [0, 4],
                [0, 5],
                [1, 1],
                [1, 2],
                [1, 3],
                [1, 4],
                [1, 5],
                [2, 2],
                [2, 3],
                [2, 4],
                [2, 5],
                [3, 3],
                [3, 4],
                [3, 5],
                [4, 4],
                [4, 5],
                [5, 5]
            ], dtype=tf.int64)
            mat = tf.scatter_nd(indices, x, shape=[6, 6])
            # Make the matrix symmetric.
            mat = mat + tf.transpose(mat) - tf.linalg.diag(tf.linalg.diag_part(mat))
            return mat

        c_inv = tf.map_fn(create_matrix, inv_cov_flat, dtype=tf.float32)  # shape: [batch, 6, 6]

        # (C) Compute the Mahalanobis distance: dp^T * c_inv * dp.
        dp_expanded = tf.expand_dims(dp, axis=-1)  # shape: [batch, 6, 1]
        temp = tf.matmul(c_inv, dp_expanded)         # shape: [batch, 6, 1]
        mahalanobis = tf.matmul(tf.transpose(dp_expanded, perm=[0, 2, 1]), temp)
        mahalanobis = tf.squeeze(mahalanobis, axis=[1, 2])  # shape: [batch]
        mahalanobis = tf.expand_dims(mahalanobis, axis=-1)  # shape: [batch, 1]

        # (D) Compute the unscaled predicted -loglikelihood.
        unscaled_values = chi0 + 0.5 * mahalanobis

        # (E) Rescale the computed function value to match target scaling.
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

# -------------------------------
# Model & Scaler Paths
# -------------------------------
model_path = os.path.join("chain_models", "chain_6D_trained_model.h5")

# Load saved scalers (stored during training)
with open(os.path.join("data", "chain_6D_param_scaler.pkl"), "rb") as f:
    param_scaler = pickle.load(f)
with open(os.path.join("data", "chain_6D_target_scaler.pkl"), "rb") as f:
    target_scaler = pickle.load(f)

# -------------------------------
# Load the Model
# -------------------------------
# compile=False since we are only using the model for prediction.
model = tf.keras.models.load_model(
    model_path,
    custom_objects={"ScaledPredictionLayer": ScaledPredictionLayer},
    compile=False
)

# -------------------------------
# Load Validation Data
# -------------------------------
# For training, 4000 points (after skipping 150 header lines) were used.
# For validation, we load additional points starting after 4150 lines.
validation_multiplicity = []
validation_neg_log_like = []
validation_param1 = []
validation_param2 = []
validation_param3 = []
validation_param4 = []
validation_param5 = []
validation_param6 = []

data_dir = os.path.join("chains", "6_param")
# Only include non-corrupted files.
filenames = [
    "2024-09-17_123456789__1.txt",
    "2024-09-17_123456789__3.txt",
    "2024-09-17_123456789__4.txt"
]

for filename in filenames:
    full_path = os.path.join(data_dir, filename)
    with open(full_path, "r") as f:
        # Skip first 4150 lines (150 header + 4000 training)
        for _ in range(4150):
            next(f)
        for line in f:
            cols = line.strip().split()
            if len(cols) < 8:
                continue
            try:
                mult = int(cols[0])
                neg_log = float(cols[1]) * 5  # Scale -loglikelihood by 5.
                p1 = float(cols[2])
                p2 = float(cols[3])
                p3 = float(cols[4])
                p4 = float(cols[5])
                p5 = float(cols[6])
                p6 = float(cols[7])
            except ValueError:
                continue
            validation_multiplicity.append(mult)
            validation_neg_log_like.append(neg_log)
            validation_param1.append(p1)
            validation_param2.append(p2)
            validation_param3.append(p3)
            validation_param4.append(p4)
            validation_param5.append(p5)
            validation_param6.append(p6)

# Combine validation parameters and targets.
val_params = np.column_stack((validation_param1, validation_param2, validation_param3,
                               validation_param4, validation_param5, validation_param6))
y_val = np.array(validation_neg_log_like)  # True -loglikelihoods in original scale

# Scale the validation parameters using the training scaler.
X_val = param_scaler.transform(val_params)

# -------------------------------
# Get Predictions and Compute Relative Error
# -------------------------------
# Model outputs are scaled predictions.
y_pred_scaled = model.predict(X_val).flatten()
# Invert target scaling to get predictions on the original scale.
y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# Compute relative error.
relative_error = (y_pred - y_val) / y_val

# -------------------------------
# Plot the Results
# -------------------------------
plot_dir = "plots"

plt.figure(figsize=(10, 8))
plt.scatter(y_val, relative_error, s=10, alpha=0.6)
plt.xlabel(r"$\chi^2_{\mathrm{true}}$")
plt.ylabel(r"$(\chi^2_{\mathrm{pred}} - \chi^2_{\mathrm{true}})/\chi^2_{\mathrm{true}}$")
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "chain_6D_model_residuals.png"), dpi=300)
plt.show()
