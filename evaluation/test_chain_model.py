import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

# -------------------------------
# Custom Layer Definition
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

        # For our 2-parameter case.
        p1 = params_unscaled[:, 0:1]
        p2 = params_unscaled[:, 1:2]

        # (B) Compute the unscaled function value.
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

        # (C) Rescale the function value to match target scaling.
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
model_path = os.path.join("chain_models", "chain_trained_model.h5")

# Load saved scalers (these were stored during training)
with open(os.path.join("data", "chain_param_scaler.pkl"), "rb") as f:
    param_scaler = pickle.load(f)
with open(os.path.join("data", "chain_target_scaler.pkl"), "rb") as f:
    target_scaler = pickle.load(f)

# -------------------------------
# Load the Model
# -------------------------------
# Note: compile=False is passed since we are only using the model for prediction.
model = tf.keras.models.load_model(
    model_path,
    custom_objects={"ScaledPredictionLayer": ScaledPredictionLayer},
    compile=False
)

# -------------------------------
# Load Validation Data
# -------------------------------
# In training, you used the first 3000 points after skipping 150 header lines.
# For validation, we load additional points (after 3150 lines) from each file.
validation_multiplicity = []
validation_neg_log_like = []
validation_param1 = []
validation_param2 = []

data_dir = os.path.join("chains", "2_param")
filenames = [
    "2024-09-17_123456789__1.txt",
    "2024-09-17_123456789__2.txt",
    "2024-09-17_123456789__3.txt",
    "2024-09-17_123456789__4.txt"
]

for filename in filenames:
    full_path = os.path.join(data_dir, filename)
    with open(full_path, "r") as f:
        # Skip first 3150 lines (150 header + 3000 training)
        for _ in range(3150):
            next(f)
        for line in f:
            cols = line.strip().split()
            if len(cols) < 4:
                continue
            try:
                mult = int(cols[0])
                neg_log = float(cols[1]) * 5  # Negative log-likelihood scaled by 5
                p1 = float(cols[2])
                p2 = float(cols[3])
            except ValueError:
                continue
            validation_multiplicity.append(mult)
            validation_neg_log_like.append(neg_log)
            validation_param1.append(p1)
            validation_param2.append(p2)

# Combine validation parameters and targets.
val_params = np.column_stack((validation_param1, validation_param2))
y_val = np.array(validation_neg_log_like)  # True negative log-likelihoods (original scale)

# Scale the validation parameters using the training scaler.
X_val = param_scaler.transform(val_params)

# -------------------------------
# Get Predictions and Compute Relative Error
# -------------------------------
# Model outputs are scaled predictions.
y_pred_scaled = model.predict(X_val).flatten()
# Invert the target scaling to get predictions in the original scale.
y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# Compute relative error.
relative_error = (y_pred - y_val) / y_val

# -------------------------------
# Plot the Results
# -------------------------------
plot_dir = "plots"

plt.figure(figsize=(10, 8))
plt.scatter(y_val, relative_error, s=10, alpha=0.6)
plt.xlabel(r"$\chi^2_{{true}}$")
plt.ylabel(r"$(\chi^2_{{pred}} - \chi^2_{{true}}) / \chi^2_{{true}}$")
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "chain_model_residuals.png"), dpi=300)
plt.show()
