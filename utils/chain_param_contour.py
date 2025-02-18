import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ---------------------------
# Custom Layer Definition
# ---------------------------
# This must match your training script’s definition.
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
# Configurable Paths
# ---------------------------
model_path = os.path.join("models", "chain_trained_model.h5")
param_scaler_path = os.path.join("data", "chain_param_scaler.pkl")
target_scaler_path = os.path.join("data", "chain_target_scaler.pkl")

# Directory to save contour plots
plot_dir = "plots_params"
os.makedirs(plot_dir, exist_ok=True)

# ---------------------------
# Load the Scalers
# ---------------------------
with open(param_scaler_path, "rb") as f:
    param_scaler = pickle.load(f)
with open(target_scaler_path, "rb") as f:
    target_scaler = pickle.load(f)

# Determine the (p1, p2) ranges based on the training data.
# For StandardScaler we use mean ± 3·std; for MinMaxScaler we use the actual min/max.
if isinstance(param_scaler, StandardScaler):
    p1_mean, p2_mean = param_scaler.mean_[0], param_scaler.mean_[1]
    p1_std,  p2_std  = param_scaler.scale_[0], param_scaler.scale_[1]
    p1_min, p1_max = p1_mean - 3 * p1_std, p1_mean + 3 * p1_std
    p2_min, p2_max = p2_mean - 3 * p2_std, p2_mean + 3 * p2_std
elif isinstance(param_scaler, MinMaxScaler):
    p1_min, p1_max = param_scaler.data_min_[0], param_scaler.data_max_[0]
    p2_min, p2_max = param_scaler.data_min_[1], param_scaler.data_max_[1]
else:
    raise ValueError("Unknown scaler type for param_scaler.")

# ---------------------------
# Load the Model
# ---------------------------
# The model uses a custom layer, so we provide it via custom_objects.
model = load_model(model_path, custom_objects={"ScaledPredictionLayer": ScaledPredictionLayer})

# Create an intermediate model that outputs the raw Gaussian parameters.
# These parameters (in order) are: chi0, p1_0, p2_0, c_inv_p1_p1, c_inv_p1_p2, c_inv_p2_p2.
intermediate_model = Model(inputs=model.input, 
                           outputs=model.get_layer("gaussian_params").output)

# ---------------------------
# Generate a Grid in the (p1, p2)-Plane
# ---------------------------
num_points = 200  # Adjust the resolution as needed
p1_vals = np.linspace(p1_min, p1_max, num_points)
p2_vals = np.linspace(p2_min, p2_max, num_points)
P1, P2 = np.meshgrid(p1_vals, p2_vals)
grid_points = np.column_stack((P1.ravel(), P2.ravel()))

# Scale the grid points using the parameter scaler.
grid_points_scaled = param_scaler.transform(grid_points)

# ---------------------------
# Predict Gaussian Parameters
# ---------------------------
# Get the six Gaussian parameters for each grid point.
pred_params = intermediate_model.predict(grid_points_scaled, verbose=0)
pred_params_reshaped = pred_params.reshape(num_points, num_points, 6)

# Unpack the parameters.
chi0        = pred_params_reshaped[:, :, 0]
p1_0        = pred_params_reshaped[:, :, 1]
p2_0        = pred_params_reshaped[:, :, 2]
c_inv_p1_p1 = pred_params_reshaped[:, :, 3]
c_inv_p1_p2 = pred_params_reshaped[:, :, 4]
c_inv_p2_p2 = pred_params_reshaped[:, :, 5]

# ---------------------------
# Plot Contour Maps for Each Gaussian Parameter
# ---------------------------
param_names = ["chi0", "p1_0", "p2_0", "c_inv_p1_p1", "c_inv_p1_p2", "c_inv_p2_p2"]
param_data  = [chi0, p1_0, p2_0, c_inv_p1_p1, c_inv_p1_p2, c_inv_p2_p2]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for i, (ax, name) in enumerate(zip(axes, param_names)):
    cf = ax.contourf(P1, P2, param_data[i], levels=50, cmap="viridis")
    fig.colorbar(cf, ax=ax)
    ax.set_title(f"Parameter: {name}")
    ax.set_xlabel("p1")
    ax.set_ylabel("p2")

plt.tight_layout()
out_file = os.path.join(plot_dir, "chain_params_contours.png")
plt.savefig(out_file, dpi=300)
plt.show()
