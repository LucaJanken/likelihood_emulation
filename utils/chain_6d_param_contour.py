import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ---------------------------
# Custom Layer Definition for 6D
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
            stds  = tf.constant(self.param_scaler_params["scale"], dtype=tf.float32)
            params_unscaled = scaled_params * stds + means
        elif self.scaler_type == "minmax":
            data_min = tf.constant(self.param_scaler_params["data_min"], dtype=tf.float32)
            data_max = tf.constant(self.param_scaler_params["data_max"], dtype=tf.float32)
            params_unscaled = scaled_params * (data_max - data_min) + data_min

        # (B) Extract predicted Gaussian parameters.
        # For 6D we expect 28 outputs:
        #   - chi0: best-fit offset (shape: [batch, 1])
        #   - bestfit: best-fit 6D point (shape: [batch, 6])
        #   - inv_cov_flat: 21 independent entries of the symmetric inverse covariance matrix (shape: [batch, 21])
        chi0     = pred_params[:, 0:1]
        bestfit  = pred_params[:, 1:7]
        inv_cov_flat = pred_params[:, 7:]

        # Compute the difference vector.
        dp = params_unscaled - bestfit  # shape: [batch, 6]

        # Reconstruct the full symmetric 6×6 inverse covariance matrix.
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

        # (C) Compute the Mahalanobis distance: dpᵀ * c_inv * dp.
        dp_expanded = tf.expand_dims(dp, axis=-1)  # [batch, 6, 1]
        temp = tf.matmul(c_inv, dp_expanded)         # [batch, 6, 1]
        mahalanobis = tf.matmul(tf.transpose(dp_expanded, perm=[0, 2, 1]), temp)
        mahalanobis = tf.squeeze(mahalanobis, axis=[1, 2])  # [batch]
        mahalanobis = tf.expand_dims(mahalanobis, axis=-1)  # [batch, 1]

        # (D) Compute the unscaled predicted -loglikelihood.
        unscaled_values = chi0 + 0.5 * mahalanobis

        # (E) Rescale to match target scaling.
        if self.scaler_type == "standard":
            target_mean = tf.constant(self.target_scaler_params["mean"], dtype=tf.float32)
            target_std  = tf.constant(self.target_scaler_params["scale"], dtype=tf.float32)
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
# Adjust these paths if needed.
model_path = os.path.join("chain_models", "chain_6D_trained_model.h5")
param_scaler_path = os.path.join("data", "chain_6D_param_scaler.pkl")
target_scaler_path = os.path.join("data", "chain_6D_target_scaler.pkl")

# Directory to save contour plots.
plot_dir = "plots_params"
os.makedirs(plot_dir, exist_ok=True)

# ---------------------------
# Load the Scalers
# ---------------------------
with open(param_scaler_path, "rb") as f:
    param_scaler = pickle.load(f)
with open(target_scaler_path, "rb") as f:
    target_scaler = pickle.load(f)

# ---------------------------
# Determine (p1, p2) Ranges & Fix Other Parameters
# ---------------------------
# For a 6D input, we vary p1 and p2 while fixing p3,...,p6.
if isinstance(param_scaler, StandardScaler):
    means = param_scaler.mean_
    scales = param_scaler.scale_
    p1_mean, p2_mean = means[0], means[1]
    p1_std, p2_std   = scales[0], scales[1]
    p1_min, p1_max = p1_mean - 3 * p1_std, p1_mean + 3 * p1_std
    p2_min, p2_max = p2_mean - 3 * p2_std, p2_mean + 3 * p2_std
    # Fix p3...p6 at their means.
    p3_fixed, p4_fixed, p5_fixed, p6_fixed = means[2], means[3], means[4], means[5]
elif isinstance(param_scaler, MinMaxScaler):
    data_min = param_scaler.data_min_
    data_max = param_scaler.data_max_
    p1_min, p1_max = data_min[0], data_max[0]
    p2_min, p2_max = data_min[1], data_max[1]
    # For p3...p6 use the mid-point.
    p3_fixed = 0.5 * (data_min[2] + data_max[2])
    p4_fixed = 0.5 * (data_min[3] + data_max[3])
    p5_fixed = 0.5 * (data_min[4] + data_max[4])
    p6_fixed = 0.5 * (data_min[5] + data_max[5])
else:
    raise ValueError("Unknown scaler type for param_scaler.")

# ---------------------------
# Load the Model & Create Intermediate Model
# ---------------------------
# The model uses a custom layer, so we pass it via custom_objects.
model = load_model(model_path, custom_objects={"ScaledPredictionLayer": ScaledPredictionLayer}, compile=False)
# The intermediate model outputs the raw Gaussian parameters.
intermediate_model = Model(inputs=model.input, 
                           outputs=model.get_layer("gaussian_params").output)
# For 6D, this intermediate layer outputs 28 parameters.

# ---------------------------
# Generate a Grid in the (p1, p2)-Plane
# ---------------------------
num_points = 200  # Resolution of the grid.
p1_vals = np.linspace(p1_min, p1_max, num_points)
p2_vals = np.linspace(p2_min, p2_max, num_points)
P1, P2 = np.meshgrid(p1_vals, p2_vals)

# Build grid points in 6D: vary p1 and p2, fix p3...p6.
grid_points = np.column_stack((
    P1.ravel(), 
    P2.ravel(),
    np.full(P1.size, p3_fixed),
    np.full(P1.size, p4_fixed),
    np.full(P1.size, p5_fixed),
    np.full(P1.size, p6_fixed)
))

# Scale the grid points.
grid_points_scaled = param_scaler.transform(grid_points)

# ---------------------------
# Predict Gaussian Parameters on the Grid
# ---------------------------
# Get the 28 Gaussian parameters for each grid point.
pred_params = intermediate_model.predict(grid_points_scaled, verbose=0)
# Reshape predictions to grid shape: (num_points, num_points, 28)
pred_params_reshaped = pred_params.reshape(num_points, num_points, 28)

# Define parameter names in order.
param_names = [
    "chi0", 
    "p1_0", "p2_0", "p3_0", "p4_0", "p5_0", "p6_0",
    "c_inv_p1_p1", "c_inv_p1_p2", "c_inv_p1_p3", "c_inv_p1_p4", "c_inv_p1_p5", "c_inv_p1_p6",
    "c_inv_p2_p2", "c_inv_p2_p3", "c_inv_p2_p4", "c_inv_p2_p5", "c_inv_p2_p6",
    "c_inv_p3_p3", "c_inv_p3_p4", "c_inv_p3_p5", "c_inv_p3_p6",
    "c_inv_p4_p4", "c_inv_p4_p5", "c_inv_p4_p6",
    "c_inv_p5_p5", "c_inv_p5_p6",
    "c_inv_p6_p6"
]

# ---------------------------
# Plot Contour Maps for Each Gaussian Parameter
# ---------------------------
num_params = len(param_names)  # 28 in total.
# Create a subplot grid: e.g. 7 rows x 4 columns.
rows = 7
cols = 4
fig, axes = plt.subplots(rows, cols, figsize=(20, 30))
axes = axes.ravel()

for i in range(num_params):
    ax = axes[i]
    # Extract parameter i from the predictions.
    Z = pred_params_reshaped[:, :, i]
    cf = ax.contourf(P1, P2, Z, levels=50, cmap="viridis")
    fig.colorbar(cf, ax=ax)
    ax.set_title(f"{param_names[i]}")
    ax.set_xlabel("p1")
    ax.set_ylabel("p2")

plt.tight_layout()
out_file = os.path.join(plot_dir, "chain_6D_params_contours.png")
plt.savefig(out_file, dpi=300)
plt.show()
