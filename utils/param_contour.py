import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle
import matplotlib.pyplot as plt

# --- Configurable Variables ---
model_path = os.path.join("models", "dg_hps_trained_model.h5")
param_scaler_path = os.path.join("data", "dg_hps_param_scaler.pkl")
target_scaler_path = os.path.join("data", "dg_hps_target_scaler.pkl")

# Create a directory to store plots
plot_dir = "plots_params"
os.makedirs(plot_dir, exist_ok=True)

# 1) Load the scalers at the top, so they exist globally
with open(param_scaler_path, "rb") as f:
    param_scaler = pickle.load(f)

with open(target_scaler_path, "rb") as f:
    target_scaler = pickle.load(f)

# 2) Define TF constants for target_scaler (these must match your training script logic)
#    and must be defined as global variables (in your script) BEFORE loading the model.
if isinstance(target_scaler, StandardScaler):
    target_mean = tf.constant(target_scaler.mean_[0], dtype=tf.float32)
    target_std = tf.constant(target_scaler.scale_[0], dtype=tf.float32)
    # If you need them:
    # target_mean_vec = tf.constant(target_scaler.mean_, dtype=tf.float32)
    # target_std_vec = tf.constant(target_scaler.scale_, dtype=tf.float32)
elif isinstance(target_scaler, MinMaxScaler):
    target_min = tf.constant(target_scaler.data_min_[0], dtype=tf.float32)
    target_max = tf.constant(target_scaler.data_max_[0], dtype=tf.float32)
    # Similarly if you need the full vector:
    # target_min_vec = tf.constant(target_scaler.data_min_, dtype=tf.float32)
    # target_max_vec = tf.constant(target_scaler.data_max_, dtype=tf.float32)

# 3) Define any custom functions. They can reference global variables. 
#    This function name must match what you used in the training script.
def inverse_transform_tf(scaled_tensor, scaler):
    if isinstance(scaler, StandardScaler):
        means = tf.constant(scaler.mean_, dtype=tf.float32)
        stds = tf.constant(scaler.scale_, dtype=tf.float32)
        return scaled_tensor * stds + means
    elif isinstance(scaler, MinMaxScaler):
        min_val = tf.constant(scaler.data_min_, dtype=tf.float32)
        max_val = tf.constant(scaler.data_max_, dtype=tf.float32)
        return scaled_tensor * (max_val - min_val) + min_val

# 4) Finally, load the model with the custom objects
model = load_model(
    model_path,
    custom_objects={
        "inverse_transform_tf": inverse_transform_tf,
        # If you have other custom objects, list them as well
    },
)

# 5) Create an intermediate model that outputs the Gaussian parameters 
intermediate_model = Model(
    inputs=model.input,
    outputs=model.get_layer("gaussian_params").output
)

# ------------------------------------------------------------------------------
# Now you can proceed with generating your (x, y) grid, scaling, predicting, etc.
# ------------------------------------------------------------------------------

# Set your evaluation range and resolution
x_min, x_max = -5, 5
y_min, y_max = -5, 5
num_points = 200  # Number of points per axis

# Generate a grid of (x, y) points
x_vals = np.linspace(x_min, x_max, num_points)
y_vals = np.linspace(y_min, y_max, num_points)
X, Y = np.meshgrid(x_vals, y_vals)

# Flatten and prepare for model prediction
grid_points = np.column_stack((X.ravel(), Y.ravel()))

# Scale the grid points
grid_points_scaled = param_scaler.transform(grid_points)

# Predict the six parameters for each (x, y) point in the grid
pred_params = intermediate_model.predict(grid_points_scaled, verbose=0)
pred_params_reshaped = pred_params.reshape(num_points, num_points, 6)

# Unpack each parameter
p0  = pred_params_reshaped[:, :, 0]
x0  = pred_params_reshaped[:, :, 1]
y0  = pred_params_reshaped[:, :, 2]
cxx = pred_params_reshaped[:, :, 3]
cxy = pred_params_reshaped[:, :, 4]
cyy = pred_params_reshaped[:, :, 5]

# Plot them as desired
param_names = ["p0", "x0", "y0", "cxx", "cxy", "cyy"]
param_data = [p0, x0, y0, cxx, cxy, cyy]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()
for i, (ax, name) in enumerate(zip(axes, param_names)):
    c = ax.contourf(X, Y, param_data[i], levels=50, cmap="viridis")
    plt.colorbar(c, ax=ax)
    ax.set_title(f"Parameter: {name}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

plt.tight_layout()
out_file = os.path.join(plot_dir, "dg_hps_params_contours.png")
plt.savefig(out_file, dpi=300)
plt.show()
