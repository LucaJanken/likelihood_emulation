import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pyDOE import lhs
import pickle
import matplotlib.pyplot as plt

# Choose scaler type: 'standard' for StandardScaler, 'minmax' for MinMaxScaler
scaler_type = "standard"  # Change to "minmax" if desired

# Set up directories and file paths
data_dir = "data"
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

model_path = os.path.join("models", "sn_trained_model.h5")
param_scaler_path = os.path.join(data_dir, "sn_param_scaler.pkl")
target_scaler_path = os.path.join(data_dir, "sn_target_scaler.pkl")

# Load scalers
with open(param_scaler_path, "rb") as f:
    param_scaler = pickle.load(f)
with open(target_scaler_path, "rb") as f:
    target_scaler = pickle.load(f)

# Define inverse transformation for both scaler types
def inverse_transform_tf(scaled_tensor, scaler):
    if isinstance(scaler, StandardScaler):
        means = tf.constant(scaler.mean_, dtype=tf.float32)
        stds = tf.constant(scaler.scale_, dtype=tf.float32)
        return scaled_tensor * stds + means
    elif isinstance(scaler, MinMaxScaler):
        min_val = tf.constant(scaler.data_min_, dtype=tf.float32)
        max_val = tf.constant(scaler.data_max_, dtype=tf.float32)
        return scaled_tensor * (max_val - min_val) + min_val

# Extract target scaling constants
if isinstance(target_scaler, StandardScaler):
    target_mean = tf.constant(target_scaler.mean_[0], dtype=tf.float32)
    target_std = tf.constant(target_scaler.scale_[0], dtype=tf.float32)
elif isinstance(target_scaler, MinMaxScaler):
    target_min = tf.constant(target_scaler.data_min_[0], dtype=tf.float32)
    target_max = tf.constant(target_scaler.data_max_[0], dtype=tf.float32)

# Load the trained sterile neutrino model
model = load_model(model_path, custom_objects={"inverse_transform_tf": inverse_transform_tf}, compile=False)

# Define the true sterile neutrino chi² function: chi² = (N_eff * m0)²
def chi_squared_sterile_neutrino(N_eff, m0):
    return (N_eff * m0) ** 2

# Define sampling ranges (in both parameters) for evaluation
# You can include both in-domain and out-of-domain regions.
sample_ranges = {
    "In-Domain [0,3]": (0, 3),
    "Out-Domain [0,4]": (0, 4),
    "Out-Domain [0,5]": (0, 5),
    "Out-Domain [0,6]": (0, 6),
    "Out-Domain [0,12]": (0, 12),
    "Out-Domain [0,24]": (0, 24)
}

num_samples = 1000        # Number of samples per region
num_samples_per_axis = 200  # Grid resolution for contour plots

# A small constant to avoid division by zero
epsilon = 1e-8

# Loop over each region to evaluate and plot results
for label, (xmin, xmax) in sample_ranges.items():
    # For symmetric ranges in N_eff and m0
    ymin, ymax = xmin, xmax

    # Generate Latin Hypercube samples within the box [xmin, xmax]²
    lhs_samples = lhs(2, samples=num_samples)
    N_eff_samples = lhs_samples[:, 0] * (xmax - xmin) + xmin
    m0_samples = lhs_samples[:, 1] * (xmax - xmin) + xmin
    params = np.column_stack((N_eff_samples, m0_samples))

    # Compute true chi² values
    chi2_true = np.array([chi_squared_sterile_neutrino(N_eff, m0) for N_eff, m0 in params])

    # Scale parameters and predict using the model
    params_scaled = param_scaler.transform(params)
    chi2_pred_scaled = model.predict(params_scaled, verbose=0)
    chi2_pred = target_scaler.inverse_transform(chi2_pred_scaled).flatten()

    # Compute relative residuals with an epsilon in the denominator to avoid division by zero
    relative_residuals = (chi2_pred - chi2_true) # / (chi2_true + epsilon)

    # Scatter plot of relative residuals vs. true chi² values
    plt.figure(figsize=(6, 5))
    plt.scatter(chi2_true, relative_residuals, s=5, alpha=0.5)
    plt.axhline(y=0, color="r", linestyle="--", linewidth=1)
    plt.xlabel(r"$\chi^2_{\rm true}$")
    #plt.ylabel(r"$(\chi^2_{\rm pred} - \chi^2_{\rm true})/\chi^2_{\rm true}$")
    plt.ylabel(r"$\chi^2_{\rm pred} - \chi^2_{\rm true}$")
    plt.title(label)
    plt.grid(True)
    plt.tight_layout()

    # Save and show the residual plot
    plot_filename = os.path.join(plot_dir, f"SN_chi2_residuals_{label.replace(' ', '_').replace('[','').replace(']','')}.png")
    plt.savefig(plot_filename, dpi=300)
    plt.show()
    print(f"Residual plot saved: {plot_filename}")

    # --- Contour Plots ---
    # Create a grid of points in the current range
    x_linspace = np.linspace(xmin, xmax, num_samples_per_axis)
    y_linspace = np.linspace(ymin, ymax, num_samples_per_axis)
    X, Y = np.meshgrid(x_linspace, y_linspace)
    grid_points = np.column_stack((X.ravel(), Y.ravel()))

    # Compute true chi² values on the grid
    chi2_true_grid = np.array([chi_squared_sterile_neutrino(N_eff, m0) for N_eff, m0 in grid_points]).reshape(X.shape)

    # Predict chi² values using the model for the grid
    params_scaled_grid = param_scaler.transform(grid_points)
    chi2_pred_scaled_grid = model.predict(params_scaled_grid, verbose=0)
    chi2_pred_grid = target_scaler.inverse_transform(chi2_pred_scaled_grid).reshape(X.shape)

    # Compute residuals grid (predicted minus true)
    residuals_grid = chi2_pred_grid - chi2_true_grid

    # Create contour plots for the true chi², predicted chi², and residuals
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # True chi² contour
    c1 = axes[0].contourf(X, Y, chi2_true_grid, levels=50, cmap="viridis")
    fig.colorbar(c1, ax=axes[0])
    axes[0].set_title(f"True $\chi^2$ ({label})")
    axes[0].set_xlabel(r"$N_{\rm eff}$")
    axes[0].set_ylabel(r"$m_0$")

    # Predicted chi² contour
    c2 = axes[1].contourf(X, Y, chi2_pred_grid, levels=50, cmap="viridis")
    fig.colorbar(c2, ax=axes[1])
    axes[1].set_title(f"Predicted $\chi^2$ ({label})")
    axes[1].set_xlabel(r"$N_{\rm eff}$")
    axes[1].set_ylabel(r"$m_0$")

    # Residuals contour
    c3 = axes[2].contourf(X, Y, residuals_grid, levels=50, cmap="coolwarm")
    fig.colorbar(c3, ax=axes[2])
    axes[2].set_title(f"Residuals: $\chi^2_{{pred}} - \chi^2_{{true}}$ ({label})")
    axes[2].set_xlabel(r"$N_{\rm eff}$")
    axes[2].set_ylabel(r"$m_0$")

    plt.tight_layout()
    contour_filename = os.path.join(plot_dir, f"SN_chi2_contours_{label.replace(' ', '_').replace('[','').replace(']','')}.png")
    plt.savefig(contour_filename, dpi=300)
    plt.show()
    print(f"Contour plot saved: {contour_filename}")
