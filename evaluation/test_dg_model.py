import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pyDOE import lhs
import pickle
import matplotlib.pyplot as plt

# Choose scaler type: 'standard' for StandardScaler, 'minmax' for MinMaxScaler
scaler_type = "standard"

# Load trained model and scalers
data_dir = "data"
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

model_path = os.path.join("models", "dg_hps_trained_model.h5")
param_scaler_path = os.path.join(data_dir, "dg_hps_param_scaler.pkl")
target_scaler_path = os.path.join(data_dir, "dg_hps_target_scaler.pkl")

# Load scalers
with open(param_scaler_path, "rb") as f:
    param_scaler = pickle.load(f)

with open(target_scaler_path, "rb") as f:
    target_scaler = pickle.load(f)

# Define inverse transformation function
def inverse_transform_tf(scaled_tensor, scaler):
    """Inverse transformation for StandardScaler and MinMaxScaler."""
    if isinstance(scaler, StandardScaler):
        means = tf.constant(scaler.mean_, dtype=tf.float32)
        stds = tf.constant(scaler.scale_, dtype=tf.float32)
        return scaled_tensor * stds + means
    elif isinstance(scaler, MinMaxScaler):
        min_val = tf.constant(scaler.data_min_, dtype=tf.float32)
        max_val = tf.constant(scaler.data_max_, dtype=tf.float32)
        return scaled_tensor * (max_val - min_val) + min_val
    else:
        raise ValueError("Unsupported scaler type.")

# Extract target scaling parameters
if isinstance(target_scaler, StandardScaler):
    target_mean = tf.constant(target_scaler.mean_[0], dtype=tf.float32)
    target_std = tf.constant(target_scaler.scale_[0], dtype=tf.float32)
elif isinstance(target_scaler, MinMaxScaler):
    target_min = tf.constant(target_scaler.data_min_[0], dtype=tf.float32)
    target_max = tf.constant(target_scaler.data_max_[0], dtype=tf.float32)

# Custom function used in Lambda layer during training
def compute_scaled_values(args):
    """
    args = [scaled_xy, pred_params]
    scaled_xy: shape (batch_size, 2)
    pred_params: shape (batch_size, 6)
    """
    scaled_xy, pred_params = args

    # Inverse transform (x, y)
    xy_unscaled = inverse_transform_tf(scaled_xy, param_scaler)
    x = xy_unscaled[:, 0:1]  
    y = xy_unscaled[:, 1:2]  

    # Extract predicted parameters
    chi0 = pred_params[:, 0:1]
    x0 = pred_params[:, 1:2]
    y0 = pred_params[:, 2:3]
    cxx = pred_params[:, 3:4]
    cxy = pred_params[:, 4:5]
    cyy = pred_params[:, 5:6]

    dx = x - x0
    dy = y - y0
    mahalanobis_dist = cxx * dx**2 + 2 * cxy * dx * dy + cyy * dy**2
    unscaled_values = chi0 + mahalanobis_dist

    # Rescale back
    if isinstance(target_scaler, StandardScaler):
        return (unscaled_values - target_mean) / target_std
    elif isinstance(target_scaler, MinMaxScaler):
        return (unscaled_values - target_min) / (target_max - target_min)

# Load model with custom objects
model = load_model(
    model_path,
    custom_objects={
        "inverse_transform_tf": inverse_transform_tf,
        "compute_scaled_values": compute_scaled_values
    }
)

# Define chi^2 function for double Gaussian
def chi_squared_double_gaussian(x, y, bestfit_point1, bestfit_point2, sigma):
    x1, y1 = bestfit_point1
    x2, y2 = bestfit_point2
    
    term1 = np.exp(-0.5 * ((x - x1) ** 2 + (y - y1) ** 2) / sigma**2)
    term2 = np.exp(-0.5 * ((x - x2) ** 2 + (y - y2) ** 2) / sigma**2)
    likelihood = (1 / (2 * np.pi * sigma**2)) * (term1 + term2)
    
    chi2 = -2 * np.log(likelihood)
    return chi2

# Function parameters
bestfit_point1 = np.array([-1.0, -1.0])
bestfit_point2 = np.array([1.0, 1.0])
sigma = np.sqrt(0.1)

# Sampling ranges
sample_ranges = {
    "In-Domain [-3,3]": (-3, 3),
    "Out-Domain [-4,4]": (-4, 4),
    "Out-Domain [-5,5]": (-5, 5),
    "Out-Domain [-6,6]": (-6, 6),
}

num_samples = 1000
num_samples_per_axis = 200

# Evaluate across regions
for label, (xmin, xmax) in sample_ranges.items():
    ymin, ymax = xmin, xmax  
    circle_radius = min(abs(xmin), abs(xmax))

    valid_samples = []

    while len(valid_samples) < num_samples:
        lhs_samples = lhs(2, samples=num_samples)
        x_samples = lhs_samples[:, 0] * (xmax - xmin) + xmin
        y_samples = lhs_samples[:, 1] * (ymax - ymin) + ymin

        distances = np.sqrt(x_samples**2 + y_samples**2)
        inside_circle = distances <= circle_radius
        valid_samples.extend(zip(x_samples[inside_circle], y_samples[inside_circle]))
        valid_samples = valid_samples[:num_samples]

    params = np.array(valid_samples)
    x_samples, y_samples = params[:, 0], params[:, 1]

    # Compute true function values
    chi2_true = np.array([
        chi_squared_double_gaussian(x, y, bestfit_point1, bestfit_point2, sigma)
        for (x, y) in zip(x_samples, y_samples)
    ])

    # Prepare model inputs
    params_scaled = param_scaler.transform(params)
    chi2_pred_scaled = model.predict(params_scaled, verbose=0)
    chi2_pred = target_scaler.inverse_transform(chi2_pred_scaled).flatten()

    # Compute relative residuals
    relative_residuals = (chi2_pred - chi2_true) / chi2_true

    # Scatter plot of residuals
    plt.figure(figsize=(6, 5))
    plt.scatter(chi2_true, relative_residuals, s=5, alpha=0.5)
    plt.axhline(y=0, color="r", linestyle="--", linewidth=1)
    plt.xlabel(r"$\chi^2_{{true}}$")
    plt.ylabel(r"$(\chi^2_{{pred}} - \chi^2_{{true}}) / \chi^2_{{true}}$")
    plt.xlim(-5, 480)
    plt.ylim(-1, 1)
    plt.title(label)
    plt.grid(True)
    plt.tight_layout()

    # Save residual plot
    plot_filename = os.path.join(plot_dir, f"DG_hps_chi2_residuals_{label.replace(' ', '_').replace('[', '').replace(']', '')}.png")
    plt.savefig(plot_filename, dpi=300)
    plt.show()
    print(f"Residual plot saved: {plot_filename}")

    # --- Contour Plots ---
    x_linspace = np.linspace(xmin, xmax, num_samples_per_axis)
    y_linspace = np.linspace(ymin, ymax, num_samples_per_axis)
    X, Y = np.meshgrid(x_linspace, y_linspace)
    grid_points = np.column_stack((X.ravel(), Y.ravel()))

    # Compute true function values for contour plot
    chi2_true_grid = np.array([
        chi_squared_double_gaussian(x, y, bestfit_point1, bestfit_point2, sigma)
        for (x, y) in grid_points
    ]).reshape(X.shape)

    # Predict using model for contour plot
    params_scaled_grid = param_scaler.transform(grid_points)
    chi2_pred_scaled_grid = model.predict(params_scaled_grid, verbose=0)
    chi2_pred_grid = target_scaler.inverse_transform(chi2_pred_scaled_grid).reshape(X.shape)

    # Compute residuals for contour plot
    residuals_grid = chi2_pred_grid - chi2_true_grid

    # Contour plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # True function contour
    c1 = axes[0].contourf(X, Y, chi2_true_grid, levels=50, cmap="viridis")
    fig.colorbar(c1, ax=axes[0])
    axes[0].set_title(f"True $\chi^2$ ({label})")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    # Predicted function contour
    c2 = axes[1].contourf(X, Y, chi2_pred_grid, levels=50, cmap="viridis")
    fig.colorbar(c2, ax=axes[1])
    axes[1].set_title(f"Predicted $\chi^2$ ({label})")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    # Residuals contour
    c3 = axes[2].contourf(X, Y, residuals_grid, levels=50, cmap="coolwarm")
    fig.colorbar(c3, ax=axes[2])
    axes[2].set_title(f"Residuals: $\chi^2_{{pred}} - \chi^2_{{true}}$ ({label})")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")

    # Adjust layout and save
    plt.tight_layout()
    contour_filename = os.path.join(plot_dir, f"DG_hps_chi2_contours_{label.replace(' ', '_').replace('[', '').replace(']', '')}.png")
    plt.savefig(contour_filename, dpi=300)
    plt.show()
    print(f"Contour plot saved: {contour_filename}")
