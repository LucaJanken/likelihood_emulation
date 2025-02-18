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

model_path = os.path.join("models", "dg_direct_trained_model.h5")
param_scaler_path = os.path.join(data_dir, "dg_direct_param_scaler.pkl")
target_scaler_path = os.path.join(data_dir, "dg_direct_target_scaler.pkl")

with open(param_scaler_path, "rb") as f:
    param_scaler = pickle.load(f)

with open(target_scaler_path, "rb") as f:
    target_scaler = pickle.load(f)

def chi2_underestimate_penalizing_loss(lambda_weight=100.0):
    def loss(y_true, y_pred):
        mse = tf.square(y_true - y_pred)
        penalty = lambda_weight * tf.square(tf.maximum(y_true - y_pred, 0))
        return tf.reduce_mean(mse + penalty)
    return loss

model = load_model(model_path, custom_objects={"loss": chi2_underestimate_penalizing_loss(lambda_weight=5.0)})

# Define the chi² function for the double Gaussian likelihood
def chi_squared_double_gaussian(x, y, bestfit_point1, bestfit_point2, sigma):
    x1, y1 = bestfit_point1
    x2, y2 = bestfit_point2
    term1 = np.exp(-0.5 * ((x - x1)**2 + (y - y1)**2) / sigma**2)
    term2 = np.exp(-0.5 * ((x - x2)**2 + (y - y2)**2) / sigma**2)
    normalization_factor = 1 / (4 * np.pi * sigma**2)
    likelihood = normalization_factor * (term1 + term2)
    likelihood = np.maximum(likelihood, 1e-300)
    chi2 = -2 * np.log(likelihood)
    return chi2

# Function parameters
bestfit_point1 = np.array([-1.0, -1.0])
bestfit_point2 = np.array([1.0, 1.0])
sigma = np.sqrt(0.1)

# Sampling regions to evaluate
sample_ranges = {
    "In-Domain [-5,5]": (-5, 5),
    "Out-Domain [-6,6]": (-6, 6),
    "Out-Domain [-7,7]": (-7, 7),
    "Out-Domain [-8,8]": (-8, 8)
}

num_samples = 1000
num_samples_per_axis = 200

# Evaluate model predictions and plot results
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
    x_vals, y_vals = params[:, 0], params[:, 1]

    # Compute true chi² values for these points
    chi2_true = np.array([
        chi_squared_double_gaussian(x, y, bestfit_point1, bestfit_point2, sigma)
        for x, y in zip(x_vals, y_vals)
    ])

    # Prepare inputs and predict with the model
    params_scaled = param_scaler.transform(params)
    chi2_pred_scaled = model.predict(params_scaled, verbose=0)
    chi2_pred = target_scaler.inverse_transform(chi2_pred_scaled).flatten()

    # Compute relative residuals
    relative_residuals = (chi2_pred - chi2_true) / chi2_true

    # Scatter plot of residuals
    plt.figure(figsize=(6, 5))
    plt.scatter(chi2_true, relative_residuals, s=5, alpha=0.5)
    plt.axhline(y=0, color="r", linestyle="--", linewidth=1)
    plt.xlabel(r"$\chi^2_{\rm true}$")
    plt.ylabel(r"$(\chi^2_{\rm pred} - \chi^2_{\rm true}) / \chi^2_{\rm true}$")
    plt.xlim(-5, 700)
    plt.ylim(-0.25, 0.25)
    plt.title(label)
    plt.grid(True)
    plt.tight_layout()

    plot_filename = os.path.join(plot_dir, f"DG_direct_chi2_residuals_{label.replace(' ', '_').replace('[', '').replace(']', '')}.png")
    plt.savefig(plot_filename, dpi=300)
    plt.show()
    print(f"Residual plot saved: {plot_filename}")

    # --- Contour Plots ---
    x_linspace = np.linspace(xmin, xmax, num_samples_per_axis)
    y_linspace = np.linspace(ymin, ymax, num_samples_per_axis)
    X, Y = np.meshgrid(x_linspace, y_linspace)
    grid_points = np.column_stack((X.ravel(), Y.ravel()))

    # Compute true chi² grid
    chi2_true_grid = np.array([
        chi_squared_double_gaussian(x, y, bestfit_point1, bestfit_point2, sigma)
        for x, y in grid_points
    ]).reshape(X.shape)

    # Predict chi² on the grid
    params_scaled_grid = param_scaler.transform(grid_points)
    chi2_pred_scaled_grid = model.predict(params_scaled_grid, verbose=0)
    chi2_pred_grid = target_scaler.inverse_transform(chi2_pred_scaled_grid).reshape(X.shape)

    residuals_grid = chi2_pred_grid - chi2_true_grid

    # Create contour plots for true, predicted, and residual values
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    c1 = axes[0].contourf(X, Y, chi2_true_grid, levels=50, cmap="viridis")
    fig.colorbar(c1, ax=axes[0])
    axes[0].set_title(f"True $\chi^2$ ({label})")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    c2 = axes[1].contourf(X, Y, chi2_pred_grid, levels=50, cmap="viridis")
    fig.colorbar(c2, ax=axes[1])
    axes[1].set_title(f"Predicted $\chi^2$ ({label})")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    c3 = axes[2].contourf(X, Y, residuals_grid, levels=50, cmap="coolwarm")
    fig.colorbar(c3, ax=axes[2])
    axes[2].set_title(f"Residuals: $\chi^2_{{pred}} - \chi^2_{{true}}$ ({label})")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")

    plt.tight_layout()
    contour_filename = os.path.join(plot_dir, f"DG_direct_chi2_contours_{label.replace(' ', '_').replace('[', '').replace(']', '')}.png")
    plt.savefig(contour_filename, dpi=300)
    plt.show()
    print(f"Contour plot saved: {contour_filename}")
