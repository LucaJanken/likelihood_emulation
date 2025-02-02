import sys
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import MinMaxScaler

##############################################################################
# 1) Adjust Python path and define/load the previously saved scalers
##############################################################################
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Load scalers (must match what you pickled in train.py)
with open(os.path.join("data", "param_scaler.pkl"), "rb") as f:
    param_scaler: MinMaxScaler = pickle.load(f)

with open(os.path.join("data", "target_scaler.pkl"), "rb") as f:
    target_scaler: MinMaxScaler = pickle.load(f)

# We also need the same TF constants your Lambda layer used for scaling
target_min = tf.constant(target_scaler.data_min_[0], dtype=tf.float32)
target_max = tf.constant(target_scaler.data_max_[0], dtype=tf.float32)

##############################################################################
# 2) Define the same custom functions used in train.py's Lambda layer
##############################################################################
def inverse_transform_tf(scaled_tensor, scaler):
    """
    Same function as in train.py. Applies the inverse of MinMaxScaler using TF ops.
    """
    min_val = tf.constant(scaler.data_min_, dtype=tf.float32)
    max_val = tf.constant(scaler.data_max_, dtype=tf.float32)
    return scaled_tensor * (max_val - min_val) + min_val

def compute_chi2(args):
    """
    Same logic as in train.py. 
    1) Unscale x,y 
    2) Parse [p0, x0, y0, cxx, cxy, cyy]
    3) Compute unscaled chi^2
    4) Re-scale chi^2 to [0,1] using (chi2 - target_min)/(target_max - target_min)
    """
    scaled_xy, pvec = args
    
    # (A) unscale (x,y)
    xy_unscaled = inverse_transform_tf(scaled_xy, param_scaler)
    x = xy_unscaled[:, 0:1]
    y = xy_unscaled[:, 1:2]
    
    # (B) parse predicted parameters
    p0  = pvec[:, 0:1]
    x0  = pvec[:, 1:2]
    y0  = pvec[:, 2:3]
    cxx = pvec[:, 3:4]
    cxy = pvec[:, 4:5]
    cyy = pvec[:, 5:6]

    # (C) compute unscaled chi^2
    dx = x - x0
    dy = y - y0
    mahalanobis = cxx*dx*dx + 2.0*cxy*dx*dy + cyy*dy*dy
    chi2_unscaled = p0 + mahalanobis

    # (D) scale chi^2 back into [0,1]
    chi2_scaled = (chi2_unscaled - target_min) / (target_max - target_min)
    return chi2_scaled

##############################################################################
# 3) Load the trained Keras model with custom_objects
##############################################################################
model_path = os.path.join("models", "trained_model.h5")
model = tf.keras.models.load_model(
    model_path,
    # Provide the functions used by the Lambda layer:
    custom_objects={
        "inverse_transform_tf": inverse_transform_tf,
        "compute_chi2": compute_chi2
    },
    compile=False
)

print("Model loaded successfully!")

##############################################################################
# 4) Define the neg-log Gaussian function for "true" reference
##############################################################################
def neg_log_gaussian_2D(x, y, bestfit_value, bestfit_point, C_inv_xx, C_inv_yy, C_inv_xy):
    """
    Computes the neg-log Gaussian likelihood function at (x, y), i.e. chi^2(x,y).
    """
    x_0, y_0 = bestfit_point
    dx = x - x_0
    dy = y - y_0
    mahalanobis_dist = C_inv_xx * dx**2 + 2 * C_inv_xy * dx * dy + C_inv_yy * dy**2
    return bestfit_value + 0.5 * mahalanobis_dist

##############################################################################
# 5) Define test regions in which we'll evaluate the model
##############################################################################
grid_size = 100
test_regions = {
    "In-Domain":         (-2,   2),
    "Extrapolation_(2.5)":(-2.5, 2.5),
    "Extrapolation_(3)":  (-3,   3),
    "Extrapolation_(3.5)":(-3.5, 3.5),
    "Extrapolation_(4)":  (-4,   4)
}

# "True" Gaussian parameters (matching those in train.py)
bestfit_value = 0.0
bestfit_point = np.array([0.0, 0.0])
sigma = np.sqrt(0.1)
C_inv_xx = 1.0 / sigma**2
C_inv_yy = 1.0 / sigma**2
C_inv_xy = 0.0

# Make a "plots" folder if not present
os.makedirs("plots", exist_ok=True)

##############################################################################
# 6) Evaluate the model over each test region
##############################################################################
for region_name, (min_val, max_val) in test_regions.items():
    print(f"\nEvaluating in {region_name} region...")

    # Create a grid of (x, y) points
    x_vals = np.linspace(min_val, max_val, grid_size)
    y_vals = np.linspace(min_val, max_val, grid_size)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Flatten the grid for batch prediction
    grid_points = np.column_stack((X.ravel(), Y.ravel()))  # shape (grid_size^2, 2)

    # Scale the grid points (x,y) as the model expects
    grid_points_scaled = param_scaler.transform(grid_points)

    # Model outputs *scaled chi^2*
    predicted_outputs_scaled = model.predict(grid_points_scaled, verbose=0)

    # Inverse-transform predictions to get original chi^2 scale
    predicted_outputs_original = target_scaler.inverse_transform(predicted_outputs_scaled)
    # We'll take the first (and only) column as the predicted chi^2
    predicted_chi2 = predicted_outputs_original[:, 0]

    # Compute the "true" chi^2 from the neg-log Gaussian
    true_chi2 = neg_log_gaussian_2D(
        grid_points[:, 0],   # x
        grid_points[:, 1],   # y
        bestfit_value,
        bestfit_point,
        C_inv_xx,
        C_inv_yy,
        C_inv_xy
    )

    # Reshape for 2D contour plotting
    Z_true = true_chi2.reshape(grid_size, grid_size)
    Z_pred = predicted_chi2.reshape(grid_size, grid_size)
    Z_diff = Z_pred - Z_true

    # -------------------------------------------------------------------------
    # 6a) 2D Contour plots of True chi^2, Predicted chi^2, and Difference
    # -------------------------------------------------------------------------
    plt.figure(figsize=(18, 6))

    # Subplot 1: True chi^2
    plt.subplot(1, 3, 1)
    plt.contourf(X, Y, Z_true, levels=100, cmap="viridis")
    plt.colorbar(label=r"$\chi^2_{\mathrm{true}}$")
    plt.title(f"True $\chi^2$ Function ({region_name})")
    plt.xlabel("X")
    plt.ylabel("Y")

    # Subplot 2: Predicted chi^2
    plt.subplot(1, 3, 2)
    plt.contourf(X, Y, Z_pred, levels=100, cmap="viridis")
    plt.colorbar(label=r"$\chi^2_{\mathrm{pred}}$")
    plt.title(f"Neural Network Emulation ({region_name})")
    plt.xlabel("X")
    plt.ylabel("Y")

    # Subplot 3: Difference = Pred - True
    plt.subplot(1, 3, 3)
    # Adjust levels depending on your typical error range
    plt.contourf(X, Y, Z_diff, levels=np.linspace(-0.5, 0.5, 31), cmap="coolwarm")
    plt.colorbar(label=r"$\chi^2_{\mathrm{pred}} - \chi^2_{\mathrm{true}}$")
    plt.title(f"Difference in $\chi^2$ ({region_name})")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.tight_layout()
    plt.savefig(os.path.join("plots", f"function_comparison_{region_name}.png"))
    plt.show()

    # -------------------------------------------------------------------------
    # 6b) Plot mean error vs. True chi^2
    # -------------------------------------------------------------------------
    bins = np.linspace(0, np.max(true_chi2), 100)
    chi2_differences = []
    true_chi2_flat = true_chi2.ravel()
    diff_flat = Z_diff.ravel()

    for b in bins:
        mask = (true_chi2_flat <= b)
        if np.any(mask):
            chi2_differences.append(np.mean(diff_flat[mask]))
        else:
            chi2_differences.append(np.nan)

    plt.figure(figsize=(8, 6))
    plt.plot(bins, chi2_differences, marker="o",
             label=r"$\langle \chi^2_{\mathrm{pred}} - \chi^2_{\mathrm{true}} \rangle$")
    plt.xlabel(r"$\chi^2_{\mathrm{true}}$")
    plt.ylabel(r"$\chi^2_{\mathrm{pred}} - \chi^2_{\mathrm{true}}$")
    plt.ylim(-2, 2)   # Adjust if your range differs
    plt.xlim(0, 100)    # Adjust if your max chi^2 is bigger/smaller
    plt.axhline(0, color='r', linestyle="--", label="Zero Error Line")
    plt.title(f"Chi² Difference vs. True Chi² ({region_name})")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join("plots", f"chi2_difference_vs_true_{region_name}.png"))
    plt.show()

print("\nEvaluation complete! All plots saved in the 'plots' directory.")
