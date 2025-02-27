import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import tensorflow as tf
import pickle
from tensorflow.keras.layers import Layer
from tqdm import trange
from scipy.stats import gaussian_kde

# -----------------------------------------------------------------------------
# Custom Keras Layer (must match training/evaluation)
# -----------------------------------------------------------------------------
class ScaledPredictionLayer(Layer):
    def __init__(self, scaler_type, param_scaler_params, target_scaler_params, **kwargs):
        super(ScaledPredictionLayer, self).__init__(**kwargs)
        self.scaler_type = scaler_type
        self.param_scaler_params = param_scaler_params
        self.target_scaler_params = target_scaler_params

    def call(self, inputs):
        scaled_params, pred_params = inputs

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

# -----------------------------------------------------------------------------
# MCMC sampler for 2D input using a TFLite model.
# -----------------------------------------------------------------------------
class MCMC2D:
    def __init__(self, initial_point, T, model_file, param_scaler_file, target_scaler_file, sigma=None):
        """
        initial_point: list/array of length 2 (original parameter values)
                     (e.g. [100*omega_b, omega_cdm])
        T: effective temperature for the sampler
        model_file: path to the .h5 model file
        param_scaler_file & target_scaler_file: corresponding pickle files
        sigma: (optional) proposal standard deviation (list/array of length 2)
        """
        self.initial_point = np.array(initial_point, dtype=np.float32)
        self.current_point = self.initial_point.copy()
        self.positions = []  # accepted positions
        self.T = T
        self.model_file = model_file

        # Convert & load the TFLite model.
        self.interpreter = self.convert_and_load_tflite_model(model_file)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Load the scalers.
        with open(param_scaler_file, "rb") as f:
            self.param_scaler = pickle.load(f)
        with open(target_scaler_file, "rb") as f:
            self.target_scaler = pickle.load(f)

        # Proposal sigma.
        if sigma is None:
            self.sigma = np.array([0.01, 0.01], dtype=np.float32)
        else:
            self.sigma = np.array(sigma, dtype=np.float32)

        # Compute the initial likelihood.
        self.L = self.likelihood(self.current_point)
        self.accepted = 0
        self.total_steps = 0

    def convert_and_load_tflite_model(self, model_file):
        model = tf.keras.models.load_model(
            model_file,
            custom_objects={"ScaledPredictionLayer": ScaledPredictionLayer},
            compile=False
        )
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        return interpreter

    def likelihood(self, point):
        """
        Scale input, predict with TFLite model, and then invert target scaling
        to get a (chi² or -log L) value.
        """
        point_scaled = self.param_scaler.transform(point.reshape(1, -1)).astype(np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], point_scaled)
        self.interpreter.invoke()
        y_pred_scaled = self.interpreter.get_tensor(self.output_details[0]['index'])
        y_pred = self.target_scaler.inverse_transform(y_pred_scaled)
        return y_pred[0, 0]

    def step(self):
        """Perform one MCMC step with a Gaussian proposal."""
        proposal = np.random.normal(self.current_point, self.sigma)
        L_new = self.likelihood(proposal)
        delta = L_new - self.L

        if delta < 0 or np.exp(-delta / self.T) > np.random.rand():
            self.current_point = proposal
            self.L = L_new
            accepted = True
        else:
            accepted = False

        self.positions.append(self.current_point.copy())
        self.total_steps += 1
        if accepted:
            self.accepted += 1
        return accepted

    def chain(self, n_steps, burn_in=1000, adaptation_interval=50, target_acceptance=0.23):
        """
        Run adaptive burn-in (to tune sigma) followed by fixed-step sampling.
        """
        if burn_in > 0:
            n_blocks = burn_in // adaptation_interval
            print("Starting adaptive burn-in:")
            for block in range(n_blocks):
                accepted_in_block = 0
                for _ in range(adaptation_interval):
                    if self.step():
                        accepted_in_block += 1
                block_acceptance = accepted_in_block / adaptation_interval
                factor = np.exp((block_acceptance - target_acceptance) * 0.1)
                self.sigma *= factor
                print(f" Block {block+1}/{n_blocks}: Acceptance = {block_acceptance:.2%}, sigma = {self.sigma}")
            # Process remaining burn-in steps.
            for _ in range(burn_in % adaptation_interval):
                self.step()
            # Clear burn-in positions.
            self.positions = []
            print(f"Burn-in complete. Final sigma: {self.sigma}. Now sampling {n_steps} steps.")

        for _ in trange(n_steps, desc="Sampling"):
            self.step()

    def acceptance_rate(self):
        return self.accepted / self.total_steps if self.total_steps > 0 else 0

# -----------------------------------------------------------------------------
# Helper functions for KDE-based plotting
# -----------------------------------------------------------------------------
def compute_contour_levels(kde, x_grid, y_grid, probs=[0.68, 0.95]):
    """
    Compute density thresholds for given cumulative probability levels.
    Returns meshgrid (X, Y), evaluated density Z and list of levels.
    """
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(positions).reshape(X.shape)
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    Z_flat = Z.ravel()
    # Sort densities in descending order.
    idx = np.argsort(Z_flat)[::-1]
    Z_sorted = Z_flat[idx]
    cumsum = np.cumsum(Z_sorted) * dx * dy
    cumsum = cumsum / cumsum[-1]
    levels = []
    for p in probs:
        try:
            level = Z_sorted[np.searchsorted(cumsum, p)]
        except IndexError:
            level = Z_sorted[-1]
        levels.append(level)
    return X, Y, Z, levels

# -----------------------------------------------------------------------------
# Combined Triangle Plot (overlaying MCMC and training-data densities)
# -----------------------------------------------------------------------------
def combined_triangle_plot(mcmc_data, training_data, training_weights, bestfit, model_filename):
    """
    Create a 2x2 triangle plot where:
      - Diagonals show 1D KDE estimates (density curves) for each parameter.
      - Off-diagonals show 2D KDE contours (credible intervals).
    Two datasets are overlaid:
      - Training Data in green (adjusted from T=5 to T=1, with weights).
      - Network in blue.
    A single bestfit point (explicitly provided) is marked in red.
    """
    labels = [r"$100\,\omega_b$", r"$\omega_{cdm}$"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # --- 1D KDEs on diagonals for parameter 1 ---
    all_param1 = np.concatenate([training_data[:, 0], mcmc_data[:, 0]])
    x_min1, x_max1 = all_param1.min(), all_param1.max()
    x_grid1 = np.linspace(x_min1, x_max1, 200)
    # Training data KDE with multiplicity weights and adjustment for T=5:
    kde_train_p1 = gaussian_kde(training_data[:, 0], weights=training_weights)
    p_train1 = kde_train_p1(x_grid1)
    p_train1_adjusted = p_train1**5  # raise to the 5th power
    p_train1_adjusted /= np.trapz(p_train1_adjusted, x_grid1)  # renormalize

    kde_network_p1 = gaussian_kde(mcmc_data[:, 0])
    axes[0, 0].plot(x_grid1, p_train1_adjusted, color="green", lw=2, label="Training Data (T fixed)")
    axes[0, 0].plot(x_grid1, kde_network_p1(x_grid1), color="blue", lw=2, label="Network")
    axes[0, 0].axvline(bestfit[0], color="green", linestyle="--", lw=2, label="Bestfit")
    axes[0, 0].set_xlabel(labels[0])
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].xaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[0, 0].legend()

    # --- 1D KDE for parameter 2 ---
    all_param2 = np.concatenate([training_data[:, 1], mcmc_data[:, 1]])
    x_min2, x_max2 = all_param2.min(), all_param2.max()
    x_grid2 = np.linspace(x_min2, x_max2, 200)
    kde_train_p2 = gaussian_kde(training_data[:, 1], weights=training_weights)
    p_train2 = kde_train_p2(x_grid2)
    p_train2_adjusted = p_train2**5
    p_train2_adjusted /= np.trapz(p_train2_adjusted, x_grid2)
    
    kde_network_p2 = gaussian_kde(mcmc_data[:, 1])
    axes[1, 1].plot(x_grid2, p_train2_adjusted, color="green", lw=2, label="Training Data (T fixed)")
    axes[1, 1].plot(x_grid2, kde_network_p2(x_grid2), color="blue", lw=2, label="Network")
    axes[1, 1].axvline(bestfit[1], color="green", linestyle="--", lw=2, label="Bestfit")
    axes[1, 1].set_xlabel(labels[1])
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].xaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[1, 1].legend()

    # --- 2D joint KDE contours on the off-diagonal ---
    x_range = [min(mcmc_data[:, 0].min(), training_data[:, 0].min()),
               max(mcmc_data[:, 0].max(), training_data[:, 0].max())]
    y_range = [min(mcmc_data[:, 1].min(), training_data[:, 1].min()),
               max(mcmc_data[:, 1].max(), training_data[:, 1].max())]
    x_grid = np.linspace(x_range[0], x_range[1], 200)
    y_grid = np.linspace(y_range[0], y_range[1], 200)

    # 2D KDE for training data with weights and for network.
    kde_train_2d = gaussian_kde(training_data.T, weights=training_weights)
    kde_network_2d = gaussian_kde(mcmc_data.T)
    
    # Compute contours for training data using the original KDE.
    X, Y, Z_train, _ = compute_contour_levels(kde_train_2d, x_grid, y_grid, probs=[0.68, 0.95])
    # Adjust training density for T=5: raise to 5th power and renormalize.
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    Z_train_adjusted = Z_train**5
    Z_train_adjusted /= np.sum(Z_train_adjusted) * dx * dy
    # Recompute cumulative thresholds for the adjusted density.
    Z_train_flat = Z_train_adjusted.ravel()
    idx = np.argsort(Z_train_flat)[::-1]
    Z_sorted = Z_train_flat[idx]
    cumsum = np.cumsum(Z_sorted) * dx * dy
    cumsum /= cumsum[-1]
    levels_train_adjusted = []
    for p in [0.68, 0.95]:
        level = Z_sorted[np.searchsorted(cumsum, p)]
        levels_train_adjusted.append(level)

    # Compute network contours (no adjustment needed).
    _, _, Z_network, levels_network = compute_contour_levels(kde_network_2d, x_grid, y_grid, probs=[0.68, 0.95])

    ax_joint = axes[1, 0]
    # Plot adjusted training-data contours.
    ct_train = ax_joint.contour(X, Y, Z_train_adjusted, levels=sorted(levels_train_adjusted), colors="green", linestyles=["-", "--"])
    # Plot network contours.
    ct_network  = ax_joint.contour(X, Y, Z_network, levels=sorted(levels_network), colors="blue", linestyles=["-", "--"])
    # Mark the bestfit point.
    ax_joint.plot(bestfit[0], bestfit[1], marker="o", color="green", markersize=8, label="Bestfit")
    ax_joint.set_xlabel(labels[0])
    ax_joint.set_ylabel(labels[1])
    ax_joint.legend()

    axes[0, 1].axis("off")

    plt.suptitle(f"Combined Triangle Plot\nModel: {os.path.basename(model_filename)}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    combined_plot_filename = os.path.splitext(os.path.basename(model_filename))[0] + "_combined_triangle.png"
    plt.savefig(combined_plot_filename)
    print(f"Combined triangle plot saved as {combined_plot_filename}")
    plt.show()

# -----------------------------------------------------------------------------
# Main: Run MCMC, load training data, and produce combined plot.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # ---------------------------
    # MCMC Sampler Setup & Run
    # ---------------------------
    initial_point = [2.160747, 0.1159404]
    T = 1.0
    model_file = os.path.join("chain_models", "model_1.h5")
    param_scaler_file = os.path.join("chain_models", "model_1_param_scaler.pkl")
    target_scaler_file = os.path.join("chain_models", "model_1_target_scaler.pkl")
    sigma = [7.031435e-03, 2.450273e-04]
    sampler = MCMC2D(initial_point, T, model_file, param_scaler_file, target_scaler_file, sigma=sigma)
    
    sampler.chain(n_steps=4*24000, burn_in=1000, adaptation_interval=10, target_acceptance=0.23)
    mcmc_data = np.array(sampler.positions)
    bestfit = sampler.initial_point

    # ---------------------------
    # Load Training Data
    # ---------------------------
    chains_dir = "chains/2_param"
    filenames = [
        "2024-09-17_123456789__1.txt",
        "2024-09-17_123456789__2.txt",
        "2024-09-17_123456789__3.txt",
        "2024-09-17_123456789__4.txt"
    ]
    n_header = 150
    n_data   = 6000  # number of data lines per file

    multiplicity = []
    neg_log_like = []
    param1 = []  # 100*omega_b
    param2 = []  # omega_cdm

    for fname in filenames:
        full_path = os.path.join(chains_dir, fname)
        with open(full_path, "r") as f:
            for _ in range(n_header):
                next(f)
            count = 0
            for line in f:
                if count >= n_data:
                    break
                cols = line.strip().split()
                if len(cols) < 4:
                    continue
                try:
                    mult = int(cols[0])
                    # Scale the negative log-likelihood by a factor of 5 (as in training).
                    neg_log = float(cols[1]) * 5  
                    p1_val = float(cols[2])
                    p2_val = float(cols[3])
                except ValueError:
                    continue
                multiplicity.append(mult)
                neg_log_like.append(neg_log)
                param1.append(p1_val)
                param2.append(p2_val)
                count += 1

    multiplicity = np.array(multiplicity)
    neg_log_like = np.array(neg_log_like)
    training_data = np.column_stack((param1, param2))
    best_index = np.argmin(neg_log_like)

    # ---------------------------
    # Create the Combined Triangle Plot
    # ---------------------------
    combined_triangle_plot(mcmc_data, training_data, multiplicity, bestfit, model_file)

    print(f"MCMC Acceptance Rate: {sampler.acceptance_rate():.2%}")
