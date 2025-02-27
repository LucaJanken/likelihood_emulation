import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from tqdm import trange
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib.ticker import MaxNLocator

# -----------------------------------------------------------------------------
# Helper function required by the Lambda layer (if used)
# -----------------------------------------------------------------------------
def inverse_transform_tf(scaled_tensor, scaler):
    if isinstance(scaler, StandardScaler):
        means = tf.constant(scaler.mean_, dtype=tf.float32)
        stds = tf.constant(scaler.scale_, dtype=tf.float32)
        return scaled_tensor * stds + means
    elif isinstance(scaler, MinMaxScaler):
        min_val = tf.constant(scaler.data_min_, dtype=tf.float32)
        max_val = tf.constant(scaler.data_max_, dtype=tf.float32)
        return scaled_tensor * (max_val - min_val) + min_val

# -----------------------------------------------------------------------------
# MCMC Sampler using a TFLite-converted model with adaptive burn-in
# -----------------------------------------------------------------------------
class MCMC2D:
    def __init__(self, initial_point, T, model_file, param_scaler_file, target_scaler_file, sigma=None):
        """
        initial_point: list/array with two values (e.g. [N_eff, m0])
        T: effective temperature for the sampler.
        model_file: path to the saved Keras model (e.g. sn_trained_model.h5)
        param_scaler_file: path to the parameter scaler pickle file (e.g. sn_param_scaler.pkl)
        target_scaler_file: path to the target scaler pickle file (e.g. sn_target_scaler.pkl)
        sigma: (optional) proposal standard deviations (list/array of length 2).
        """
        self.initial_point = np.array(initial_point, dtype=np.float32)
        self.current_point = self.initial_point.copy()
        self.positions = [self.current_point.copy()]  # to store accepted positions
        self.T = T

        # Load scalers.
        with open(param_scaler_file, "rb") as f:
            self.param_scaler = pickle.load(f)
        with open(target_scaler_file, "rb") as f:
            self.target_scaler = pickle.load(f)
        
        # Set globals for the Lambda layer (if it uses them).
        global param_scaler, target_scaler, target_mean, target_std, target_min, target_max
        param_scaler = self.param_scaler
        target_scaler = self.target_scaler
        if isinstance(self.target_scaler, StandardScaler):
            target_mean = tf.constant(self.target_scaler.mean_[0], dtype=tf.float32)
            target_std = tf.constant(self.target_scaler.scale_[0], dtype=tf.float32)
        elif isinstance(self.target_scaler, MinMaxScaler):
            target_min = tf.constant(self.target_scaler.data_min_[0], dtype=tf.float32)
            target_max = tf.constant(self.target_scaler.data_max_[0], dtype=tf.float32)

        # Convert and load the model as TFLite.
        self.interpreter = self.convert_and_load_tflite_model(model_file)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Set proposal sigma.
        if sigma is None:
            self.sigma = np.array([0.01, 0.01], dtype=np.float32)
        else:
            self.sigma = np.array(sigma, dtype=np.float32)

        # Compute initial log-likelihood value.
        self.logL = self.log_likelihood(self.current_point)
        self.accepted = 0
        self.total_steps = 0

    def convert_and_load_tflite_model(self, model_file):
        """Load the Keras model (with custom objects) and convert it to TFLite."""
        model = tf.keras.models.load_model(
            model_file,
            compile=False,
            custom_objects={"inverse_transform_tf": inverse_transform_tf}
        )
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        return interpreter

    def log_likelihood(self, point):
        """
        Compute the log-likelihood at a given point using the traditional formulation:
          log L = -chi²/2.
        Steps:
          1. Scale the input using the parameter scaler.
          2. Run inference via the TFLite interpreter.
          3. Invert target scaling to recover chi².
          4. Return log L = -chi²/2.
        """
        point_scaled = self.param_scaler.transform(point.reshape(1, -1)).astype(np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], point_scaled)
        self.interpreter.invoke()
        y_pred_scaled = self.interpreter.get_tensor(self.output_details[0]['index'])
        chi2 = self.target_scaler.inverse_transform(y_pred_scaled)[0, 0]
        return - chi2 / 2.0

    def step(self):
        """Perform one MCMC step with a Gaussian proposal and enforce parameter boundaries [0,10]."""
        proposal = np.random.normal(self.current_point, self.sigma)
        # Enforce boundaries: both parameters must be between 0 and 10.
        if proposal[0] < 0 or proposal[0] > 10 or proposal[1] < 0 or proposal[1] > 10:
            accepted = False
        else:
            logL_new = self.log_likelihood(proposal)
            delta = logL_new - self.logL
            # A higher log-likelihood (i.e. lower chi²) is favorable.
            if delta > 0 or np.exp(delta / self.T) > np.random.rand():
                self.current_point = proposal
                self.logL = logL_new
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
        Run adaptive burn-in to tune sigma followed by fixed-step sampling.
        Prints block-level progress and uses a progress bar for the sampling phase.
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
            for _ in range(burn_in % adaptation_interval):
                self.step()
            self.positions = []  # clear burn-in positions
            print(f"Burn-in complete. Final sigma: {self.sigma}. Now sampling {n_steps} steps.")

        for _ in trange(n_steps, desc="Sampling"):
            self.step()

    def acceptance_rate(self):
        return self.accepted / self.total_steps if self.total_steps > 0 else 0

# -----------------------------------------------------------------------------
# Simple Triangle Plot: Histograms on diagonals, scatter plot off-diagonals
# -----------------------------------------------------------------------------
def triangle_plot(chain):
    """
    Create a triangle plot for 2D MCMC chain data:
      - Diagonals: 1D histograms for each parameter.
      - Off-diagonals: a scatter plot showing the sampled points.
    The plot is saved to the 'mcmc_plots' folder.
    """
    labels = [r"$N_{\mathrm{eff}}$", r"$m_0$"]
    chain = np.array(chain)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # Histogram for parameter 1.
    axes[0, 0].hist(chain[:, 0], bins=50, density=True, color='blue', alpha=0.7)
    axes[0, 0].set_xlabel(labels[0])
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].xaxis.set_major_locator(MaxNLocator(nbins=5))
    
    # Histogram for parameter 2.
    axes[1, 1].hist(chain[:, 1], bins=50, density=True, color='blue', alpha=0.7)
    axes[1, 1].set_xlabel(labels[1])
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].xaxis.set_major_locator(MaxNLocator(nbins=5))
    
    # Scatter plot for joint distribution.
    axes[1, 0].scatter(chain[:, 0], chain[:, 1], color='blue', s=5, alpha=0.5)
    axes[1, 0].set_xlabel(labels[0])
    axes[1, 0].set_ylabel(labels[1])
    axes[1, 0].set_xlim(-0.25, 10.25)
    axes[1, 0].set_ylim(-0.25, 10.25)
    
    # Turn off the top-right plot.
    axes[0, 1].axis('off')
    
    plt.suptitle("Triangle Plot of MCMC Chain")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    output_dir = "mcmc_plots"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "triangle_plot.png")
    plt.savefig(plot_path)
    print(f"Triangle plot saved as {plot_path}")
    plt.show()

# -----------------------------------------------------------------------------
# Main: Run the sampler and generate the triangle plot.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # MCMC settings.
    initial_point = [1.5, 1.5]          # Starting point (parameters between 0 and 10)
    T = 1                             # Effective temperature
    model_file = os.path.join("models", "sn_trained_model.h5")
    param_scaler_file = os.path.join("data", "sn_param_scaler.pkl")
    target_scaler_file = os.path.join("data", "sn_target_scaler.pkl")
    sigma = [0.05, 0.05]              # Initial proposal sigma

    # Initialize and run the sampler.
    sampler = MCMC2D(initial_point, T, model_file, param_scaler_file, target_scaler_file, sigma=sigma)
    n_steps = 100000
    sampler.chain(n_steps, burn_in=1000, adaptation_interval=25, target_acceptance=0.05)
    print(f"Acceptance rate: {sampler.acceptance_rate() * 100:.2f}%")
    
    # Generate the triangle plot.
    chain_data = np.array(sampler.positions)
    triangle_plot(chain_data)
