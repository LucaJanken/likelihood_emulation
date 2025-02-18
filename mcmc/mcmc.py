import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from tensorflow.keras.layers import Layer
from tqdm import trange
from matplotlib.ticker import MaxNLocator

# (Re)define the custom layer exactly as in your training/evaluation code.
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
# MCMC sampler for 2D input using a TFLite model for faster evaluation.
# -----------------------------------------------------------------------------
class MCMC2D:
    def __init__(self, initial_point, T, model_file, param_scaler_file, target_scaler_file, sigma=None):
        """
        initial_point: list or array of length 2 (original parameter values)
                     (For this application, initial_point corresponds to [100*omega_b, omega_cdm])
        T: effective temperature for the sampler
        model_file: path to the .h5 model file
        param_scaler_file & target_scaler_file: paths to the corresponding pickle files
        sigma: (optional) proposal standard deviation (2-element list/array)
        """
        # Store the best-fit values as provided.
        self.initial_point = np.array(initial_point, dtype=np.float32)
        # Start the chain at the best-fit.
        self.current_point = self.initial_point.copy()
        self.positions = []  # to store accepted positions
        self.T = T
        self.model_file = model_file

        # Convert and load the TFLite model for faster evaluation.
        self.interpreter = self.convert_and_load_tflite_model(model_file)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Load the scalers (which were saved during training)
        with open(param_scaler_file, "rb") as f:
            self.param_scaler = pickle.load(f)
        with open(target_scaler_file, "rb") as f:
            self.target_scaler = pickle.load(f)

        # Proposal sigma: if not provided, use a default small step.
        if sigma is None:
            self.sigma = np.array([0.01, 0.01], dtype=np.float32)
        else:
            self.sigma = np.array(sigma, dtype=np.float32)

        # Calculate the initial "energy" (chi² or -log L). Lower is better.
        self.L = self.likelihood(self.current_point)
        self.accepted = 0
        self.total_steps = 0

    def convert_and_load_tflite_model(self, model_file):
        """
        Load the Keras model, convert it to TFLite, and return a TFLite Interpreter.
        This function works on a copy of your model, leaving the original untouched.
        """
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
        Given a proposal in the original parameter space (2D),
        scale it using the training parameter scaler, predict with the TFLite model,
        and then invert the target scaling to get a chi² value.
        """
        point_scaled = self.param_scaler.transform(point.reshape(1, -1)).astype(np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], point_scaled)
        self.interpreter.invoke()
        y_pred_scaled = self.interpreter.get_tensor(self.output_details[0]['index'])
        y_pred = self.target_scaler.inverse_transform(y_pred_scaled)
        return y_pred[0, 0]

    def step(self):
        """Perform one MCMC step using a Gaussian proposal.
           Returns True if the proposal is accepted; False otherwise.
        """
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
        Run adaptive burn-in followed by fixed-step sampling.
        During burn-in the proposal step size (sigma) is adjusted every 'adaptation_interval'
        steps to approach the target acceptance rate.
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
                # Update sigma multiplicatively.
                # The update factor is based on the deviation from the target acceptance.
                factor = np.exp((block_acceptance - target_acceptance) * 0.1)
                self.sigma *= factor
                print(f" Block {block+1}/{n_blocks}: Block acceptance = {block_acceptance:.2%}, "
                      f"updated sigma = {self.sigma}")
            # Process any remaining burn-in steps.
            remainder = burn_in % adaptation_interval
            for _ in range(remainder):
                self.step()
            # Clear burn-in positions to start sampling fresh.
            self.positions = []
            print(f"Burn-in complete. Final sigma: {self.sigma}. Starting sampling for {n_steps} steps.")

        for _ in trange(n_steps, desc="Sampling"):
            self.step()

    def plot_triangle(self):
        """
        Create a simple 'corner' plot for 2D:
          - Diagonals: Histograms of each parameter with a dashed red line marking the best-fit.
          - Off-diagonals: A scatter plot of the joint distribution with the best-fit shown as a red dot.
        The plot is saved as a PNG file in the current folder.
        """
        positions = np.array(self.positions)
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))

        # LaTeX labels for parameters
        labels = [r"$100\,\omega_b$", r"$\omega_{cdm}$"]
        bestfit = self.initial_point  # best-fit values

        # Histogram for parameter 1 (100*omega_b)
        axes[0, 0].hist(positions[:, 0], bins=50, color="skyblue")
        axes[0, 0].axvline(bestfit[0], color="red", linestyle="--", linewidth=2)
        axes[0, 0].set_xlabel(labels[0])
        axes[0, 0].xaxis.set_major_locator(MaxNLocator(nbins=5))

        # Histogram for parameter 2 (omega_cdm)
        axes[1, 1].hist(positions[:, 1], bins=50, color="skyblue")
        axes[1, 1].axvline(bestfit[1], color="red", linestyle="--", linewidth=2)
        axes[1, 1].set_xlabel(labels[1])
        axes[1, 1].xaxis.set_major_locator(MaxNLocator(nbins=5))

        # Scatter plot (joint distribution)
        axes[1, 0].scatter(positions[:, 0], positions[:, 1], s=2, alpha=0.5)
        axes[1, 0].scatter(bestfit[0], bestfit[1], color="red", s=50, marker="o", label="Best-fit")
        axes[1, 0].set_xlabel(labels[0])
        axes[1, 0].set_ylabel(labels[1])
        axes[1, 0].yaxis.set_major_locator(MaxNLocator(nbins=5))
        axes[1, 0].legend()

        # Upper right subplot unused.
        axes[0, 1].axis("off")

        plt.suptitle(f"Triangle Plot for Model: {os.path.basename(self.model_file)}", fontsize=14)
        plt.tight_layout()

        # Save the plot as a PNG file in the current directory.
        plot_filename = os.path.splitext(os.path.basename(self.model_file))[0] + "_triangle.png"
        plt.savefig(plot_filename)
        print(f"Triangle plot saved as {plot_filename}")
        plt.show()

    def acceptance_rate(self):
        """Return the fraction of accepted proposals."""
        return self.accepted / self.total_steps if self.total_steps > 0 else 0

# -----------------------------------------------------------------------------
# Example usage for one model (repeat or loop over models as desired)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Best-fit values for the parameters:
    #   100*omega_b = 2.160747
    #   omega_cdm  = 0.1159404
    initial_point = [2.160747, 0.1159404]

    # Temperature parameter for the sampler.
    T = 3.0

    # Define paths for one model (adjust paths as needed).
    model_file = os.path.join("chain_models", "model_1.h5")
    param_scaler_file = os.path.join("chain_models", "model_1_param_scaler.pkl")
    target_scaler_file = os.path.join("chain_models", "model_1_target_scaler.pkl")

    # (Optional) specify proposal sigma for each parameter.
    sigma = [7.031435e-03, 2.450273e-04]

    # Create an MCMC sampler instance.
    sampler = MCMC2D(initial_point, T, model_file, param_scaler_file, target_scaler_file, sigma=sigma)

    # Run the chain with adaptive burn-in (e.g., burn-in of 1000 steps, then 24000 sampling steps).
    sampler.chain(n_steps=24000, burn_in=1000, adaptation_interval=10, target_acceptance=0.23)

    # Show and save the triangle plot.
    sampler.plot_triangle()

    # Report the overall acceptance rate.
    print(f"Acceptance Rate: {sampler.acceptance_rate():.2%}")
