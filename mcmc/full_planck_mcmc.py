import os
import re
import numpy as np
import tensorflow as tf
import pickle
from ast import literal_eval
import matplotlib.pyplot as plt
from tqdm import trange
from matplotlib.ticker import MaxNLocator

# ----------------------------
# Custom Layers and Functions (must match training/evaluation)
# ----------------------------
class InverseScalingLayer(tf.keras.layers.Layer):
    def __init__(self, scale, mean, **kwargs):
        super(InverseScalingLayer, self).__init__(**kwargs)
        self.scale = tf.constant(scale, dtype=tf.float32)
        self.mean = tf.constant(mean, dtype=tf.float32)
    
    def call(self, inputs):
        return inputs * self.scale + self.mean

    def get_config(self):
        config = super(InverseScalingLayer, self).get_config()
        config.update({
            'scale': self.scale.numpy().tolist(),
            'mean': self.mean.numpy().tolist()
        })
        return config

class ScaledTanh(tf.keras.layers.Layer):
    def __init__(self, initial_alpha=1.0, **kwargs):
        super(ScaledTanh, self).__init__(**kwargs)
        self.initial_alpha = initial_alpha

    def build(self, input_shape):
        self.alpha = self.add_weight(
            name="alpha",
            shape=(1,),
            initializer=tf.keras.initializers.Constant(self.initial_alpha),
            trainable=True
        )
        super(ScaledTanh, self).build(input_shape)

    def call(self, inputs):
        return tf.math.tanh(self.alpha * inputs)

    def get_config(self):
        config = super(ScaledTanh, self).get_config()
        config.update({"initial_alpha": self.initial_alpha})
        return config

@tf.function
def reconstruct_sym_matrix(flat, dim):
    """
    Reconstruct a symmetric matrix from its flat upper-triangular part.
    flat: Tensor of shape (batch_size, dim*(dim+1)//2)
    returns: Tensor of shape (batch_size, dim, dim)
    """
    batch_size = tf.shape(flat)[0]
    # Create a (dim, dim) boolean mask for the upper triangle.
    upper_tri_mask = tf.linalg.band_part(tf.ones((dim, dim), dtype=tf.bool), 0, -1)
    idx = tf.cast(tf.where(upper_tri_mask), tf.int32)
    num_elements = tf.shape(idx)[0]
    batch_range = tf.range(batch_size, dtype=tf.int32)
    batch_idx = tf.reshape(tf.tile(batch_range[:, None], [1, num_elements]), (-1, 1))
    idx_tile = tf.tile(tf.expand_dims(idx, 0), [batch_size, 1, 1])
    idx_tile = tf.reshape(idx_tile, (-1, 2))
    full_idx = tf.concat([batch_idx, idx_tile], axis=1)
    M = tf.zeros((batch_size, dim, dim), dtype=flat.dtype)
    flat_reshaped = tf.reshape(flat, (-1,))
    M = tf.tensor_scatter_nd_update(M, full_idx, flat_reshaped)
    M_sym = M + tf.transpose(M, perm=[0, 2, 1]) - tf.linalg.diag(tf.linalg.diag_part(M))
    return M_sym

@tf.function
def predicted_neg_loglike(x, raw_output):
    """
    Compute the predicted -loglike value using the Gaussian approximation:
      pred_scaled = 0.5 * (offset + (x - x0)^T M (x - x0))
    x: input parameters (batch_size, dim)
    raw_output: raw network output (batch_size, output_dim)
    """
    dim = tf.shape(x)[-1]
    offset = raw_output[:, 0]               # (batch_size,)
    x0 = raw_output[:, 1:1+dim]              # (batch_size, dim)
    flat_inv_cov = raw_output[:, 1+dim:]     # (batch_size, dim*(dim+1)//2)
    M = reconstruct_sym_matrix(flat_inv_cov, dim)
    diff = x - x0
    quad = tf.einsum('bi,bij,bj->b', diff, M, diff)
    pred_scaled = 0.5 * (offset + quad)
    return pred_scaled

def parse_log_param(file_path, param_keys):
    """
    Reads the log.param file and returns a dictionary mapping parameter names
    to their specification list [mean, min, max, sigma, scale, role] for keys in param_keys.
    """
    params = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("data.parameters["):
                match = re.match(r"data\.parameters\['([^']+)'\]\s*=\s*(\[.*\])", line)
                if match:
                    key = match.group(1)
                    if key in param_keys:
                        try:
                            values = literal_eval(match.group(2))
                        except Exception as e:
                            print(f"Error parsing line: {line}")
                            continue
                        params[key] = values
    return params

# ----------------------------
# MCMC Sampler using TFLite-converted Model with Adaptive Covariance Proposal
# ----------------------------
class MCMCPlanck:
    def __init__(self, initial_point, T, model_file, param_scaler_file, target_scaler_file, sigma):
        """
        initial_point: array-like of length dim (original parameter values)
        T: effective temperature (typically set to 1)
        model_file: path to the saved .h5 model
        param_scaler_file & target_scaler_file: paths to the corresponding pickle files
        sigma: proposal standard deviation (array-like of length dim)
        """
        self.dim = len(initial_point)
        self.initial_point = np.array(initial_point, dtype=np.float32)
        self.current_point = self.initial_point.copy()
        self.positions = [self.current_point.copy()]
        self.T = T
        
        self.interpreter = self.convert_and_load_tflite_model(model_file)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        with open(param_scaler_file, "rb") as f:
            self.param_scaler = pickle.load(f)
        with open(target_scaler_file, "rb") as f:
            self.target_scaler = pickle.load(f)
        
        self.sigma = np.array(sigma, dtype=np.float32)
        d = self.dim
        init_cov = np.diag(self.sigma**2)
        self.optimal_scale = (2.38**2) / d
        self.adapt_scaling = 1.0
        self.epsilon = 1e-6 * np.eye(d)
        self.proposal_cov = self.optimal_scale * init_cov + self.epsilon
        
        self.L = self.likelihood(self.current_point)
        self.loglkl_history = [self.L]
        self.accepted = 0
        self.total_steps = 0

    def convert_and_load_tflite_model(self, model_file):
        model = tf.keras.models.load_model(
            model_file,
            custom_objects={'InverseScalingLayer': InverseScalingLayer, 'ScaledTanh': ScaledTanh},
            compile=False
        )
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        tflite_model = converter.convert()
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        return interpreter

    def likelihood(self, point):
        point_scaled = self.param_scaler.transform(point.reshape(1, -1)).astype(np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], point_scaled)
        self.interpreter.invoke()
        pred = self.interpreter.get_tensor(self.output_details[0]['index'])
        return np.squeeze(pred)

    def step(self):
        proposal = np.random.multivariate_normal(self.current_point, self.proposal_cov)
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
        self.loglkl_history.append(self.L)
        return accepted

    def chain(self, n_steps, burn_in=1000, adaptation_interval=50, target_acceptance=0.234):
        if burn_in > 0:
            n_blocks = burn_in // adaptation_interval
            print("Starting adaptive burn-in:")
            for block in range(n_blocks):
                accepted_in_block = 0
                for _ in range(adaptation_interval):
                    if self.step():
                        accepted_in_block += 1
                block_acceptance = accepted_in_block / adaptation_interval
                self.adapt_scaling *= np.exp((block_acceptance - target_acceptance) * 0.1)
                samples_array = np.array(self.positions)
                if samples_array.shape[0] > 1:
                    empirical_cov = np.cov(samples_array.T)
                    self.proposal_cov = self.adapt_scaling * self.optimal_scale * empirical_cov + self.epsilon
                print(f" Block {block+1}/{n_blocks}: Acceptance = {block_acceptance:.2%}, adapt_scaling = {self.adapt_scaling:.3f}")
            for _ in range(burn_in % adaptation_interval):
                self.step()
            self.positions = []  # clear burn-in samples
            print(f"Burn-in complete. Final proposal covariance:\n{self.proposal_cov}\nNow sampling {n_steps} steps.")
        
        for _ in trange(n_steps, desc="Sampling"):
            self.step()

    def acceptance_rate(self):
        return self.accepted / self.total_steps if self.total_steps > 0 else 0

# ----------------------------
# Corner Plot Function (for plotting the first few parameters)
# ----------------------------
def corner_plot(mcmc_data, training_data, training_weights, bestfit, labels, dims_to_plot=None, output_filename="planck_corner_plot.png"):
    """
    dims_to_plot: list or array of indices of parameters to plot.
                  If None, plot all dimensions.
    """
    if dims_to_plot is not None:
        mcmc_data = mcmc_data[:, dims_to_plot]
        training_data = training_data[:, dims_to_plot]
        bestfit = bestfit[dims_to_plot]
        labels = [labels[i] for i in dims_to_plot]
        
    ndim = training_data.shape[1]
    fig, axes = plt.subplots(ndim, ndim, figsize=(15, 15))
    bins_1d = 50
    bins_2d = 50

    for i in range(ndim):
        for j in range(ndim):
            ax = axes[i, j]
            if i == j:
                data_net = mcmc_data[:, i]
                data_train = training_data[:, i]
                x_min = min(np.min(data_net), np.min(data_train))
                x_max = max(np.max(data_net), np.max(data_train))
                bins = np.linspace(x_min, x_max, bins_1d+1)
                ax.hist(data_net, bins=bins, color="blue", alpha=0.5, density=True, label="Network")
                ax.hist(data_train, bins=bins, weights=training_weights, color="green", alpha=0.5, density=True, label="Training")
                ax.axvline(bestfit[i], color="red", lw=2)
                ax.xaxis.set_major_locator(MaxNLocator(5))
                if i == 0:
                    ax.legend(fontsize=8)
            elif i > j:
                x_net = mcmc_data[:, j]
                y_net = mcmc_data[:, i]
                x_train = training_data[:, j]
                y_train = training_data[:, i]
                x_min = min(np.min(x_net), np.min(x_train))
                x_max = max(np.max(x_net), np.max(x_train))
                y_min = min(np.min(y_net), np.min(y_train))
                y_max = max(np.max(y_net), np.max(y_train))
                x_bins = np.linspace(x_min, x_max, bins_2d+1)
                y_bins = np.linspace(y_min, y_max, bins_2d+1)
                hist_net, x_edges, y_edges = np.histogram2d(x_net, y_net, bins=[x_bins, y_bins])
                hist_train, _, _ = np.histogram2d(x_train, y_train, bins=[x_bins, y_bins], weights=training_weights)
                def compute_levels(hist, fractions=[0.68, 0.95]):
                    hist_flat = hist.flatten()
                    idx_sorted = np.argsort(hist_flat)[::-1]
                    hist_sorted = hist_flat[idx_sorted]
                    cumsum = np.cumsum(hist_sorted)
                    cumsum /= cumsum[-1]
                    levels = []
                    for frac in fractions:
                        level_idx = np.where(cumsum >= frac)[0][0]
                        threshold = hist_sorted[level_idx]
                        levels.append(threshold)
                    levels = sorted(levels)
                    for i in range(1, len(levels)):
                        if levels[i] <= levels[i-1]:
                            levels[i] = levels[i-1] + 1e-8
                    return levels
                levels_net = compute_levels(hist_net)
                levels_train = compute_levels(hist_train)
                x_centers = (x_edges[:-1] + x_edges[1:]) / 2.
                y_centers = (y_edges[:-1] + y_edges[1:]) / 2.
                X, Y = np.meshgrid(x_centers, y_centers)
                ax.contour(X, Y, hist_net.T, levels=levels_net, colors="blue")
                ax.contour(X, Y, hist_train.T, levels=levels_train, colors="green", linestyles="dashed")
            else:
                ax.set_visible(False)
            if j == 0 and i < ndim:
                ax.set_ylabel(labels[i], fontsize=12)
            if i == ndim - 1 and j < ndim:
                ax.set_xlabel(labels[j], fontsize=12)
    plt.suptitle("Corner Plot", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_filename)
    plt.show()

# ----------------------------
# Main Execution: Run MCMC and Produce Corner Plot
# ----------------------------
if __name__ == "__main__":
    # Define the desired number of parameters (e.g. 6 for the first six cosmological parameters)
    dim = 27
    
    # Full list of parameter names (assumed ordering remains unchanged)
    full_param_order = [
        "omega_b",
        "omega_cdm",
        "100*theta_s",
        "ln10^{10}A_s",
        "n_s",
        "tau_reio",
        "A_cib_217",
        "xi_sz_cib",
        "A_sz",
        "ps_A_100_100",
        "ps_A_143_143",
        "ps_A_143_217",
        "ps_A_217_217",
        "ksz_norm",
        "gal545_A_100",
        "gal545_A_143",
        "gal545_A_143_217",
        "gal545_A_217",
        "galf_TE_A_100",
        "galf_TE_A_100_143",
        "galf_TE_A_100_217",
        "galf_TE_A_143",
        "galf_TE_A_143_217",
        "galf_TE_A_217",
        "calib_100T",
        "calib_217T",
        "A_planck"
    ]
    # Use only the first dim parameters
    param_order = full_param_order[:dim]
    
    # Parse the log.param file for the required parameters
    log_param_file = os.path.join("chains", f"{dim}_param", "log.param")
    parsed_params = parse_log_param(log_param_file, param_order)
    
    initial_point = []
    proposal_sigma = []
    for key in param_order:
        if key not in parsed_params:
            raise ValueError(f"Parameter {key} not found in log.param file.")
        values = parsed_params[key]
        initial_point.append(float(values[0]))
        proposal_sigma.append(float(values[3]))
    initial_point = np.array(initial_point, dtype=np.float32)
    proposal_sigma = np.array(proposal_sigma, dtype=np.float32)
    
    print("Initial point from log.param:", initial_point)
    print("Proposal sigma from log.param:", proposal_sigma)
    
    # File paths for model and scalers (assumed to correspond to the same dim)
    T = 1
    model_file = os.path.join("planck_models", f"planck_model.h5")
    param_scaler_file = os.path.join("data", f"planck_param_scaler.pkl")
    target_scaler_file = os.path.join("data", f"planck_target_scaler.pkl")
    
    # Initialize MCMC sampler
    print("Starting MCMC with initial point and proposal sigma from log.param")
    sampler = MCMCPlanck(
        initial_point=initial_point,
        T=T,
        model_file=model_file,
        param_scaler_file=param_scaler_file,
        target_scaler_file=target_scaler_file,
        sigma=proposal_sigma
    )
    
    # Run MCMC: adaptive burn-in then fixed sampling
    sampler.chain(n_steps=5000, burn_in=1000, adaptation_interval=50, target_acceptance=0.25)
    mcmc_data = np.array(sampler.positions)
    
    # Plot evolution of predicted -loglike.
    plt.figure(figsize=(10, 6))
    plt.plot(sampler.loglkl_history, lw=1.5)
    plt.xlabel("Step")
    plt.ylabel("Predicted -loglike")
    plt.title("Evolution of Predicted -loglike over Steps")
    plt.grid(True)
    plt.savefig("loglkl_evolution.png")
    plt.show()
    
    # Load training chain data along with multiplicity weights.
    chains_dir = os.path.join("chains", f"{dim}_param")
    chain_files = [os.path.join(chains_dir, f) for f in os.listdir(chains_dir) if f.endswith(".txt")]
    training_chain_list = []
    training_weights_list = []
    for fname in chain_files:
        data_all = np.loadtxt(fname, skiprows=500)
        # data_all columns: 0=multiplicity, 1=-loglike, columns 2:2+full_dim.
        # We use only the first dim parameters.
        training_weights_list.append(data_all[:, 0])
        training_chain_list.append(data_all[:, 2:2+dim])
    training_data = np.concatenate(training_chain_list, axis=0)
    training_weights = np.concatenate(training_weights_list, axis=0)
    
    # For the corner plot we also need labels for the parameters.
    # Assume labels_27 is a list with 27 labels
    labels_27 = [r"$10^{-2}\omega_b$", r"$\omega_{cdm}$", r"$100\,\theta_s$", r"$\ln10^{10}A_s$", r"$n_s$",
                r"$\tau_{reio}$", r"$A_{cib,217}$", r"$\xi_{sz,cib}$", r"$A_{sz}$", r"$ps_{A,100,100}$",
                r"$ps_{A,143,143}$", r"$ps_{A,143,217}$", r"$ps_{A,217,217}$", r"$ksz_{norm}$", 
                r"$gal545_{A,100}$", r"$gal545_{A,143}$", r"$gal545_{A,143,217}$", r"$gal545_{A,217}$", 
                r"$galf_{TE,A,100}$", r"$galf_{TE,A,100,143}$", r"$galf_{TE,A,100,217}$", r"$galf_{TE,A,143}$",
                r"$galf_{TE,A,143,217}$", r"$galf_{TE,A,217}$", r"$10^{-3}calib_{100T}$", r"$10^{-3}calib_{217T}$", r"$A_{planck}$"]

    
    # For dim=6, best-fit is the initial point
    bestfit = initial_point
    plot_dim = 6
    dims_to_plot = list(range(plot_dim))

    corner_plot(mcmc_data, training_data, training_weights, bestfit, labels_27, dims_to_plot, output_filename="planck_corner_plot.png")
    
    print(f"MCMC Acceptance Rate: {sampler.acceptance_rate():.2%}")
