import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

# ----------------------------
# Global Parameter Definition
# ----------------------------
dim = 6  # Change this to the desired number of parameters

# ----------------------------
# Custom Function Definitions
# ----------------------------
@tf.function
def reconstruct_sym_matrix(flat, dim):
    """
    Reconstruct a symmetric matrix from its flat upper-triangular part.
    flat: Tensor of shape (batch_size, dim*(dim+1)//2)
    returns: Tensor of shape (batch_size, dim, dim)
    """
    batch_size = tf.shape(flat)[0]
    # Initialize a zero matrix for each sample.
    M = tf.zeros((batch_size, dim, dim), dtype=flat.dtype)
    idx = []
    for i in range(dim):
        for j in range(i, dim):
            idx.append((i, j))
    idx = tf.constant(idx, dtype=tf.int32)
    num_elements = idx.shape[0]
    batch_idx = tf.reshape(tf.range(batch_size), (-1, 1))
    batch_idx = tf.tile(batch_idx, (1, num_elements))
    batch_idx = tf.reshape(batch_idx, (-1, 1))
    idx = tf.tile(tf.expand_dims(idx, 0), (batch_size, 1, 1))
    idx = tf.reshape(idx, (-1, 2))
    full_idx = tf.concat([batch_idx, idx], axis=1)
    flat_reshaped = tf.reshape(flat, (-1,))
    M = tf.tensor_scatter_nd_update(M, full_idx, flat_reshaped)
    # Mirror the upper triangle to the lower triangle to make it symmetric.
    M_sym = M + tf.transpose(M, perm=[0, 2, 1]) - tf.linalg.diag(tf.linalg.diag_part(M))
    return M_sym

@tf.function
def predicted_neg_loglike(x, raw_output):
    """
    Compute the predicted -loglike value using the Gaussian approximation:
      prediction_scaled = 0.5 * (offset + (x - x0)^T M (x - x0))
    This function returns predictions in the scaled space.
    """
    offset = raw_output[:, 0]                   # shape (batch_size,)
    x0 = raw_output[:, 1:1+dim]                   # shape (batch_size, dim)
    flat_inv_cov = raw_output[:, 1+dim:]          # shape (batch_size, dim*(dim+1)//2)
    M = reconstruct_sym_matrix(flat_inv_cov, dim) # shape (batch_size, dim, dim)
    diff = x - x0                               # shape (batch_size, dim)
    quad = tf.einsum('bi,bij,bj->b', diff, M, diff)
    pred_scaled = 0.5 * (offset + quad)           # prediction in scaled space
    return pred_scaled

# ----------------------------
# Custom Inverse Scaling Layer
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

# ----------------------------
# Custom Trainable Scaled Tanh Activation Layer
# ----------------------------
class ScaledTanh(tf.keras.layers.Layer):
    def __init__(self, initial_alpha=1.0, **kwargs):
        super(ScaledTanh, self).__init__(**kwargs)
        self.initial_alpha = initial_alpha

    def build(self, input_shape):
        # Create a trainable parameter alpha
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
        config.update({
            "initial_alpha": self.initial_alpha
        })
        return config

# ----------------------------
# Directories for Data, Models, and Plots
# ----------------------------
data_dir = "data"
model_dir = "n_dim_models"
plots_dir = "plots"
chains_dir = os.path.join("chains", f"{dim}_param")  # Adjusted for arbitrary dim

# ----------------------------
# Load Saved Scalers and Trained Model
# ----------------------------
with open(os.path.join(data_dir, "6d_param_scaler.pkl"), "rb") as f:
    scaler_X = pickle.load(f)
with open(os.path.join(data_dir, "6d_target_scaler.pkl"), "rb") as f:
    scaler_y = pickle.load(f)

model_path = os.path.join(model_dir, "6d_model.h5")
# Include custom functions and layers in custom_objects so they can be reconstructed.
custom_objects = {
    'predicted_neg_loglike': predicted_neg_loglike,
    'reconstruct_sym_matrix': reconstruct_sym_matrix,
    'InverseScalingLayer': InverseScalingLayer,
    'ScaledTanh': ScaledTanh
}
final_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)

# ----------------------------
# Load and Combine Chain Data for Evaluation
# ----------------------------
chain_files = [os.path.join(chains_dir, f) for f in os.listdir(chains_dir) if f.endswith(".txt")]

# Define parameters for loading evaluation data:
lines_per_file_eval = None  # Maximum number of rows to load per file for evaluation
burn_in_lines = 150         # Number of lines to skip in each chain file
skip_training_lines = 4000 + burn_in_lines  # Skip training and burn-in lines

chain_data_list = []
for file in chain_files:
    try:
        # Assumes each file has (2 + dim) columns: multiplicity, -loglike, and 'dim' parameters.
        data = np.loadtxt(file, skiprows=skip_training_lines, max_rows=lines_per_file_eval, usecols=range(2 + dim))
        chain_data_list.append(data)
    except Exception as e:
        print(f"Error loading {file}: {e}")

if not chain_data_list:
    raise ValueError("No evaluation data loaded. Check file paths and parameters.")

combined_chain = np.concatenate(chain_data_list, axis=0)

# ----------------------------
# Compute Flat-Sampling Weights and Sample Evaluation Points
# ----------------------------
T = 5.0  # Temperature parameter (adjust as needed)

num_samples = 5000
multiplicity = combined_chain[:, 0]
neg_loglike = combined_chain[:, 1] * T  # Multiply by T to get the original -loglike values

# Compute sampling weights: weight ∝ multiplicity * exp(neg_loglike / T)
exponent = neg_loglike / T
exponent_stable = exponent - np.max(exponent)
weights = multiplicity * np.exp(exponent_stable)
weights_normalized = weights / np.sum(weights)

indices = np.random.choice(len(weights_normalized), size=num_samples, replace=False, p=weights_normalized)
sampled_chain = combined_chain[indices, :]

# ----------------------------
# Prepare Evaluation Data
# ----------------------------
# Extract parameters (columns 2 to 2+dim) and true -loglike values (column 1).
X_eval = sampled_chain[:, 2:2+dim]  # shape: (num_samples, dim)
y_true = sampled_chain[:, 1] * T   # shape: (num_samples,)

# Scale the parameters using the loaded parameter scaler.
X_eval_scaled = scaler_X.transform(X_eval)

# ----------------------------
# Get Predictions and Compute Relative Error
# ----------------------------
# The final_model outputs predictions in the original target space.
y_pred = final_model.predict(X_eval_scaled).flatten()
# Relative error in the original -loglike space.
relative_error = (y_pred - y_true) / y_true

# Convert the -loglike values to chi^2 using chi^2 = 2 * (-loglike)
chi2_true = 2 * y_true
chi2_pred = 2 * y_pred

# ----------------------------
# Plot the Results using chi^2
# ----------------------------
plt.figure(figsize=(10, 8))
plt.scatter(chi2_true, relative_error, s=10, alpha=0.6)
plt.xlabel(r"$\chi^2_{\mathrm{true}}$")
plt.ylabel(r"$(\chi^2_{\mathrm{pred}} - \chi^2_{\mathrm{true}})/\chi^2_{\mathrm{true}}$")
plt.title("6d Model Prediction Residuals in $\chi^2$")
plt.grid(True)
plot_path = os.path.join(plots_dir, "6d_model_residuals.png")
plt.savefig(plot_path, dpi=300)
plt.show()

# ----------------------------
# Print Summary Statistics
# ----------------------------
print("Mean relative error:", np.mean(relative_error))
print("Standard deviation of relative error:", np.std(relative_error))
