import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle

# ----------------------------
# Custom InverseScalingLayer Definition
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
# (Existing Code) Directories, Data Loading, and Preprocessing
# ----------------------------
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)
model_dir = "planck_models"
os.makedirs(model_dir, exist_ok=True)
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

# 1. Load TeX Labels from the .paramnames File
dim = 27  # number of parameters (excluding multiplicity and -loglike)
chains_dir = os.path.join("chains", f"{dim}_param")
paramnames_files = [f for f in os.listdir(chains_dir) if f.endswith(".paramnames")]
paramnames_file = os.path.join(chains_dir, paramnames_files[0])
with open(paramnames_file, "r") as f:
    lines = f.readlines()
tex_labels = [line.strip().split(maxsplit=1)[1].strip() for line in lines if line.strip()]
tex_labels = tex_labels[:dim]  # keep only the first dim labels

# 2. Load and Combine Chain Data
lines_to_skip = 500  # Adjust as needed
chain_files = [os.path.join(chains_dir, f) for f in os.listdir(chains_dir) if f.endswith(".txt")]
num_files = len(chain_files)
total_lines = 3 * (10 ** 5)  # total lines to load from all files
lines_per_file = total_lines // num_files  # integer division

chain_data_list = []
for file in chain_files:
    data = np.loadtxt(file, skiprows=lines_to_skip, max_rows=lines_per_file, usecols=range(2 + dim))
    chain_data_list.append(data)
combined_chain = np.concatenate(chain_data_list, axis=0)

# 3. Compute Flat-Sampling Weights and Sample Points
T = 1.0         # Temperature (adjust as needed)
num_samples = 50000
multiplicity = combined_chain[:, 0]
neg_loglike  = combined_chain[:, 1]  # already positive
exponent = neg_loglike / T
exponent_stable = exponent - np.max(exponent)
weights = multiplicity * np.exp(exponent_stable)
weights_normalized = weights / weights.sum()
indices = np.random.choice(len(weights_normalized), size=num_samples, replace=False, p=weights_normalized)
sampled_chain = combined_chain[indices, :]

names_dict = {i: tex_labels[i] for i in range(dim)}
flat_chain = {}
flat_chain["multiplicity"] = sampled_chain[:, 0]
flat_chain["-loglike"]     = sampled_chain[:, 1]
for i in range(dim):
    flat_chain[i] = sampled_chain[:, 2 + i]
flat_chain["names"] = names_dict

# 4. Scale Parameter Data; Keep Targets in Original Space
X = np.column_stack([flat_chain[i] for i in range(dim)])
y = flat_chain["-loglike"]  # use original -loglike values for training

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
scaler_y.fit(y.reshape(-1, 1))

# Save scalers for later use.
with open(os.path.join(data_dir, "planck_param_scaler.pkl"), "wb") as f:
    pickle.dump(scaler_X, f)
with open(os.path.join(data_dir, "planck_target_scaler.pkl"), "wb") as f:
    pickle.dump(scaler_y, f)

# ----------------------------
# 5. Build the Neural Network Model with Gaussian Approximation Output
# ----------------------------
# Output dimension: 1 (offset) + dim (best-fit) + dim*(dim+1)//2 (upper triangular of inverse covariance)
output_dim = 1 + dim + (dim * (dim + 1)) // 2

@tf.function
def reconstruct_sym_matrix(flat, dim):
    """
    Reconstruct a symmetric matrix from its flat upper-triangular part.
    flat: Tensor of shape (batch_size, dim*(dim+1)//2)
    returns: Tensor of shape (batch_size, dim, dim)
    """
    batch_size = tf.shape(flat)[0]
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
    M_sym = M + tf.transpose(M, perm=[0, 2, 1]) - tf.linalg.diag(tf.linalg.diag_part(M))
    return M_sym

@tf.function
def predicted_neg_loglike(x, raw_output):
    """
    Compute the predicted -loglike value using the Gaussian approximation:
      prediction_scaled = 0.5 * (offset + (x - x0)^T M (x - x0))
    This function returns predictions in the scaled space.
    """
    offset = raw_output[:, 0]               # shape (batch_size,)
    x0 = raw_output[:, 1:1+dim]              # shape (batch_size, dim)
    flat_inv_cov = raw_output[:, 1+dim:]     # shape (batch_size, dim*(dim+1)//2)
    M = reconstruct_sym_matrix(flat_inv_cov, dim)  # shape (batch_size, dim, dim)
    diff = x - x0                          # shape (batch_size, dim)
    quad = tf.einsum('bi,bij,bj->b', diff, M, diff)
    pred_scaled = 0.5 * (offset + quad)      # prediction in scaled space
    return pred_scaled

# Build the core model that predicts in scaled space.
input_layer = Input(shape=(dim,))

# Replace the Dense(tanh) layers with Dense layers followed by ScaledTanh
hidden = Dense(1024)(input_layer)
hidden = ScaledTanh()(hidden)
hidden = Dense(1024)(hidden)
hidden = ScaledTanh()(hidden)
hidden = Dense(1024)(hidden)
hidden = ScaledTanh()(hidden)
hidden = Dense(1024)(hidden)
hidden = ScaledTanh()(hidden)
hidden = Dense(1024)(hidden)
hidden = ScaledTanh()(hidden)

raw_output = Dense(output_dim, activation='linear')(hidden)
scaled_prediction = Lambda(lambda inputs: predicted_neg_loglike(inputs[0], inputs[1]))([input_layer, raw_output])
unscaled_output = InverseScalingLayer(scaler_y.scale_[0], scaler_y.mean_[0])(scaled_prediction)

# Build and compile the final model.
model = Model(inputs=input_layer, outputs=unscaled_output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse')

# ----------------------------
# 6. Train the Model with Early Stopping (loss computed in original space)
# ----------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=250, restore_best_weights=True)
history = model.fit(X_scaled, y,
                    epochs=5000,
                    batch_size=256,
                    validation_split=0.1,
                    verbose=2,
                    callbacks=[early_stop])

# Save training history.
with open(os.path.join(data_dir, "planck_training_history.pkl"), "wb") as f:
    pickle.dump(history.history, f)

# ----------------------------
# 7. Plot and Save Training History as a .png File
# ----------------------------
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Training History')
plt.savefig(os.path.join(plots_dir, "planck_training_history.png"))
plt.close()

# ----------------------------
# 8. Save the Trained Model
# ----------------------------
model.save(os.path.join(model_dir, "planck_model.h5"))
