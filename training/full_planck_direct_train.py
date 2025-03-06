import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle

# ----------------------------
# Custom InverseScalingLayer Definition (unchanged)
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
# Custom Alsing Activation Layer
# (Implements the activation from equation (2.3) in the article)
# ----------------------------
class AlsingActivation(tf.keras.layers.Layer):
    def __init__(self, initial_beta=1.0, initial_gamma=0.1, **kwargs):
        super(AlsingActivation, self).__init__(**kwargs)
        self.initial_beta = initial_beta
        self.initial_gamma = initial_gamma

    def build(self, input_shape):
        param_shape = (input_shape[-1],)
        self.beta = self.add_weight(
            name="beta",
            shape=param_shape,
            initializer=tf.keras.initializers.Constant(self.initial_beta),
            trainable=True
        )
        self.gamma = self.add_weight(
            name="gamma",
            shape=param_shape,
            initializer=tf.keras.initializers.Constant(self.initial_gamma),
            trainable=True
        )
        super(AlsingActivation, self).build(input_shape)

    def call(self, inputs):
        # f(x) = (gamma + (1 + exp(-beta*x))^(-1) * (1 - gamma)) * x
        factor = self.gamma + (1 - self.gamma) / (1 + tf.exp(-self.beta * inputs))
        return factor * inputs

    def get_config(self):
        config = super(AlsingActivation, self).get_config()
        config.update({
            "initial_beta": self.initial_beta,
            "initial_gamma": self.initial_gamma
        })
        return config

# ----------------------------
# Data Loading, Preprocessing and Target Scaling
# ----------------------------
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)
model_dir = "planck_models"
os.makedirs(model_dir, exist_ok=True)
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

dim = 27  # number of parameters (excluding multiplicity and -loglike)
chains_dir = os.path.join("chains", f"{dim}_param")
paramnames_files = [f for f in os.listdir(chains_dir) if f.endswith(".paramnames")]
paramnames_file = os.path.join(chains_dir, paramnames_files[0])
with open(paramnames_file, "r") as f:
    lines = f.readlines()
tex_labels = [line.strip().split(maxsplit=1)[1].strip() for line in lines if line.strip()]
tex_labels = tex_labels[:dim]

lines_to_skip = 500
chain_files = [os.path.join(chains_dir, f) for f in os.listdir(chains_dir) if f.endswith(".txt")]
num_files = len(chain_files)
total_lines = 3 * (10 ** 5)
lines_per_file = total_lines // num_files

chain_data_list = []
for file in chain_files:
    data = np.loadtxt(file, skiprows=lines_to_skip, max_rows=lines_per_file, usecols=range(2 + dim))
    chain_data_list.append(data)
combined_chain = np.concatenate(chain_data_list, axis=0)

T = 1.0
num_samples = 50000
multiplicity = combined_chain[:, 0]
neg_loglike  = combined_chain[:, 1]  # target values
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

X = np.column_stack([flat_chain[i] for i in range(dim)])
y = flat_chain["-loglike"]

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
# Scale the targets
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Save scalers with "direct" added to avoid overwriting originals.
with open(os.path.join(data_dir, "planck_direct_param_scaler.pkl"), "wb") as f:
    pickle.dump(scaler_X, f)
with open(os.path.join(data_dir, "planck_direct_target_scaler.pkl"), "wb") as f:
    pickle.dump(scaler_y, f)

# ----------------------------
# Build the Neural Network Model
# ----------------------------
input_layer = Input(shape=(dim,))
hidden = Dense(1024)(input_layer)
hidden = AlsingActivation()(hidden)
hidden = Dense(1024)(hidden)
hidden = AlsingActivation()(hidden)
hidden = Dense(1024)(hidden)
hidden = AlsingActivation()(hidden)
hidden = Dense(1024)(hidden)
hidden = AlsingActivation()(hidden)
hidden = Dense(1024)(hidden)
hidden = AlsingActivation()(hidden)

# Final output predicts the scaled -loglike value
raw_output = Dense(1, activation='linear')(hidden)
# Convert scaled prediction back to original space
unscaled_output = InverseScalingLayer(scaler_y.scale_[0], scaler_y.mean_[0])(raw_output)

model = Model(inputs=input_layer, outputs=unscaled_output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse')

# ----------------------------
# Train the Model with Early Stopping
# ----------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=250, restore_best_weights=True)
history = model.fit(X_scaled, y_scaled,
                    epochs=5000,
                    batch_size=256,
                    validation_split=0.1,
                    verbose=2,
                    callbacks=[early_stop])

with open(os.path.join(data_dir, "planck_direct_training_history.pkl"), "wb") as f:
    pickle.dump(history.history, f)

# ----------------------------
# Plot and Save Training History
# ----------------------------
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Direct Training History')
plt.savefig(os.path.join(plots_dir, "planck_direct_training_history.png"))
plt.close()

# ----------------------------
# Save the Trained Model
# ----------------------------
model.save(os.path.join(model_dir, "planck_direct_model.h5"))
