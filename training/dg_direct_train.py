import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pyDOE import lhs
import matplotlib.pyplot as plt
import pickle

# Set paths
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# Choose scaler type: 'standard' or 'minmax'
scaler_type = "standard"  # change to 'minmax' if needed

def get_scaler(scaler_type):
    if scaler_type == "standard":
        return StandardScaler()
    elif scaler_type == "minmax":
        return MinMaxScaler(feature_range=(0, 1))
    else:
        raise ValueError("Invalid scaler_type. Choose 'standard' or 'minmax'.")

# Define chi² function for double Gaussian likelihood
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

# Choose function parameters
bestfit_point1 = np.array([-1.0, -1.0])  # First Gaussian center
bestfit_point2 = np.array([1.0, 1.0])    # Second Gaussian center
sigma = np.sqrt(0.1)                      # Variance for both Gaussians

# Define sample ranges and generate Latin Hypercube samples
x_min, x_max = -5, 5
y_min, y_max = -5, 5
circle_radius = 5
num_samples = 2000
valid_samples = []

while len(valid_samples) < num_samples:
    lhs_samples = lhs(2, samples=num_samples)
    x_samples = lhs_samples[:, 0] * (x_max - x_min) + x_min
    y_samples = lhs_samples[:, 1] * (y_max - y_min) + y_min
    distances = np.sqrt(x_samples**2 + y_samples**2)
    inside_circle = distances <= circle_radius
    valid_samples.extend(zip(x_samples[inside_circle], y_samples[inside_circle]))
    valid_samples = valid_samples[:num_samples]

params = np.array(valid_samples)
data = np.array([
    chi_squared_double_gaussian(x, y, bestfit_point1, bestfit_point2, sigma)
    for x, y in params
])
targets = data.reshape(-1, 1)

# Instantiate and apply scalers
param_scaler = get_scaler(scaler_type)
target_scaler = get_scaler(scaler_type)

params_scaled = param_scaler.fit_transform(params)
targets_scaled = target_scaler.fit_transform(targets)

# Save scalers for later use
with open(os.path.join(data_dir, "dg_direct_param_scaler.pkl"), "wb") as f:
    pickle.dump(param_scaler, f)
with open(os.path.join(data_dir, "dg_direct_target_scaler.pkl"), "wb") as f:
    pickle.dump(target_scaler, f)

# Define the custom loss function
def chi2_underestimate_penalizing_loss(lambda_weight=5.0):
    def loss(y_true, y_pred):
        mse = tf.square(y_true - y_pred)
        penalty = lambda_weight * tf.square(tf.maximum(y_true - y_pred, 0))  # Extra penalty for underestimates
        return tf.reduce_mean(mse + penalty)
    return loss

# Define a model that directly predicts the scaled chi² value
inputs_scaled = layers.Input(shape=(2,), name="scaled_xy")
hidden = layers.Dense(416, activation="relu")(inputs_scaled)
hidden = layers.Dense(480, activation="relu")(hidden)
predicted_value = layers.Dense(1, name="predicted_value")(hidden)

model = models.Model(inputs=inputs_scaled, outputs=predicted_value)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00018582), 
              loss=chi2_underestimate_penalizing_loss(lambda_weight=100.0))

model.summary()

# Train the model
early_stopping = EarlyStopping(monitor="loss", patience=100, restore_best_weights=True)
history = model.fit(
    params_scaled, 
    targets_scaled, 
    epochs=1000, 
    batch_size=32, 
    validation_split=0.1, 
    verbose=2, 
    callbacks=[early_stopping]
)

# Save the trained model and training history
os.makedirs("models", exist_ok=True)
model.save(os.path.join("models", "dg_direct_trained_model.h5"))

with open(os.path.join(data_dir, "dg_direct_training_history.pkl"), "wb") as f:
    pickle.dump(history.history, f)

# Plot training history
plt.figure(figsize=(8, 6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.yscale("log")
plt.grid(True)

os.makedirs("plots", exist_ok=True)
plt.savefig(os.path.join("plots", "dg_direct_training_loss.png"))
plt.show()
