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

# Choose scaler type: 'standard' for StandardScaler, 'minmax' for MinMaxScaler
scaler_type = "standard"  # Change this to 'minmax' to switch to MinMaxScaler

# Function to select the scaler dynamically
def get_scaler(scaler_type):
    if scaler_type == "standard":
        return StandardScaler()
    elif scaler_type == "minmax":
        return MinMaxScaler(feature_range=(0, 1))
    else:
        raise ValueError("Invalid scaler_type. Choose 'standard' or 'minmax'.")

# Define chi^2 function for double Gaussian likelihood
def chi_squared_double_gaussian(x, y, bestfit_point1, bestfit_point2, sigma):
    x1, y1 = bestfit_point1
    x2, y2 = bestfit_point2
    
    # Compute the Gaussian terms
    term1 = np.exp(-0.5 * ((x - x1) ** 2 + (y - y1) ** 2) / sigma**2)
    term2 = np.exp(-0.5 * ((x - x2) ** 2 + (y - y2) ** 2) / sigma**2)
    
    # Properly normalize the likelihood
    normalization_factor = 1 / (4 * np.pi * sigma**2)  # Factor of 2 for the sum of two Gaussians
    likelihood = normalization_factor * (term1 + term2)
    
    # Avoid taking log of zero
    likelihood = np.maximum(likelihood, 1e-300)  # Prevent log(0) issues
    
    # Compute chi^2
    chi2 = -2 * np.log(likelihood)
    
    return chi2

# Choose function parameters
bestfit_point1 = np.array([-1.0, -1.0])  # First Gaussian center
bestfit_point2 = np.array([1.0, 1.0])  # Second Gaussian center
sigma = np.sqrt(0.1)  # Same variance for both Gaussians

# Define sample ranges
x_min, x_max = -5, 5
y_min, y_max = -5, 5
circle_radius = 5

# Generate Latin Hypercube samples
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

# Compute chi^2 values
data = np.array([
    chi_squared_double_gaussian(x, y, bestfit_point1, bestfit_point2, sigma)
    for x, y in params
])

# Reshape for scaling
targets = data.reshape(-1, 1)

# Instantiate scalers
param_scaler = get_scaler(scaler_type)
target_scaler = get_scaler(scaler_type)

params_scaled = param_scaler.fit_transform(params)
targets_scaled = target_scaler.fit_transform(targets)

# Save scalers
with open(os.path.join(data_dir, "cl_dg_param_scaler.pkl"), "wb") as f:
    pickle.dump(param_scaler, f)
with open(os.path.join(data_dir, "cl_dg_target_scaler.pkl"), "wb") as f:
    pickle.dump(target_scaler, f)

# Define model
inputs_scaled = layers.Input(shape=(2,), name="scaled_xy")
hidden = layers.Dense(416, activation="tanh")(inputs_scaled)
hidden = layers.Dense(448, activation="sigmoid")(hidden)
hidden = layers.Dense(416, activation="sigmoid")(hidden)
hidden = layers.Dense(256, activation="sigmoid")(hidden)
hidden = layers.Dense(448, activation="relu")(hidden)
pred_params = layers.Dense(6, name="gaussian_params")(hidden)

def compute_scaled_values(args):
    scaled_xy, pred_params = args
    xy_unscaled = scaled_xy * (param_scaler.scale_ + param_scaler.mean_)
    x, y = xy_unscaled[:, 0:1], xy_unscaled[:, 1:2]
    
    chi0 = pred_params[:, 0:1]
    x0 = pred_params[:, 1:2]
    y0 = pred_params[:, 2:3]
    cxx = pred_params[:, 3:4]
    cxy = pred_params[:, 4:5]
    cyy = pred_params[:, 5:6]
    
    dx = x - x0
    dy = y - y0
    mahalanobis_dist = cxx * dx**2 + 2 * cxy * dx * dy + cyy * dy**2
    unscaled_values = chi0 + mahalanobis_dist
    
    # Rescale function values
    if isinstance(target_scaler, StandardScaler):
        scaled_values = (unscaled_values - target_scaler.mean_) / target_scaler.scale_
    elif isinstance(target_scaler, MinMaxScaler):
        scaled_values = (unscaled_values - target_scaler.data_min_) / (target_scaler.data_max_ - target_scaler.data_min_)
    
    return scaled_values

scaled_values = layers.Lambda(compute_scaled_values, name="predicted_values")([inputs_scaled, pred_params])
model = models.Model(inputs=inputs_scaled, outputs=scaled_values)
#model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), loss="mse")

# Define the custom loss function
def chi2_underestimate_penalizing_loss(lambda_weight=5.0):
    def loss(y_true, y_pred):
        mse = tf.square(y_true - y_pred)
        penalty = lambda_weight * tf.square(tf.maximum(y_true - y_pred, 0))  # Extra penalty for underestimates
        return tf.reduce_mean(mse + penalty)
    return loss

# Compile model with custom loss
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), 
    loss=chi2_underestimate_penalizing_loss(lambda_weight=100.0)
)

# Model summary
model.summary()

# Train model
early_stopping = EarlyStopping(monitor="loss", patience=100, restore_best_weights=True)
history = model.fit(
    params_scaled, 
    targets_scaled, 
    epochs=1500, 
    batch_size=32, 
    validation_split=0.1, 
    verbose=2, 
    callbacks=[early_stopping]
    )

# Save model
os.makedirs("models", exist_ok=True)
model.save(os.path.join("models", "cl_dg_trained_model.h5"))

# Save training history
with open(os.path.join(data_dir, "cl_dg_training_history.pkl"), "wb") as f:
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

# Save plot
os.makedirs("plots", exist_ok=True)
plt.savefig(os.path.join("plots", "cl_dg_training_loss.png"))
plt.show()
