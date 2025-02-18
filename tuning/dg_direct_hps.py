import sys
import os
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pyDOE import lhs
import pickle

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Set paths
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# Choose scaler type: 'standard' for StandardScaler, 'minmax' for MinMaxScaler
scaler_type = "standard"

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
    normalization_factor = 1 / (4 * np.pi * sigma**2)
    likelihood = normalization_factor * (term1 + term2)
    
    # Avoid taking log of zero
    likelihood = np.maximum(likelihood, 1e-300)
    
    # Compute chi^2
    chi2 = -2 * np.log(likelihood)
    
    return chi2

# Choose function parameters
bestfit_point1 = np.array([-1.0, -1.0])   # First Gaussian center
bestfit_point2 = np.array([1.0, 1.0])       # Second Gaussian center
sigma = np.sqrt(0.1)                        # Common variance

# Generate samples
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
data = np.array([chi_squared_double_gaussian(x, y, bestfit_point1, bestfit_point2, sigma)
                 for x, y in valid_samples])
targets = data.reshape(-1, 1)

# Scale parameters and targets
param_scaler = get_scaler(scaler_type)
target_scaler = get_scaler(scaler_type)
params_scaled = param_scaler.fit_transform(params)
targets_scaled = target_scaler.fit_transform(targets)

with open(os.path.join(data_dir, "dg_direct_hps_param_scaler.pkl"), "wb") as f:
    pickle.dump(param_scaler, f)
with open(os.path.join(data_dir, "dg_direct_hps_target_scaler.pkl"), "wb") as f:
    pickle.dump(target_scaler, f)

# Build a model that directly predicts the function value
def build_model(hp):
    inputs_scaled = layers.Input(shape=(2,), name="scaled_xy")
    num_layers = hp.Int("num_layers", min_value=2, max_value=5)
    activation_options = ["relu", "tanh", "sigmoid"]

    # First hidden layer
    hidden = layers.Dense(
        hp.Int("units_1", min_value=32, max_value=512, step=32),
        activation=hp.Choice("activation_1", values=activation_options)
    )(inputs_scaled)

    # Additional hidden layers
    for i in range(2, num_layers + 1):
        hidden = layers.Dense(
            hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
            activation=hp.Choice(f"activation_{i}", values=activation_options)
        )(hidden)

    # Directly output a single value corresponding to the chi^2
    output = layers.Dense(1, name="predicted_value")(hidden)

    model = models.Model(inputs=inputs_scaled, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Float("learning_rate", min_value=1e-5, max_value=1e-2, sampling="log")
        ),
        loss="mse"
    )
    return model

# Hyperparameter tuning
tuner = kt.RandomSearch(
    build_model,
    objective="val_loss",
    max_trials=5000,
    executions_per_trial=1,
    directory="random_search_results",
    project_name="dg_hyperparameter_search"
)

early_stopping = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

tuner.search(
    params_scaled, targets_scaled,
    epochs=200,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stopping]
)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)

history = best_model.fit(
    params_scaled,
    targets_scaled,
    epochs=500,
    batch_size=64,
    validation_split=0.1,
    verbose=2,
    callbacks=[early_stopping]
)

# Save the best model and training history
os.makedirs("models", exist_ok=True)
best_model.save(os.path.join("models", "dg_direct_hps_trained_model.h5"))

with open(os.path.join(data_dir, "dg_direct_hps_training_history.pkl"), "wb") as f:
    pickle.dump(history.history, f)
