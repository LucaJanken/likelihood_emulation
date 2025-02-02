import sys
import os

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the project root to sys.path if not already added
if project_root not in sys.path:
    sys.path.append(project_root)

# Now import modules AFTER modifying sys.path
from training.network import build_model, gaussian_nll_loss
from training.data_loader import generate_lhs_data, normalize_data
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pickle

# Set paths
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# Generate and normalize data
params, targets, outputs = generate_lhs_data(n_samples=2000, x_range=(-2, 2), y_range=(-2, 2))
params_scaled, targets_scaled, param_scaler, target_scaler = normalize_data(params, targets)

# Save scalers for later use
with open(os.path.join(data_dir, "param_scaler.pkl"), "wb") as f:
    pickle.dump(param_scaler, f)

with open(os.path.join(data_dir, "target_scaler.pkl"), "wb") as f:
    pickle.dump(target_scaler, f)

# Convert outputs to a NumPy array (ensuring correct shape)
outputs = np.array(outputs)

# Ensure correct shape: outputs should be (N, 6)
if outputs.shape[1] != 6:
    raise ValueError("Expected outputs to have shape (N, 6), but got:", outputs.shape)

# The correct target for training is `outputs`, not just `targets`
targets_combined = outputs  # Now correctly using the Gaussian parameters

# Build the model
model = build_model()

# Compile the model using the custom loss function
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=gaussian_nll_loss)

# Define Early Stopping
early_stopping = EarlyStopping(
    monitor="val_loss",  # Monitor validation loss
    patience=20,         # Stop if val_loss doesn't improve for 20 epochs
    restore_best_weights=True  # Restore the best weights at the end
)

# Train the model with Early Stopping
history = model.fit(
    params_scaled, targets_combined,  # Using extended targets
    epochs=500, 
    batch_size=64, 
    verbose=2, 
    validation_split=0.1,
    callbacks=[early_stopping]  # Add the callback
)

# Save trained model
os.makedirs("models", exist_ok=True)
model.save(os.path.join("models", "trained_model.h5"))

# Save training history
with open(os.path.join(data_dir, "training_history.pkl"), "wb") as f:
    pickle.dump(history.history, f)

# Plot training loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.yscale('log')
plt.grid(True)

# Save and show plot
os.makedirs("plots", exist_ok=True)  # Ensure directory exists
plt.savefig(os.path.join("plots", "training_loss.png"))
plt.show()
