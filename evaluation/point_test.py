import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load the trained model
model_path = "models/trained_model.h5"
model = tf.keras.models.load_model(model_path)

# Load the scalers
with open("data/param_scaler.pkl", "rb") as f:
    param_scaler = pickle.load(f)

with open("data/target_scaler.pkl", "rb") as f:
    target_scaler = pickle.load(f)

# Define the function for ground truth comparison
def neg_log_gaussian_2D(x, y):
    bestfit_value = 0.0
    bestfit_point = np.array([0.0, 0.0])
    sigma = np.sqrt(0.1)
    C_inv_xx = 1 / sigma**2
    C_inv_yy = 1 / sigma**2
    C_inv_xy = 0.0

    x_0, y_0 = bestfit_point
    dx = x - x_0
    dy = y - y_0
    mahalanobis_dist = C_inv_xx * dx**2 + 2 * C_inv_xy * dx * dy + C_inv_yy * dy**2
    return bestfit_value + 0.5 * mahalanobis_dist

# Input (x, y) in original space
x_input, y_input = 2.01, 2.01  # Change these values to test different inputs

# Scale input
input_scaled = param_scaler.transform([[x_input, y_input]])

# Predict using the model
pred_scaled = model.predict(input_scaled)

# Inverse transform the prediction to get the true-scale function value
pred_unscaled = target_scaler.inverse_transform(pred_scaled)

# Compute the true function value
true_value = neg_log_gaussian_2D(x_input, y_input)

# Print results
print(f"Input (x, y): ({x_input}, {y_input})")
print(f"True function value: {true_value:.6f}")
print(f"Model predicted value: {pred_unscaled[0, 0]:.6f}")
print(f"Absolute error: {abs(true_value - pred_unscaled[0, 0]):.6f}")
