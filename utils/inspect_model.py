from tensorflow.keras.models import load_model
import pickle
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf

# Choose scaler type: 'standard' for StandardScaler, 'minmax' for MinMaxScaler
scaler_type = "standard"  # Change this to 'minmax' to use MinMaxScaler

# Load trained model and scalers
data_dir = "data"
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)  # Ensure the directory exists

#model_path = os.path.join("models", "trained_model.h5")
model_path = os.path.join("models", "hps_trained_model.h5")
#param_scaler_path = os.path.join(data_dir, "param_scaler.pkl")
#target_scaler_path = os.path.join(data_dir, "target_scaler.pkl")
param_scaler_path = os.path.join(data_dir, "hps_param_scaler.pkl")
target_scaler_path = os.path.join(data_dir, "hps_target_scaler.pkl")

# Load scalers
with open(param_scaler_path, "rb") as f:
    param_scaler = pickle.load(f)

with open(target_scaler_path, "rb") as f:
    target_scaler = pickle.load(f)

# Define inverse transformation for both scaler types
def inverse_transform_tf(scaled_tensor, scaler):
    if isinstance(scaler, StandardScaler):
        means = tf.constant(scaler.mean_, dtype=tf.float32)
        stds = tf.constant(scaler.scale_, dtype=tf.float32)
        return scaled_tensor * stds + means
    elif isinstance(scaler, MinMaxScaler):
        min_val = tf.constant(scaler.data_min_, dtype=tf.float32)
        max_val = tf.constant(scaler.data_max_, dtype=tf.float32)
        return scaled_tensor * (max_val - min_val) + min_val
    
# Extract appropriate constants for rescaling
if isinstance(target_scaler, StandardScaler):
    target_mean = tf.constant(target_scaler.mean_[0], dtype=tf.float32)
    target_std = tf.constant(target_scaler.scale_[0], dtype=tf.float32)
elif isinstance(target_scaler, MinMaxScaler):
    target_min = tf.constant(target_scaler.data_min_[0], dtype=tf.float32)
    target_max = tf.constant(target_scaler.data_max_[0], dtype=tf.float32)

# Load the best-trained model
model = load_model("models/hps_trained_model.h5")

# Print the architecture
#model.summary()

# Print layer details
for i, layer in enumerate(model.layers):
    if isinstance(layer, tf.keras.layers.Dense):  # Only show Dense layers
        print(f"Layer {i}: {layer.name}")
        print(f"  Neurons: {layer.units}")
        print(f"  Activation: {layer.activation.__name__}")
        print("-" * 40)

# Get optimizer configuration
optimizer = model.optimizer
learning_rate = float(tf.keras.backend.get_value(optimizer.lr))

print(f"Learning Rate: {learning_rate}")

