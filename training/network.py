import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def build_model(hp=None):
    """
    Builds and returns a neural network model.
    If hp is None, build a default model without tuning.
    If hp is provided, search over number of layers, units, and dropout rate.
    """
    model = Sequential()

    if hp is not None:
        # Number of hidden layers
        num_layers = hp.Int("num_layers", min_value=1, max_value=5, step=1)
        # First layer units
        first_units = hp.Int("units_0", min_value=32, max_value=512, step=32)
        # Dropout rate
        dropout_rate = hp.Float("dropout_rate", min_value=0.0, max_value=0.5, step=0.1)

        # First layer + input
        model.add(Dense(first_units, activation='relu', input_shape=(2,)))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

        # Additional hidden layers
        for i in range(1, num_layers):
            units_i = hp.Int(f"units_{i}", min_value=32, max_value=512, step=32)
            model.add(Dense(units_i, activation='relu'))
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))

    else:
        # Default architecture if no hp is passed
        model.add(Dense(128, activation='relu', input_shape=(2,)))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))

    # Output layer: 6 neurons
    # First 3 neurons (p_0, x_0, y_0) use linear activation (unbounded)
    # Last 3 neurons (C^-1_xx, C^-1_yy, C^-1_xy) use tanh activation (bounded)
    model.add(Dense(6, activation=['linear', 'linear', 'linear', 'tanh', 'tanh', 'tanh']))

    return model

def gaussian_nll_loss(y_true, y_pred):
    """
    Custom loss function that computes the Mean Squared Error (MSE)
    between the true and predicted negative log-likelihood functions.

    Parameters:
    - y_true: True negative log-likelihood values (batch_size, 1)
    - y_pred: Predicted Gaussian parameters (batch_size, 6)
      (p_0, x_0, y_0, C^-1_xx, C^-1_yy, C^-1_xy)

    Returns:
    - MSE loss
    """
    # Extract predicted parameters
    p_0_pred = y_pred[:, 0]  # Best-fit value
    x_0_pred = y_pred[:, 1]  # Best-fit point (x)
    y_0_pred = y_pred[:, 2]  # Best-fit point (y)
    C_inv_xx_pred = y_pred[:, 3]  # Inverse covariance entry C^-1_xx
    C_inv_yy_pred = y_pred[:, 4]  # Inverse covariance entry C^-1_yy
    C_inv_xy_pred = y_pred[:, 5]  # Inverse covariance entry C^-1_xy

    # Extract input (x, y) values (assuming they are stored in y_true[:, 1] and y_true[:, 2])
    x_input = y_true[:, 1]
    y_input = y_true[:, 2]

    # Extract true negative log-likelihood values
    true_nll = y_true[:, 0]  # The actual neg-log-likelihood values

    # Compute squared Mahalanobis distance using predicted inverse covariance matrix
    dx = x_input - x_0_pred
    dy = y_input - y_0_pred
    mahalanobis_dist = (
        C_inv_xx_pred * dx**2 + 2 * C_inv_xy_pred * dx * dy + C_inv_yy_pred * dy**2
    )

    # Compute predicted negative log-likelihood
    predicted_nll = p_0_pred + 0.5 * mahalanobis_dist

    # Compute MSE loss
    loss = tf.reduce_mean(tf.square(true_nll - predicted_nll))

    return loss
