import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pyDOE import lhs

def neg_loglike_gaussian_2D(x, y, bestfit_value, bestfit_point, sigma):
    """
    Computes the negative log-likelihood function for an isotropic 2D Gaussian.
    Also returns the Gaussian parameters needed for training.
    """
    x_0, y_0 = bestfit_point
    variance = sigma**2
    squared_distance = (x - x_0)**2 + (y - y_0)**2
    neg_log_likelihood = bestfit_value + 0.5 * (squared_distance / variance)

    # Isotropic Gaussian: inverse covariance matrix
    C_inv_xx = 1 / variance  # Since it's isotropic, C^-1_xx = C^-1_yy
    C_inv_yy = 1 / variance
    C_inv_xy = 0  # No correlation in isotropic case

    return neg_log_likelihood, bestfit_value, x_0, y_0, C_inv_xx, C_inv_yy, C_inv_xy

def generate_lhs_data(n_samples=2000, x_range=(-2, 2), y_range=(-2, 2)):
    """
    Generates Latin Hypercube sampled data for training.
    """
    bestfit_value = 0.0  # Fixed best-fit negative log-likelihood value
    bestfit_point = np.array([0.0, 0.0])  # True center of Gaussian
    sigma = np.sqrt(0.1)  # Standard deviation

    # Generate LHS samples in [0,1] range
    lhs_samples = lhs(2, samples=n_samples)
    x_samples = lhs_samples[:, 0] * (x_range[1] - x_range[0]) + x_range[0]
    y_samples = lhs_samples[:, 1] * (y_range[1] - y_range[0]) + y_range[0]

    # Compute function values and parameters
    data = [neg_loglike_gaussian_2D(x, y, bestfit_value, bestfit_point, sigma) for x, y in zip(x_samples, y_samples)]
    data = np.array(data)  # Convert list of tuples to NumPy array

    # Extract outputs separately
    targets = data[:, 0].reshape(-1, 1)  # True neg-log-likelihood values
    params = np.column_stack((x_samples, y_samples))  # Input coordinates
    outputs = data[:, 1:]  # (p_0, x_0, y_0, C^-1_xx, C^-1_yy, C^-1_xy)

    return params, targets, outputs  # Returning parameters separately for better scaling

def normalize_data(params):
    """
    Normalizes only the input parameters.
    """
    param_scaler = MinMaxScaler(feature_range=(0,1))
    param_scaler.fit(np.array([[-10, -10], [10, 10]]))  # Fixed input range

    params_scaled = param_scaler.transform(params)
    return params_scaled, param_scaler

