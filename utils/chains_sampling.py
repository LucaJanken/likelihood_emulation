#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# ---------------------------
# Data Loading & Processing
# ---------------------------
chains_dir = "chains/2_param"
filenames = [
    "2024-09-17_123456789__1.txt",
    "2024-09-17_123456789__2.txt",
    "2024-09-17_123456789__3.txt",
    "2024-09-17_123456789__4.txt"
]

multiplicity = []
neg_log_like = []
param1 = []
param2 = []

# Read data from each file: skip the first 150 header lines and take 3000 data lines
header_lines = 150
max_lines = 3000

for filename in filenames:
    full_path = os.path.join(chains_dir, filename)
    with open(full_path, "r") as file:
        for _ in range(header_lines):
            next(file)
        count = 0
        for line in file:
            if count >= max_lines:
                break
            columns = line.strip().split()
            if len(columns) < 4:
                continue
            try:
                mult = int(columns[0])
                # Scale negative log-likelihoods by factor 5, as in training script.
                neg_val = float(columns[1]) * 5  
                p1_val = float(columns[2])
                p2_val = float(columns[3])
            except ValueError:
                continue
            multiplicity.append(mult)
            neg_log_like.append(neg_val)
            param1.append(p1_val)
            param2.append(p2_val)
            count += 1

multiplicity = np.array(multiplicity)
neg_log_like = np.array(neg_log_like)
param1 = np.array(param1)
param2 = np.array(param2)
params = np.column_stack((param1, param2))

# ---------------------------
# Sampling Strategy Function
# ---------------------------
def sample_indices(sampling_method, num_samples):
    """
    Choose indices based on the desired sampling strategy.
    Strategies:
      - "multiplicity": sample proportional to multiplicity.
      - "no_multiplicity": uniform random sampling.
      - "flat": weights = multiplicity / exp(-neg_log_like) (counteracts MCMC bias).
      - "no_multiplicity_flat": weights = 1 / exp(-neg_log_like) (ignores multiplicity).
    """
    indices = np.arange(len(multiplicity))
    
    if sampling_method == "multiplicity":
        probabilities = np.exp(multiplicity) / np.sum(np.exp(multiplicity))
    elif sampling_method == "no_multiplicity":
        probabilities = np.ones_like(multiplicity) / len(multiplicity)
    elif sampling_method == "flat":
        max_neg = np.max(neg_log_like)
        safe_neg = neg_log_like - max_neg
        weights = np.exp(multiplicity) * np.exp(safe_neg)
        probabilities = weights / np.sum(weights)
    elif sampling_method == "no_multiplicity_flat":
        max_neg = np.max(neg_log_like)
        safe_neg = neg_log_like - max_neg
        weights = np.exp(safe_neg)
        probabilities = weights / np.sum(weights)
    else:
        raise ValueError("Invalid sampling method selected.")
    
    sampled_indices = np.random.choice(indices, size=num_samples, p=probabilities, replace=False)
    return sampled_indices

# ---------------------------
# Create Contour Grid of Underlying Function
# ---------------------------
grid_size = 100
p1_min, p1_max = param1.min(), param1.max()
p2_min, p2_max = param2.min(), param2.max()
p1_grid = np.linspace(p1_min, p1_max, grid_size)
p2_grid = np.linspace(p2_min, p2_max, grid_size)
P1, P2 = np.meshgrid(p1_grid, p2_grid)

# Interpolate the function values (neg_log_like) over the grid.
grid_z = griddata(points=params, values=neg_log_like, xi=(P1, P2), method='cubic')

# ---------------------------
# Plotting: Underlying Function & Sampling Distributions
# ---------------------------
strategies = ["multiplicity", "no_multiplicity", "flat", "no_multiplicity_flat"]
num_samples = 2000  # Number of points to sample for each strategy

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, strategy in zip(axes.flatten(), strategies):
    # Sample indices using the current strategy.
    idx = sample_indices(strategy, num_samples)
    sampled_params = params[idx]
    
    # Plot the contour map of the underlying function.
    contour = ax.contourf(P1, P2, grid_z, levels=50, cmap='viridis')
    # Overlay the sampled points.
    ax.scatter(sampled_params[:, 0], sampled_params[:, 1], color='red', s=10, alpha=0.6, label=strategy)
    ax.set_title(f"Strategy: {strategy}")
    ax.set_xlabel("Parameter 1")
    ax.set_ylabel("Parameter 2")
    ax.legend()

# Adjust layout to make room for a single colorbar on the right.
plt.tight_layout(rect=[0, 0, 0.95, 1])

# Add a colorbar on the right side that spans all subplots.
cbar = fig.colorbar(contour, ax=axes.ravel().tolist(), orientation="vertical", fraction=0.046, pad=0.04)
cbar.set_label("Scaled Negative Log-Likelihood")

fig.suptitle("Underlying Function Contour with Sampling Distributions", fontsize=16)
plt.subplots_adjust(top=0.92, right=0.88)
plt.savefig("plots/sampling_strategies.png", dpi=300)
plt.show()
