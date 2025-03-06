import os
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Configuration and Directories
# ----------------------------
dim = 2  # number of parameters
T = 5.0  # Temperature used for weight computation
num_samples = 2000
total_lines = 2 * 10**6  # total lines to load from all files
burn_in = 150  # number of lines to skip (burn-in)

# Directory containing the chain files (assumes a folder structure as in your training code)
chains_dir = os.path.join("chains", f"{dim}_param")
chain_files = [os.path.join(chains_dir, f) for f in os.listdir(chains_dir) if f.endswith(".txt")]
num_files = len(chain_files)
lines_per_file = total_lines // num_files  # integer division

# ----------------------------
# Load Chain Data (Skipping Burn-in)
# ----------------------------
# Each chain file is expected to have columns:
# Column 0: multiplicity, Column 1: -loglike, Columns 2,3: parameter values
chain_data_list = []
for file in chain_files:
    # Skip the first 'burn_in' lines in each file.
    data = np.loadtxt(file, skiprows=burn_in, max_rows=lines_per_file, usecols=range(2 + dim))
    chain_data_list.append(data)
combined_chain = np.concatenate(chain_data_list, axis=0)

# ----------------------------
# Compute Flat-Sampling Weights and Sample Points
# ----------------------------
multiplicity = combined_chain[:, 0]
# The -loglike values are scaled by 5 (as in the original code)
neg_loglike = combined_chain[:, 1] * 5
exponent = neg_loglike / (T)
exponent_stable = exponent - np.max(exponent)
weights = multiplicity * np.exp(exponent_stable)
weights_normalized = weights / weights.sum()

# Sample indices based on the normalized weights
indices = np.random.choice(len(weights_normalized), size=num_samples, replace=False, p=weights_normalized)
sampled_chain = combined_chain[indices, :]

# ----------------------------
# Prepare Data for Plotting
# ----------------------------
# Extract parameter columns (columns 2 and 3)
chain_params = combined_chain[:, 2:2+dim]
sampled_params = sampled_chain[:, 2:2+dim]

# Create a 2D histogram of the chain parameters to represent the overall chain density
num_bins = 100
H, xedges, yedges = np.histogram2d(chain_params[:, 0], chain_params[:, 1],
                                    bins=num_bins, density=True)
# Meshgrid for contour plotting (exclude the last bin edge)
X, Y = np.meshgrid(xedges[:-1], yedges[:-1])

# ----------------------------
# Plotting the Chain Contour and Flat-Sampled Points
# ----------------------------
plt.figure(figsize=(8, 6))
# Create filled contours from the 2D histogram; transpose H to align axes correctly.
contour = plt.contourf(X, Y, H.T, levels=50, cmap='viridis')
plt.colorbar(contour, label='Density')
# Overlay the flat-sampled points in red with white edges
plt.scatter(sampled_params[:, 0], sampled_params[:, 1],
            color='red', s=10, edgecolor='white', alpha=0.7,
            label='Flat Sampled Points')
plt.xlabel("Parameter 0")
plt.ylabel("Parameter 1")
plt.title("Flat Sampling over Chain Contour (Burn-in Skipped)")
plt.legend()
plt.savefig("flat_sampling_chain_contour.png", dpi=300)
plt.show()
