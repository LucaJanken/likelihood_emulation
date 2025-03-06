import os
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Settings and Directories
# ----------------------------
dim = 6  # number of parameters (excluding multiplicity and -loglike)
chains_dir = os.path.join("chains", f"{dim}_param")
lines_to_skip = 150  # Adjust as needed
total_lines = 24000  # total lines to load from all files

# ----------------------------
# Load Chain Data
# ----------------------------
chain_files = [os.path.join(chains_dir, f) for f in os.listdir(chains_dir) if f.endswith(".txt")]
num_files = len(chain_files)
lines_per_file = total_lines // num_files

chain_data_list = []
for file in chain_files:
    data = np.loadtxt(file, skiprows=lines_to_skip, max_rows=lines_per_file, usecols=range(2 + dim))
    chain_data_list.append(data)
combined_chain = np.concatenate(chain_data_list, axis=0)

# ----------------------------
# Compute Flat-Sampling Weights
# ----------------------------
# Here, multiplicity is in column 0 and the -loglike is in column 1.
# We adjust -loglike by a temperature T.
T = 5.0
multiplicity = combined_chain[:, 0]
neg_loglike = combined_chain[:, 1] * T  # recover original -loglike values

# For stability in the exponent, subtract the max value
exponent = neg_loglike / T
exponent_stable = exponent - np.max(exponent)
weights = multiplicity * np.exp(exponent_stable)
weights_normalized = weights / weights.sum()

# ----------------------------
# Perform Weighted Sampling
# ----------------------------
num_samples = 5000
indices = np.random.choice(len(weights_normalized), size=num_samples, replace=True, p=weights_normalized)
sampled_chain = combined_chain[indices, :]

# ----------------------------
# Visualization: Histogram of -loglike values
# ----------------------------
plt.figure(figsize=(12, 5))

# Histogram of original chain's -loglike values
plt.subplot(1, 2, 1)
plt.hist(neg_loglike, bins=50, color="skyblue", edgecolor="black")
plt.xlabel("-loglike")
plt.ylabel("Frequency")
plt.title("Original Chain -loglike Distribution")

# Histogram of sampled chain's -loglike values
plt.subplot(1, 2, 2)
sampled_neg_loglike = sampled_chain[:, 1] * T  # adjust for T as before
plt.hist(sampled_neg_loglike, bins=50, color="lightgreen", edgecolor="black")
plt.xlabel("-loglike")
plt.ylabel("Frequency")
plt.title("Flat-Sampled Chain -loglike Distribution")

plt.tight_layout()
plt.savefig("flat_sampling_check.png")
plt.show()

# ----------------------------
# Optional: Check marginal distributions for each parameter
# ----------------------------
# This block creates a figure with one subplot per parameter.
param_data = [sampled_chain[:, 2 + i] for i in range(dim)]
plt.figure(figsize=(15, 10))
for i in range(dim):
    plt.subplot(2, 3, i+1)
    plt.hist(param_data[i], bins=40, color="salmon", edgecolor="black")
    plt.xlabel(f"Parameter {i}")
    plt.ylabel("Frequency")
    plt.title(f"Parameter {i} Distribution")
plt.tight_layout()
plt.savefig("parameter_marginals.png")
plt.show()
