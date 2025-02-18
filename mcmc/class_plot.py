import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ---------------------------
# Settings & Data Paths
# ---------------------------
chains_dir = "chains/2_param"
filenames = [
    "2024-09-17_123456789__1.txt",
    "2024-09-17_123456789__2.txt",
    "2024-09-17_123456789__3.txt",
    "2024-09-17_123456789__4.txt"
]
# We will read the first 5000 data lines (after skipping 150 header lines) from each file.
n_header = 150
n_data   = 6000

# ---------------------------
# Load Training Data
# ---------------------------
multiplicity = []
neg_log_like = []
param1 = []  # will be 100*omega_b
param2 = []  # will be omega_cdm

for fname in filenames:
    full_path = os.path.join(chains_dir, fname)
    with open(full_path, "r") as f:
        # Skip header lines
        for _ in range(n_header):
            next(f)
        count = 0
        for line in f:
            if count >= n_data:
                break
            cols = line.strip().split()
            if len(cols) < 4:
                continue
            try:
                mult = int(cols[0])
                # Scale negative log-likelihoods by factor 5 (as in your training code).
                neg_log = float(cols[1]) * 5  
                p1_val = float(cols[2])
                p2_val = float(cols[3])
            except ValueError:
                continue
            multiplicity.append(mult)
            neg_log_like.append(neg_log)
            param1.append(p1_val)
            param2.append(p2_val)
            count += 1

# Convert lists to numpy arrays.
multiplicity = np.array(multiplicity)
neg_log_like = np.array(neg_log_like)
params = np.column_stack((param1, param2))

# ---------------------------
# Determine Best-fit
# ---------------------------
# We define the best-fit as the sample with the smallest negative log-likelihood.
best_index = np.argmin(neg_log_like)
bestfit = params[best_index]  # [100*omega_b, omega_cdm]

# ---------------------------
# Create Triangle Plot
# ---------------------------
# For 2D, we'll create a 2x2 grid: histograms on the diagonals and a scatter plot in the lower-left.
fig, axes = plt.subplots(2, 2, figsize=(8, 8))

# LaTeX labels for parameters.
labels = [r"$100\,\omega_b$", r"$\omega_{cdm}$"]

# Histogram for parameter 1 (100*omega_b) on the upper-left.
axes[0, 0].hist(params[:, 0], bins=50, color="skyblue")
axes[0, 0].axvline(bestfit[0], color="red", linestyle="--", linewidth=2)
axes[0, 0].set_xlabel(labels[0])
axes[0, 0].xaxis.set_major_locator(MaxNLocator(nbins=5))

# Histogram for parameter 2 (omega_cdm) on the lower-right.
axes[1, 1].hist(params[:, 1], bins=50, color="skyblue")
axes[1, 1].axvline(bestfit[1], color="red", linestyle="--", linewidth=2)
axes[1, 1].set_xlabel(labels[1])
axes[1, 1].xaxis.set_major_locator(MaxNLocator(nbins=5))

# Scatter plot (joint distribution) in the lower-left.
axes[1, 0].scatter(params[:, 0], params[:, 1], s=2, alpha=0.5)
axes[1, 0].scatter(bestfit[0], bestfit[1], color="red", s=50, marker="o", label="Best-fit")
axes[1, 0].set_xlabel(labels[0])
axes[1, 0].set_ylabel(labels[1])
axes[1, 0].yaxis.set_major_locator(MaxNLocator(nbins=5))
axes[1, 0].legend()

# The upper-right subplot is unused.
axes[0, 1].axis("off")

plt.suptitle("Triangle Plot of Training Data", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the plot as a PNG file in the current directory.
plot_filename = "training_data_triangle.png"
plt.savefig(plot_filename)
print(f"Triangle plot saved as {plot_filename}")
plt.show()
