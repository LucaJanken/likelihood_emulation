import os
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Analytical chi² function.
# -----------------------------------------------------------------------------
def chi_squared_sterile_neutrino(N_eff, m0):
    return (N_eff * m0) ** 2

# -----------------------------------------------------------------------------
# Load the MCMC chain from file.
# -----------------------------------------------------------------------------
chain_file = os.path.join("mcmc_plots", "mcmc_chain.txt")
if not os.path.exists(chain_file):
    raise FileNotFoundError(f"MCMC chain file not found: {chain_file}")

# The chain file is expected to have columns:
# multiplicity, chi², param_1, param_2 (with a header starting with '#')
chain_data = np.loadtxt(chain_file, comments='#')

# -----------------------------------------------------------------------------
# Sampling functions
# -----------------------------------------------------------------------------
def random_sample_from_chain(chain, n_samples):
    """
    Randomly sample n_samples rows from the chain without weighting.
    Sampling is done without replacement if possible.
    """
    if len(chain) >= n_samples:
        indices = np.random.choice(len(chain), size=n_samples, replace=False)
    else:
        indices = np.random.choice(len(chain), size=n_samples, replace=True)
    return chain[indices]

def flat_sample_from_chain_no_scaling(chain, n_samples, T=5.0):
    """
    Flat sampling using raw weighting based on the traditional likelihood.
    
    The MCMC chain was generated with a likelihood:
      L_T ∝ exp(-χ²/(2T))
    so each sample appears with multiplicity proportional to L_T.
    
    To recover a flat (uniform) sample we assign each sample a weight:
      weight = multiplicity * exp(χ²/(2T))
    """
    multiplicities = chain[:, 0]
    chi2_vals = chain[:, 1]
    
    weights = multiplicities * np.exp(chi2_vals / (2 * T))
    weights /= np.sum(weights)
    
    if len(chain) >= n_samples:
        indices = np.random.choice(len(chain), size=n_samples, replace=False, p=weights)
    else:
        indices = np.random.choice(len(chain), size=n_samples, replace=True, p=weights)
    return chain[indices]

def flat_sample_from_chain_soft(chain, n_samples, alpha=0.3, T=5.0):
    """
    Flat sampling with a softening parameter alpha.
    
    Here the weight is defined as:
      weight = multiplicity * exp(alpha * χ²/(2T))
    With alpha=1 you recover the full reweighting,
    and alpha=0 gives a weighting based solely on multiplicity.
    """
    multiplicities = chain[:, 0]
    chi2_vals = chain[:, 1]
    
    weights = multiplicities * np.exp(alpha * chi2_vals / (2 * T))
    weights /= np.sum(weights)
    
    if len(chain) >= n_samples:
        indices = np.random.choice(len(chain), size=n_samples, replace=False, p=weights)
    else:
        indices = np.random.choice(len(chain), size=n_samples, replace=True, p=weights)
    return chain[indices]

# -----------------------------------------------------------------------------
# Generate samples from the chain.
# -----------------------------------------------------------------------------
n_samples = 250
random_samples = random_sample_from_chain(chain_data, n_samples)
flat_samples_raw = flat_sample_from_chain_no_scaling(chain_data, n_samples, T=5.0)
alpha = 0.15  # Adjust this value to soften the weighting effect.
flat_samples_soft = flat_sample_from_chain_soft(chain_data, n_samples, alpha=alpha, T=5.0)

# Extract parameters (columns 2 and 3)
params_random = random_samples[:, 2:4]
params_flat_raw = flat_samples_raw[:, 2:4]
params_flat_soft = flat_samples_soft[:, 2:4]

# -----------------------------------------------------------------------------
# Create a contour grid for the analytical chi² function.
# -----------------------------------------------------------------------------
N_eff_range = np.linspace(0, 10, 200)
m0_range = np.linspace(0, 10, 200)
X, Y = np.meshgrid(N_eff_range, m0_range)
Z = chi_squared_sterile_neutrino(X, Y)

# -----------------------------------------------------------------------------
# Plotting: Create three separate subplots.
# -----------------------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7), sharey=True)

# Subplot 1: Randomly-sampled points.
cf1 = ax1.contourf(X, Y, Z, levels=50, cmap='viridis')
ax1.contour(X, Y, Z, levels=10, colors='k', linewidths=0.5)
ax1.scatter(params_random[:, 0], params_random[:, 1],
            color='blue', s=10, alpha=0.6)
ax1.set_title("Randomly-Sampled Points")
ax1.set_xlabel(r'$N_{\mathrm{eff}}$')
ax1.set_ylabel(r'$m_0$')
fig.colorbar(cf1, ax=ax1, label=r'$\chi^2$')

# Subplot 2: Flat-sampled points with raw weighting.
cf2 = ax2.contourf(X, Y, Z, levels=50, cmap='viridis')
ax2.contour(X, Y, Z, levels=10, colors='k', linewidths=0.5)
ax2.scatter(params_flat_raw[:, 0], params_flat_raw[:, 1],
            color='blue', s=10, alpha=0.6)
ax2.set_title(r"Flat-Sampled (Raw Weighting: $w=\mathrm{mult}\,\exp(\chi^2/(2T))$)")
ax2.set_xlabel(r'$N_{\mathrm{eff}}$')
fig.colorbar(cf2, ax=ax2, label=r'$\chi^2$')

# Subplot 3: Flat-sampled points with soft weighting.
cf3 = ax3.contourf(X, Y, Z, levels=50, cmap='viridis')
ax3.contour(X, Y, Z, levels=10, colors='k', linewidths=0.5)
ax3.scatter(params_flat_soft[:, 0], params_flat_soft[:, 1],
            color='blue', s=10, alpha=0.6)
ax3.set_title(r"Soft-Weighted Flat Sampling: $w=\mathrm{mult}\,\exp(\alpha\,\chi^2/(2T))$, $\alpha=%s$" % alpha)
ax3.set_xlabel(r'$N_{\mathrm{eff}}$')
fig.colorbar(cf3, ax=ax3, label=r'$\chi^2$')

plt.suptitle(r'Comparison of Sampling Methods over $\chi^2=(N_{\mathrm{eff}}\cdot m_0)^2$', fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# -----------------------------------------------------------------------------
# Save and display the plot.
# -----------------------------------------------------------------------------
output_plot = os.path.join("mcmc_plots", "three_sampling_methods_contour_with_weights.png")
plt.savefig(output_plot)
print(f"Plot saved to {output_plot}")
plt.show()
