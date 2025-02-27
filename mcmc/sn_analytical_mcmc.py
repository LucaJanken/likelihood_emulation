import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from matplotlib.ticker import MaxNLocator

# -----------------------------------------------------------------------------
# Analytical chi² function.
# -----------------------------------------------------------------------------
def chi_squared_analytical(point):
    # point is [N_eff, m0]
    return (point[0] * point[1])**2

# -----------------------------------------------------------------------------
# Log-likelihood using the traditional formulation:
# -log L = chi²/2  so that log L = -chi²/2.
# -----------------------------------------------------------------------------
def log_likelihood_analytical(point):
    return - chi_squared_analytical(point) / 2.0

# -----------------------------------------------------------------------------
# MCMC Sampler using the analytical function with traditional likelihood.
# -----------------------------------------------------------------------------
class MCMC2DAnalytical:
    def __init__(self, initial_point, T, sigma=None):
        """
        initial_point: list/array with two values, e.g. [N_eff, m0]
        T: effective temperature for the sampler.
        sigma: (optional) proposal standard deviations (list/array of length 2).
        """
        self.initial_point = np.array(initial_point, dtype=np.float32)
        self.current_point = self.initial_point.copy()
        self.positions = [self.current_point.copy()]  # store accepted positions
        self.T = T
        if sigma is None:
            self.sigma = np.array([0.01, 0.01], dtype=np.float32)
        else:
            self.sigma = np.array(sigma, dtype=np.float32)
        
        # Compute the initial log-likelihood value.
        self.logL = log_likelihood_analytical(self.current_point)
        self.accepted = 0
        self.total_steps = 0

    def log_likelihood(self, point):
        """Compute the analytical log-likelihood value at a given point using the traditional formulation."""
        return log_likelihood_analytical(point)
    
    def step(self):
        """Perform one MCMC step with a Gaussian proposal, enforcing that both parameters lie within [0, 10]."""
        proposal = np.random.normal(self.current_point, self.sigma)
        # Enforce physical boundaries: both N_eff and m0 must be between 0 and 10.
        if proposal[0] < 0 or proposal[1] < 0 or proposal[0] > 10 or proposal[1] > 10:
            accepted = False
        else:
            logL_new = self.log_likelihood(proposal)
            delta = logL_new - self.logL
            # Higher logL is more favorable; accept if delta > 0 or with probability exp(delta/T).
            if delta > 0 or np.exp(delta / self.T) > np.random.rand():
                self.current_point = proposal
                self.logL = logL_new
                accepted = True
            else:
                accepted = False

        self.positions.append(self.current_point.copy())
        self.total_steps += 1
        if accepted:
            self.accepted += 1
        return accepted

    def chain(self, n_steps, burn_in=1000, adaptation_interval=50, target_acceptance=0.23):
        """
        Run adaptive burn-in (to tune sigma) followed by fixed-step sampling.
        Prints block-level progress and uses a progress bar for the sampling phase.
        """
        if burn_in > 0:
            n_blocks = burn_in // adaptation_interval
            print("Starting adaptive burn-in:")
            for block in range(n_blocks):
                accepted_in_block = 0
                for _ in range(adaptation_interval):
                    if self.step():
                        accepted_in_block += 1
                block_acceptance = accepted_in_block / adaptation_interval
                factor = np.exp((block_acceptance - target_acceptance) * 0.1)
                self.sigma *= factor
                print(f" Block {block+1}/{n_blocks}: Acceptance = {block_acceptance:.2%}, sigma = {self.sigma}")
            for _ in range(burn_in % adaptation_interval):
                self.step()
            # Clear burn-in positions.
            self.positions = []
            print(f"Burn-in complete. Final sigma: {self.sigma}. Now sampling {n_steps} steps.")

        for _ in trange(n_steps, desc="Sampling"):
            self.step()

    def acceptance_rate(self):
        return self.accepted / self.total_steps if self.total_steps > 0 else 0

# -----------------------------------------------------------------------------
# Function to save MCMC chain to a text file.
# -----------------------------------------------------------------------------
def save_chain_txt(chain, filename, chi_squared_func):
    """
    Save the MCMC chain as a text file with columns: multiplicity, chi², param_1, param_2.
    Consecutive duplicate samples are grouped together and assigned a multiplicity.
    """
    with open(filename, 'w') as f:
        # Write header
        f.write("# multiplicity  chi_squared  param_1  param_2\n")
        if len(chain) == 0:
            return
        current_sample = chain[0]
        multiplicity = 1
        # Loop over chain elements (starting from the second sample)
        for sample in chain[1:]:
            if np.array_equal(sample, current_sample):
                multiplicity += 1
            else:
                chi2_val = chi_squared_func(current_sample)
                f.write(f"{multiplicity} {chi2_val} {current_sample[0]} {current_sample[1]}\n")
                current_sample = sample
                multiplicity = 1
        # Write the last group
        chi2_val = chi_squared_func(current_sample)
        f.write(f"{multiplicity} {chi2_val} {current_sample[0]} {current_sample[1]}\n")
    print(f"MCMC chain saved to {filename}")

# -----------------------------------------------------------------------------
# Triangle Plot Function: Diagonals as histograms; off-diagonals as scatter plot.
# -----------------------------------------------------------------------------
def triangle_plot(chain):
    """
    Create a triangle plot for 2D MCMC chain data:
      - Diagonals: 1D histograms for each parameter.
      - Off-diagonals: a scatter plot of the sampled points.
    The plot is saved to the 'mcmc_plots' folder.
    """
    labels = [r"$N_{\mathrm{eff}}$", r"$m_0$"]
    chain = np.array(chain)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # Histogram for parameter 1.
    axes[0, 0].hist(chain[:, 0], bins=50, density=True, color='blue', alpha=0.7)
    axes[0, 0].set_xlabel(labels[0])
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].xaxis.set_major_locator(MaxNLocator(nbins=5))
    
    # Histogram for parameter 2.
    axes[1, 1].hist(chain[:, 1], bins=50, density=True, color='blue', alpha=0.7)
    axes[1, 1].set_xlabel(labels[1])
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].xaxis.set_major_locator(MaxNLocator(nbins=5))
    
    # Scatter plot for joint distribution.
    axes[1, 0].scatter(chain[:, 0], chain[:, 1], color='blue', s=5, alpha=0.5)
    axes[1, 0].set_xlabel(labels[0])
    axes[1, 0].set_ylabel(labels[1])
    
    # Turn off the top-right subplot.
    axes[0, 1].axis('off')
    
    plt.suptitle("Triangle Plot of MCMC Chain (Analytical Function)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_dir = "mcmc_plots"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "triangle_plot_analytical.png")
    plt.savefig(plot_path)
    print(f"Triangle plot saved as {plot_path}")
    plt.show()

# -----------------------------------------------------------------------------
# Main: Run the sampler using the analytical likelihood, generate the triangle plot,
# and save the chain to a text file.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # MCMC settings.
    initial_point = [1.5, 1.5]  # Starting point (both parameters must be between 0 and 10)
    T = 5.0                   # Effective temperature; adjust as needed
    sigma = [0.05, 0.05]      # Initial proposal sigma

    # Initialize and run the sampler.
    sampler = MCMC2DAnalytical(initial_point, T, sigma=sigma)
    n_steps = 1000000
    sampler.chain(n_steps, burn_in=1000, adaptation_interval=25, target_acceptance=0.05)
    print(f"Acceptance rate: {sampler.acceptance_rate() * 100:.2f}%")
    
    # Generate the triangle plot.
    chain_data = np.array(sampler.positions)
    triangle_plot(chain_data)
    
    # Save the chain to a text file.
    output_filename = os.path.join("mcmc_plots", "mcmc_chain.txt")
    save_chain_txt(chain_data, output_filename, chi_squared_analytical)
