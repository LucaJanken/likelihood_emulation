import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the simplified likelihood function with isotropic Gaussians
def likelihood(theta1, theta2, mu1, mu2, mu1_prime, mu2_prime, sigma):
    term1 = (1 / (2 * np.pi * sigma**2)) * np.exp(-0.5 * ((theta1 - mu1)**2 + (theta2 - mu2)**2) / sigma**2)
    term2 = (1 / (2 * np.pi * sigma**2)) * np.exp(-0.5 * ((theta1 - mu1_prime)**2 + (theta2 - mu2_prime)**2) / sigma**2)
    return term1 + term2

# Define the chi-squared function
def chi_squared(likelihood_values):
    chi2 = -2 * np.log(likelihood_values)
    chi2[np.isinf(chi2)] = np.nan  # Avoid log(0) issues by setting to NaN
    return chi2

# Parameters
mu1, mu2 = -2, 0
mu1_prime, mu2_prime = 2, 0
sigma = 1

# Define the grid of values
theta1_vals = np.linspace(-6, 6, 200)
theta2_vals = np.linspace(-6, 6, 200)
Theta1, Theta2 = np.meshgrid(theta1_vals, theta2_vals)
Likelihood = likelihood(Theta1, Theta2, mu1, mu2, mu1_prime, mu2_prime, sigma)
Chi2 = chi_squared(Likelihood)

# Plot the likelihood function
fig1 = plt.figure(figsize=(10, 7))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_surface(Theta1, Theta2, Likelihood, cmap='viridis', edgecolor='none')

ax1.set_xlabel(r'$\theta_1$')
ax1.set_ylabel(r'$\theta_2$')
ax1.set_zlabel(r'$\mathcal{L}(\theta_1, \theta_2)$')
#ax1.set_title('3D Plot of Simplified Likelihood Function')

plt.savefig('likelihood_plot.png')

# Plot the chi-squared function
fig2 = plt.figure(figsize=(10, 7))
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_surface(Theta1, Theta2, Chi2, cmap='plasma', edgecolor='none')

ax2.set_xlabel(r'$\theta_1$')
ax2.set_ylabel(r'$\theta_2$')
ax2.set_zlabel(r'$\chi^2(\theta_1, \theta_2)$')
#ax2.set_title('3D Plot of Corresponding Chi-Squared Function')


# Save and show the plot
plt.tight_layout()
plt.savefig('double_gaussian_plot.png')
plt.show()