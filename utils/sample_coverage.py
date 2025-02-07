import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs

# Define sample ranges
x_min, x_max = -4, 4
y_min, y_max = -4, 4
circle_radius = 4
num_samples = 2000

# Generate Latin Hypercube samples
valid_samples = []
while len(valid_samples) < num_samples:
    lhs_samples = lhs(2, samples=num_samples)
    x_samples = lhs_samples[:, 0] * (x_max - x_min) + x_min
    y_samples = lhs_samples[:, 1] * (y_max - y_min) + y_min
    
    distances = np.sqrt(x_samples**2 + y_samples**2)
    inside_circle = distances <= circle_radius
    valid_samples.extend(zip(x_samples[inside_circle], y_samples[inside_circle]))
    valid_samples = valid_samples[:num_samples]

params = np.array(valid_samples)

# Define the function for visualization
def neg_log_double_gaussian_2D(x, y):
    bestfit_point1 = np.array([-1.0, 0.0])
    bestfit_point2 = np.array([1.0, 0.0])
    sigma = np.sqrt(0.1)
    C_inv = 1 / sigma**2
    Q1 = C_inv * ((x - bestfit_point1[0])**2 + (y - bestfit_point1[1])**2)
    Q2 = C_inv * ((x - bestfit_point2[0])**2 + (y - bestfit_point2[1])**2)
    likelihood = (1 / (2 * np.pi * sigma**2)) * (np.exp(-0.5 * Q1) + np.exp(-0.5 * Q2))
    return -2 * np.log(likelihood)

# Generate a grid for visualization
x = np.linspace(x_min, x_max, 100)
y = np.linspace(y_min, y_max, 100)
X, Y = np.meshgrid(x, y)
Z = neg_log_double_gaussian_2D(X, Y)

# Plot sample coverage and function contours
plt.figure(figsize=(8, 6))
plt.scatter(params[:, 0], params[:, 1], s=5, color='blue', alpha=0.5, label='Sampled Points')
plt.contour(X, Y, Z, levels=20, cmap='hot', alpha=0.6)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sample Coverage and Function Contours')
plt.legend()
plt.grid(True)
plt.savefig('sample_coverage.png')
plt.show()
