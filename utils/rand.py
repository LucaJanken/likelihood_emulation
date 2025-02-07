import numpy as np

# Define chi^2 function for double Gaussian
def chi_squared_double_gaussian(x, y, bestfit_point1, bestfit_point2, sigma):
    x1, y1 = bestfit_point1
    x2, y2 = bestfit_point2
    
    term1 = np.exp(-0.5 * ((x - x1) ** 2 + (y - y1) ** 2) / sigma**2)
    term2 = np.exp(-0.5 * ((x - x2) ** 2 + (y - y2) ** 2) / sigma**2)
    likelihood = (1 / (2 * np.pi * sigma**2)) * (term1 + term2)
    
    chi2 = -2 * np.log(likelihood)
    return chi2

# Function parameters
bestfit_point1 = np.array([-1.0, -1.0])
bestfit_point2 = np.array([1.0, 1.0])
sigma = np.sqrt(0.1)

print(chi_squared_double_gaussian(-1, -1, bestfit_point1, bestfit_point2, sigma))
print(chi_squared_double_gaussian(1, 1, bestfit_point1, bestfit_point2, sigma))