import numpy as np


def piecewise1():
    """
    Generates a piecewise function and its domain
    """
    x_start = 0
    x_stop = 3

    def f(x):
        return np.piecewise(
            x, [x < 1.7, x >= 1.7],
            [lambda x: x**2.5, lambda x: 0.9*x**1.2])

    return (x_start, x_stop), f


def piecewise2():
    """
    Generates a piecewise function and its domain
    """
    x_start = 0
    x_stop = 10

    def f(x):
        return np.piecewise(
            x, [x < 5, x >= 5],
            [lambda x: np.sin(x), lambda x: np.sin(x) + 3])

    return (x_start, x_stop), f


def piecewise3():
    """
    Generates a piecewise function and its domain
    """
    x_start = 0
    x_stop = 3

    def f(x):
        return np.piecewise(
            x, [x < 1.2, (x >= 1.2) & (x < 2.2), x >= 2.2],
            [lambda x: 1 * x**2.5,
             lambda x: 0.5 * x**1.5 - 2,
             lambda x: 1 * np.exp(0.5 * (x - 2)) + 1]
             )

    return (x_start, x_stop), f


def nonstationary1():
    """
    Generates a nonstationary function and its domain
    """
    x_start = 0
    x_stop = 20

    def f(x):
        return np.where(x < 10, np.sin(np.pi * x / 5) + 0.2 * np.cos(4 * np.pi * x / 4.9), x / 15 - 1)

    return (x_start, x_stop), f


def nonstationary2():
    """
    Generates a nonstationary function and its domain
    """
    x_start = -7
    x_stop = 7

    def f(x):
        y_smooth = np.sin(0.7 * x) * (np.abs(x) >= 2)
        y_non_smooth = np.sin(10 * x) * np.exp(-np.abs(2 * x)) * (np.abs(x) < 2)
        return y_smooth + y_non_smooth

    return (x_start, x_stop), f


def nonstationary3():
    """
    Generates a nonstationary function and its domain
    """
    x_start = 0
    x_stop = 10

    def f(x):
        spike_params = [
            (1, 2, 0.1),   # (amplitude, position, width)
            (-0.5, 4, 0.15),
            (1.5, 6, 0.1),
            (-1, 8, 0.1),
            (0.75, 9, 0.1)
        ]
        # Generate the smooth base curve
        y = np.sin(x)
        # Add each spike to the base curve
        for a, b, c in spike_params:
            y += a * np.exp(-((x - b)**2) / (2 * c**2))
        return y

    return (x_start, x_stop), f


def rays2d(grid_density=50):
    """
    Generates a function to compute the intensity of rays over a 2D domain, given a set of indices. Returns
    the domain of points and the generated function. This function is designed for scenarios where the evaluation
    of phase transition behavior is needed at specific points within a larger, predefined domain,
    such as in optimization or active learning contexts.
    """
    origin = (0, 0)  # Origin of the rays
    num_rays = 30  # Total number of rays
    width_factor_range = (0.03, 0.15)  # Range for the width factors of the rays
    intensity_range = (1, 3)  # Range for the intensity factors

    # Define a domain of points
    x_values = np.linspace(0.5, 5, grid_density)
    y_values = np.linspace(0.5, 5, grid_density)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    X_domain = np.vstack((x_grid.flatten(), y_grid.flatten())).T

    # Pre-calculate width and intensity factors for consistency
    np.random.seed(42)
    width_factors = np.random.uniform(*width_factor_range, size=num_rays)
    intensity_factors = np.random.uniform(*intensity_range, size=num_rays)

    def rays_function(indices):
        """
        Calculates the intensity of rays at specified indices within the domain.

        Parameters:
        - indices (np.ndarray): An array of indices for which to calculate the ray intensities.
                                These indices correspond to points in `X_domain`.

        Returns:
        - np.ndarray: An array of ray intensities for the specified indices.
        """
        x, y = X_domain[indices].T

        angles = np.arctan2(y - origin[1], x - origin[0]) % (2 * np.pi)
        distance = np.sqrt((x - origin[0])**2 + (y - origin[1])**2)

        rays = np.zeros_like(angles)
        for i in range(num_rays):
            angle_position = angles / (2 * np.pi) * num_rays
            distance_to_ray_center = np.abs(angle_position - i)
            distance_to_ray_center = np.minimum(distance_to_ray_center, num_rays - distance_to_ray_center)
            smoothing = np.exp(- (distance_to_ray_center**2) / (2 * width_factors[i]**2))
            intensity_modulation = 2 * (np.cos(distance * intensity_factors[i]) + 1.5) / 2
            rays += smoothing * intensity_modulation

        return rays

    return X_domain, rays_function


def phases2d(grid_density=50):
    """
    Generates a function to emulate a behavior of a 2D phase transition system at specified indices within a predefined domain.
    Returns the domain of points and the generated function. This function is designed for scenarios where the evaluation
    of phase transition behavior is needed at specific points within a larger, predefined domain, such as in optimization
    or active learning contexts.
    """
    # Define a domain of points
    X = np.linspace(-3, 3, grid_density)
    Y = np.linspace(-3, 3, grid_density)
    XX, YY = np.meshgrid(X, Y)
    X_domain = np.vstack((XX.flatten(), YY.flatten())).T

    def phases2d_function(indices):
        """
        Computes the values of a 2D phase transition function at points specified by their indices within the predefined domain.

        Parameters:
        - indices (np.ndarray): An array of indices for which to calculate the phase transition values. These indices correspond
                                to points in `X_domain`.

        Returns:
        - np.ndarray: The function values at the specified points, reflecting the distinct behaviors above and below the
                      transition curve.
        """
        # Select the points based on the provided indices and transpose for easy unpacking
        x, y = X_domain[indices].T

        curve = np.sin(x)
        above_curve = y > curve

        # Polynomial behavior for points above the curve
        z_above = 0.1 * x**2 + 0.5 * y**2

        # Exponential decay for points below the curve
        z_below = np.exp(-(x**2 + y**2))

        # Combine the two regions based on the position relative to the curve
        z = np.where(above_curve, z_above, z_below)

        return z

    return X_domain, phases2d_function


def get_ising_data(array_key):
    data = np.load("/home/ubuntu/code/NeuroBayes/ising_data.npz")
    array = data[array_key]

    height, width = array.shape
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    X_full = np.column_stack((y_coords.ravel(), x_coords.ravel()))
    y_full = array.ravel()

    return X_full, y_full

def ising_mag():
    return get_ising_data("magnetization")

def ising_heat():
    return get_ising_data("heat_capacity")
