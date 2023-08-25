"""Module for generating multi-dimensional probability density functions with 
various methods"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def get_samples_gaussian(n_samples: int, n_dims: int) -> np.ndarray:
    """Return a samples uniformly distributed on a hypersphere using the method
    of gaussian normalization. The hypersphere is of unit length and centered
    on the origin.

    George Marsaglia. "Choosing a Point from the Surface of a Sphere." Ann.
    Math. Statist. 43 (2) 645 - 646, April, 1972.
    https://doi.org/10.1214/aoms/1177692644

    Args:
        n_samples: the number of samples to generate
        n_dims: the dimension of the hypersphere surface

    Returns:
        A NxD numpy array of samples where N is the number of samples and D is
        the number of dimensions the hypersphere is embedded in. The coordinates
        for each sample are cartesian. The hypersphere is assumed to be of unit
        length and centered at the origin.
    """

    unnormalized_samples = np.random.standard_normal((n_samples, n_dims))
    samples = unnormalized_samples / np.repeat(
        np.linalg.norm(unnormalized_samples, axis=1).reshape(-1, 1),
        repeats=n_dims,
        axis=1,
    )
    return samples


def plot_cartesian_histograms(samples: np.ndarray) -> None:
    """Plot a histogram of each cartesian coordinate for an array of samples.

    Args:
        samples: an NxD numpy array of samples where N is the number of samples
            and D is the number of dimensions. Each sample is assumed to
            represent a cartesian coordinate.

    Returns:
        None
    """
    num_samples, dimensions = samples.shape
    fig, axs = plt.subplots(dimensions)

    for dim, ax in enumerate(axs):
        ax.hist(samples[:, dim], 20)
        ax.set_ylabel(f"X{dim}")
    ax.set_xlabel("Location")
    fig.suptitle(f"Cartesian Histograms | {num_samples} Samples")


def plot_spherical_histograms(samples: np.ndarray) -> None:
    """Plot a histogram of each spherical coordinate for an array of samples.

    Args:
        samples: an NxD numpy array of samples where N is the number of samples
            and D is the number of dimensions. Each sample is assumed to
            represent a cartesian coordinate.

    Returns:
        None
    """
    num_samples, dimensions = samples.shape

    fig, axs = plt.subplots(dimensions)
    bins = 10

    for dim, ax in enumerate(axs):
        ax.hist(samples[:, dim], bins)
        if dim < dimensions - 1 and dim > 0:
            domain = np.linspace(0, np.pi, 1000)
            ax.plot(domain, np.sin(domain) * num_samples / bins * 2)

        if dim == 0:
            ax.set_ylabel("r")
        else:
            ax.set_ylabel(r"$\phi_{0}$".format(dim))
    ax.set_xlabel("Angle")
    fig.suptitle(f"Spherical Histograms | {num_samples} Samples")


def cartesian_to_spherical(cartesian_samples: np.ndarray) -> np.ndarray:
    """Given cartesian coordinates, return spherical coordinates

    Args:
        cartesian_samples: an NxD numpy array of cartesian samples where N is
            the number of samples and D is the number of dimensions.

    Returns:
        An NxD numpy array of samples in hyperspherical coordinates where N is
        the number of samples and D is the number of dimensions. The spherical
        dimensions are ordered as [r, phi_1, phi_2, ..., phi_n-1] where r is
        the radius, and phi_d is the angle from the dth cartesian axis to the
        projection of the sample onto the plane spanned by x_d and x_d+1"""
    _, dimensions = cartesian_samples.shape

    r = np.linalg.norm(cartesian_samples, axis=1)
    sphere_coords = [r]

    for dim in range(dimensions - 2):
        projected_length = np.linalg.norm(cartesian_samples[:, dim + 1 :], axis=1)
        phi = np.arctan2(projected_length, cartesian_samples[:, dim])
        sphere_coords.append(phi)

    # The final coordinate has a larger domain, so we use the half-angle formula
    sphere_coords.append(
        2
        * np.arctan2(
            cartesian_samples[:, -1],
            cartesian_samples[:, -2]
            + np.linalg.norm(cartesian_samples[:, -2:], axis=1),
        )
    )

    spherical_samples = np.array(sphere_coords).T
    return spherical_samples


def spherical_to_cartesian(spherical_samples: np.ndarray) -> np.ndarray:
    """Given spherical coordinates, return cartesian coordinates

    Args:
        spherical_samples: An NxD numpy array of samples in hyperspherical
        coordinates where N is the number of samples and D is the number of
        dimensions. The spherical dimensions are ordered as [r, phi_1, phi_2,
        ..., phi_n-1] where r is the radius, and phi_d is the angle from the
        dth cartesian axis to the projection of the sample onto the plane
        spanned by x_d and x_d+1.

    Returns:
        An NxD numpy array of cartesian samples where N is the number of samples
        and D is the number of dimensions."""
    _, dimensions = spherical_samples.shape
    cartesian_coords = []

    sin_chain = spherical_samples[:, 0]
    for dim in range(dimensions - 1):
        phi = spherical_samples[:, dim + 1]
        x = sin_chain * np.cos(phi)
        cartesian_coords.append(x)
        sin_chain = sin_chain * np.sin(phi)
    cartesian_coords.append(sin_chain)
    cartesian_samples = np.array(cartesian_coords).T

    return cartesian_samples
