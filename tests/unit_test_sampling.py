"""Unit tests for probability_density_functions.py"""

import sampling
import numpy as np


def test_get_samples_gaussian():
    samples = sampling.get_samples_gaussian(10000, 100)
    assert np.all(np.linalg.norm(samples, axis=1) - 1 < 1e-9)


def test_coordinate_transformation():
    samples = sampling.get_samples_gaussian(10000, 100)
    spherical_coords = sampling.cartesian_to_spherical(samples)
    cartesian_coords = sampling.spherical_to_cartesian(spherical_coords)
    assert np.all(np.abs(samples - cartesian_coords) < 1e-9)
    assert samples.shape == spherical_coords.shape == cartesian_coords.shape
