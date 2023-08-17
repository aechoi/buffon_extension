"""Module for generating multi-dimensional probability density functions with 
various methods"""

import numpy as np
from scipy import stats


def get_samples_gaussian(n_samples: int, n_dims: int) -> np.ndarray:
    """Return a samples uniformly distributed on a hypersphere using the method
    of gaussian normalization. The hypersphere is of unit length and centered
    on the origin.

    Args:
        n_samples: the number of samples to generate
        n_dims: the dimension of the hypersphere surface

    Returns:
        A NxD numpy array of samples where N is the number of samples and D is
        the number of dimensions the hypersphere is embedded in. The hypersphere
        is assumed to be of unit length and centered at the origin.
    """

    unnormalized_samples = np.random.standard_normal((n_samples, n_dims))
    samples = unnormalized_samples / np.repeat(
        np.linalg.norm(unnormalized_samples, axis=1).reshape(-1, 1),
        repeats=n_dims,
        axis=1,
    )
    return samples
