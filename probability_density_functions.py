"""Module for generating multi-dimensional probability density functions with 
various methods"""

import numpy as np
from scipy import stats


def get_samples_gaussian(n_samples: int, n_dims: int) -> np.ndarray:
    """Return a samples uniformly distributed on a hypersphere using the method
    of gaussian normalization.

    Args:
        n_samples: the number of samples to generate
        n_dims: the dimension of the hypersphere surface

    Returns:
        A NxD numpy array of samples where N is the number of samples and D is
        the number of dimensions the hypersphere is embedded in.
    """

    unnormalized_samples = np.random.standard_normal((n_samples, n_dims))
    samples = unnormalized_samples / np.linalg.norm(unnormalized_samples, axis=1)
    return samples
