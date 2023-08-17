"""Unit tests for probability_density_functions.py"""

import probability_density_functions as pdfs
import numpy as np


def test_get_samples_gaussian():
    samples = pdfs.get_samples_gaussian(10000, 100)
    assert np.all(np.linalg.norm(samples, axis=1) - 1 < 1e-9)
