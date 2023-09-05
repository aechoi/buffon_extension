"""Module for modeling the probabilites as opposed to simulating"""

import numpy as np
from scipy import special as spec


def prob_of_crossing_vs_length(
    lengths: np.array, dim: int, C: int = 0, N: int = 1, S=np.array([1])
):
    """Calculate the probability of crossing more than some number of
    hyperplanes with a hyperdimensional needle. Lots to do to clean this function up

    Args:
        lengths: an array of lengths of the needle
        dim: the dimension of the R^D space that the needle is embedded in
        C: the number of hyperplanes the needle must cross
        N: the number of orthogonal sets of parallel hyperplanes
        S: the spacing between the parallel hyperplanes of any given set (can
            be different for each set)

    Returns:
        A numpy array the same length as `lengths` with the probability of
        meeting the crossing criteria."""
    if dim <= 1:
        raise ValueError("dim must be greater than 1")
    if N == 1:
        i_s = lengths < S[0] * (C - 1)
        i_m = (lengths >= S[0] * (C - 1)) * (lengths < S[0] * C)
        i_l = lengths >= S[0] * C

        r_s = lengths[i_s]
        r_m = lengths[i_m]
        r_l = lengths[i_l]

        prob_s = np.zeros_like(r_s)
        prob_m = prob_of_small_r(r_m, dim, C, N, S)
        prob_l = prob_of_small_r(r_l, dim, C, N, S) - prob_of_small_r(
            r_l, dim, C + 1, N, S
        )

        probability = np.concatenate((prob_s, prob_m, prob_l))
    else:
        probability = np.zeros_like(lengths)
    return probability


def prob_of_small_r(lengths, dim, C, N, S):
    if N == 1:
        r = lengths[lengths > S[0] * (C - 1)]
        gamma = S[0] * (C - 1) / r
        loop_sum = np.sum(
            [
                spec.beta((dim - 2 * i) / 2, 0.5) / (1 - gamma**2) ** i
                for i in range(1, int((dim - 2) / 2) + 1)
            ],
            axis=0,
        )
        if dim % 2 == 0:
            parity_shape = 2 * gamma * np.arccos(gamma)
        else:
            parity_shape = np.pi * gamma * (1 - gamma)
        probability = (
            r
            / (np.pi * S[0])
            * (
                (1 - gamma**2) ** ((dim - 1) / 2)
                * (spec.beta(dim / 2, 0.5) + gamma**2 * loop_sum)
                - parity_shape
            )
        )
    return probability


def prob_of_crossing_vs_dim(
    length: int, dims: np.array, C: int = 0, N: int = 1, S=None
):
    """Calculate the probability of crossing more than some number of
    hyperplanes with a hyperdimensional needle. Lots to do to clean this function up

    Args:
        lengths: the length of the needle
        dims: the dimensions of the R^D space that the needle is embedded in
        C: the number of hyperplanes the needle must cross
        N: the number of orthogonal sets of parallel hyperplanes
        S: the spacing between the parallel hyperplanes of any given set (can
            be different for each set)

    Returns:
        A numpy array the same length as `lengths` with the probability of
        meeting the crossing criteria."""
    raise (NotImplementedError)
