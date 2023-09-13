"""Module for modeling the probabilites as opposed to simulating"""

from itertools import combinations
import numpy as np
from scipy import special as spec


def prob_of_crossing_single_set(lengths: np.array, dim: int, C: int, S: int):
    """Calculate the probability of crossing more than some number of
    hyperplanes with a hyperdimensional needle. Assume there is only a
    single set of parallel hyperplanes.

    Args:
        lengths: an array of lengths of the needle
        dim: the dimension of the R^D space that the needle is embedded in
        C: the number of hyperplanes the needle must cross
        S: the spacing between the parallel hyperplanes

    Returns:
        A numpy array the same length as `lengths` with the probability of
        meeting the crossing criteria."""
    if dim <= 1:
        raise ValueError("dim must be greater than 1")
    i_s = lengths < (S * (C - 1))
    i_m = (lengths >= (S * (C - 1))) * (lengths < (S * C))
    i_l = lengths >= (S * C)

    r_s = lengths[i_s]
    r_m = lengths[i_m]
    r_l = lengths[i_l]

    prob_s = np.zeros_like(r_s)
    prob_m = prob_of_crossing_single_set_small_r(r_m, dim, C, S)
    prob_l = prob_of_crossing_single_set_small_r(
        r_l, dim, C, S
    ) - prob_of_crossing_single_set_small_r(r_l, dim, C + 1, S)

    probability = np.concatenate((prob_s, prob_m, prob_l))
    return probability


def prob_of_crossing_single_set_small_r(lengths, dim, C, S):
    """Calculate the probability of crossing more than some number of
    hyperplanes with a hyperdimensional needle. Assume there is only a
    single set of parallel hyperplanes and that the length is less than
    C*S. The lengths need not be actually less than C*S.

    Args:
        lengths: an array of lengths of the needle
        dim: the dimension of the R^D space that the needle is embedded in
        C: the number of hyperplanes the needle must cross
        S: the spacing between the parallel hyperplanes

    Returns:
        A numpy array the same length as `lengths` with the probability of
        meeting the crossing criteria."""
    r = lengths
    gamma = S * (C - 1) / r
    loop_sum = np.sum(
        [
            spec.beta((dim - 2 * i) / 2, 0.5) / (1 - gamma**2) ** i
            for i in range(1, int((dim - 2) / 2) + 1)
        ],
        axis=0,
    )
    if dim % 2 == 0:
        parity_shape = 2 / np.pi * np.arccos(gamma)
    else:
        parity_shape = 1 - gamma
    probability = (
        r
        / S
        * (
            (1 - gamma**2) ** ((dim - 1) / 2)
            / np.pi
            * (spec.beta(dim / 2, 0.5) + gamma**2 * loop_sum)
            - gamma * parity_shape
        )
    )
    return probability


def sum_of_combo_reciprocals(S: np.array, num_choose: int):
    """Given an array of array of values, return the sum of the reciprocals of
    the combo products.

    eg. S = [1, 2, 3], num_choose = 2: return 1/(1*2) + 1/(1*3) + 1/(2*3)"""
    reciprocal_sum = 0
    for combos in combinations(S, num_choose):
        reciprocal_sum += 1 / np.prod(combos)
    return reciprocal_sum


def prob_of_crossing_small_r(lengths: np.array, dim: int, c: int, N: int, S: np.array):
    """Return the probability of crossing exactly c hyperplanes.

    Args:
        - lengths: lengths of the line segment to calculate the prob of.
            For now must be <= all S
        - dim: the dimension of the space the needle is embedded in
        - c: the number of hyperplanes
        - N: the number of sets of parallel D-1 dimensional hyperplanes
        - S: the spacing for each of the N hyperplanes. Must have length N.

    Returns:
        The probability that the line segment has at least 1 crossing in all
        sets of hyperplanes."""
    probability = 0
    for idx in range(N - c + 1):
        probability += (
            (-1) ** idx
            * lengths ** (c + idx)
            * spec.gamma(dim / 2)
            / (c + idx + 1)
            / np.pi ** ((c + idx) / 2)
            / spec.beta(c + 1, idx + 1)
            / spec.gamma((dim + c + idx) / 2)
            * sum_of_combo_reciprocals(S, c + idx)
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
