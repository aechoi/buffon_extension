"""Module for modeling the probabilites as opposed to simulating"""

import numpy as np
from scipy import special as spec


def prob_of_crossing_vs_length(
    lengths: np.array, dim: int, C: int = 0, N: int = 1, S=None
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
    if dim < 3:
        return None
    if C == 0 and N == 1:
        r_s = lengths[lengths <= 1]
        r_l = lengths[lengths > 1]
        p_s = r_s * spec.factorial2(dim - 2) / spec.factorial2(dim - 1)
        if dim % 2 == 0:
            scale = 2 / np.pi
            loop_base = np.arccos(1 / r_l)
        else:
            scale = 1
            loop_base = 1 - 1 / r_l

        loop_sum = 0
        for j in range(1, 1 + int((dim - 2) / 2)):
            loop_sum += (
                spec.factorial2(dim - 2 - 2 * j)
                / spec.factorial2(dim - 1 - 2 * j)
                * (r_l**2 - 1) ** ((dim - 1 - 2 * j) / 2)
                / r_l ** (dim - 2 * j)
            )
        p_l = (
            spec.factorial2(dim - 2)
            / spec.factorial2(dim - 1)
            * r_l
            * (1 - ((r_l**2 - 1) ** 0.5 / r_l) ** (dim - 1))
            + loop_base
            - loop_sum
        )
        probability = scale * np.concatenate((p_s, p_l))

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
    if C == 0 and N == 1:
        r = length
        p_s = length * spec.factorial2(dims - 2) / spec.factorial2(dims - 1)
        if dim % 2 == 0:
            scale = 2 / np.pi
            loop_base = np.arccos(1 / r_l)
        else:
            scale = 1
            loop_base = 1 - 1 / r_l

        loop_sum = 0
        for j in range(1, 1 + int((dim - 2) / 2)):
            loop_sum += (
                spec.factorial2(dim - 2 - 2 * j)
                / spec.factorial2(dim - 1 - 2 * j)
                * (r_l**2 - 1) ** ((dim - 1 - 2 * j) / 2)
                / r_l ** (dim - 2 * j)
            )
        p_l = (
            spec.factorial2(dim - 2)
            / spec.factorial2(dim - 1)
            * r_l
            * (1 - ((r_l**2 - 1) ** 0.5 / r_l) ** (dim - 1))
            + loop_base
            - loop_sum
        )
        probability = scale * np.concatenate((p_s, p_l))

    return probability
