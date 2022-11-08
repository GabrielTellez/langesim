import numpy as np
from scipy.integrate import simps


def make_sampler(pdf, range=(-25, 25), bins=10000001):
    """Generates a sampler of random samples distributed with pdf
    using the inverse transform sampling method
    Adapted from:
    https://towardsdatascience.com/random-sampling-using-scipy-and-numpy-part-i-f3ce8c78812e


    Args:
        pdf (function(x)): probability density function to draw the
          samples. Does not need to be normalized
        range (tuple, optional): range for the random variable values. Defaults to (-25,25).
        bins (int, optional): bins to build the discretized inverse
        cumulative distribution function. Defaults to 10000001.

    Returns:
        sampler_single, sampler_multi (tuple of functions): samplers that
        provide random numbers distributed with the given pdf.
          sampler_single() returns one single value.
          sampler_multi(n) return n random values.
    """

    def normalisation(x):
        return simps(pdf(x), x)

    xs = np.linspace(*range, bins)
    # define function to normalise our pdf to sum to 1 so it satisfies a distribution
    norm_constant = normalisation(xs)
    # create pdf
    my_pdfs = pdf(xs) / norm_constant
    # create cdf then ensure it is bounded at [0,1]
    my_cdf = np.cumsum(my_pdfs)
    my_cdf = my_cdf / my_cdf[-1]

    def sampler_single():
        rand = np.random.random_sample()
        return np.interp(rand, my_cdf, xs)

    def sampler_multi(size: int):
        rand = np.random.random_sample(size)
        return np.interp(rand, my_cdf, xs)

    return sampler_single, sampler_multi
