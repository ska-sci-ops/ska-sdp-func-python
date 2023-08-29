# pylint: disable=invalid-name
"""
Utilities to support ionospheric calibration and the generation of ionospheric
phase screens.
"""

import logging
import math

import numpy

log = logging.getLogger("func-python-logger")


def decompose_phasescreen(x, y, r_0, beta=5.0 / 3.0):
    """
    Generation and eigen-decomposition of pierce-point covariance matrix with
    Kolmogorov statistics. Used for generation of turbulent phase shifts and
    phase screens. Based on the the algorithm from "Fast simulation of a
    kolmogorov phase screen," Harding, Johnston & Lane (1999) Applied Optics,
    38 (11).

    This can be used to generate random phase shfits for pierce points between
    stations and radio sources, however the processing time increases rapidly
    as the number of pierce points increases. For many pierce points it is
    recommended to use this function to generate phase shifts for vertices of
    a coarse 2D grid, and functions interpolate_phasescreen and
    displace_phasescreen to extend the Kolmogorov statistics to higher
    resolutions. Such grids also allow for sidereal motion of pierce points
    across them, resulting in time variation of the phases.

    :param x: coordinates of 1st pierce-point dimension
    :param y: coordinates of 2nd pierce-point dimension
    :param r_0: diffractive scale [m]
    :param beta: exponent of the power spectrum (defaults to 5/3)
    :return: eigenvector matrix
    :return: sqrt(eigenvalues) vector

    """

    log.debug("generating phase screen eigenvectors and eigenvalues")

    # stack 2D array into a vector
    N = numpy.size(x)
    x = numpy.reshape(x, N)
    y = numpy.reshape(y, N)

    # calculate pierce-point offsets
    u = numpy.tile(x[:, numpy.newaxis], (1, N)) - numpy.tile(
        x[numpy.newaxis, :], (N, 1)
    )
    v = numpy.tile(y[:, numpy.newaxis], (1, N)) - numpy.tile(
        y[numpy.newaxis, :], (N, 1)
    )
    D = (numpy.sqrt(u * u + v * v) / r_0) ** beta
    Dint = numpy.mean(D, axis=0)
    Dsum = numpy.tile(Dint[:, numpy.newaxis], (1, N)) + numpy.tile(
        Dint[numpy.newaxis, :], (N, 1)
    )
    # and the final covariance matrix
    C = -0.5 * D + 0.5 * Dsum

    # eigen decomposition
    S, U = numpy.linalg.eigh(C, UPLO="U")
    S[0] = 0.0

    return U, numpy.sqrt(S)


def standardise_eigenvectors(evec):
    """
    Different eigenvector solvers genearte vectors with arbitrary scaling by
    +1 or -1. And for numpy, this has been seen to change from one version to
    the next. While the scaling is arbitrary and won't affect the desired
    statistics, it will affect the resulting phase screens. This routine
    standardises the scaling to avoid randomness during testing.

    :param evec: eigenvector matrix to standardise
    """

    # scale to make the first element of each vector positive
    # for k in range(evec.shape[1]):
    #     evec[:,k] *= numpy.sign(evec[0,k])
    evec *= numpy.sign(evec[0, :])


def interpolate_phasescreen(input_screen):
    """
    Bilinear interpolation of a two dimensinal phase screen. This could, for
    instance, be a screen generated using decompose_phasescreen and the new
    interpolated phases will in general have extra phase distortions added in a
    subsequent displace_phasescreen call to maintain Kolmogorov statistics at
    the new resoluton. It follows the approach described in  "Fast simulation
    of a kolmogorov phase screen," Harding, Johnston & Lane (1999) Applied
    Optics, 38 (11).

    :param input_screen: input phase screen. The NxN screen should have odd N
    :return: interpolated phase screen

    """
    # increase resolution by 2x using standard bilinear interpolation
    Nside = 2 * input_screen.shape[0] - 1
    interpolated_screen = numpy.zeros((Nside, Nside))

    # set the known phases
    interpolated_screen[0:Nside:2, 0:Nside:2] = input_screen

    Nsub1 = Nside - 1
    Nsub2 = Nside - 2
    Nsub3 = Nside - 3

    # interpolate new edge phases -- each an average of two input edge phases
    # edge columns:
    interpolated_screen[1:Nsub1:2, [0, -1]] = 0.5 * (
        input_screen[1:, [0, -1]] + input_screen[:-1, [0, -1]]
    )
    # edge rows:
    interpolated_screen[[0, -1], 1:Nsub1:2] = 0.5 * (
        input_screen[[0, -1], 1:] + input_screen[[0, -1], :-1]
    )

    # interpolate centre of squares -- each an average of four corners
    # new phase between input rows and columns:
    interpolated_screen[1:Nside:2, 1:Nside:2] = 0.25 * (
        interpolated_screen[0:Nsub1:2, 0:Nsub1:2]
        + interpolated_screen[0:Nsub1:2, 2:Nside:2]
        + interpolated_screen[2:Nside:2, 0:Nsub1:2]
        + interpolated_screen[2:Nside:2, 2:Nside:2]
    )

    # interpolate centre of diamonds -- each an average of four corners
    # on input columns and between input rows:
    interpolated_screen[1:Nside:2, 2:Nsub2:2] = 0.25 * (
        interpolated_screen[1:Nside:2, 1:Nsub3:2]
        + interpolated_screen[1:Nside:2, 3:Nsub1:2]
        + interpolated_screen[0:Nsub2:2, 2:Nsub2:2]
        + interpolated_screen[2:Nside:2, 2:Nsub2:2]
    )
    # on input rows and between input columns:
    interpolated_screen[2:Nsub2:2, 1:Nside:2] = 0.25 * (
        interpolated_screen[1:Nsub3:2, 1:Nside:2]
        + interpolated_screen[3:Nsub1:2, 1:Nside:2]
        + interpolated_screen[2:Nsub2:2, 0:Nsub2:2]
        + interpolated_screen[2:Nsub2:2, 2:Nside:2]
    )

    return interpolated_screen


def displace_phasescreen(interpolated_screen, res, r_0, beta):
    """
    Add extra random phase shifts to phasescreen elements interpolated with
    interpolate_phasescreen. This follows the midpoint displacement method
    described in "Simulation of a Kolmogorov phase screen," Lane, Glindemann
    & Dainty (1992) Waves in Random Media, 2, 209--224.

    The bilinear interpolation from interpolate_phasescreen results in new
    points that have a certain amount of extra variance relative to the points
    being interpolated. And this extra variance will be different from that
    required to maintain the structure function.

    Midpoint Displacement is the technique of adding a little more variation to
    be consistent with r_0 and beta at the interpolated scale.

    For the interpolated centres of squares in interpolate_phasescreen, the
    four initial points are each assumed to be a gaussian random variable plus
    a second random variable connecting it to the sample on the other size of
    the midpoint.
        a b
         m
        c d
    Midpoint m sees the independent variable of a and half the correlated
    variable for a and d, along with the independent variable of b and half
    the correlated variable for b and c. This can all be combined and compared
    with the required variation given by the structure function:
    var(separation) = (separation / r_0)**beta.

    :param interpolated_screen: input interpolated phase screen
    :param res: resolution of phase screen prior to interpolation [m]
    :param r_0: indiffractive scale [m]reen
    :param beta: exponent of the power spectrum
    :return: displaced phase screen

    """
    displaced_screen = interpolated_screen.copy()

    Nside = displaced_screen.shape[0]

    Nsub1 = Nside - 1
    Nsub2 = Nside - 2

    # new edge phases -- points interpolated were separated by res
    # edge columns:
    displaced_screen[1:Nsub1:2, [0, -1]] += numpy.random.normal(
        loc=0,
        scale=numpy.sqrt(0.0650 * (res / r_0) ** beta),
        size=displaced_screen[1:Nsub1:2, [0, -1]].shape,
    )
    # edge rows:
    displaced_screen[[0, -1], 1:Nsub1:2] += numpy.random.normal(
        0,
        numpy.sqrt(0.0650 * (res / r_0) ** beta),
        displaced_screen[[0, -1], 1:Nsub1:2].shape,
    )

    # centre of squares -- points interpolated were separated by res
    # new phase between input rows and columns:
    displaced_screen[1:Nside:2, 1:Nside:2] += numpy.random.normal(
        0,
        numpy.sqrt(0.0885 * (res / r_0) ** beta),
        displaced_screen[1:Nside:2, 1:Nside:2].shape,
    )

    # centre of diamonds -- points interpolated were separated by res/sqrt(2)
    # on input columns and between input rows:
    displaced_screen[1:Nside:2, 2:Nsub2:2] += numpy.random.normal(
        0,
        numpy.sqrt(0.0885 * (res / numpy.sqrt(2) / r_0) ** beta),
        displaced_screen[1:Nside:2, 2:Nsub2:2].shape,
    )
    # on input rows and between input columns:
    displaced_screen[2:Nsub2:2, 1:Nside:2] += numpy.random.normal(
        0,
        numpy.sqrt(0.0885 * (res / numpy.sqrt(2) / r_0) ** beta),
        displaced_screen[2:Nsub2:2, 1:Nside:2].shape,
    )

    return displaced_screen


def zern(m, n, rho, phi):
    """
    Generate Zernike polynomial values

    :param m: standard Zernike m integer (azimuthal degree)
    :param n: standard Zernike n integer (radial degree)
    :param rho: array of polar radial coordinates
    :param phi: array of polar angular coordinates
    :return array of Zernike polynomial values:

    """
    if m >= 0:
        func = numpy.cos
    else:
        func = numpy.sin
    m = numpy.abs(m)
    R = numpy.zeros(rho.shape)
    for k in range((n - m) // 2 + 1):
        R += (
            (-1) ** k
            * math.factorial(n - k)
            // (
                math.factorial(k)
                * math.factorial((n + m) // 2 - k)
                * math.factorial((n - m) // 2 - k)
            )
            * rho ** (n - 2 * k)
        )

    return R * func(m * phi)


def zern_array(nm, x, y, noll_order=False):
    """
    Generate an array of all zernike upto a given degree

    :param nm: maximum degree. If noll_order is selected, this sets the maximum
        radial degree (n), otherwise it set the maximum values of n+|m|
    :param x: array of cartesian x coordinates
    :param y: array of cartesian y coordinates
    :param noll_order: whether to limit the Zernike polynomial order by
        n (True) or n+|m| (False)
    :return array of Zernike polynomial values:

    """
    # get normalised polar coords for all of the stations in cluster 0
    x -= numpy.mean(x)
    y -= numpy.mean(y)
    phi = numpy.arctan2(y, x)
    rho = numpy.sqrt(x * x + y * y)
    rho /= numpy.amax(rho)

    # coeff = [None] * len(x)
    # for stn in range(len(x)):
    #     coeff[stn] = []
    #     count = 0
    #     if noll_order is True:
    #         for n in range(nm + 1):
    #             for m in range(-n, n + 1, 2):
    #                 coeff[stn].append(zern(m, n, rho[stn], phi[stn]))
    #                 count += 1
    #     else:
    #         for n in range(nm + 1):
    #             for m in range(-n, n + 1, 2):
    #                 if n + numpy.abs(m) > nm:
    #                     continue
    #                 coeff[stn].append(zern(m, n, rho[stn], phi[stn]))
    #                 count += 1
    # return numpy.array(coeff)

    coeff = numpy.array([])
    count = 0
    if noll_order is True:
        for n in range(nm + 1):
            for m in range(-n, n + 1, 2):
                coeff = numpy.append(coeff, zern(m, n, rho, phi))
                count += 1

    else:
        for n in range(nm + 1):
            for m in range(-n, n + 1, 2):
                if n + numpy.abs(m) > nm:
                    continue
                coeff = numpy.append(coeff, zern(m, n, rho, phi))
                count += 1

    return numpy.reshape(coeff, (count, len(x))).T
